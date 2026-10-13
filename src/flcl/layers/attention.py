import logging
import os
import warnings

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from typing import Union, Tuple, Dict, Optional

XFORMERS_AVAILABLE = False



class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
        self.num_anchor_tokens = 0

        # --- SSC scoring variant state (training-free) ---
        # 'ffn'        : original repr_shift score for new tokens (default, unchanged behaviour)
        # 'attn'       : new-token importance = attention mass received within the current frame
        # 'attn_decay' : like 'attn', plus a per-frame-decayed attention score kept for cached
        #                tokens and combined with the cosine-diversity score during eviction.
        self.ssc_score_mode = 'ffn'
        self.decay_gamma = 0.9
        self.decay_beta = 0.5
        # Per-token importance aligned with this layer's KV cache (attn_decay only). [B, N_cache]
        self.layer_importance = None
        # Frame-level retention: anchor/recent pins (composition ablation)
        self.frame_keep_anchor = True
        self.frame_keep_recent = True
        self.frame_random_seed = 0

    def _reset_cache_state(self):
        self.num_anchor_tokens = 0
        self.layer_importance = None

    def intra_frame_prune(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        importance: torch.Tensor,
        keep_count: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Prune current frame tokens based on repr_shift importance.

        This is Stage 1 of the two-stage eviction strategy:
        - Intra-frame pruning: prune current frame tokens BEFORE storing to KV cache
        - Only affects what gets stored, not what's used for current layer attention

        Args:
            k (torch.Tensor): Current frame K [B, H, N, D]
            v (torch.Tensor): Current frame V [B, H, N, D]
            importance (torch.Tensor): Importance scores [B, N] from repr_shift
                Higher = more important = KEEP
            keep_count (int): Number of tokens to keep

        Returns:
            (pruned_k, pruned_v, kept_indices)
            pruned_k: [B, H, keep_count, D]
            pruned_v: [B, H, keep_count, D]
            kept_indices: [B, keep_count] indices of kept tokens, or None if no pruning
        """
        B, H, N, D = k.shape

        if keep_count >= N:
            return k, v, None

        # Keep at least 1 token
        keep_count = max(keep_count, 1)

        # Keep top-k by importance (higher = more important)
        _, top_indices = torch.topk(importance, k=keep_count, dim=-1)
        top_indices = top_indices.sort(dim=-1).values  # Maintain temporal order

        # Gather pruned K, V
        expanded_indices = top_indices.unsqueeze(1).unsqueeze(-1).expand(B, H, keep_count, D)
        pruned_k = torch.gather(k, 2, expanded_indices)
        pruned_v = torch.gather(v, 2, expanded_indices)

        return pruned_k, pruned_v, top_indices

    def eviction(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        cache_budget: int,
        num_anchor_tokens: int,
        importance_scores: torch.Tensor = None,
        num_new_tokens: int = 0,
        importance_weight: float = 0.5,
        window_token_count: int = 0,
        token_importance: torch.Tensor = None,
        decay_beta: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, Optional[torch.Tensor]]:
        """
        Evicts tokens from the key-value cache using hybrid scoring strategy.

        Hybrid strategy:
        - Old tokens (from past frames): use baseline (cosine diversity) weighted by (1 - importance_weight)
        - New tokens (from current frame): use repr_shift importance weighted by importance_weight

        Args:
            k (torch.Tensor): The key tensor of shape [B, H, N, D].
            v (torch.Tensor): The value tensor of shape [B, H, N, D].
            cache_budget (int): The maximum number of tokens to retain.
            num_anchor_tokens (int): The number of initial tokens to preserve.
            importance_scores (torch.Tensor): Optional [B, N_new] tensor for new tokens.
                Higher = more important = KEEP.
            num_new_tokens (int): Number of new tokens in the current frame.
            importance_weight (float): Weight for current frame importance scores (default=0.5).
                New token scores = importance_weight * normalized_importance
                Old token scores = (1 - importance_weight) * normalized_diversity

        Returns:
            A tuple of (pruned_k, pruned_v, avg_scores, kept_indices).
            kept_indices: [B, cache_budget] global indices of kept tokens, or None if no eviction.
        """
        B, H, N, D = k.shape

        if N <= cache_budget:
            return k, v, None, None

        if cache_budget <= num_anchor_tokens:
            anchor_k = k[:, :, :num_anchor_tokens, :]
            anchor_v = v[:, :, :num_anchor_tokens, :]
            anchor_indices = torch.arange(num_anchor_tokens, device=k.device).unsqueeze(0).expand(B, -1)
            return anchor_k, anchor_v, None, anchor_indices

        window_token_count = 0 if window_token_count is None else int(window_token_count)
        window_token_count = max(window_token_count, 0)
        max_tail = max(N - num_anchor_tokens, 0)
        tail_count = min(window_token_count, max(cache_budget - num_anchor_tokens, 0), max_tail)
        tail_start = N - tail_count if tail_count > 0 else N

        anchor_k = k[:, :, :num_anchor_tokens, :]
        anchor_v = v[:, :, :num_anchor_tokens, :]
        tail_k = k[:, :, tail_start:, :] if tail_count > 0 else None
        tail_v = v[:, :, tail_start:, :] if tail_count > 0 else None
        candidate_k = k[:, :, num_anchor_tokens:tail_start, :]
        candidate_v = v[:, :, num_anchor_tokens:tail_start, :]

        num_candidates = candidate_k.shape[2]
        num_to_keep_from_candidates = min(max(cache_budget - num_anchor_tokens - tail_count, 0), num_candidates)
        num_new_candidates = max(num_new_tokens - tail_count, 0)
        num_old_candidates = max(num_candidates - num_new_candidates, 0)

        # Edge case: if we can keep all candidates
        if num_to_keep_from_candidates >= num_candidates:
            return k, v, None, None
        
        # Edge case: budget is too small (occupied by anchors), keep only anchors
        if num_to_keep_from_candidates <= 0:
            anchor_indices = torch.arange(num_anchor_tokens, device=k.device).unsqueeze(0).expand(B, -1)
            if tail_count > 0:
                tail_indices = torch.arange(tail_start, N, device=k.device).unsqueeze(0).expand(B, -1)
                kept_indices = torch.cat([anchor_indices, tail_indices], dim=-1)
                final_k = torch.cat([anchor_k, tail_k], dim=2)
                final_v = torch.cat([anchor_v, tail_v], dim=2)
                return final_k, final_v, None, kept_indices
            return anchor_k, anchor_v, None, anchor_indices

        # Compute baseline scores (cosine diversity) for ALL candidates
        candidate_k_norm = F.normalize(candidate_k, p=2, dim=-1)
        mean_vector = torch.mean(candidate_k_norm, dim=2, keepdim=True)
        baseline_scores = torch.sum(candidate_k_norm * mean_vector, dim=-1)  # [B, H, N_cand]
        # Convert to diversity: lower similarity = higher diversity = keep
        baseline_diversity = 1.0 - baseline_scores  # [B, H, N_cand]
        # Average across heads for unified scoring
        baseline_diversity_avg = baseline_diversity.mean(dim=1)  # [B, N_cand]

        # Check if we can use hybrid scoring
        if importance_scores is not None and num_new_candidates > 0:
            importance_scores = importance_scores[:, :num_new_candidates]
        use_hybrid = (
            importance_scores is not None
            and importance_scores.shape[1] == num_new_candidates
            and num_new_candidates > 0
            and num_old_candidates > 0
        )

        if use_hybrid:
            # Hybrid strategy: baseline for old, repr_shift for new
            old_scores = baseline_diversity_avg[:, :num_old_candidates]  # [B, N_old]
            new_importance = importance_scores  # [B, N_new_candidates]

            # Normalize both to [0, 1] range for fair comparison
            old_min, old_max = old_scores.min(dim=-1, keepdim=True)[0], old_scores.max(dim=-1, keepdim=True)[0]
            old_normalized = (old_scores - old_min) / (old_max - old_min + 1e-8)

            # attn_decay variant: blend the (per-frame-decayed) attention importance carried by
            # cached tokens with their cosine-diversity score.
            #   old_score = beta * diversity + (1 - beta) * decayed_attn
            if token_importance is not None:
                # reuse the anchor/candidate/tail partition computed above (tail_start, num_anchor_tokens)
                cand_importance = token_importance[:, num_anchor_tokens:tail_start]  # [B, N_cand]
                old_importance = cand_importance[:, :num_old_candidates]  # [B, N_old]
                if old_importance.shape[1] == num_old_candidates and num_old_candidates > 0:
                    oi_min = old_importance.min(dim=-1, keepdim=True)[0]
                    oi_max = old_importance.max(dim=-1, keepdim=True)[0]
                    old_importance_norm = (old_importance - oi_min) / (oi_max - oi_min + 1e-8)
                    old_normalized = decay_beta * old_normalized + (1.0 - decay_beta) * old_importance_norm

            new_min, new_max = new_importance.min(dim=-1, keepdim=True)[0], new_importance.max(dim=-1, keepdim=True)[0]
            new_normalized = (new_importance - new_min) / (new_max - new_min + 1e-8)

            # Apply importance_weight: current frame weighted by importance_weight, past frames by (1 - importance_weight)
            weighted_old = (1.0 - importance_weight) * old_normalized
            weighted_new = importance_weight * new_normalized

            # Combine: [B, N_cand]
            combined_scores = torch.cat([weighted_old, weighted_new], dim=-1)
            avg_scores = combined_scores.mean().item()

            # Keep tokens with HIGHEST combined score
            _, top_indices = torch.topk(combined_scores, k=num_to_keep_from_candidates, dim=-1)
            top_indices_sorted = top_indices.sort(dim=-1).values  # Maintain temporal order

            # Expand for gather across heads: [B, H, num_to_keep, D]
            expanded_indices = top_indices_sorted.unsqueeze(1).unsqueeze(-1).expand(B, H, num_to_keep_from_candidates, D)

            # Build kept_indices
            anchor_indices = torch.arange(num_anchor_tokens, device=k.device).unsqueeze(0).expand(B, -1)
            kept_candidate_indices = top_indices_sorted + num_anchor_tokens
            if tail_count > 0:
                tail_indices = torch.arange(tail_start, N, device=k.device).unsqueeze(0).expand(B, -1)
                kept_indices = torch.cat([anchor_indices, kept_candidate_indices, tail_indices], dim=-1)
            else:
                kept_indices = torch.cat([anchor_indices, kept_candidate_indices], dim=-1)
        else:
            # Fallback: pure baseline (cosine diversity)
            avg_scores = baseline_scores.mean().item()

            # Keep tokens with LOWEST similarity (most diverse)
            _, top_indices = torch.topk(-baseline_scores, k=num_to_keep_from_candidates, dim=-1)
            top_indices_sorted = top_indices.sort(dim=-1).values

            expanded_indices = top_indices_sorted.unsqueeze(-1).expand(B, H, num_to_keep_from_candidates, D)

            # Build kept_indices (use first head's indices)
            anchor_indices = torch.arange(num_anchor_tokens, device=k.device).unsqueeze(0).expand(B, -1)
            kept_candidate_indices = top_indices_sorted[:, 0, :] + num_anchor_tokens
            if tail_count > 0:
                tail_indices = torch.arange(tail_start, N, device=k.device).unsqueeze(0).expand(B, -1)
                kept_indices = torch.cat([anchor_indices, kept_candidate_indices, tail_indices], dim=-1)
            else:
                kept_indices = torch.cat([anchor_indices, kept_candidate_indices], dim=-1)

        kept_candidate_k = torch.gather(candidate_k, 2, expanded_indices)
        kept_candidate_v = torch.gather(candidate_v, 2, expanded_indices)

        if tail_count > 0:
            final_k = torch.cat([anchor_k, kept_candidate_k, tail_k], dim=2)
            final_v = torch.cat([anchor_v, kept_candidate_v, tail_v], dim=2)
        else:
            final_k = torch.cat([anchor_k, kept_candidate_k], dim=2)
            final_v = torch.cat([anchor_v, kept_candidate_v], dim=2)

        return final_k, final_v, avg_scores, kept_indices

    def frame_eviction(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        cache_budget: int,
        tokens_per_frame: int,
        q: torch.Tensor = None,
        mode: str = 'cov',
        hybrid_alpha: float = 0.5,
    ):
        """
        Frame-coherent retention (FLCL): the KV cache of this layer is treated as
        a concatenation of whole per-frame blocks (each ``tokens_per_frame`` tokens).

        By default the first frame (world anchor) and the most-recent frame are
        always kept, and the remaining frame budget is filled from the middle
        frames by:
          - 'cov'    : farthest-first coverage over normalized mean-key signatures
                       (the FLCL rule used in the paper).
          - 'attn'   : per-frame cross-attention relevance from the current queries
                       (ablation).
          - 'hybrid' : alpha*coverage + (1-alpha)*relevance (ablation).
          - 'random' : deterministic random middle-frame selection (ablation).
        Whole frames are kept or dropped, never split, so each retained view stays
        a coherent observation.
        Batch-1 only (evaluation setting); otherwise falls back to no eviction.
        """
        B, H, N, D = k.shape
        tpf = int(tokens_per_frame)
        if B != 1 or tpf <= 0 or N % tpf != 0:
            return k, v, None, None
        num_frames = N // tpf
        frame_budget = max(cache_budget // tpf, 2)
        if num_frames <= frame_budget:
            return k, v, None, None

        # The anchor/recent pins can be switched off for the composition ablation.
        keep_anchor = getattr(self, 'frame_keep_anchor', True)
        keep_recent = getattr(self, 'frame_keep_recent', True)
        anchor, recent = 0, num_frames - 1
        forced = []
        if keep_anchor:
            forced.append(anchor)
        if keep_recent:
            forced.append(recent)
        forced = sorted(set(forced))
        pool_frames = [f for f in range(num_frames) if f not in forced]
        n_select = max(frame_budget - len(forced), 0)

        mk = k.view(B, H, num_frames, tpf, D).mean(dim=3).mean(dim=1)[0]
        mk = F.normalize(mk, dim=-1, eps=1e-8)

        def cos_dist(i, j):
            return (1.0 - (mk[i] * mk[j]).sum()).item()

        rel = None
        if mode in ('attn', 'hybrid') and q is not None:
            cross = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            cross = cross.softmax(dim=-1)
            tok_rel = cross.sum(dim=2).mean(dim=1)[0]
            rel = tok_rel.view(num_frames, tpf).mean(dim=1)

        ref = forced
        if mode == 'cov':
            # All pairwise distances in one shot + a single GPU->CPU sync, so the
            # greedy loop below does not trigger O(F^2) device syncs per layer.
            dist_all = (1.0 - (mk.unsqueeze(1) * mk.unsqueeze(0)).sum(dim=-1)).cpu()
            selected, pool = [], pool_frames[:]
            while len(selected) < n_select and pool:
                best, best_s = None, None
                for c in pool:
                    seeds = ref + selected
                    if seeds:
                        dmin = min(dist_all[c, r].item() for r in seeds)
                    else:
                        dmin = (1.0 - (mk[c] * mk[pool].mean(dim=0)).sum()).item()
                    if best_s is None or dmin > best_s:
                        best_s, best = dmin, c
                selected.append(best); pool.remove(best)
        elif mode == 'attn':
            selected = sorted(pool_frames, key=lambda c: rel[c].item(), reverse=True)[:n_select]
        elif mode == 'hybrid':
            cov = torch.tensor(
                [(min(cos_dist(c, r) for r in ref) if ref else 1.0) for c in pool_frames],
                device=k.device)
            rl = torch.tensor([rel[c].item() for c in pool_frames], device=k.device)
            nrm = lambda t: (t - t.min()) / (t.max() - t.min() + 1e-8)
            score = hybrid_alpha * nrm(cov) + (1.0 - hybrid_alpha) * nrm(rl)
            top = torch.topk(score, k=min(n_select, len(pool_frames))).indices.tolist()
            selected = [pool_frames[i] for i in top]
        elif mode == 'random':
            g = torch.Generator(device='cpu')
            g.manual_seed(int(getattr(self, 'frame_random_seed', 0)) * 1000003 + N)
            perm = torch.randperm(len(pool_frames), generator=g).tolist()
            selected = [pool_frames[i] for i in perm[:n_select]]
        else:
            return k, v, None, None

        keep = sorted(set(forced + selected))
        idx = torch.cat([torch.arange(f * tpf, (f + 1) * tpf, device=k.device) for f in keep])
        kept_indices = idx.unsqueeze(0).expand(B, -1)
        kk = k.index_select(2, idx)
        vv = v.index_select(2, idx)
        # Per-layer diversity signal = mean pairwise cosine SIMILARITY of the
        # per-frame mean-key signatures over the pre-eviction pool (paper Eq. 8).
        # The aggregator maps eta^l = 1 - score, so returning similarity gives the
        # correct diversity direction (higher diversity -> larger budget). The
        # selection mode only changes WHICH frames are kept, never this signal.
        # mk is already L2-normalized (see above).
        if num_frames >= 2:
            mk_f = mk.float()                          # [F, D], normalized
            sim = mk_f @ mk_f.t()                      # [F, F], cosine; diag ~1
            off = sim.sum() - sim.diagonal().sum()     # exclude self-pairs
            avg = float((off / (num_frames * (num_frames - 1))).item())
        else:
            avg = 0.0
        return kk, vv, avg, kept_indices

    def forward(self,
        x: torch.Tensor,
        pos=None,
        attn_mask=None,
        past_key_values=None,
        use_cache=False,
        cache_budget=2000,
        importance_scores=None,
        intra_frame_keep_ratio=1.0,
        defer_eviction=False,
        anchor_token_count: int = None,
        importance_weight: float = 0.5,
        window_token_count: int = 0,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple]]:
        """
        Forward pass with optional two-stage eviction support.

        Args:
            intra_frame_keep_ratio: Ratio of current frame tokens to keep (1.0 = keep all)
            defer_eviction: If True, skip inter-frame eviction here (handled by Block)

        When defer_eviction=True, returns additional info for Block to handle eviction:
            (output, (k_full, v_full, k_current, v_current, past_kv_or_none), scores)
        """
        B, N, C = x.shape
        num_new_tokens = N  # Number of new tokens from current frame
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scores = None
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if use_cache and self.num_anchor_tokens == 0:
            self.num_anchor_tokens = k.shape[2]

        k_current, v_current = k.clone(), v.clone() 
        past_kv_for_block = None

        if use_cache:
            # --- SSC variant: attention-based new-token importance (current frame only) ---
            # Keep FlashAttention for the main output below; here we do a small, score-only
            # pass on just the current frame's query x key block ([B,H,N_new,N_new]).
            new_attn_score = None
            if self.ssc_score_mode in ('attn', 'attn_decay'):
                attn_logits = torch.matmul(q, k_current.transpose(-2, -1)) * self.scale  # [B,H,N_new,N_new]
                attn_w = attn_logits.softmax(dim=-1)
                # importance = attention mass each current-frame token receives from current queries
                new_attn_score = attn_w.sum(dim=2).mean(dim=1)  # [B, N_new]

            if past_key_values is not None:
                past_k, past_v = past_key_values
                past_kv_for_block = (past_k, past_v)
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)

            # For the attn variants, override the (FFN-derived) new-token importance.
            if new_attn_score is not None:
                importance_scores = new_attn_score

            # For attn_decay, keep a per-token importance aligned with the full cache:
            # decay previously stored scores by gamma, then append the new-token scores.
            token_importance = None
            if self.ssc_score_mode == 'attn_decay':
                if self.layer_importance is not None:
                    token_importance = torch.cat(
                        [self.layer_importance * self.decay_gamma, new_attn_score], dim=1
                    )  # [B, N]
                else:
                    token_importance = new_attn_score  # first frame
            elif self.ssc_score_mode == 'attn_relevance':
                # Score EVERY cached token by how much the current frame's queries attend to it
                # (cross-attention over the full cache). Recomputed fresh each frame -> no decay
                # / no persisted state; directly measures "does the current view still need this token".
                cross_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,N_new,N_full]
                cross_w = cross_logits.softmax(dim=-1)
                token_importance = cross_w.sum(dim=2).mean(dim=1)  # [B, N_full]
                importance_scores = token_importance[:, -num_new_tokens:]  # new-token part
            elif self.ssc_score_mode == 'token_cov':
                # Token-level coverage ablation: score tokens purely by cosine
                # diversity / farthest-first in key space -- the token analogue of
                # frame_cov, no importance/attention blend. Nulling importance_scores
                # routes eviction() through its pure-diversity fallback.
                importance_scores = None

            kept_indices = None
            if not defer_eviction:
                # Legacy path: apply eviction in attention
                if cache_budget is not None and k.shape[2] > cache_budget:
                    # Use anchor_token_count if provided (History Anchor), otherwise fall back to num_anchor_tokens
                    effective_anchor_count = anchor_token_count if anchor_token_count is not None else self.num_anchor_tokens
                    if self.ssc_score_mode.startswith('frame_'):
                        # Frame-coherent retention (FLCL)
                        k, v, scores, kept_indices = self.frame_eviction(
                            k, v, cache_budget,
                            tokens_per_frame=num_new_tokens,
                            q=q,
                            mode=self.ssc_score_mode[len('frame_'):],
                            hybrid_alpha=self.decay_beta,
                        )
                    else:
                        k, v, scores, kept_indices = self.eviction(
                            k, v, cache_budget, effective_anchor_count,
                            importance_scores=importance_scores,
                            num_new_tokens=num_new_tokens,
                            importance_weight=importance_weight,
                            window_token_count=window_token_count,
                            token_importance=token_importance,
                            decay_beta=self.decay_beta,
                        )
                new_kv = (k, v, kept_indices)

                # Keep the attn_decay per-token importance aligned with the (possibly evicted) cache.
                if self.ssc_score_mode == 'attn_decay':
                    if kept_indices is not None:
                        self.layer_importance = torch.gather(token_importance, 1, kept_indices)
                    else:
                        self.layer_importance = token_importance
            else:
                # Two-stage path: return info for Block to handle eviction
                # Block will apply: 1) intra-frame pruning, 2) inter-frame eviction
                new_kv = (k, v, k_current, v_current, past_kv_for_block)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )

        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            # Mask
            if attn_mask is not None:
                assert attn_mask.shape[-2:] == (N, N), f"Expected mask shape [..., {N}, {N}], got {attn_mask.shape}"
                attn = attn + attn_mask

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if use_cache:
                return x, new_kv, scores
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)

        return x