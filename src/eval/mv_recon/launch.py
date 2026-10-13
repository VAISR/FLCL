import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import time
import torch
import argparse
import numpy as np
import open3d as o3d
import os.path as osp
from torch.utils.data import DataLoader
from add_ckpt_path import add_path_to_dust3r
from accelerate import Accelerator
from torch.utils.data._utils.collate import default_collate
import tempfile
from tqdm import tqdm
import uuid
import json
from collections import defaultdict

def get_args_parser():
    parser = argparse.ArgumentParser("3D Reconstruction evaluation", add_help=False)
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="ckpt name",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument(
        "--conf_thresh", type=float, default=0.0, help="confidence threshold"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="value for outdir",
    )
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--revisit", type=int, default=1, help="revisit times")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--max_frames", type=int, default=None, help="max frames limit")
    parser.add_argument("--use_proj", action="store_true")
    parser.add_argument("--scene", type=str, default=None, help="restrict to a single 7scenes scene, e.g. chess (default: all)")
    parser.add_argument("--kf_every", type=int, default=2, help="keyframe sampling stride")
    parser.add_argument("--voxel_size", type=float, default=0.0,
                        help="if >0, voxel-downsample fused pred/gt clouds before ICP/metric (caps RAM for long sequences; 0=off, exact original)")
    parser.add_argument("--save_artifacts", action="store_true",
                        help="write per-sequence .ply/.npy debug artifacts (~1.4GB/seq). OFF by default to avoid filling disk.")
    parser.add_argument("--dataset", type=str, default="7scenes", choices=["7scenes", "NRGBD", "eth3d"])
    parser.add_argument("--root_7scenes", type=str, default="./data/7scenes")
    parser.add_argument("--root_nrgbd", type=str, default="./data/neural_rgbd")
    parser.add_argument("--root_eth3d", type=str, default="./data/eth3d")
    # --- StreamVGGT / FrameVGGT baseline memory knobs (optional, needs FRAMEVGGT_SRC) ---
    parser.add_argument("--recent_frames", type=int, default=2)
    parser.add_argument("--middle_frames", type=int, default=4)
    parser.add_argument("--anchor_frames", type=int, default=2)
    parser.add_argument("--anchor_every", type=int, default=50)
    parser.add_argument("--fv_score_mode", type=str, default="cosine_diversity",
                        choices=["cosine_diversity", "importance", "relevance", "hybrid"])
    # --- OVGGT streaming / SSC variant controls ---
    parser.add_argument("--max_token_budget", type=int, default=200000, help="OVGGT total KV cache token budget")
    parser.add_argument("--camera_budget", type=int, default=384, help="OVGGT camera-head KV cache budget")
    parser.add_argument("--dap", type=str, default="off", choices=["on", "off"],
                        help="Dynamic Anchor Protection: 'on'=coverage (original), 'off'=none. "
                             "Default: off (required by frame_* modes).")
    parser.add_argument("--ssc_score", type=str, default="frame_cov",
                        choices=["ffn", "attn", "attn_decay", "attn_relevance",
                                 "frame_cov", "frame_attn", "frame_hybrid",
                                 "token_cov", "frame_random"],
                        help="SSC scoring. token-level: ffn=repr_shift (original), attn=current-frame attention, "
                             "attn_decay=attn+time decay, attn_relevance=cross-attn relevance. "
                             "frame-level: frame_cov=mean-key farthest-first coverage (FLCL, default), "
                             "frame_attn=per-frame cross-attn relevance, frame_hybrid=coverage+relevance, "
                             "frame_random=random middle frames; token_cov=token-level coverage")
    parser.add_argument("--decay_gamma", type=float, default=0.9, help="per-frame decay factor for attn_decay variant")
    parser.add_argument("--decay_beta", type=float, default=0.5,
                        help="attn_decay: old-token score = beta*diversity + (1-beta)*decayed_attn")
    parser.add_argument("--importance_weight", type=float, default=0.5,
                        help="eviction mix: new-token score weight (old-token weight = 1 - this)")
    # --- frame-level retention: anchor-composition toggles + random selection (ablations) ---
    parser.add_argument("--frame_keep_anchor", type=str, default="on", choices=["on", "off"],
        help="frame-level retention: always keep the first frame (world anchor). Default on.")
    parser.add_argument("--frame_keep_recent", type=str, default="on", choices=["on", "off"],
        help="frame-level retention: always keep the most-recent frame. Default on.")
    parser.add_argument("--frame_random_seed", type=int, default=0,
        help="deterministic seed base for --ssc_score frame_random.")
    # --- layer budget allocation (paper Eq. 8-9) ---
    parser.add_argument("--budget_mode", type=str, default="uniform", choices=["adaptive", "uniform"],
                        help="per-layer KV budget: 'uniform'=Theta/L per layer (default, theta=1 in the paper); "
                             "'adaptive'=temperature softmax over per-layer diversity (paper Eq. 9)")
    parser.add_argument("--budget_temperature", type=float, default=0.5,
                        help="theta in [0,1]; theta=1 => uniform, smaller => concentrated "
                             "(softmax(eta*(1-theta)/theta), paper Eq. 9)")
    parser.add_argument("--budget_norm", type=str, default="none", choices=["none", "zscore", "minmax"],
                        help="cross-layer normalization of eta before softmax (aggregator only): "
                             "'none'=paper; 'zscore'/'minmax' amplify relative spread when eta is narrow")
    return parser


def main(args):
    add_path_to_dust3r(args.weights)
    from eval.mv_recon.data import SevenScenes, NRGBD, ETH3D
    from eval.mv_recon.utils import accuracy, completion

    if args.size == 512:
        resolution = (512, 384)
    elif args.size == 224:
        resolution = 224
    elif args.size == 518:
        resolution = (518, 392)
        # resolution = (518, 336)
    else:
        raise NotImplementedError
    if args.dataset == "7scenes":
        datasets_all = {
            "7scenes": SevenScenes(
                split="test",
                ROOT=args.root_7scenes,
                resolution=resolution,
                num_seq=1,
                full_video=True,
                kf_every=args.kf_every,
                test_id=args.scene,
                max_frames=args.max_frames,
            ),
        }
    elif args.dataset == "NRGBD":
        datasets_all = {
            "NRGBD": NRGBD(
                split="test",
                ROOT=args.root_nrgbd,
                resolution=resolution,
                num_seq=1,
                full_video=True,
                kf_every=args.kf_every,
                test_id=args.scene,
                max_frames=args.max_frames,
            ),
        }
    elif args.dataset == "eth3d":
        # ETH3D has a different loader interface than SevenScenes/NRGBD: its __init__
        # takes no split/test_id/max_frames (loads every scene under ROOT, one strided
        # full-video per scene). full_video=True + kf_every strides frames. No DTU-style
        # scale hack -> default eval threshold.
        datasets_all = {
            "eth3d": ETH3D(
                ROOT=args.root_eth3d,
                resolution=resolution,
                num_seq=1,
                full_video=True,
                kf_every=args.kf_every,
            ),
        }
        # --scene single-scene support: ETH3D's loader has no test_id, so filter
        # scene_list post-construction. BaseStereoViewDataset.__len__ == len(self.scenes)
        # and _get_views indexes self.scene_list, so keep both in sync.
        if args.scene:
            _d = datasets_all["eth3d"]
            _d.scene_list = [s for s in _d.scene_list if s == args.scene]
            _d.scenes = _d.scene_list
            assert len(_d.scene_list) == 1, f"scene {args.scene} not in ETH3D"
    _unused_old = {
        "_disabled": None,
        # "NRGBD": NRGBD(
        #     split="test",
        #     ROOT="./data/neural_rgbd_data",
        #     resolution=resolution,
        #     num_seq=1,
        #     full_video=True,
        #     kf_every=500,
        # ),
    }

    accelerator = Accelerator()
    device = accelerator.device
    model_name = args.model_name
    if model_name == "OVGGT":
        from flcl.models.ovggt import OVGGT
        from flcl.utils.pose_enc import pose_encoding_to_extri_intri
        from flcl.utils.geometry import unproject_depth_map_to_point_map
        from eval.mv_recon.criterion import Regr3D_t_ScaleShiftInv, L21
        from dust3r.utils.geometry import geotrf
        from copy import deepcopy
        model = OVGGT(
            total_budget=args.max_token_budget,
            camera_budget=args.camera_budget,
            ssc_score_mode=args.ssc_score,
            decay_gamma=args.decay_gamma,
            decay_beta=args.decay_beta,
            importance_weight=args.importance_weight,
            budget_mode=args.budget_mode,
            budget_temperature=args.budget_temperature,
            budget_norm=args.budget_norm,
            frame_keep_anchor=(args.frame_keep_anchor == "on"),
            frame_keep_recent=(args.frame_keep_recent == "on"),
            frame_random_seed=args.frame_random_seed,
        )
        ckpt = torch.load(args.weights, map_location=device, weights_only=False)
        model.load_state_dict(ckpt, strict=True)
        model.eval()
        model = model.to("cuda")
    elif model_name == "VGGT":
        from vggt.models.vggt import VGGT
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        from eval.mv_recon.criterion import Regr3D_t_ScaleShiftInv, L21
        from dust3r.utils.geometry import geotrf
        from copy import deepcopy
        model = VGGT()
        ckpt = torch.load(args.weights, map_location=device, weights_only=False)
        model.load_state_dict(ckpt, strict=True)
        model.eval()
        model = model.to("cuda")

    elif model_name == "StreamVGGT":
        # FrameVGGT baseline (StreamVGGT + BlockMemoryManager) run through this eval
        # harness so all methods share identical metrics. Point FRAMEVGGT_SRC at the
        # FrameVGGT source tree; it is appended AFTER this tree so flcl/eval/dust3r stay local.
        import os as _os, sys as _sys
        _fv = _os.environ.get("FRAMEVGGT_SRC", "")
        if _fv and _fv not in _sys.path:
            _sys.path.append(_fv)
        from streamvggt.models.streamvggt import StreamVGGT
        from eval.mv_recon.criterion import Regr3D_t_ScaleShiftInv, L21
        from dust3r.utils.geometry import geotrf
        from copy import deepcopy
        model = StreamVGGT()
        ckpt = torch.load(args.weights, map_location=device, weights_only=False)
        model.load_state_dict(ckpt, strict=True)
        model.eval()
        model = model.to("cuda")
        model._eval_score_mode = args.fv_score_mode
        model.set_memory_config(
            recent_frames=args.recent_frames,
            middle_frames=args.middle_frames,
            anchor_frames=args.anchor_frames,
            anchor_every=args.anchor_every,
        )
    else:
        raise NotImplementedError
    del ckpt
    os.makedirs(args.output_dir, exist_ok=True)

    criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)

    with torch.no_grad():
        for name_data, dataset in datasets_all.items():
            save_path = osp.join(args.output_dir, name_data)
            os.makedirs(save_path, exist_ok=True)
            log_file = osp.join(save_path, f"logs_{accelerator.process_index}.txt")

            acc_all = 0
            acc_all_med = 0
            comp_all = 0
            comp_all_med = 0
            nc1_all = 0
            nc1_all_med = 0
            nc2_all = 0
            nc2_all_med = 0

            fps_all = []
            time_all = []

            with accelerator.split_between_processes(list(range(len(dataset)))) as idxs:
                for data_idx in tqdm(idxs):
                    batch = default_collate([dataset[data_idx]])
                    ignore_keys = set(
                        [
                            "depthmap",
                            "dataset",
                            "label",
                            "instance",
                            "idx",
                            "true_shape",
                            "rng",
                        ]
                    )
                    for view in batch:
                        for name in view.keys():  # pseudo_focal
                            if name in ignore_keys:
                                continue
                            if isinstance(view[name], tuple) or isinstance(
                                view[name], list
                            ):
                                view[name] = [
                                    x.to(device, non_blocking=True) for x in view[name]
                                ]
                            else:
                                view[name] = view[name].to(device, non_blocking=True)

                    pts_all = []
                    pts_gt_all = []
                    images_all = []
                    masks_all = []
                    conf_all = []
                    in_camera1 = None  

                    if model_name == "stream3r" or "VGGT":
                        revisit = args.revisit
                        update = not args.freeze
                        if revisit > 1:
                            # repeat input for 'revisit' times
                            new_views = []
                            for r in range(revisit):
                                for i in range(len(batch)):
                                    new_view = deepcopy(batch[i])
                                    new_view["idx"] = [
                                        (r * len(batch) + i)
                                        for _ in range(len(batch[i]["idx"]))
                                    ]
                                    new_view["instance"] = [
                                        str(r * len(batch) + i)
                                        for _ in range(len(batch[i]["instance"]))
                                    ]
                                    if r > 0:
                                        if not update:
                                            new_view["update"] = torch.zeros_like(
                                                batch[i]["update"]
                                            ).bool()
                                    new_views.append(new_view)
                            batch = new_views
                        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
                        with torch.cuda.amp.autocast(dtype=dtype):
                            if isinstance(batch, dict) and "img" in batch:
                                batch["img"] = (batch["img"] + 1.0) / 2.0
                            elif isinstance(batch, list) and all(isinstance(v, dict) and "img" in v for v in batch):
                                for view in batch:
                                    view["img"] = (view["img"] + 1.0) / 2.0

                        with torch.cuda.amp.autocast(dtype=dtype):
                            with torch.no_grad():
                                if model_name == "StreamVGGT":
                                    results = model.inference(batch)
                                else:
                                    results = model.inference(
                                        batch,
                                        history_anchor_strategy=("coverage" if args.dap == "on" else "none"),
                                    )

                            preds, batch = results.ress, results.views

                            if model_name == "StreamVGGT":
                                # StreamVGGT returns predictions (and possibly views) on CPU;
                                # move both preds and batch tensors to device for the criterion.
                                def _to_dev(_d):
                                    for _k in list(_d.keys()):
                                        _v = _d[_k]
                                        if torch.is_tensor(_v):
                                            _d[_k] = _v.to(device)
                                        elif isinstance(_v, (list, tuple)) and len(_v) and torch.is_tensor(_v[0]):
                                            _d[_k] = [x.to(device) if torch.is_tensor(x) else x for x in _v]
                                for _r in preds:
                                    _to_dev(_r)
                                for _b in batch:
                                    _to_dev(_b)

                            if args.use_proj:
                                pose_enc = torch.stack([preds[s]["camera_pose"] for s in range(len(preds))], dim=1)
                                depth_map = torch.stack([preds[s]["depth"] for s in range(len(preds))], dim=1)
                                depth_conf = torch.stack([preds[s]["depth_conf"] for s in range(len(preds))], dim=1)
                                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc,
                                                                                    batch[0]["img"].shape[-2:])

                                if "DTU" in name_data:
                                    depth_map = depth_map * 1000.0
                                    extrinsic[..., :3, 3] *= 1000.0

                                point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0),
                                                                                                extrinsic.squeeze(0),
                                                                                                intrinsic.squeeze(0))
                            valid_length = len(preds) // args.revisit
                            if args.revisit > 1:
                                preds = preds[-valid_length:]
                                batch = batch[-valid_length:]
                                

                        # Evaluation
                        print(f"Evaluation for {name_data} {data_idx+1}/{len(dataset)}")
                        gt_pts, pred_pts, gt_factor, pr_factor, masks, monitoring = (
                            criterion.get_all_pts3d_t(batch, preds)
                        )

                        in_camera1 = None
                        pts_all = []
                        pts_gt_all = []
                        images_all = []
                        masks_all = []
                        conf_all = []

                        for j, view in enumerate(batch):
                            if in_camera1 is None:
                                in_camera1 = view["camera_pose"][0].cpu()

                            image = view["img"].permute(0, 2, 3, 1).cpu().numpy()[0]
                            mask = view["valid_mask"].cpu().numpy()[0]

                            if args.use_proj:
                                pts = point_map_by_unprojection[j]
                                conf = depth_conf[0, j].cpu().data.numpy()
                            else:
                                pts = pred_pts[j].cpu().numpy()[0]
                                conf = preds[j]["conf"].cpu().data.numpy()[0]

                            # mask = mask & (conf > 1.8)

                            pts_gt = gt_pts[j].detach().cpu().numpy()[0]

                            H, W = image.shape[:2]
                            cx = W // 2
                            cy = H // 2
                            l, t = cx - 112, cy - 112
                            r, b = cx + 112, cy + 112
                            image = image[t:b, l:r]
                            mask = mask[t:b, l:r]
                            pts = pts[t:b, l:r]
                            pts_gt = pts_gt[t:b, l:r]

                            # Align predicted 3D points to the ground truth
                            # pts = geotrf(in_camera1, pts)
                            # pts_gt = geotrf(in_camera1, pts_gt)

                            images_all.append(image[None, ...])
                            pts_all.append(pts[None, ...])
                            pts_gt_all.append(pts_gt[None, ...])
                            masks_all.append(mask[None, ...])
                            conf_all.append(conf[None, ...])

                    images_all = np.concatenate(images_all, axis=0)
                    pts_all = np.concatenate(pts_all, axis=0)
                    pts_gt_all = np.concatenate(pts_gt_all, axis=0)
                    masks_all = np.concatenate(masks_all, axis=0)

                    scene_id = view["label"][0].rsplit("/", 1)[0]

                    save_params = {}
                    if args.save_artifacts:
                        save_params["images_all"] = images_all
                        save_params["pts_all"] = pts_all
                        save_params["pts_gt_all"] = pts_gt_all
                        save_params["masks_all"] = masks_all
                        np.save(
                            os.path.join(save_path, f"{scene_id.replace('/', '_')}.npy"),
                            save_params,
                        )

                    if "DTU" in name_data:
                        threshold = 100
                    else:
                        threshold = 0.1

                    pts_all_masked = pts_all[masks_all > 0]
                    pts_gt_all_masked = pts_gt_all[masks_all > 0]
                    images_all_masked = images_all[masks_all > 0]

                    mask = np.isfinite(pts_all_masked)  
                    pts_all_masked = pts_all_masked[mask]

                    mask_gt = np.isfinite(pts_gt_all_masked)
                    pts_gt_all_masked = pts_gt_all_masked[mask]

                    if args.use_proj:
                        def umeyama_alignment(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
                            assert src.shape == dst.shape
                            N, dim = src.shape

                            mu_src = src.mean(axis=0)
                            mu_dst = dst.mean(axis=0)
                            src_c = src - mu_src
                            dst_c = dst - mu_dst

                            Sigma = dst_c.T @ src_c / N  # (3,3)

                            U, D, Vt = np.linalg.svd(Sigma) 

                            S = np.eye(dim)
                            if np.linalg.det(U) * np.linalg.det(Vt) < 0:
                                S[-1, -1] = -1

                            R = U @ S @ Vt

                            if with_scale:
                                var_src = (src_c ** 2).sum() / N
                                s = (D * S.diagonal()).sum() / var_src
                            else:
                                s = 1.0

                            t = mu_dst - s * R @ mu_src

                            return s, R, t

                        pts_all_masked = pts_all_masked.reshape(-1, 3)
                        pts_gt_all_masked = pts_gt_all_masked.reshape(-1, 3)
                        s, R, t = umeyama_alignment(pts_all_masked, pts_gt_all_masked, with_scale=True)
                        pts_all_aligned = (s * (R @ pts_all_masked.T)).T + t  # (N,3)
                        pts_all_masked = pts_all_aligned

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(
                        pts_all_masked.reshape(-1, 3)
                    )
                    pcd.colors = o3d.utility.Vector3dVector(
                        images_all_masked.reshape(-1, 3)
                    )
                    if args.save_artifacts:
                        o3d.io.write_point_cloud(
                            os.path.join(save_path, f"{scene_id.replace('/', '_')}-mask.ply"),
                            pcd,
                        )

                    pcd_gt = o3d.geometry.PointCloud()
                    pcd_gt.points = o3d.utility.Vector3dVector(
                        pts_gt_all_masked.reshape(-1, 3)
                    )
                    pcd_gt.colors = o3d.utility.Vector3dVector(
                        images_all_masked.reshape(-1, 3)
                    )
                    if args.save_artifacts:
                        o3d.io.write_point_cloud(
                            os.path.join(save_path, f"{scene_id.replace('/', '_')}-gt.ply"),
                            pcd_gt,
                        )

                    if getattr(args, "voxel_size", 0.0) and args.voxel_size > 0:
                        pcd = pcd.voxel_down_sample(args.voxel_size)
                        pcd_gt = pcd_gt.voxel_down_sample(args.voxel_size)

                    trans_init = np.eye(4)

                    reg_p2p = o3d.pipelines.registration.registration_icp(
                        pcd,
                        pcd_gt,
                        threshold,
                        trans_init,
                        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    )

                    transformation = reg_p2p.transformation

                    pcd = pcd.transform(transformation)

                    if args.save_artifacts:
                        o3d.io.write_point_cloud(
                            os.path.join(save_path, f"{scene_id.replace('/', '_')}-mask_align.ply"),
                            pcd,
                        )

                    pcd.estimate_normals()
                    pcd_gt.estimate_normals()

                    gt_normal = np.asarray(pcd_gt.normals)
                    pred_normal = np.asarray(pcd.normals)

                    acc, acc_med, nc1, nc1_med = accuracy(
                        pcd_gt.points, pcd.points, gt_normal, pred_normal
                    )
                    comp, comp_med, nc2, nc2_med = completion(
                        pcd_gt.points, pcd.points, gt_normal, pred_normal
                    )
                    print(
                        f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}"
                    )
                    with open(log_file, "a") as _lf:
                        _lf.write(
                            f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}\n"
                        )

                    acc_all += acc
                    comp_all += comp
                    nc1_all += nc1
                    nc2_all += nc2

                    acc_all_med += acc_med
                    comp_all_med += comp_med
                    nc1_all_med += nc1_med
                    nc2_all_med += nc2_med

                    # release cuda memory
                    torch.cuda.empty_cache()

                    # release CPU memory between sequences — the eval fuses all frames into
                    # large point clouds; without this the process leaks ~1GB/sequence and OOMs
                    # on long sequences (30GB RAM machine).
                    import gc
                    del pcd, pcd_gt, gt_normal, pred_normal, reg_p2p
                    del pts_all, pts_gt_all, images_all, masks_all
                    del pts_all_masked, pts_gt_all_masked, images_all_masked
                    del save_params, preds, results
                    gc.collect()

            accelerator.wait_for_everyone()
            # Get depth from pcd and run TSDFusion
            if accelerator.is_main_process:
                to_write = ""
                # Copy the error log from each process to the main error log
                for i in range(8):
                    if not os.path.exists(osp.join(save_path, f"logs_{i}.txt")):
                        break
                    with open(osp.join(save_path, f"logs_{i}.txt"), "r") as f_sub:
                        to_write += f_sub.read()

                with open(osp.join(save_path, f"logs_all.txt"), "w") as f:
                    log_data = to_write
                    metrics = defaultdict(list)
                    for line in log_data.strip().split("\n"):
                        match = regex.match(line)
                        if match:
                            data = match.groupdict()
                            # Exclude 'scene_id' from metrics as it's an identifier
                            for key, value in data.items():
                                if key != "scene_id":
                                    metrics[key].append(float(value))
                            metrics["nc"].append(
                                (float(data["nc1"]) + float(data["nc2"])) / 2
                            )
                            metrics["nc_med"].append(
                                (float(data["nc1_med"]) + float(data["nc2_med"])) / 2
                            )
                    mean_metrics = {
                        metric: sum(values) / len(values)
                        for metric, values in metrics.items()
                    }

                    c_name = "mean"
                    print_str = f"{c_name.ljust(20)}: "
                    for m_name in mean_metrics:
                        print_num = np.mean(mean_metrics[m_name])
                        print_str = print_str + f"{m_name}: {print_num:.3f} | "
                    # Peak GPU memory for the whole run (constant-memory check).
                    try:
                        import torch as _torch
                        if _torch.cuda.is_available():
                            peak_gb = _torch.cuda.max_memory_allocated() / (1024 ** 3)
                            print_str = print_str + f"peak_mem_gb: {peak_gb:.3f} | "
                    except Exception:
                        pass
                    print_str = print_str + "\n"
                    f.write(to_write + print_str)
                    print(print_str, flush=True)



from collections import defaultdict
import re

pattern = r"""
    Idx:\s*(?P<scene_id>[^,]+),\s*
    Acc:\s*(?P<acc>[^,]+),\s*
    Comp:\s*(?P<comp>[^,]+),\s*
    NC1:\s*(?P<nc1>[^,]+),\s*
    NC2:\s*(?P<nc2>[^,]+)\s*-\s*
    Acc_med:\s*(?P<acc_med>[^,]+),\s*
    Compc_med:\s*(?P<comp_med>[^,]+),\s*
    NC1c_med:\s*(?P<nc1_med>[^,]+),\s*
    NC2c_med:\s*(?P<nc2_med>[^,]+)
"""

regex = re.compile(pattern, re.VERBOSE)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)