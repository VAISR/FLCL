import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import math
import cv2
import numpy as np
import torch
import argparse

from copy import deepcopy
from eval.video_depth.metadata import dataset_metadata
from eval.video_depth.utils import save_depth_maps
from accelerate import PartialState
from add_ckpt_path import add_path_to_dust3r
import time
from tqdm import tqdm

def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        type=str,
        help="path to the model weights",
        default="",
    )

    parser.add_argument("--device", type=str, default="cuda", help="pytorch device")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="value for outdir",
    )
    parser.add_argument(
        "--no_crop", type=bool, default=True, help="whether to crop input data"
    )

    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="sintel",
        choices=list(dataset_metadata.keys()),
    )
    parser.add_argument("--size", type=int, default="224")

    parser.add_argument(
        "--pose_eval_stride", default=1, type=int, help="stride for pose evaluation"
    )
    parser.add_argument(
        "--full_seq",
        action="store_true",
        default=False,
        help="use full sequence for pose evaluation",
    )
    parser.add_argument(
        "--seq_list",
        nargs="+",
        default=None,
        help="list of sequences for pose evaluation",
    )
    # --- model selection + StreamVGGT / FrameVGGT baseline memory knobs (mirrors eval/mv_recon/launch.py) ---
    parser.add_argument("--model_name", type=str, default="OVGGT", choices=["OVGGT", "VGGT", "StreamVGGT"])
    parser.add_argument("--recent_frames", type=int, default=2)
    parser.add_argument("--middle_frames", type=int, default=4)
    parser.add_argument("--anchor_frames", type=int, default=2)
    parser.add_argument("--anchor_every", type=int, default=50)
    parser.add_argument("--fv_score_mode", type=str, default="cosine_diversity",
                        choices=["cosine_diversity", "importance", "relevance", "hybrid"])
    # --- OVGGT streaming / SSC variant controls (mirrors eval/mv_recon/launch.py) ---
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


def eval_pose_estimation(args, model, save_dir=None):
    metadata = dataset_metadata.get(args.eval_dataset)
    img_path = metadata["img_path"]
    mask_path = metadata["mask_path"]

    ate_mean, rpe_trans_mean, rpe_rot_mean = eval_pose_estimation_dist(
        args, model, save_dir=save_dir, img_path=img_path, mask_path=mask_path
    )
    return ate_mean, rpe_trans_mean, rpe_rot_mean


def eval_pose_estimation_dist(args, model, img_path, save_dir=None, mask_path=None):
    from dust3r.inference import loss_of_one_batch

    metadata = dataset_metadata.get(args.eval_dataset)
    anno_path = metadata.get("anno_path", None)

    seq_list = args.seq_list
    if seq_list is None:
        if metadata.get("full_seq", False):
            args.full_seq = True
        else:
            seq_list = metadata.get("seq_list", [])
        if args.full_seq:
            seq_list = os.listdir(img_path)
            seq_list = [
                seq for seq in seq_list if os.path.isdir(os.path.join(img_path, seq))
            ]
        seq_list = sorted(seq_list)

    if save_dir is None:
        save_dir = args.output_dir

    distributed_state = PartialState()
    model.to(distributed_state.device)
    device = distributed_state.device

    with distributed_state.split_between_processes(seq_list) as seqs:
        ate_list = []
        rpe_trans_list = []
        rpe_rot_list = []
        load_img_size = args.size
        assert load_img_size == 518
        error_log_path = f"{save_dir}/_error_log_{distributed_state.process_index}.txt"  # Unique log file per process
        bug = False
        for seq in tqdm(seqs):
            try:
                dir_path = metadata["dir_path_func"](img_path, seq)

                # Handle skip_condition
                skip_condition = metadata.get("skip_condition", None)
                if skip_condition is not None and skip_condition(save_dir, seq):
                    continue

                mask_path_seq_func = metadata.get(
                    "mask_path_seq_func", lambda mask_path, seq: None
                )
                mask_path_seq = mask_path_seq_func(mask_path, seq)

                filelist = [
                    os.path.join(dir_path, name) for name in os.listdir(dir_path)
                ]
                filelist.sort()
                filelist = filelist[:: args.pose_eval_stride]

                views = prepare_input(
                    filelist,
                    [True for _ in filelist],
                    size=load_img_size,
                    crop=not args.no_crop,
                )
                for view in views:
                    view["img"] = (view["img"] + 1.0) / 2.0
                start = time.time()
                # Mirror eval/mv_recon/launch.py: call model.inference directly so the
                # --dap flag threads through as history_anchor_strategy (loss_of_one_batch
                # hardcodes the inference() default 'coverage'). views carry no valid_mask
                # here, so query_points stays None -- same as loss_of_one_batch's path.
                dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
                with torch.no_grad():
                    with torch.cuda.amp.autocast(dtype=dtype):
                        if args.model_name == "StreamVGGT":
                            _out = model.inference(views)
                        else:
                            _out = model.inference(
                                views,
                                query_points=None,
                                history_anchor_strategy=("coverage" if args.dap == "on" else "none"),
                            )
                outputs = dict(views=_out.views, pred=_out.ress)
                end = time.time()
                # fps = len(filelist) / (end - start)
                with torch.cuda.amp.autocast(dtype=torch.float32):
                    (
                        pts3ds_self,
                        conf_self,
                    ) = prepare_output(outputs)

                    os.makedirs(f"{save_dir}/{seq}", exist_ok=True)
                    save_depth_maps(pts3ds_self, f"{save_dir}/{seq}", conf_self=conf_self)

            except Exception as e:
                if "out of memory" in str(e):
                    # Handle OOM
                    torch.cuda.empty_cache()  # Clear the CUDA memory
                    with open(error_log_path, "a") as f:
                        f.write(
                            f"OOM error in sequence {seq}, skipping this sequence.\n"
                        )
                    print(f"OOM error in sequence {seq}, skipping...")
                elif "Degenerate covariance rank" in str(
                    e
                ) or "Eigenvalues did not converge" in str(e):
                    # Handle Degenerate covariance rank exception and Eigenvalues did not converge exception
                    with open(error_log_path, "a") as f:
                        f.write(f"Exception in sequence {seq}: {str(e)}\n")
                    print(f"Traj evaluation error in sequence {seq}, skipping.")
                else:
                    raise e  # Rethrow if it's not an expected exception
    return None, None, None


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    add_path_to_dust3r(args.weights)
    from dust3r.utils.image import load_images_for_eval as load_images
    from dust3r.post_process import estimate_focal_knowing_depth
    from dust3r.model import ARCroco3DStereo
    from dust3r.utils.camera import pose_encoding_to_camera

    from flcl.models.ovggt import OVGGT
    from flcl.utils.pose_enc import pose_encoding_to_extri_intri
    from flcl.utils.geometry import unproject_depth_map_to_point_map
    from eval.mv_recon.criterion import Regr3D_t_ScaleShiftInv, L21
    from dust3r.utils.geometry import geotrf
    from copy import deepcopy

    if args.eval_dataset == "sintel":
        args.full_seq = True
    else:
        args.full_seq = False
    args.no_crop = True

    def prepare_input(
        img_paths,
        img_mask,
        size,
        raymaps=None,
        raymap_mask=None,
        revisit=1,
        update=True,
        crop=True,
    ):
        images = load_images(img_paths, size=size, crop=crop)
        views = []
        if raymaps is None and raymap_mask is None:
            num_views = len(images)

            for i in range(num_views):
                view = {
                    "img": images[i]["img"].to(device='cuda'),
                    "ray_map": torch.full(
                        (
                            images[i]["img"].shape[0],
                            6,
                            images[i]["img"].shape[-2],
                            images[i]["img"].shape[-1],
                        ),
                        torch.nan,
                    ).to(device='cuda'),
                    "true_shape": torch.from_numpy(images[i]["true_shape"]).to(device='cuda'),
                    "idx": i,
                    "instance": str(i),
                    "camera_pose": torch.from_numpy(
                        np.eye(4).astype(np.float32)
                    ).unsqueeze(0).to(device='cuda'),
                    "img_mask": torch.tensor(True).unsqueeze(0).to(device='cuda'),
                    "ray_mask": torch.tensor(False).unsqueeze(0).to(device='cuda'),
                    "update": torch.tensor(True).unsqueeze(0).to(device='cuda'),
                    "reset": torch.tensor(False).unsqueeze(0).to(device='cuda'),
                }
                views.append(view)
        else:

            num_views = len(images) + len(raymaps)
            assert len(img_mask) == len(raymap_mask) == num_views
            assert sum(img_mask) == len(images) and sum(raymap_mask) == len(raymaps)

            j = 0
            k = 0
            for i in range(num_views):
                view = {
                    "img": (
                        images[j]["img"].to(device='cuda')
                        if img_mask[i]
                        else torch.full_like(images[0]["img"], torch.nan).to(device='cuda')
                    ),
                    "ray_map": (
                        raymaps[k].to(device='cuda')
                        if raymap_mask[i]
                        else torch.full_like(raymaps[0], torch.nan).to(device='cuda')
                    ),
                    "true_shape": (
                        torch.from_numpy(images[j]["true_shape"]).to(device='cuda')
                        if img_mask[i]
                        else torch.from_numpy(np.int32([raymaps[k].shape[1:-1][::-1]])).to(device='cuda')
                    ),
                    "idx": i,
                    "instance": str(i),
                    "camera_pose": torch.from_numpy(
                        np.eye(4).astype(np.float32)
                    ).unsqueeze(0).to(device='cuda'),
                    "img_mask": torch.tensor(img_mask[i]).unsqueeze(0).to(device='cuda'),
                    "ray_mask": torch.tensor(raymap_mask[i]).unsqueeze(0).to(device='cuda'),
                    "update": torch.tensor(img_mask[i]).unsqueeze(0).to(device='cuda'),
                    "reset": torch.tensor(False).unsqueeze(0).to(device='cuda'),
                }
                if img_mask[i]:
                    j += 1
                if raymap_mask[i]:
                    k += 1
                views.append(view)
            assert j == len(images) and k == len(raymaps)

        if revisit > 1:
            # repeat input for 'revisit' times
            new_views = []
            for r in range(revisit):
                for i in range(len(views)):
                    new_view = deepcopy(views[i])
                    new_view["idx"] = r * len(views) + i
                    new_view["instance"] = str(r * len(views) + i)
                    if r > 0:
                        if not update:
                            new_view["update"] = torch.tensor(False).unsqueeze(0)
                    new_views.append(new_view)
            return new_views
        return views

    def prepare_output(outputs, revisit=1):
        valid_length = len(outputs["pred"]) // revisit
        outputs["pred"] = outputs["pred"][-valid_length:]
        outputs["views"] = outputs["views"][-valid_length:]

        pts3ds_self = [output["depth"].cpu() for output in outputs["pred"]]
        conf_self = [output["depth_conf"].cpu() for output in outputs["pred"]]
        pts3ds_self = torch.cat(pts3ds_self, 0)
        return (
            pts3ds_self,
            conf_self,
        )

    # Three-model construction block, mirroring eval/mv_recon/launch.py (OVGGT / VGGT /
    # StreamVGGT). Load/eval/to-cuda consolidated (identical semantics to mv_recon's
    # per-branch load); StreamVGGT gets its FrameVGGT memory config afterwards.
    model_name = args.model_name
    if model_name == "OVGGT":
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
    elif model_name == "VGGT":
        from vggt.models.vggt import VGGT
        model = VGGT()
    elif model_name == "StreamVGGT":
        # FrameVGGT baseline: point FRAMEVGGT_SRC at the FrameVGGT source tree.
        import os as _os, sys as _sys
        _fv = _os.environ.get("FRAMEVGGT_SRC", "")
        if _fv and _fv not in _sys.path:
            _sys.path.append(_fv)
        from streamvggt.models.streamvggt import StreamVGGT
        model = StreamVGGT()
    else:
        raise NotImplementedError
    ckpt = torch.load(args.weights, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    model = model.to("cuda")
    del ckpt
    if model_name == "StreamVGGT":
        model._eval_score_mode = args.fv_score_mode
        model.set_memory_config(
            recent_frames=args.recent_frames,
            middle_frames=args.middle_frames,
            anchor_frames=args.anchor_frames,
            anchor_every=args.anchor_every,
        )
    with torch.no_grad():
        eval_pose_estimation(args, model, save_dir=args.output_dir)
