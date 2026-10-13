#!/usr/bin/env python
"""ETH3D single-scene demo for FLCL.

Runs FLCL streaming inference on one ETH3D scene (default: kicker) and saves
three deliverables at full model resolution (392x518, no 224 center-crop):

  1) depth maps      -> <out>/depth/           per-frame colored PNG + raw .npy + grid
  2) point clouds    -> <out>/pointcloud/      fused predicted cloud (.ply, +voxel)
  3) trajectory plot -> <out>/trajectory/      predicted vs GT camera path (Umeyama-aligned)

This is a visualization/demo script; it does NOT touch the metric eval harness
(eval/mv_recon/launch.py). Default config = the paper setting:
  frame_cov + dap off (history_anchor_strategy='none') + uniform budget + 200k tokens.
"""
import os
import sys
import json
import argparse

import numpy as np
import torch

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(PROJ, "src")
sys.path.insert(0, SRC_DIR)  # so `flcl`, `eval`, `dust3r` resolve to this working tree


def umeyama(src, dst, with_scale=True):
    """Similarity transform aligning src -> dst (both Nx3). Returns s, R, t."""
    assert src.shape == dst.shape
    N = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    Sigma = dc.T @ sc / N
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    s = (D * S.diagonal()).sum() / (sc ** 2).sum() * N if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="kicker")
    ap.add_argument("--kf_every", type=int, default=1, help="frame sampling stride (kicker=31 imgs)")
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(PROJ, "outputs/eth3d_demo"))
    ap.add_argument("--weights", default=os.path.join(PROJ, "ckpt/checkpoints.pth"))
    ap.add_argument("--eth3d_root", default=os.path.join(SRC_DIR, "data/eth3d"))
    ap.add_argument("--conf_percentile", type=float, default=30.0,
                    help="drop the lowest-confidence X%% of predicted points from the fused cloud")
    ap.add_argument("--voxel_size", type=float, default=0.01, help="voxel size for the downsampled cloud")
    # ---- FLCL defaults (paper setting) ----
    ap.add_argument("--max_token_budget", type=int, default=200000)
    ap.add_argument("--ssc_score", default="frame_cov")
    ap.add_argument("--budget_mode", default="uniform")
    ap.add_argument("--budget_norm", default="none")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import open3d as o3d

    from add_ckpt_path import add_path_to_dust3r
    add_path_to_dust3r(args.weights)
    from eval.mv_recon.data import ETH3D
    from flcl.models.ovggt import OVGGT
    from flcl.utils.pose_enc import pose_encoding_to_extri_intri
    from torch.utils.data._utils.collate import default_collate

    device = "cuda"
    resolution = (518, 392)

    d_depth = os.path.join(args.out, "depth")
    d_pcd = os.path.join(args.out, "pointcloud")
    d_traj = os.path.join(args.out, "trajectory")
    for d in (d_depth, d_pcd, d_traj):
        os.makedirs(d, exist_ok=True)

    print(f">>> scene={args.scene} kf_every={args.kf_every} budget={args.max_token_budget} "
          f"ssc={args.ssc_score} budget_mode={args.budget_mode}", flush=True)

    # ---- dataset (restrict to the one requested scene) ----
    ds = ETH3D(split="test", ROOT=args.eth3d_root, resolution=resolution,
               num_seq=1, full_video=True, kf_every=args.kf_every)
    ds.scene_list = [args.scene]
    ds.scenes = [args.scene]
    views = ds[0]
    if args.max_frames:
        views = views[: args.max_frames]
    S = len(views)
    frame_names = [os.path.splitext(os.path.basename(v["label"].split("/", 1)[1]))[0] for v in views]
    print(f">>> loaded {S} frames: {frame_names}", flush=True)

    # ---- model: FLCL ----
    model = OVGGT(
        total_budget=args.max_token_budget,
        ssc_score_mode=args.ssc_score,
        budget_mode=args.budget_mode,
        budget_norm=args.budget_norm,
    )
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    model.load_state_dict(ckpt, strict=True)
    model.eval().to(device)
    del ckpt

    # collate one sample -> list[S] of dicts each with batch dim 1 (matches launch.py)
    batch = default_collate([views])
    ignore = {"depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng"}
    for view in batch:
        for k in list(view.keys()):
            if k in ignore:
                continue
            v = view[k]
            if isinstance(v, (list, tuple)):
                view[k] = [x.to(device, non_blocking=True) if torch.is_tensor(x) else x for x in v]
            elif torch.is_tensor(v):
                view[k] = v.to(device, non_blocking=True)
        # normalize img [-1,1] -> [0,1] exactly like the eval harness
        view["img"] = (view["img"] + 1.0) / 2.0

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        results = model.inference(batch, history_anchor_strategy="none")  # dap off
    preds, pviews = results.ress, results.views
    print(f">>> inference done, {len(preds)} predictions", flush=True)

    # ============ 1) DEPTH MAPS ============
    depth_maps = []
    for s in range(S):
        depth = preds[s]["depth"].float().squeeze().cpu().numpy()  # [H,W]
        depth_maps.append(depth)
        np.save(os.path.join(d_depth, f"frame{s:02d}_{frame_names[s]}.npy"), depth.astype(np.float32))
        valid = depth > 0
        if valid.any():
            lo, hi = np.percentile(depth[valid], [2, 98])
        else:
            lo, hi = 0.0, 1.0
        norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
        colored = plt.get_cmap("turbo")(norm)[..., :3]
        colored[~valid] = 0
        plt.imsave(os.path.join(d_depth, f"frame{s:02d}_{frame_names[s]}_depth.png"), colored)
    # contact-sheet grids (RGB + depth)
    cols = min(6, S)
    rows = int(np.ceil(S / cols))
    for kind, getter in (("rgb", None), ("depth", None)):
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.0))
        axes = np.atleast_1d(axes).ravel()
        for s in range(rows * cols):
            axes[s].axis("off")
            if s >= S:
                continue
            if kind == "rgb":
                img = pviews[s]["img"][0].permute(1, 2, 0).float().cpu().numpy()
                axes[s].imshow(np.clip(img, 0, 1))
            else:
                dm = depth_maps[s]
                v = dm > 0
                lo, hi = np.percentile(dm[v], [2, 98]) if v.any() else (0, 1)
                axes[s].imshow(np.clip((dm - lo) / max(hi - lo, 1e-6), 0, 1), cmap="turbo")
            axes[s].set_title(frame_names[s], fontsize=6)
        fig.suptitle(f"ETH3D/{args.scene} — {kind} ({S} frames)", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(d_depth, f"{kind}_grid.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)
    print(f">>> saved {S} depth maps + grids -> {d_depth}", flush=True)

    # ============ 2) POINT CLOUD (fused predicted) ============
    pts_list, col_list, conf_list = [], [], []
    for s in range(S):
        pts = preds[s]["pts3d_in_other_view"].float().squeeze(0).cpu().numpy().reshape(-1, 3)
        col = pviews[s]["img"][0].permute(1, 2, 0).float().cpu().numpy().reshape(-1, 3)
        conf = preds[s]["conf"].float().squeeze(0).cpu().numpy().reshape(-1)
        pts_list.append(pts)
        col_list.append(col)
        conf_list.append(conf)
    pts_all = np.concatenate(pts_list, 0)
    col_all = np.clip(np.concatenate(col_list, 0), 0, 1)
    conf_all = np.concatenate(conf_list, 0)
    finite = np.isfinite(pts_all).all(1)
    thr = np.percentile(conf_all[finite], args.conf_percentile)
    keep = finite & (conf_all >= thr)
    print(f">>> point cloud: {keep.sum()}/{keep.size} pts kept (conf>={thr:.3f}, "
          f"drop lowest {args.conf_percentile:.0f}%)", flush=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_all[keep])
    pcd.colors = o3d.utility.Vector3dVector(col_all[keep])
    ply_full = os.path.join(d_pcd, f"{args.scene}_pred.ply")
    o3d.io.write_point_cloud(ply_full, pcd)
    pcd_v = pcd.voxel_down_sample(args.voxel_size)
    ply_voxel = os.path.join(d_pcd, f"{args.scene}_pred_voxel{args.voxel_size}.ply")
    o3d.io.write_point_cloud(ply_voxel, pcd_v)
    print(f">>> saved point clouds: {ply_full} ({len(pcd.points)} pts), "
          f"{ply_voxel} ({len(pcd_v.points)} pts)", flush=True)

    # ============ 3) TRAJECTORY (predicted vs GT, Umeyama-aligned) ============
    pose_enc = torch.stack([preds[s]["camera_pose"] for s in range(S)], dim=1).float()  # [1,S,9]
    H, W = depth_maps[0].shape
    extr, _ = pose_encoding_to_extri_intri(pose_enc, (H, W))  # extr: [1,S,3,4] world->cam
    extr = extr[0].cpu().numpy()
    cen_pred = np.stack([-e[:3, :3].T @ e[:3, 3] for e in extr], 0)  # pred cam centers

    gt_ok = True
    try:
        cen_gt = np.stack([np.asarray(pviews[s]["camera_pose"][0].cpu())[:3, 3] for s in range(S)], 0)
        if not np.isfinite(cen_gt).all():
            gt_ok = False
    except Exception:
        gt_ok = False

    ate = None
    cen_pred_aln = cen_pred
    if gt_ok:
        s_, R_, t_ = umeyama(cen_pred, cen_gt, with_scale=True)
        cen_pred_aln = (s_ * (R_ @ cen_pred.T)).T + t_
        ate = float(np.sqrt(((cen_pred_aln - cen_gt) ** 2).sum(1).mean()))
        print(f">>> trajectory ATE-RMSE (aligned) = {ate:.4f} m", flush=True)

    np.save(os.path.join(d_traj, "poses_pred_extr_w2c.npy"), extr)
    np.save(os.path.join(d_traj, "centers_pred_aligned.npy"), cen_pred_aln)
    if gt_ok:
        np.save(os.path.join(d_traj, "centers_gt.npy"), cen_gt)

    fig = plt.figure(figsize=(13, 5.5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.plot(*cen_pred_aln.T, "-o", ms=3, color="tab:red", label="pred (aligned)")
    if gt_ok:
        ax3d.plot(*cen_gt.T, "-o", ms=3, color="tab:green", label="GT")
    ax3d.set_title(f"ETH3D/{args.scene} camera trajectory (3D)")
    ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")
    ax3d.legend(fontsize=8)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(cen_pred_aln[:, 0], cen_pred_aln[:, 2], "-o", ms=3, color="tab:red", label="pred (aligned)")
    if gt_ok:
        ax2.plot(cen_gt[:, 0], cen_gt[:, 2], "-o", ms=3, color="tab:green", label="GT")
    for i in range(S):
        ax2.annotate(str(i), (cen_pred_aln[i, 0], cen_pred_aln[i, 2]), fontsize=5)
    ax2.set_title("top-down (x-z)" + (f"  |  ATE={ate:.3f} m" if ate is not None else ""))
    ax2.set_xlabel("x"); ax2.set_ylabel("z"); ax2.axis("equal"); ax2.legend(fontsize=8)
    fig.tight_layout()
    traj_png = os.path.join(d_traj, f"{args.scene}_trajectory.png")
    fig.savefig(traj_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f">>> saved trajectory plot -> {traj_png}", flush=True)

    # ---- manifest ----
    meta = {
        "scene": args.scene, "dataset": "ETH3D", "num_frames": S, "frames": frame_names,
        "resolution_hw": [int(H), int(W)],
        "method": {"model": "OVGGT", "ssc_score": args.ssc_score, "dap": "off",
                   "budget_mode": args.budget_mode, "budget_norm": args.budget_norm,
                   "max_token_budget": args.max_token_budget},
        "point_cloud": {"kept_points": int(keep.sum()), "total_points": int(keep.size),
                        "conf_percentile_dropped": args.conf_percentile,
                        "voxel_size": args.voxel_size},
        "trajectory_ate_rmse_m": ate,
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f">>> DONE. outputs under {args.out}", flush=True)


if __name__ == "__main__":
    main()
