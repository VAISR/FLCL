#!/usr/bin/env python
"""Camera-frustum trajectory plot (predicted only) for pipeline figures.

Reads predicted world->cam extrinsics ([S,3,4]) and draws one wireframe camera
frustum per frame plus the camera-center trajectory. No GT, clean background,
time-ordered color gradient. Intended to be dropped into a method/pipeline figure.

    python plot_camera_frustums.py \
        --extr <path/poses_pred_extr_w2c.npy> --out <dir> [--elev 22 --azim -64]
"""
import os
import argparse
import numpy as np


def catmull_rom(P, per=24, alpha=0.5):
    """Smooth curve through points P (centripetal Catmull-Rom, no self-overshoot)."""
    P = np.asarray(P, float)
    if len(P) < 3:
        return P
    P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])  # phantom endpoints

    def _t(ti, pi, pj):
        return ti + max(float(np.linalg.norm(pj - pi)), 1e-8) ** alpha

    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        t0 = 0.0
        t1 = _t(t0, p0, p1)
        t2 = _t(t1, p1, p2)
        t3 = _t(t2, p2, p3)
        t = np.linspace(t1, t2, per, endpoint=False)[:, None]
        A1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        A2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        A3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
        B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
        out.append((t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2)
    out.append(P[-2][None])
    return np.vstack(out)


def main():
    PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    default_extr = os.path.join(PROJ, "outputs/eth3d_demo/trajectory/poses_pred_extr_w2c.npy")
    ap = argparse.ArgumentParser()
    ap.add_argument("--extr", default=default_extr, help="[S,3,4] world->cam extrinsics .npy")
    ap.add_argument("--out", default=os.path.join(PROJ, "outputs/eth3d_demo/trajectory"))
    ap.add_argument("--name", default="kicker")
    ap.add_argument("--num_cams", type=int, default=0, help="use only the first N cameras (0 = all)")
    ap.add_argument("--aspect", type=float, default=518.0 / 392.0, help="image W/H for frustum shape")
    ap.add_argument("--vfov_deg", type=float, default=48.0, help="vertical FoV used for frustum shape")
    ap.add_argument("--frustum_scale", type=float, default=0.0,
                    help="frustum depth in world units; 0 = auto (fraction of trajectory extent)")
    ap.add_argument("--frustum_frac", type=float, default=0.045, help="auto frustum size as frac of bbox diag")
    ap.add_argument("--up_axis", choices=["y", "z"], default="y",
                    help="which data axis is vertical; 'y' => x-z is the ground plane (default)")
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--traj", choices=["curve", "line"], default="curve",
                    help="connect cameras with a smooth spline ('curve') or straight segments")
    ap.add_argument("--elev", type=float, default=22.0)
    ap.add_argument("--azim", type=float, default=-64.0)
    ap.add_argument("--sweep", action="store_true",
                    help="render a montage over many (elev,azim) views to pick an angle")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    extr = np.load(args.extr)  # [S,3,4] world->cam
    if args.num_cams > 0:
        extr = extr[: args.num_cams]
    S = extr.shape[0]
    name = args.name if args.num_cams <= 0 else f"{args.name}_first{S}"
    R = extr[:, :3, :3]                       # [S,3,3]
    t = extr[:, :3, 3]                        # [S,3]
    C = np.einsum("sij,sj->si", np.transpose(R, (0, 2, 1)), -t)  # camera centers, world

    bbox = C.max(0) - C.min(0)
    diag = float(np.linalg.norm(bbox)) or 1.0
    d = args.frustum_scale if args.frustum_scale > 0 else args.frustum_frac * diag
    hh = d * np.tan(np.deg2rad(args.vfov_deg) / 2.0)  # half height at frustum depth
    hw = hh * args.aspect                             # half width

    # frustum corners in camera coords (OpenCV: +z forward), apex at origin
    corners_cam = np.array([
        [-hw, -hh, d], [hw, -hh, d], [hw, hh, d], [-hw, hh, d]
    ])  # [4,3]

    colors = cm.get_cmap(args.cmap)(np.linspace(0, 1, S))

    # Put the chosen "up" data-axis into matplotlib's 3rd (vertical) slot so the
    # ground plane is x-z when up_axis='y'. perm maps data (x,y,z) -> plot axes.
    perm, labs = ({"y": ([0, 2, 1], ("x", "z", "y")),
                   "z": ([0, 1, 2], ("x", "y", "z"))})[args.up_axis]

    def render(ax, transparent_axes, elev=None, azim=None):
        Cp = C[:, perm]
        # trajectory through camera centers: smooth spline or straight segments
        traj = catmull_rom(Cp, per=24) if args.traj == "curve" else Cp
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], "-", color="0.35",
                lw=1.6, alpha=0.95, zorder=1)
        edges, faces, face_colors = [], [], []
        all_pts = [Cp]
        for s in range(S):
            Rw = R[s].T                    # cam->world rotation
            base = ((corners_cam @ Rw.T) + C[s])[:, perm]   # [4,3] frustum base, plot axes
            apex = Cp[s]
            # 4 side edges apex->corner + 4 base edges
            for k in range(4):
                edges.append([apex, base[k]])
                edges.append([base[k], base[(k + 1) % 4]])
            faces.append(base)
            face_colors.append(colors[s])
            all_pts.append(base)
        lc = Line3DCollection(edges, colors="0.15", linewidths=0.8, zorder=3)
        ax.add_collection3d(lc)
        pc = Poly3DCollection(faces, facecolors=face_colors, edgecolors="none", alpha=0.55, zorder=2)
        ax.add_collection3d(pc)
        # camera centers as dots
        ax.scatter(Cp[:, 0], Cp[:, 1], Cp[:, 2], s=8, c=colors, depthshade=False, zorder=4)

        # tight equal-aspect limits hugging the actual frustum geometry
        all_pts = np.concatenate(all_pts, 0)
        lo, hi = all_pts.min(0), all_pts.max(0)
        ctr = (lo + hi) / 2.0
        r = float((hi - lo).max()) * 0.5 * 1.08   # 8% margin
        ax.set_xlim(ctr[0] - r, ctr[0] + r)
        ax.set_ylim(ctr[1] - r, ctr[1] + r)
        ax.set_zlim(ctr[2] - r, ctr[2] + r)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        ax.view_init(elev=args.elev if elev is None else elev,
                     azim=args.azim if azim is None else azim)
        if transparent_axes:
            ax.set_axis_off()
        else:
            ax.set_xlabel(labs[0]); ax.set_ylabel(labs[1]); ax.set_zlabel(labs[2])
            ax.grid(True, alpha=0.25)
            for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
                pane.pane.set_alpha(0.04)

    os.makedirs(args.out, exist_ok=True)

    if args.sweep:
        elevs = [20, 45, 70]
        azims = [-90, -60, -30, 0]
        fig = plt.figure(figsize=(len(azims) * 3.2, len(elevs) * 3.2))
        for i, el in enumerate(elevs):
            for j, az in enumerate(azims):
                ax = fig.add_subplot(len(elevs), len(azims), i * len(azims) + j + 1,
                                     projection="3d")
                render(ax, transparent_axes=True, elev=el, azim=az)
                ax.set_title(f"elev={el}, azim={az}", fontsize=9)
        fig.suptitle(f"{name}: view sweep (pick elev/azim)", fontsize=12)
        fig.tight_layout()
        p_sweep = os.path.join(args.out, f"{name}_frustums_sweep.png")
        fig.savefig(p_sweep, dpi=115, bbox_inches="tight")
        plt.close(fig)
        print(f">>> {S} frustums, frustum_depth={d:.4f}")
        print(f">>> saved sweep -> {p_sweep}")
        return

    # (a) labeled version (white bg, faint axes) — for inspection
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    render(ax, transparent_axes=False)
    ax.set_title(f"{name}: predicted camera trajectory ({S} frustums)", fontsize=11)
    p_labeled = os.path.join(args.out, f"{name}_frustums.png")
    fig.savefig(p_labeled, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # (b) clean transparent version — for the pipeline figure
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    render(ax, transparent_axes=True)
    p_clean = os.path.join(args.out, f"{name}_frustums_transparent.png")
    fig.savefig(p_clean, dpi=240, bbox_inches="tight", transparent=True)
    plt.close(fig)

    print(f">>> {S} frustums, frustum_depth={d:.4f} (bbox diag={diag:.3f})")
    print(f">>> saved:\n  {p_labeled}\n  {p_clean}")


if __name__ == "__main__":
    main()
