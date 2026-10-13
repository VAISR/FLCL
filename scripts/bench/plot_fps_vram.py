"""Plot the 5-method FPS / VRAM / wall-clock comparison from the benchmark CSVs.

Reads one or more benchmark CSVs written by bench_fps_vram.py (same columns)
and renders three charts into --outdir:
  1. fps_vs_frames      — average FPS vs frames processed
  2. vram_vs_frames     — peak allocated VRAM vs frames processed
  3. time_vs_frames     — cumulative inference time vs frames processed

Palette/marks follow the dataviz reference palette (validated 5-slot categorical,
light surface); low-contrast slots get relief via direct end labels.
"""

import argparse
import csv
import os
from collections import defaultdict

import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Optional: set FLCL_FONT_DIR to a directory of Inter-*.otf files to match the
# paper's typeface; otherwise matplotlib's default font is used.
for _f in glob.glob(os.path.join(os.environ.get("FLCL_FONT_DIR", ""), "Inter-*.otf")):
    font_manager.fontManager.addfont(_f)
if any(f.name == "Inter" for f in font_manager.fontManager.ttflist):
    plt.rcParams["font.family"] = "Inter"

SURFACE = "#fcfcfb"
TEXT_1 = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e9e8e6"
AXIS = "#b0afa9"

# Fixed entity->color mapping (categorical slots 1-5, validated order).
# linestyle separates FLCL from OVGGT where the two curves nearly coincide.
SERIES = [
    # key            label            color      lw   ls    z
    ("FLCL",         "FLCL (Ours)",   "#2a78d6", 2.6, "-",  5),
    ("FrameVGGT",    "FrameVGGT",     "#eb6834", 1.8, "-",  4),
    ("OVGGT",        "OVGGT",         "#1baf7a", 1.8, "--", 3),
    ("InfiniteVGGT", "InfiniteVGGT",  "#eda100", 1.8, "-",  2),
    ("StreamVGGT",   "StreamVGGT",    "#e87ba4", 1.8, "-",  2),
]

FOOTNOTE = "7-Scenes · 1000-frame stream · 518×392 · RTX 5090 32 GB · bf16"

# Legend ordering (colors stay bound to the entity regardless of this order).
LEGEND_ORDER = ["StreamVGGT", "InfiniteVGGT", "FrameVGGT", "OVGGT", "FLCL"]


def load(csv_paths):
    data = defaultdict(list)  # model -> list of row dicts
    for path in csv_paths:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                data[row["model"]].append(
                    {k: float(v) for k, v in row.items() if k != "model"}
                )
    for rows in data.values():
        rows.sort(key=lambda r: r["actual_frames"])
    return data


def style_axis(ax, ylab, surface=SURFACE, label_size=10, tick_size=9):
    ax.set_facecolor(surface)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=TEXT_2, labelsize=tick_size)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Frames processed", color=TEXT_2, fontsize=label_size)
    ax.set_ylabel(ylab, color=TEXT_2, fontsize=label_size)


def new_fig(ylab):
    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_axis(ax, ylab)
    return fig, ax


def draw_series(ax, data, ycol, oom_note=True, ann_size=8.5):
    ends = []
    for key, label, color, lw, ls, z in SERIES:
        rows = data.get(key)
        if not rows:
            continue
        x = [r["actual_frames"] for r in rows]
        y = [r[ycol] for r in rows]
        ax.plot(x, y, color=color, lw=lw, ls=ls, zorder=z, label=label,
                marker="o", markersize=3.2, markevery=max(1, len(x) // 8),
                markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=0.6,
                solid_capstyle="round")
        # StreamVGGT's stream dies early (OOM on a 32 GB card): mark the end.
        if key == "StreamVGGT" and x[-1] < 900:
            ax.plot(x[-1], y[-1], marker="x", color=color, markersize=8,
                    markeredgewidth=2.2, zorder=6)
            if oom_note:
                ax.annotate("OOM", (x[-1], y[-1]), textcoords="offset points",
                            xytext=(8, 4), color=TEXT_1, fontsize=ann_size)
        ends.append((y[-1], x[-1], label))
    return ends


def end_labels(ax, ends, min_gap_frac=0.055):
    """Direct labels at each line's end, pushed apart to avoid collisions
    (relief for the low-contrast palette slots)."""
    lo, hi = ax.get_ylim()
    min_gap = (hi - lo) * min_gap_frac
    ends = sorted(ends)  # by end y
    ys = [e[0] for e in ends]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < min_gap:
            ys[i] = ys[i - 1] + min_gap
    for (y0, x0, label), y in zip(ends, ys):
        ax.annotate(label, (x0, y), textcoords="offset points", xytext=(7, -3),
                    color=TEXT_1, fontsize=8.5, ha="left")


def ordered_legend(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    key = {label: k for k, label, *_ in SERIES}
    order = sorted(range(len(labels)),
                   key=lambda i: LEGEND_ORDER.index(key.get(labels[i], labels[i])))
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              frameon=False, labelcolor=TEXT_2, **kwargs)


def finish(fig, ax, out_base):
    ordered_legend(ax, loc="best", fontsize=8.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", facecolor=SURFACE)
    plt.close(fig)
    print(f"[plot] wrote {out_base}.png/.pdf")


def paper_figure(data, out_base):
    """Combined two-panel (throughput + peak VRAM) figure sized for a
    \\linewidth float in docs/latex5; no titles/footnotes (caption carries)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.4), dpi=200)
    fig.patch.set_facecolor("white")
    for ax, ylab in ((ax1, "Average FPS"), (ax2, "Peak VRAM (GB)")):
        style_axis(ax, ylab, surface="white", label_size=12, tick_size=10.5)

    draw_series(ax1, data, "avg_fps", oom_note=True, ann_size=10)
    ax1.set_xlim(0, 1050)
    ax1.set_ylim(bottom=0)

    draw_series(ax2, data, "peak_alloc_gb", oom_note=True, ann_size=10)
    ax2.set_xlim(0, 1050)
    ax2.set_ylim(0, 34.5)
    ax2.axhline(32, color=AXIS, lw=1.0, ls=(0, (4, 3)))
    ax2.annotate("GPU capacity (32 GB)", (30, 32), textcoords="offset points",
                 xytext=(0, 5), color=TEXT_2, fontsize=9.5)
    # legend lives in the VRAM panel's empty band between the capacity line
    # (32 GB) and the InfiniteVGGT plateau (15 GB)
    ordered_legend(ax2, loc="upper right", bbox_to_anchor=(1.0, 0.88),
                   fontsize=9.5)

    fig.tight_layout(w_pad=2.2)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", facecolor="white")
    plt.close(fig)
    print(f"[plot] wrote {out_base}.png/.pdf (paper figure)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", nargs="+",
                   default=["eval_results/bench/fps_vram.csv"])
    p.add_argument("--outdir", default="eval_results/bench")
    args = p.parse_args()

    data = load(args.csv)
    print(f"[plot] models: { {k: len(v) for k, v in data.items()} }")
    os.makedirs(args.outdir, exist_ok=True)

    # 1. throughput
    fig, ax = new_fig("Average FPS")
    ends = draw_series(ax, data, "avg_fps")
    ax.set_xlim(0, 1150)
    ax.set_ylim(bottom=0)
    end_labels(ax, [e for e in ends if e[1] > 900])
    finish(fig, ax, os.path.join(args.outdir, "fps_vs_frames"))

    # 2. peak VRAM
    fig, ax = new_fig("Peak allocated VRAM (GB)")
    ends = draw_series(ax, data, "peak_alloc_gb")
    ax.set_xlim(0, 1150)
    ax.set_ylim(0, 34.5)
    ax.axhline(32, color=AXIS, lw=1.0, ls=(0, (4, 3)))
    ax.annotate("GPU capacity (32 GB)", (20, 32), textcoords="offset points",
                xytext=(0, 5), color=TEXT_2, fontsize=8)
    end_labels(ax, [e for e in ends if e[1] > 900])
    finish(fig, ax, os.path.join(args.outdir, "vram_vs_frames"))

    # 3. cumulative wall-clock time
    fig, ax = new_fig("Total inference time (s)")
    ends = draw_series(ax, data, "time_s")
    ax.set_xlim(0, 1150)
    ax.set_ylim(bottom=0)
    end_labels(ax, [e for e in ends if e[1] > 900])
    finish(fig, ax, os.path.join(args.outdir, "time_vs_frames"))

    # 4. combined two-panel figure for the paper (FPS + VRAM)
    paper_figure(data, os.path.join(args.outdir, "efficiency"))


if __name__ == "__main__":
    main()
