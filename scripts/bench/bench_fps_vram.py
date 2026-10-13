"""FPS / peak-VRAM benchmark for streaming models on one 7-Scenes sequence.

Methodology (the same protocol is used for every method reported in the paper):
  - one long test sequence (default: chess test seq, kf_every=1, 1000 frames),
    prefix-sliced to each N in --frames_list (SevenScenes max_frames is a prefix
    cut, so slicing the full load is bit-identical to reloading with max_frames=N)
  - per N: gc + empty_cache + reset_peak_memory_stats, then one full streaming
    inference under bf16 autocast, cuda.synchronize, record wall time and
    torch.cuda.max_memory_allocated / max_memory_reserved
  - frames live on CPU and are moved to GPU one at a time (DeviceStream), so the
    whole video never sits in VRAM — same footprint shape as the baseline rows
  - cache_results=False: per-frame results are still computed and copied to CPU
    (same work as eval) but not accumulated, keeping host RAM flat at 1000f

Run via scripts/bench/run_bench_fps_vram.sh, which sets cwd/PYTHONPATH to src/.
The FrameVGGT option needs the FrameVGGT source tree on PYTHONPATH.
"""

import argparse
import csv
import gc
import os
import time

import torch
from torch.utils.data._utils.collate import default_collate

DEFAULT_GRID = ",".join(
    [str(n) for n in range(10, 101, 10)]
    + [str(n) for n in range(150, 501, 50)]
    + [str(n) for n in range(600, 1001, 100)]
)


def parse_args():
    p = argparse.ArgumentParser("FPS/VRAM benchmark for FLCL / FrameVGGT")
    p.add_argument("--model", required=True, choices=["FLCL", "FrameVGGT"])
    p.add_argument("--weights", required=True)
    p.add_argument("--root_7scenes", required=True)
    p.add_argument("--scene", default="chess")
    p.add_argument("--kf_every", type=int, default=1)
    p.add_argument("--frames_list", default=DEFAULT_GRID,
                   help="comma-separated max_frames checkpoints")
    p.add_argument("--out", required=True, help="output csv (appended)")
    p.add_argument("--warmup_frames", type=int, default=5,
                   help="frames for one unmeasured warm-up inference (0=off)")
    # FLCL knobs (defaults = paper setting)
    p.add_argument("--max_token_budget", type=int, default=200000)
    p.add_argument("--camera_budget", type=int, default=384)
    p.add_argument("--ssc_score", default="frame_cov")
    p.add_argument("--budget_mode", default="uniform")
    p.add_argument("--budget_norm", default="none")
    p.add_argument("--dap", default="off", choices=["on", "off"])
    # FrameVGGT knobs (baseline; defaults follow the FrameVGGT paper)
    p.add_argument("--recent", type=int, default=2)
    p.add_argument("--middle", type=int, default=4)
    p.add_argument("--anchor", type=int, default=2)
    p.add_argument("--anchor_every", type=int, default=50)
    p.add_argument("--fv_score_mode", default="cosine_diversity")
    return p.parse_args()


class DeviceStream:
    """Lazily moves one CPU frame at a time onto the GPU, mimicking a real
    stream so the full video never counts toward peak VRAM."""

    def __init__(self, views, device):
        self.views = views
        self.device = device

    def __len__(self):
        return len(self.views)

    def __getitem__(self, i):
        return self._move(self.views[i])

    def __iter__(self):
        for v in self.views:
            yield self._move(v)

    def _move(self, view):
        return {
            k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in view.items()
        }


def load_frames(args, n_frames):
    from eval.mv_recon.data import SevenScenes

    dataset = SevenScenes(
        split="test",
        ROOT=args.root_7scenes,
        resolution=(518, 392),
        num_seq=1,
        full_video=True,
        kf_every=args.kf_every,
        test_id=args.scene,
        max_frames=n_frames,
    )
    views = default_collate([dataset[0]])
    label = views[0]["label"]
    if isinstance(label, (list, tuple)):
        label = label[0]
    # inference only reads frame["img"]; keep frames light and pre-normalise
    # to [0,1] exactly like eval launch.py does ((img+1)/2)
    frames = [{"img": ((v["img"] + 1.0) / 2.0)} for v in views]
    return frames, label


def build_model(args, device):
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    if args.model == "FLCL":
        from flcl.models.ovggt import OVGGT

        model = OVGGT(
            total_budget=args.max_token_budget,
            camera_budget=args.camera_budget,
            ssc_score_mode=args.ssc_score,
            budget_mode=args.budget_mode,
            budget_norm=args.budget_norm,
        )
        model.load_state_dict(ckpt, strict=True)
        anchor_strategy = "coverage" if args.dap == "on" else "none"

        def infer(frames):
            return model.inference(
                frames,
                cache_results=False,
                history_anchor_strategy=anchor_strategy,
            )
    else:
        from streamvggt.models.streamvggt import StreamVGGT

        model = StreamVGGT()
        model.load_state_dict(ckpt, strict=True)
        model._eval_score_mode = args.fv_score_mode
        model.set_memory_config(
            recent_frames=args.recent,
            middle_frames=args.middle,
            anchor_frames=args.anchor,
            anchor_every=args.anchor_every,
        )

        def infer(frames):
            return model.inference(
                frames,
                cache_results=False,
                recent_frames=args.recent,
                middle_frames=args.middle,
                anchor_frames=args.anchor,
                anchor_every=args.anchor_every,
            )

    model.eval()
    model.to(device)
    del ckpt
    gc.collect()
    return model, infer


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "benchmark requires a CUDA device"
    device = torch.device("cuda")
    grid = [int(x) for x in args.frames_list.split(",") if x.strip()]

    print(f"[bench] model={args.model} scene={args.scene} kf_every={args.kf_every} "
          f"grid={grid}", flush=True)

    frames_cpu, label = load_frames(args, max(grid))
    print(f"[bench] loaded {len(frames_cpu)} frames from {label}", flush=True)
    assert len(frames_cpu) >= max(grid), (
        f"sequence too short: {len(frames_cpu)} < {max(grid)}"
    )

    model, infer = build_model(args, device)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    write_header = not os.path.exists(args.out)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with torch.no_grad():
        if args.warmup_frames > 0:
            with torch.cuda.amp.autocast(dtype=dtype):
                infer(DeviceStream(frames_cpu[: args.warmup_frames], device))
            torch.cuda.synchronize()
            print(f"[bench] warm-up done ({args.warmup_frames} frames)", flush=True)

        for n in grid:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            stream = DeviceStream(frames_cpu[:n], device)
            with torch.cuda.amp.autocast(dtype=dtype):
                t0 = time.time()
                infer(stream)
                torch.cuda.synchronize()
                elapsed = time.time() - t0

            peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
            peak_res = torch.cuda.max_memory_reserved() / 1024**3
            row = [args.model, n, n, round(elapsed, 4), round(n / elapsed, 4),
                   round(peak_alloc, 4), round(peak_res, 4)]
            with open(args.out, "a", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(["model", "max_frames", "actual_frames", "time_s",
                                "avg_fps", "peak_alloc_gb", "peak_res_gb"])
                    write_header = False
                w.writerow(row)
            print(f"[bench] N={n} time={elapsed:.2f}s fps={n / elapsed:.2f} "
                  f"alloc={peak_alloc:.2f}GB res={peak_res:.2f}GB", flush=True)

    print("[bench] DONE", flush=True)


if __name__ == "__main__":
    main()
