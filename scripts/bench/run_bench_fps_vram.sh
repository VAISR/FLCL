#!/bin/bash
# Streaming throughput / peak-VRAM benchmark for FLCL on one 7-Scenes sequence.
# Each checkpoint length in FRAMES_LIST is measured independently (fresh CUDA stats),
# frames are moved to the GPU one at a time, bf16 autocast.
#
# Environment variables:
#   OUT          output csv (appended)             default: eval_results/bench/fps_vram.csv
#   FRAMES_LIST  comma-separated frame counts      default: 10..100 step 10, 150..500 step 50, 600..1000 step 100
#   SCENE        7-Scenes scene                    default: chess
#   CKPT         checkpoint path                   default: ckpt/checkpoints.pth
#   EXTRA        extra flags forwarded to bench_fps_vram.py
#
# Example (smoke): FRAMES_LIST=10 bash scripts/bench/run_bench_fps_vram.sh
set -e

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC_DIR="$PROJ/src"
CKPT=${CKPT:-$PROJ/ckpt/checkpoints.pth}
OUT=${OUT:-$PROJ/eval_results/bench/fps_vram.csv}
SCENE=${SCENE:-chess}
mkdir -p "$(dirname "$OUT")"

cd "$SRC_DIR"
export PYTHONPATH="$SRC_DIR${PYTHONPATH:+:$PYTHONPATH}"
python "$PROJ/scripts/bench/bench_fps_vram.py" \
  --model FLCL --weights "$CKPT" --root_7scenes "$SRC_DIR/data/7scenes" \
  --scene "$SCENE" --out "$OUT" \
  ${FRAMES_LIST:+--frames_list "$FRAMES_LIST"} ${EXTRA:-}

echo ">>> DONE. csv: $OUT"
echo ">>> Plot with: python scripts/bench/plot_fps_vram.py --csv $OUT --outdir $(dirname "$OUT")"
