#!/bin/bash
# Multi-view 3D reconstruction evaluation (7-Scenes / NRGBD / ETH3D) on a single GPU.
#
# Environment variables:
#   TAG         output tag (output directory name suffix)          default: flcl
#   MODEL_NAME  OVGGT | VGGT | StreamVGGT                          default: OVGGT
#   DATASET     7scenes | NRGBD | eth3d                             default: 7scenes
#   SCENE       restrict to one scene, e.g. chess                   default: all scenes
#   MAX_FRAMES  cap on frames per sequence                          default: full sequence
#   KF_EVERY    keyframe sampling stride                            default: 2
#   CKPT        checkpoint path                                     default: ckpt/checkpoints.pth
#   EXTRA       extra flags forwarded to launch.py (see README "Flags")
#
# Examples:
#   TAG=smoke SCENE=chess MAX_FRAMES=20 bash scripts/eval/run_mv_recon.sh
#   TAG=7s_1000 MAX_FRAMES=1000 KF_EVERY=1 bash scripts/eval/run_mv_recon.sh
#   TAG=nrgbd_100 DATASET=NRGBD MAX_FRAMES=100 bash scripts/eval/run_mv_recon.sh
#   TAG=abl_token_cov MAX_FRAMES=1000 KF_EVERY=1 EXTRA="--ssc_score token_cov" bash scripts/eval/run_mv_recon.sh
set -e

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC_DIR="$PROJ/src"
CKPT=${CKPT:-$PROJ/ckpt/checkpoints.pth}
MODEL_NAME=${MODEL_NAME:-OVGGT}
DATASET=${DATASET:-7scenes}
TAG=${TAG:-flcl}
KF_EVERY=${KF_EVERY:-2}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJ/eval_results/mv_recon_${TAG}}

ARGS=(--weights "$CKPT" --output_dir "$OUTPUT_DIR" --model_name "$MODEL_NAME" --dataset "$DATASET" --kf_every "$KF_EVERY")
[ -n "$SCENE" ]      && ARGS+=(--scene "$SCENE")
[ -n "$MAX_FRAMES" ] && ARGS+=(--max_frames "$MAX_FRAMES")
[ -n "$EXTRA" ]      && ARGS+=($EXTRA)

echo ">>> model=$MODEL_NAME dataset=$DATASET tag=$TAG scene=${SCENE:-all} max_frames=${MAX_FRAMES:-full} kf_every=$KF_EVERY"
echo ">>> output_dir=$OUTPUT_DIR"
echo ">>> launch args: ${ARGS[*]}"

cd "$SRC_DIR"
export PYTHONPATH="$SRC_DIR${PYTHONPATH:+:$PYTHONPATH}"
python ./eval/mv_recon/launch.py "${ARGS[@]}" "$@"

echo ">>> DONE. Summary: $OUTPUT_DIR/$DATASET/logs_all.txt (last 'mean' line)"
