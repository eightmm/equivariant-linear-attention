#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${1:-artifacts/egnn-matched-baseline-development-20260718}"

cd "$repo_root"
mkdir -p "$output_dir"

common=(
  uv run --locked python scripts/train_compare.py
  --dataset synthetic
  --num-samples 64
  --train-size 44
  --val-size 10
  --batch-size 8
  --steps 20
  --num-layers 3
  --split-seed 42
  --model-seed 42
  --seed 42
  --device cpu
  --skip-test-eval
)

run_arm() {
  local name="$1"
  shift
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "${common[@]}" "$@" \
    --metrics-out "$output_dir/cpu-$name.json"
}

run_arm ggg-learned \
  --hidden-dim 16 --num-heads 4 --routing ggg \
  --global-transport-mode learned --bounded-diagnostics
run_arm lgg-learned \
  --hidden-dim 16 --num-heads 4 --routing lgg \
  --global-transport-mode learned --bounded-diagnostics
run_arm ggl-learned \
  --hidden-dim 16 --num-heads 4 --routing ggl \
  --global-transport-mode learned --bounded-diagnostics
run_arm lgl-learned \
  --hidden-dim 16 --num-heads 4 --routing lgl \
  --global-transport-mode learned --bounded-diagnostics
run_arm lgl-uniform \
  --hidden-dim 16 --num-heads 4 --routing lgl \
  --global-transport-mode uniform --bounded-diagnostics
run_arm lgl-none \
  --hidden-dim 16 --num-heads 4 --routing lgl \
  --global-transport-mode none --bounded-diagnostics
run_arm egnn-static \
  --benchmark-model internal_static_egnn_baseline --hidden-dim 16
run_arm matched-lgl-h64 \
  --hidden-dim 64 --num-heads 4 --routing lgl \
  --global-transport-mode learned
run_arm matched-egnn-h91 \
  --benchmark-model internal_static_egnn_baseline --hidden-dim 91
