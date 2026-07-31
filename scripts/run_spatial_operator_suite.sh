#!/usr/bin/env bash
set -euo pipefail

MODE="${SPATIAL_SUITE_MODE:-smoke}"
DEVICE="${SPATIAL_SUITE_DEVICE:-$(uv run python - <<'PY'
import torch
print('cuda' if torch.cuda.is_available() else 'cpu')
PY
)}"
DTYPE="${SPATIAL_SUITE_DTYPE:-$(if [[ "$DEVICE" == cuda* ]]; then echo bfloat16; else echo float64; fi)}"
RUN_DIR="${1:-artifacts/spatial-operator-comparison/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$RUN_DIR"

case "$MODE" in
  smoke)
    SEEDS="0"
    TRAIN_GRAPHS=8
    VALIDATION_GRAPHS=4
    NODES_PER_GRAPH=8
    HIDDEN_DIM=16
    LAYERS=1
    LOCAL_RANK=2
    STEPS=5
    EVAL_INTERVAL=1
    PROFILE_WARMUP=1
    PROFILE_REPEATS=2
    SCALE_NODES="64,128"
    SCALE_DEPTHS="1,2"
    SCALE_BLOCKS="1"
    SCALE_DEGREES="4"
    APPROX_NODES=64
    FRAGMENT_BASE_NODES=16
    FRAGMENT_NODES=4
    ;;
  full)
    SEEDS="${SPATIAL_SUITE_SEEDS:-0,1,2,3,4}"
    TRAIN_GRAPHS="${SPATIAL_SUITE_TRAIN_GRAPHS:-64}"
    VALIDATION_GRAPHS="${SPATIAL_SUITE_VALIDATION_GRAPHS:-32}"
    NODES_PER_GRAPH="${SPATIAL_SUITE_NODES_PER_GRAPH:-24}"
    HIDDEN_DIM="${SPATIAL_SUITE_HIDDEN_DIM:-64}"
    LAYERS="${SPATIAL_SUITE_LAYERS:-4}"
    LOCAL_RANK="${SPATIAL_SUITE_LOCAL_RANK:-4}"
    STEPS="${SPATIAL_SUITE_STEPS:-1000}"
    EVAL_INTERVAL="${SPATIAL_SUITE_EVAL_INTERVAL:-50}"
    PROFILE_WARMUP="${SPATIAL_SUITE_PROFILE_WARMUP:-10}"
    PROFILE_REPEATS="${SPATIAL_SUITE_PROFILE_REPEATS:-30}"
    SCALE_NODES="${SPATIAL_SUITE_SCALE_NODES:-256,512,1024,2048,4096}"
    SCALE_DEPTHS="${SPATIAL_SUITE_SCALE_DEPTHS:-4,8,16,32}"
    SCALE_BLOCKS="${SPATIAL_SUITE_SCALE_BLOCKS:-4,8}"
    SCALE_DEGREES="${SPATIAL_SUITE_SCALE_DEGREES:-8,16,32,64}"
    APPROX_NODES="${SPATIAL_SUITE_APPROX_NODES:-512}"
    FRAGMENT_BASE_NODES="${SPATIAL_SUITE_FRAGMENT_BASE_NODES:-64}"
    FRAGMENT_NODES="${SPATIAL_SUITE_FRAGMENT_NODES:-16}"
    ;;
  *)
    echo "SPATIAL_SUITE_MODE must be smoke or full" >&2
    exit 2
    ;;
esac

{
  echo "mode=$MODE"
  echo "device=$DEVICE"
  echo "dtype=$DTYPE"
  uv run python --version
  uv --version
  uv pip freeze
  nvidia-smi 2>/dev/null || true
} > "$RUN_DIR/environment.txt"

{
  git status --short
  git rev-parse HEAD
  git diff --stat
} > "$RUN_DIR/git.txt"

uv run pytest \
  tests/test_spatial_ablation.py \
  tests/test_spatial_benchmarks.py \
  tests/test_spatial_comparison.py \
  tests/test_fragment_locality.py \
  tests/test_implicit_spatial.py \
  tests/test_implicit_spatial_chunks.py \
  tests/test_implicit_spatial_gradients.py \
  tests/test_implicit_spatial_permutation.py \
  tests/test_implicit_spatial_residual.py \
  tests/test_implicit_spatial_validation.py \
  tests/test_scaling_contract.py \
  tests/test_scaling_memory.py \
  2>&1 | tee "$RUN_DIR/focused-tests.log"

if [[ "$DEVICE" == cuda* ]]; then
  uv run pytest tests/test_implicit_spatial_cuda.py \
    2>&1 | tee "$RUN_DIR/cuda-tests.log"
fi

if [[ "$MODE" == "full" ]]; then
  bash scripts/check.sh fast 2>&1 | tee "$RUN_DIR/fast-gate.log"
  if [[ "$DEVICE" == cuda* ]]; then
    bash scripts/check.sh gpu 2>&1 | tee "$RUN_DIR/gpu-gate.log"
  fi
fi

uv run python scripts/compare_spatial_operators.py \
  --tasks local_directional,smooth_gaussian,mixed \
  --seeds "$SEEDS" \
  --train-graphs "$TRAIN_GRAPHS" \
  --validation-graphs "$VALIDATION_GRAPHS" \
  --nodes-per-graph "$NODES_PER_GRAPH" \
  --hidden-dim "$HIDDEN_DIM" \
  --layers "$LAYERS" \
  --heads 4 \
  --local-rank "$LOCAL_RANK" \
  --cutoff 1.75 \
  --candidate-skin 0 \
  --gaussian-scale 2.5 \
  --implicit-scales 2,4,8 \
  --implicit-scale-init 0 \
  --steps "$STEPS" \
  --evaluation-interval "$EVAL_INTERVAL" \
  --profile-warmup "$PROFILE_WARMUP" \
  --profile-repeats "$PROFILE_REPEATS" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --output "$RUN_DIR/result.json" \
  > "$RUN_DIR/comparison.log"

uv run python -m json.tool "$RUN_DIR/result.json" > /dev/null
uv run python scripts/validate_spatial_operator_result.py \
  "$RUN_DIR/result.json" \
  2>&1 | tee "$RUN_DIR/protocol-validation.log"

uv run python scripts/report_spatial_operator_comparison.py \
  "$RUN_DIR/result.json" \
  --output "$RUN_DIR/report.md" \
  --decision-json "$RUN_DIR/decision.json" \
  > "$RUN_DIR/report.log"

uv run python scripts/evaluate_implicit_spatial.py \
  --nodes "$APPROX_NODES" \
  --scales 2,4,8 \
  --cutoff 1.75 \
  --output "$RUN_DIR/implicit-accuracy.json" \
  > "$RUN_DIR/implicit-accuracy.log"

uv run python scripts/evaluate_fragment_locality.py \
  --base-nodes "$FRAGMENT_BASE_NODES" \
  --fragment-nodes "$FRAGMENT_NODES" \
  --fragment-distance 20 \
  --value-width 16 \
  --cutoff 1.75 \
  --scales 2,4,8 \
  --output "$RUN_DIR/fragment-locality.json" \
  > "$RUN_DIR/fragment-locality.log"

uv run python scripts/benchmark_scaling.py \
  --modes base,attnres,implicit \
  --nodes "$SCALE_NODES" \
  --graphs 1 \
  --depths "$SCALE_DEPTHS" \
  --blocks "$SCALE_BLOCKS" \
  --degrees "$SCALE_DEGREES" \
  --warmup "$PROFILE_WARMUP" \
  --repeats "$PROFILE_REPEATS" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --output "$RUN_DIR/scaling.json" \
  > "$RUN_DIR/scaling.log"

uv run python scripts/benchmark_scaling.py \
  --modes base,attnres,implicit \
  --nodes "$SCALE_NODES" \
  --graphs 1 \
  --depths "$SCALE_DEPTHS" \
  --blocks "$SCALE_BLOCKS" \
  --degrees "$SCALE_DEGREES" \
  --warmup "$PROFILE_WARMUP" \
  --repeats "$PROFILE_REPEATS" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --backward \
  --output "$RUN_DIR/scaling-backward.json" \
  > "$RUN_DIR/scaling-backward.log"

cat > "$RUN_DIR/manifest.json" <<EOF
{
  "schema_version": 1,
  "suite": "spatial_operator_comparison",
  "mode": "$MODE",
  "device": "$DEVICE",
  "dtype": "$DTYPE",
  "result_schema": "schemas/spatial_operator_comparison.schema.json",
  "files": [
    "result.json",
    "report.md",
    "decision.json",
    "implicit-accuracy.json",
    "fragment-locality.json",
    "scaling.json",
    "scaling-backward.json",
    "environment.txt",
    "git.txt",
    "focused-tests.log",
    "protocol-validation.log",
    "comparison.log",
    "report.log",
    "implicit-accuracy.log",
    "fragment-locality.log",
    "scaling.log",
    "scaling-backward.log"
  ]
}
EOF

printf 'Spatial operator suite complete: %s\n' "$RUN_DIR"
