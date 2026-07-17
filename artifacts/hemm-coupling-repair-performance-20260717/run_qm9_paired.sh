#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="$ROOT/artifacts/hemm-coupling-repair-performance-20260717"
OUT_DIR="$RUN_DIR/qm9"
LEDGER="$RUN_DIR/qm9-runs.jsonl"
EXPECTED_HEAD="a8bda61868cf118b93b6a605001fb401c23f46c1"

cd "$ROOT"
mkdir -p "$OUT_DIR"
if [[ "$(git rev-parse HEAD)" != "$EXPECTED_HEAD" ]]; then
  echo "unexpected git commit" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "tracked worktree must be clean" >&2
  exit 2
fi

started_all=$SECONDS
for seed in 41 42 43; do
  for routing in ggg lgl; do
    if (( SECONDS - started_all >= 1500 )); then
      echo "25-minute new-arm stop reached" >&2
      exit 3
    fi
    metrics="$OUT_DIR/${routing}-m1-seed${seed}.json"
    log="$OUT_DIR/${routing}-m1-seed${seed}.log"
    started_run=$SECONDS
    oms research-runner \
      --file "$LEDGER" \
      --metrics "$metrics" \
      --question "Does M=1 lgl improve QM9 gap validation MAE over matched M=1 ggg at 2,000 steps? Record ${routing} seed ${seed}." \
      --hypothesis "Across seeds 41-43, lgl lowers paired mean validation MAE by at least 0.010 eV, improves at least two seeds, and never regresses more than 0.020 eV." \
      --prediction "The local-global-local route preserves global context while improving short-range routing enough to meet the registered aggregate threshold." \
      --baseline "Matched ggg M=1 route on commit ${EXPECTED_HEAD}, split seed 42, model seed ${seed}." \
      --metric "val_mae/QM9 gap eV, seeded random-row warm validation split" \
      --success "Finite validation metrics with test_evaluated=false; final decision uses mean delta >=0.010 eV, at least 2/3 positive deltas, and minimum delta >=-0.020 eV." \
      --change "Change only routing from ggg to lgl within the paired seed; memory count remains one." \
      --no-gate \
      --reason "Clean commit ${EXPECTED_HEAD} passed scripts/check.sh fast and gpu, and the registered five-process CUDA ceiling; avoid repeating unchanged preflight per arm." \
      -- uv run --locked python scripts/train_compare.py \
        --dataset qm9 \
        --data-root data/qm9 \
        --qm9-target-index 4 \
        --num-samples 130000 \
        --train-size 110000 \
        --val-size 10000 \
        --batch-size 64 \
        --steps 2000 \
        --hidden-dim 64 \
        --num-layers 3 \
        --num-heads 4 \
        --split-seed 42 \
        --model-seed "$seed" \
        --device cuda \
        --amp-dtype none \
        --skip-test-eval \
        --bounded-diagnostics \
        --diagnostic-max-nodes 32 \
        --routing "$routing" \
        --memory-count 1 \
        --metrics-out "$metrics" >"$log"
    elapsed_run=$((SECONDS - started_run))
    if (( elapsed_run > 180 )); then
      echo "individual 2k run exceeded 180 seconds: ${routing} seed ${seed}" >&2
      exit 4
    fi
    jq -r '"'"$routing"' seed='"$seed"' val_mae=\(.val_mae) train_seconds=\(.elapsed_seconds) test_evaluated=\(.test_evaluated)"' "$metrics"
  done
done
echo "paired_qm9_wall_seconds=$((SECONDS - started_all))"
