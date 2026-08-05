#!/usr/bin/env bash
# Project verification contract. Agents run this before claiming work done.
#   fast      CPU-only, under 60 seconds, safe to run anytime.
#   ml-smoke  ML interface smoke: import/config/data/model/loss one-batch checks.
#   gpu       Short GPU smoke; wrapped in a transient srun on Slurm machines.
# Fill the TODO blocks as the project takes shape. An empty contract fails
# loudly on purpose -- never let "no checks" look like a pass.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODE="${1:-fast}"

run_fast() {
  local ran=0
  local dirs=()
  # The fast contract is intentionally CPU-only even on a workstation that
  # happens to have a visible accelerator. GPU kernels have a separate,
  # explicitly scheduled gate below.
  export CUDA_VISIBLE_DEVICES=""
  [ -d src ] && dirs+=(src)
  [ -d scripts ] && dirs+=(scripts)

  if [ -f pyproject.toml ] && command -v uv >/dev/null 2>&1; then
    uv run python - <<'PY'
import equivariant_linear_attention as ela
from equivariant_linear_attention import ELA, ELAGraph

assert ELA is not None
assert ELAGraph is not None
assert set(ela.__all__) == {"ELA", "ELAGraph"}
PY
    ran=1
  fi

  # Invoked through bash so a checkout that lost the executable bit still runs
  # the isolated wheel build/install/import instead of silently skipping it.
  if [ -f scripts/wheel_smoke.sh ] && command -v uv >/dev/null 2>&1; then
    bash scripts/wheel_smoke.sh
    ran=1
  fi

  if [ "${#dirs[@]}" -gt 0 ] && [ -f pyproject.toml ] && command -v uv >/dev/null 2>&1; then
    uv run python -m compileall -q "${dirs[@]}"
    ran=1
  fi
  # Lint is not probed behind a silent availability test: an unresolvable ruff
  # used to skip the whole stage while the gate still printed "ok".
  if [ -f pyproject.toml ] && command -v uv >/dev/null 2>&1; then
    uv run ruff check .
    ran=1
  fi

  if [ -d tests ] && [ -f pyproject.toml ] && command -v uv >/dev/null 2>&1; then
    uv run pytest -q --cov=equivariant_linear_attention --cov-report=term-missing --cov-fail-under=80.01
    ran=1
  fi

  if [ -f scripts/ml_smoke.py ] && [ -f pyproject.toml ] && command -v uv >/dev/null 2>&1; then
    uv run python scripts/ml_smoke.py
    ran=1
  fi

  if [ "$ran" -eq 0 ]; then
    echo "check fast: no checks ran; configure scripts/check.sh" >&2
    exit 1
  fi
  echo "check fast: ok"
}

run_ml_smoke() {
  local ran=0

  if [ -f scripts/ml_smoke.py ] && [ -f pyproject.toml ] && command -v uv >/dev/null 2>&1; then
    uv run python scripts/ml_smoke.py
    ran=1
  elif [ -f scripts/ml_smoke.py ]; then
    python3 scripts/ml_smoke.py
    ran=1
  fi

  # TODO ML project: implement scripts/ml_smoke.py or replace this function
  # with a CPU-only one-batch check covering config load, dataloader sample,
  # model forward, loss, backward, eval mode, and checkpoint save/load.
  if [ "$ran" -eq 0 ]; then
    echo "check ml-smoke: no ML smoke configured; add scripts/ml_smoke.py or edit scripts/check.sh" >&2
    exit 1
  fi
  echo "check ml-smoke: ok"
}

run_gpu() {
  export UV_LOCKED=1
  if command -v srun >/dev/null 2>&1; then
    srun --gres=gpu:1 --time=00:10:00 uv run --extra triton python -c \
      'import torch; from equivariant_linear_attention.kernels import triton_available; assert torch.cuda.is_available(), "CUDA unavailable"; assert torch.cuda.is_bf16_supported(), "CUDA BF16 unavailable"; assert triton_available(), "Triton unavailable"'
    srun --gres=gpu:1 --time=00:10:00 uv run --extra triton python scripts/ml_smoke.py cuda bf16
    srun --gres=gpu:1 --time=00:10:00 uv run --extra triton python scripts/ml_smoke.py cuda auto
    srun --gres=gpu:1 --time=00:10:00 uv run --extra triton pytest -q \
      tests/test_gpu_completion.py \
      tests/test_canonical_cuda.py \
      tests/test_kernel_triton_cuda.py \
      tests/test_triton_equivariance_cuda.py \
      tests/test_inference.py \
      tests/test_unified_multipole_precision.py
  else
    uv run --extra triton python -c \
      'import torch; from equivariant_linear_attention.kernels import triton_available; assert torch.cuda.is_available(), "CUDA unavailable"; assert torch.cuda.is_bf16_supported(), "CUDA BF16 unavailable"; assert triton_available(), "Triton unavailable"'
    uv run --extra triton python scripts/ml_smoke.py cuda bf16
    uv run --extra triton python scripts/ml_smoke.py cuda auto
    uv run --extra triton pytest -q \
      tests/test_gpu_completion.py \
      tests/test_canonical_cuda.py \
      tests/test_kernel_triton_cuda.py \
      tests/test_triton_equivariance_cuda.py \
      tests/test_inference.py \
      tests/test_unified_multipole_precision.py
  fi
  echo "check gpu: ok"
}

case "$MODE" in
  fast) run_fast ;;
  ml-smoke) run_ml_smoke ;;
  gpu) run_gpu ;;
  *)
    echo "usage: scripts/check.sh [fast|ml-smoke|gpu]" >&2
    exit 2
    ;;
esac
