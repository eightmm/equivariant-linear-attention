# SETUP

## Tooling

- Package manager: `uv`
- Python: `>=3.12`
- Install: `uv sync`
- CPU verification: `scripts/check.sh fast`
- GPU smoke: `scripts/check.sh gpu`
- Attention benchmark: `uv run python scripts/bench_attention.py --device cuda`
- BF16/compile smoke: `uv run python scripts/ml_smoke.py cuda bf16 compile`

## Core Dependencies

- `torch`
- `cuequivariance`
- `cuequivariance-torch`
- `cuequivariance-ops-torch-cu13`
- `e3nn`
- `pytest`, `ruff`

## Hardware Target

- CUDA 13 GPU path is enabled through cuEquivariance CUDA 13 ops.
- CPU path remains valid through cuEquivariance/e3nn fallback behavior.

## First Run

```bash
uv sync
scripts/check.sh fast
scripts/check.sh gpu
```
