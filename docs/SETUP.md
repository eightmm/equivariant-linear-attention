# Setup

## Core

```bash
uv sync --locked
scripts/check.sh fast
```

Core runtime dependencies are Python 3.12+ and PyTorch. The active model does
not require cuEquivariance, e3nn, RDKit, or PyTorch Geometric.

The same fast gate is available through the manually dispatched
`.github/workflows/ci.yml`; pushes and pull requests do not start it
automatically.

The workflow pins `astral-sh/setup-uv@v8.3.2`; the floating `@v8` reference is
not published by the action repository and caused the first v0.7 CI run to fail
before project setup.

## Historical dataset extras

```bash
uv sync --locked --extra qm9
uv sync --locked --extra pdbbind
```

These dependency groups remain in the lockfile for historical environment
reproduction. The architecture-only package no longer ships QM9 or PDBBind
loaders; current datasets enter through mappings collated into `ELABatch`.

## GPU smoke

```bash
scripts/check.sh gpu
```

The GPU smoke checks explicit bf16 model forward/backward/inference, plus an
FP32 training backward and FP32-parameter auto-autocast inference lane. A
successful CPU run does not establish CUDA mixed-precision correctness.

`prepare_for_inference(dtype="auto")` keeps parameters in float32 and uses CUDA
autocast when available. Explicit `dtype="bf16"` or `dtype="fp16"` still
performs whole-model conversion for controlled lanes. The helper preserves the
caller's TF32 policy by default. `allow_tf32=True` or `False` is an explicit,
process-global CUDA opt-in and is ignored for CPU targets.
