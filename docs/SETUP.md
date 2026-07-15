# Setup

## Core

```bash
uv sync --locked
scripts/check.sh fast
```

Core runtime dependencies are Python 3.12+ and PyTorch. The active model does
not require cuEquivariance, e3nn, RDKit, or PyTorch Geometric.

## QM9 extra

```bash
uv sync --locked --extra qm9
```

This adds RDKit and PyTorch Geometric only for the local QM9 loader.

## GPU smoke

```bash
scripts/check.sh gpu
```

The GPU smoke checks bf16 forward/backward and inference. A successful CPU run
does not establish CUDA mixed-precision correctness.
