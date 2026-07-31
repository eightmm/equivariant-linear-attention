# Implicit spatial kernel handoff

Branch: `agent/implicit-spatial-kernel`

Base branch: `agent/conditioned-se3-layer`

This branch deliberately does not replace the canonical sparse-local model. It
adds a mathematically separate edge-free spatial approximation, an exact
on-the-fly backend specification, scaling contracts, and validation tools so a
follow-up agent can decide whether and where to integrate the approximation.

## Implemented files

### Core

- `src/equivariant_attention/implicit_spatial.py`
  - multiscale order-two Gaussian--Taylor feature map;
  - no edge input, neighbor discovery, or pair matrix;
  - chunked per-graph sufficient-statistic transport;
  - self exclusion and three normalization modes;
  - implicit relative `1o` and `2e` moments;
  - parity-complete hidden-state transport.
- `src/equivariant_attention/implicit_spatial_residual.py`
  - zero-initializable per-copy residual integration boundary.
- `src/equivariant_attention/scaling_contract.py`
  - explicit base, AttnRes, and implicit complexity contracts;
  - log-log power-law fitting.

### Evaluation scripts

- `scripts/benchmark_scaling.py`
  - node, depth, degree, graph-count, forward/backward, and peak-memory sweeps;
  - base, fixed-`B` AttnRes, `B=L` AttnRes, and implicit modes;
  - separate prepared-model and optional CSR-pack timings;
  - no neighbor-discovery claim.
- `scripts/evaluate_implicit_spatial.py`
  - dense exact Gaussian-mixture comparison;
  - C2 compact-cutoff comparison;
  - kernel/message relative errors and top-k overlap.

### Specifications

- `docs/IMPLICIT_SPATIAL_KERNEL.md`
- `docs/ON_THE_FLY_NEIGHBOR_KERNEL.md`
- `docs/SCALING.md`

## Tests

Focused tests cover:

- finite-feature factorization versus dense feature-kernel evaluation;
- local-regime Gaussian approximation;
- self exclusion and singleton graphs;
- graph isolation;
- translation, O(3), parity-sector, and node-permutation equivariance;
- value, coordinate, scale-weight, and double-backward gradients;
- chunk-size equivalence;
- BF16 CUDA accumulation/mass dtype contract;
- zero-initialized residual identity and wake-up gradient;
- explicit conditional scaling contracts and synthetic slope fitting.

## Recommended commands

```bash
uv run pytest \
  tests/test_implicit_spatial.py \
  tests/test_implicit_spatial_chunks.py \
  tests/test_implicit_spatial_gradients.py \
  tests/test_implicit_spatial_permutation.py \
  tests/test_implicit_spatial_residual.py \
  tests/test_implicit_spatial_validation.py \
  tests/test_scaling_contract.py \
  tests/test_scaling_memory.py
```

CUDA:

```bash
uv run pytest tests/test_implicit_spatial_cuda.py
```

Approximation diagnostic:

```bash
uv run python scripts/evaluate_implicit_spatial.py \
  --nodes 512 \
  --scales 1,2,4 \
  --cutoff 3 \
  --output artifacts/implicit-spatial-accuracy.json
```

Scaling:

```bash
uv run python scripts/benchmark_scaling.py \
  --modes base,attnres,implicit \
  --nodes 256,512,1024,2048,4096 \
  --graphs 1,8,32 \
  --depths 4,8,16,32 \
  --blocks 4,8 \
  --degrees 8,16,32,64 \
  --device cuda \
  --dtype bfloat16 \
  --output artifacts/scaling.json
```

## Integration choices

### A. Additive auxiliary residual

Use `ImplicitSpatialResidual` after the exact linear-attention/local branch. This
is the lowest-risk experiment because zero initialization preserves the
incumbent function.

### B. Replace only static node multipoles

Use implicit relative moments as an edge-free initialization/context path while
retaining the exact content-conditioned sparse local residual.

### C. Replace sparse local transport

This yields a fully edge-free spatial stack but changes hard local semantics to
a smooth graph-global kernel. It requires the strongest downstream validation.

### D. Hybrid exact short-range plus implicit long-range

Keep exact sparse local interactions below a cutoff and use the implicit kernel
as a long-range smooth complement. This is the most physically interpretable
integration for molecular and particle workloads, but it still requires an
exact neighbor backend for the short-range part.

## Non-claims

The branch does not establish:

- superiority over the explicit sparse local operator;
- equivalence to a radius graph;
- production GPU speedup;
- periodic-cell support;
- an exact on-the-fly neighbor kernel;
- end-to-end linear time including arbitrary neighbor discovery.

Those claims require the documented experiments and, for exact edge-free API
semantics, the backend specified in `ON_THE_FLY_NEIGHBOR_KERNEL.md`.
