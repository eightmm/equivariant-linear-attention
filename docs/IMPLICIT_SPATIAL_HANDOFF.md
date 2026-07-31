# Implicit spatial kernel handoff

Branch: `agent/implicit-spatial-kernel`

Base branch: `agent/conditioned-se3-layer`

This branch deliberately does not replace the canonical sparse-local model. It
adds a mathematically separate edge-free spatial approximation, an exact
on-the-fly backend specification, scaling contracts, and a complete
explicit/implicit/hybrid comparison packet.

## Implemented files

### Core implicit operator

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

### Matched operator comparison

- `src/equivariant_attention/spatial_ablation.py`
  - one parameter-schema-matched model for explicit, implicit, and hybrid arms;
  - precomputed no-edge topology for uncontaminated implicit timing;
  - exact state-dict hashing and audit metadata.
- `src/equivariant_attention/spatial_benchmarks.py`
  - deterministic local-directional, smooth-Gaussian, and mixed tasks;
  - dataset hashing and fixed sparse candidate construction.
- `src/equivariant_attention/spatial_comparison.py`
  - protocol validation;
  - paired seed deltas;
  - bounded synthetic promotion gates;
  - Markdown report rendering.

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
- `scripts/compare_spatial_operators.py`
  - same data, initial state, parameter schema, optimizer, and budget for all
    three arms;
  - training curves, best/final metrics, clipping, latency, and peak memory;
  - initial explicit/hybrid identity and implicit edge-independence audits.
- `scripts/report_spatial_operator_comparison.py`
  - report and machine-readable decision generation.

### Specifications

- `docs/IMPLICIT_SPATIAL_KERNEL.md`
- `docs/ON_THE_FLY_NEIGHBOR_KERNEL.md`
- `docs/SCALING.md`
- `docs/SPATIAL_OPERATOR_COMPARISON.md`

## Focused tests

```bash
uv run pytest \
  tests/test_spatial_ablation.py \
  tests/test_spatial_comparison.py \
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

The tests cover:

- finite-feature factorization versus dense feature-kernel evaluation;
- local-regime Gaussian approximation;
- self exclusion, singleton graphs, graph isolation, and node permutation;
- translation, proper/improper O(3), and all parity-sector transport;
- value, coordinate, scale-weight, and double-backward gradients;
- chunk-size equivalence and BF16 accumulation dtype;
- zero-initialized implicit residual identity and wake-up gradient;
- identical parameter schema/hash across comparison arms;
- zero-init explicit/hybrid function equality;
- implicit execution independence from explicit edge metadata;
- conditional scaling contracts and report gate behavior.

## Complete comparison workflow

```bash
RUN=artifacts/spatial-operator-comparison/$(date +%Y%m%d-%H%M%S)
mkdir -p "$RUN"

uv run python scripts/compare_spatial_operators.py \
  --tasks local_directional,smooth_gaussian,mixed \
  --seeds 0,1,2,3,4 \
  --train-graphs 64 \
  --validation-graphs 32 \
  --nodes-per-graph 24 \
  --hidden-dim 64 \
  --layers 4 \
  --heads 4 \
  --local-rank 4 \
  --cutoff 1.75 \
  --candidate-skin 0.25 \
  --gaussian-scale 2.5 \
  --implicit-scales 2,4,8 \
  --implicit-scale-init 0 \
  --steps 1000 \
  --evaluation-interval 50 \
  --device cuda \
  --dtype bfloat16 \
  --output "$RUN/result.json"

uv run python scripts/report_spatial_operator_comparison.py \
  "$RUN/result.json" \
  --output "$RUN/report.md" \
  --decision-json "$RUN/decision.json"
```

Approximation diagnostic:

```bash
uv run python scripts/evaluate_implicit_spatial.py \
  --nodes 512 \
  --scales 2,4,8 \
  --cutoff 1.75 \
  --output "$RUN/implicit-accuracy.json"
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
  --output "$RUN/scaling.json"
```

## Integration choices

### A. Additive auxiliary residual

Use `ImplicitSpatialResidual` after the exact linear-attention/local branch. This
is the lowest-risk experiment because zero initialization preserves the
incumbent function. The `hybrid` arm evaluates this choice.

### B. Replace only static node multipoles

Use implicit relative moments as an edge-free initialization/context path while
retaining the exact content-conditioned sparse local residual.

### C. Replace sparse local transport

Use an empty explicit topology and the implicit residual. The `implicit` arm
evaluates this choice. It changes hard local semantics to a smooth graph-global
kernel and therefore has the strongest promotion requirements.

### D. Hybrid exact short-range plus implicit long-range

Keep exact sparse local interactions below a cutoff and use implicit scales
larger than that cutoff. This is the most physically interpretable first
integration for molecular and particle workloads.

### E. Integrate spatial features into global linear attention

If the hybrid residual consistently helps, concatenate the positive spatial
feature map with the existing global query/key feature map. This avoids a full
second value-transport pass and keeps equivariant linear attention as the single
global sufficient-statistic operator. Mathematical and performance equivalence
must be tested before replacing the auxiliary residual.

## Promotion boundary

Synthetic gates only advance an arm to real-task evaluation. Before any
canonical change, run at least:

- one general 3D task, such as point cloud, field, mesh, or registration;
- one molecular/protein task, with leakage-controlled splits where applicable.

Keep splits, initialization, parameter schema, optimizer, step budget, and
hardware paired. Record task metric, latency, peak memory, clipping, and scaling
slopes.

## Non-claims

The branch does not establish:

- superiority over the explicit sparse local operator;
- equivalence to a radius graph;
- production GPU speedup;
- periodic-cell support;
- an exact on-the-fly neighbor kernel;
- end-to-end linear time including arbitrary neighbor discovery;
- downstream architecture promotion.

Those claims require the documented experiments and, for exact edge-free API
semantics, the backend specified in `ON_THE_FLY_NEIGHBOR_KERNEL.md`.
