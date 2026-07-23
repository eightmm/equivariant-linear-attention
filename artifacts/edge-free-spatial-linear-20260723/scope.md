# Edge-free spatial-linear performance contract

## Decision

- Question: can the existing global factorized attention accept a fixed-rank
  multi-scale Euclidean spatial kernel without any edge list or pair matrix,
  preserve the registered symmetry contracts, and improve the large-edge
  latency/peak-memory tradeoff against the private static EGNN?
- Deliverable: opt-in implementation, intentional RED/green tests, bounded
  CPU/CUDA benchmark, raw metrics, interpretation, provenance, and independent
  review.
- Evidence cutoff: 2026-07-23 Asia/Seoul.
- Inputs: deterministic synthetic typed 3D point sets. No external dataset or
  label is used and no test split is evaluated.

## Intervention

- Add one opt-in boolean to the existing `EquivariantAttention` architecture.
- When enabled, every head receives one fixed positive scale from a registered
  log-spaced range and builds a ten-dimensional degree-two Gaussian-Taylor
  feature map from graph-centered/RMS-normalized coordinates.
- Add the resulting positive spatial dot-product kernel to the incumbent
  content/vector kernel through graph-segmented sufficient statistics.
- Do not materialize `edge_index`, an `N x N` pair tensor, or a node-pair
  feature tensor.
- Keep the option off by default. With it off, parameter names, values, and
  outputs must be unchanged for matched initialization.
- Require an all-global route and learned global transport in this first
  implementation. Memory interaction and local heads are excluded.
- The existing coordinate updater remains optional and recomputes global
  geometry after every coordinate step.

## Falsifiable hypothesis and prediction

- H1: the factorized spatial kernel matches its materialized dense kernel below
  `1e-10` maximum absolute error in float64, including one-cycle key balancing.
- H2: full-model scalar/vector/tensor outputs and optional positions satisfy
  O(3), translation, permutation, and batch-isolation error below `1e-6`.
- H3: at fixed width/depth/rank, edge-free latency and peak allocated memory do
  not depend on the EGNN edge multiplier `k`.
- Performance prediction: at `N=8192`, the edge-free candidate is faster and
  uses less incremental peak allocated CUDA memory than static EGNN for at
  least `k=64` and `k=128`. Failure is retained as a negative result.
- Overhead prediction: relative to current edge-free GGG, the spatial candidate
  costs at most `2.5x` median forward latency and `2.5x` incremental peak CUDA
  memory at `N=8192`. This is diagnostic, not a promotion rule.

## Registered benchmark

- Hardware: one local CUDA GPU selected by the existing environment.
- Environment: repository `.venv` and lockfile; no new dependency, download,
  network host, container, or external data.
- Models:
  - current static all-global GGG;
  - static edge-free spatial GGG;
  - coordinate-updating edge-free spatial GGG;
  - near-parameter-matched private static EGNN.
- Width/depth: attention hidden width 64, four heads, three layers; EGNN hidden
  width 91, three layers.
- Node sizes: `N={128,512,2048,8192}`.
- EGNN edge multipliers: `k={4,16,64,128}`, admitted only when `k<=N`.
- Timing: synchronized forward-only median after warmup; graph generation is
  outside timing. Peak memory is maximum allocated bytes above the
  post-warmup live-allocation baseline.
- Seed: model seed 20260723; graph seed 20260723 with deterministic per-cell
  derivation.
- Resource ceiling: at most 300 cumulative GPU-wall seconds for correctness
  smoke plus the registered benchmark.
- Stop on nonfinite output, contract failure, repeated OOM, or wall ceiling.

## Acceptance

- Implementation completion requires H1 and H2, focused tests, `scripts/check.sh
  fast`, and `scripts/check.sh gpu`.
- Benchmark completion requires every admitted cell to record a completed,
  failed, or skipped status with its exact configuration.
- Scientific claims are limited to the recorded hardware, model shapes,
  forward workload, and synthetic coordinates.
- No default changes based only on speed/memory.

## Non-goals and limitations

- No QM9 training, validation, test evaluation, accuracy, force, energy, or
  coordinate-quality claim.
- No claim that removing edges preserves arbitrary graph topology. Two inputs
  with the same node features/coordinates but different omitted adjacency are
  indistinguishable.
- No neighbor-builder timing claim for EGNN in the primary kernel comparison.
- No backward, optimizer, batching-throughput, compile, multi-GPU, protein, or
  production claim.
- The ten-dimensional truncated feature map is a soft spatial bias, not an
  exact Gaussian or exact hard neighborhood cutoff.
