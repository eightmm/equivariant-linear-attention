# Exact-edge-multiplier scaling contract

## Decision

- Question: on deterministic pseudo-random directed 3D graphs with exactly
  `E = kN` candidate edges including one self edge per node, how do EC-LGL and
  the private static EGNN scale when both receive the identical edge tensor?
- Decision use: separate fixed model overhead, per-edge cost, memory growth,
  and any same-edge crossover before choosing an optimization target.
- Evidence cutoff: 2026-07-23 Asia/Seoul, current branch source plus this
  task-scoped diff.
- Population: one single-graph synthetic workload at each registered `(N, k)`
  pair, with 11 invariant node features and 3D coordinates whose diameter is
  below the local cutoff.
- Non-goals: task accuracy, QM9/protein/point-cloud generalization, neighbor
  construction time, official EGNN reproduction, training throughput, backward
  latency, or a claim that the two models perform identical computations.

## Claims and falsifiers

- `C-201` (implementation): the generator returns deterministic, duplicate-free
  directed edges with complete self coverage and exactly `min(kN, N^2)` edges.
  Any count, range, duplicate, determinism, or self-coverage failure falsifies it.
- `C-202` (fair comparison): each `(N, k)` row passes one identical prevalidated
  edge tensor to EC-LGL and EGNN, and graph construction is outside timing.
  Distinct edge hashes/counts or timed construction falsify it.
- `C-203` (measured scaling): the saved grid is sufficient to estimate latency
  and peak-memory dependence on both `N` and `E=kN` for each model on this
  hardware. Missing admitted rows, nonfinite metrics, or an unrecorded OOM
  falsifies it.
- `C-204` (optimization decision): fixed overhead and per-edge growth are
  reported separately enough to decide whether fusion alone is sufficient.
  A conclusion based only on one graph size or one density falsifies it.

## Falsifiable hypothesis and prediction

- Hypothesis: same-edge EC-LGL has a larger fixed cost and remains slower than
  EGNN over the registered grid, although its latency ratio narrows as `k`
  increases. The previously observed advantage against complete-edge EGNN is
  therefore primarily avoided edge work plus factorized global context, not a
  cheaper local edge kernel.
- Prediction: no same-edge runtime crossover for `k <= 128`; the EC-LGL/EGNN
  latency ratio at a fixed `N` decreases between the lowest and highest admitted
  multipliers. Both paths remain approximately linear in `E` at fixed width,
  with residuals and fit quality reported rather than thresholded post hoc.
- Result labels: `met`, `not_met`, or `mixed`; a crossover is valid evidence
  against the prediction rather than a reason to change the grid.

## Frozen execution matrix

- Nodes: `N in {128, 512, 2048, 8192}`.
- Edge multipliers: `k in {4, 8, 16, 32, 64, 128}`, admitted only when `k <= N`.
- Exact candidate edges: `E=kN`, including `N` self edges; nonself senders are
  sampled without replacement by a seeded affine permutation independently
  for each receiver, so every receiver has exactly `k` candidate edges. Graph
  seed: `20260723` plus a deterministic `(N, k)` offset. Model initialization
  uses the separate fixed seed `20260723`, and both model-state hashes are
  recorded.
- Models: current width-64, three-layer EC-LGL with local counts `(4,0,4)` and
  edge-conditioned transport; current width-91, three-layer private static EGNN.
- Precision/device: FP32 model execution on local CUDA, eval/inference mode.
- Timing: three warmups and seven synchronized repeats; median milliseconds.
- Resources: CUDA peak allocated-byte delta, candidate/effective edge counts,
  degree distribution, and edge tensor SHA-256.
- CPU smoke: `N in {16,32}`, `k in {2,4}`, zero warmups, one repeat.
- GPU ceiling: 120 cumulative wall seconds for this packet. Stop and retain
  partial rows on OOM or ceiling exhaustion; do not silently shrink the grid.

## Acceptance

- Generator unit tests cover exact count, self coverage, uniqueness, range,
  seed determinism, changed-seed sensitivity, and `k>N` saturation.
- A CPU smoke emits strict finite JSON and proves identical-edge accounting.
- Every admitted CUDA cell records both models or an explicit failure state.
- The report includes per-cell ratios, first crossover if any, fixed-`N`
  latency-versus-edge fits, memory behavior, prediction verdict, and limitations.
- Project fast tests and GPU smoke pass after implementation.
- The run bundle validates and receives independent record/source/method review.

## Protocol amendment before authoritative rerun

The first implementation used one affine traversal over the globally flattened
nonself-pair universe. A post-implementation degree check, added before final
analysis and review, found avoidable receiver-load skew (for example, degree
1--32 at registered `N=512,k=8`). Exact edge count and uniqueness were valid,
but this topology could confound scatter contention with edge-count scaling.
The initial raw measurements are retained with an `affine-exploratory` label.
The authoritative rerun uses the per-receiver sampler described above and adds
exact receiver-degree tests. The registered `(N,k)`, seed family, models,
timing, resource ceiling, hypothesis, and interpretation boundaries are
otherwise unchanged; the amendment was made after observing the initial timing
outcome, so the retained initial run is disclosed rather than overwritten.

## Safety and interpretation boundaries

- No network, dependency, dataset, or lockfile change is allowed.
- Synthetic graphs are systems probes, not molecule/protein distributions.
- `edge_index_is_validated=True` is used only for generator output checked by
  unit tests and packet invariants; unchecked external edges remain unsafe.
- Same-edge rows control topology but not computation: EC-LGL additionally has
  factorized global transport and different local equations/readout.
