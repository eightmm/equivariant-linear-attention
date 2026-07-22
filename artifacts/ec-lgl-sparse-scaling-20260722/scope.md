# Scaling-aware EC-LGL packet

## Decision contract

- Question: can the public architecture preserve all-node global context with
  factorized `O(N)` attention, cap expensive local geometric work at `O(E_local)`
  with `E_local = O(kN)`, and improve the current QM9 validation gap without
  losing permutation/O(3)/translation correctness?
- Decision: first establish an honest sparse execution and crossover boundary;
  then admit a bounded accuracy comparison only if the new operator is active,
  symmetric, finite and parameter bounded.
- Prediction unit for accuracy: one QM9 molecule/conformer and graph-level gap
  target in eV.
- Scaling systems: synthetic single 3D graphs with controlled `N`, local degree
  and edge density. Synthetic timings do not establish molecule/protein/point-
  cloud accuracy.
- Evidence cutoff: 2026-07-22 Asia/Seoul.
- Source base: Git revision
  `4adea3c5a982eaa24efd3e78a48de634d21725b1` plus the task-scoped diff.

## Architecture intervention

The candidate remains the one public `EquivariantAttention` architecture. It
uses the three-stage `LGL` route:

```text
bounded sparse EC local sum -> factorized global moment attention
                            -> bounded sparse EC local sum
```

Each local stage owns an independent invariant edge MLP with frozen shape
`Linear(144, 12) -> SiLU -> Linear(12, 76)`. Inputs are 64 receiver scalars,
64 sender scalars and 16 distance RBFs. Outputs are a 64-channel scalar edge
message plus four gates each for sender vectors, relative displacement vectors
and relative symmetric-traceless rank-2 bases. Nonself messages use the existing
smooth cutoff and `sum` aggregation. Self information stays in the node
residual. Width, route, number of heads, RBF count and MLP shape cannot be
changed after outcome inspection.

The candidate local graph is deliberately bounded. Domain edges beyond the
local budget are not enumerated in the local operator; long-range communication
uses the exact factorized global stage. Therefore this candidate is not
functionally equivalent to a dense-edge EGNN and any crossover against dense
EGNN is a systems/inductive-bias comparison, not an identical-computation speedup.

## Sparse execution contract

- `GraphSample` and `GraphBatch` carry optional receiver/sender `edge_index`.
- Collation offsets node indices and preserves graph isolation and edge order.
- `.to(...)`, training, evaluation and inference forward the exact supplied
  candidate edges.
- Repo convention remains row 0 receiver, row 1 sender. A PyG source/target
  tensor requires an explicit row swap at an adapter boundary.
- QM9 geometric candidates are precomputed once per sample at load time with
  the frozen 2.5 Angstrom cutoff and self edges. This removes forward-time
  Cartesian discovery but does not support a linear-time graph-builder claim.
- Large synthetic scaling inputs construct bounded ring/k-nearest-style edges
  directly without dense pair discovery.

## Preregistered hypotheses

### H1: correctness and sparse plumbing

- Baseline: current complete-candidate fallback and current local moment layer.
- Prediction: supplied-edge candidate and dense-reference outputs agree on tiny
  graphs; the new training path completes even when fallback candidate discovery
  is replaced by a test sentinel that raises.
- Success: float64 dense/sparse, node-permutation, edge-order, graph-ID reorder,
  mixed-batch, independent rotation/reflection/translation errors below `1e-6`;
  every intended EC branch has a finite nonzero gradient.

### H2: linear global scaling and edge-density crossover

- Same-computation baseline: materialized dense evaluation of the exact same
  registered finite kernel.
- Systems controls: static EGNN on the same bounded sparse edge set and static
  EGNN on increasingly dense edge sets.
- Sizes: the largest safe subset of `N={32,64,128,256,512,1024,2048,4096}`;
  bounded local degree `k=16`; edge-density controls are preregistered fractions
  `{1/N, 4/N, 16/N, 0.25, 1.0}` after clipping to valid directed edges.
- Primary scaling metrics: synchronized median forward latency, peak allocated
  memory, retained/materialized pair elements, and log-log slope over the last
  three feasible sizes. Warmup and repeat counts are fixed before execution.
- Success for the same kernel: dense/factorized float64 maximum error below
  `1e-10`; factorized path materializes no `N x N` tensor; at the largest common
  size it uses at most half the measured/declared pair storage and either is
  faster or has a runtime slope at least `0.5` lower than dense.
- EGNN crossover is descriptive unless the same edge set is used. The bounded-
  edge comparison reports constant factors; the dense-edge comparison tests the
  intended edge-count regime but cannot attribute differences to attention alone.

### H3: bounded QM9 accuracy

- Screen: EC-LGL and incumbent static LGL, seed 42, 500 updates, each repeated
  once to expose CUDA drift. Static coordinates, FP32, batch 64, AdamW
  `lr=3e-4`, weight decay `0.01`, clip `1.0`, no scheduler, test disabled.
- Admission: all gates pass; candidate/EGNN trainable-parameter ratio at most
  `1.05`; candidate mean screen validation MAE is no more than `0.020 eV` worse
  than incumbent.
- Conditional confirmation: EC-LGL, incumbent static LGL and private static
  EGNN width 91 at seeds 41--45 and 2,000 updates. All 15 arms are generated by
  this packet; no historical substitution.
- Accuracy promotion requires candidate mean at most EGNN mean minus
  `0.010 eV`, at least three of five EGNN paired wins, worst EGNN paired
  regression no larger than `0.020 eV`, incumbent mean improvement at least
  `0.050 eV`, and at least three of five incumbent paired wins.
- Small-QM9 latency is reported but is not a veto on a verified large-graph
  crossover. No test labels are evaluated.

## Resource and stop contract

- Existing locked environment; no dependency or lockfile change.
- CPU work: focused tests, one fast project gate and bounded synthetic scaling.
- GPU: one local GPU, cumulative measured wall time at most 1,200 seconds across
  scaling, screen and any admitted confirmation. Stop before starting an arm
  that cannot fit the remaining ceiling.
- Outputs: `artifacts/ec-lgl-sparse-scaling-20260722/` plus registered source,
  tests and documentation in the worktree.
- Cancellation: interrupt only the process launched by this packet; preserve
  completed records and mark the interrupted arm inconclusive.

## Non-goals and inference boundary

- No final test evaluation, 10,000-step result, checkpoint publication,
  dependency change or default promotion without the frozen gates.
- No official EGNN reproduction or claim that EC-LGL computes the same function
  as dense-edge EGNN.
- No claim that radius/kNN preprocessing, data loading or arbitrary domain-edge
  construction is linear.
- No protein-family, scaffold/conformer, chirality, point-cloud scene or force/
  dynamics generalization claim.
- Current O(3) reflection-invariant scalar boundary remains parity safe and
  cannot distinguish geometry-only enantiomers.

## Objective completion criteria

1. Sparse edge plumbing is exercised end to end and fallback discovery is not
   called when supplied edges exist.
2. EC local transport passes dense-reference, gradient and public-boundary
   symmetry/permutation/batch tests.
3. A reproducible scaling table separates same-kernel factorization, same-edge
   constant factors and different-density systems comparisons.
4. The bounded QM9 screen is executed; confirmation is executed only if admitted
   and within budget.
5. Source, tests, commands, failures, metrics, claims and review are hash-linked
   in a valid independently reviewed bundle.
