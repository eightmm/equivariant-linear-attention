# ELA completion research contract

## Decision

- Question: can the canonical `ELAGraph -> ELA -> ELAGraph` implementation close
  every item in the 2026-08-04 completion checklist without weakening O(3),
  translation, permutation, autograd, sparse-memory, or single-public-API
  contracts?
- Deliverable: integrated source, focused and repository tests, installable-wheel
  smoke, end-to-end and numerical-stack benchmarks, architecture ablation entry
  points, documentation, and an independent review receipt.
- Baseline: clean `main` at `29241e9c0ee1bafb5ae19b19a36d765cc9379916`.
- Evidence boundary: local checked-in code, generated run artifacts, CPU checks,
  and only user-approved GPU or long-running experiments.
- Non-goals: arbitrary `l > 2`, periodic/triclinic radius construction, a second
  public model/graph API, favorable accuracy claims without paired evidence, or
  automatic Triton promotion without a measured winning regime.

## Falsifiable claims

- C1: source import, installed-wheel import, and the exact two-symbol public
  surface all pass from isolated environments.
- C2: the benchmark reports public `model(graph)` ingestion separately for cold
  radius, prepared-cache reuse, explicit topology, and moving-coordinate
  execution, including the exact number of interstage rebuilds.
- C3: graph-major collation can enter a trusted grouped preparation path, and
  automatic radius construction can emit receiver-major CSR without a
  post-discovery COO repack while matching exact reference topology. The cell
  backend's one receiver-grouping sort remains in scope and is not claimed away.
- C4: the private numerical core has a current-main compiled execution contract;
  training local fusion and BF16 CUDA ragged grouped-MM inference avoid their
  documented Python or payload bottlenecks while matching eager numerics. The
  exact tiled ragged training fallback remains explicit.
- C5: `update_positions=True` updates coordinates at layer boundaries while
  preserving hidden state, refreshing geometry/topology as required, and retaining
  translation/O(3)/permutation equivariance.
- C6: relation conditioning has a controlled on/off ablation; explicit Cartesian
  `1 x 2` closure and multi-scale sparse local lanes are implemented behind
  deterministic, identity-compatible controls and have mechanics/resource
  comparisons.
- C7: the complete implementation passes the project fast gate, wheel smoke,
  focused symmetry/autograd tests, and independent review with no open blocking
  finding.

## Smallest falsifiers and acceptance

- Any public API expansion beyond `ELA` and `ELAGraph`, numerical mismatch outside
  dtype-appropriate tolerance, cross-graph leakage, broken improper-O(3) behavior,
  or failed first/double backward falsifies promotion.
- Direct CSR must produce the same receiver/sender/relation order and row pointers
  as the stable COO reference on dense, cell-list, batched, empty, capped, tied,
  and permuted cases.
- Ragged global execution must contain no Python loop over individual graphs in
  the numerical hot path and must match the outer-scatter reference.
- Stagewise coordinate execution must call each layer once per stage, carry the
  returned hidden state forward, and rebuild only topology whose provenance is no
  longer reusable.
- Architecture ablations are complete when disabled compatibility, active
  sensitivity, gradients, symmetry, parameter/resource deltas, and a runnable
  paired task harness are recorded. They need not win; a null result is retained.
- GPU latency/memory and real-data training claims remain pending. The user has
  explicitly reserved the local GPU for another task, so no current G1-G5
  execution is allowed.

## Frozen CUDA profiler protocol

Future execution is split into separately authorized `gpu` (G1-G2), `data`
(G3-G5), and CPU-only `finalize` (G6) phases. The frozen packet runner verifies
the packet SHA-256 and canonical argv before any subprocess, never exposes an
`all` phase, and halts before G2 when G1 lacks a passing source-bound receipt.

The canonical GPU smoke runs through a source-bound receipt wrapper. The final
adjudicator accepts neither a caller-supplied success code nor a missing result:
G1 and G2 must identify the same verified source manifest, and malformed,
non-finite, missing, device-incompatible, or over-budget evidence exits 2 while
retaining a failure receipt.

The planned CUDA profiler uses seed 20260804, three warmups, seven alternating
samples, CUDA events plus synchronized wall time, and
`torch.cuda.max_memory_allocated`. The `<10 minute` wall-time is a configuration
target, not a pre-execution guarantee. Every timed lane has a hard
`peak_allocated < 16 GiB` acceptance gate. Failed, slower, or over-budget lanes
remain in the receipt; thresholds will not be moved after observing results.
The two `ratio <= 1.0` latency gates intentionally have no noise margin. A
slower observation is retained as a failed promotion lane, not automatically
rerun or reinterpreted.

- Prepared reuse: the safe-default graph must exactly validate externally
  aliasable content, and an unsealed DLPack alias mutation must invalidate the
  cache and rebuild exact topology. Separately,
  `ELAGraph.assume_immutable()` must clone topology-bearing storage and admit
  the same packed template without an O(E) edge flip or batch reconstruction.
  Mutating or exporting a mutable alias of the sealed return violates that
  explicit lifetime contract.
- Ragged global: direct native dispatch must differ from itself by exactly zero,
  both transport and integrated balanced-attention relative L2 error must be at
  most `0.05`, the integrated balanced path must call grouped MM exactly once,
  and native grouped-MM inference median must be no slower than the same-input
  segmented inference median (`ratio <= 1.0`). Segmented training forward plus
  first backward is reported separately and has no native-training comparator.
- Radius/direct CSR: automatic and explicit paths must have identical canonical
  receiver/sender topology; output maximum absolute and relative L2 errors must
  each be at most `1e-5`. Local output projections are activated before this
  comparison so edge/local errors cannot be hidden by zero initialization.
  Latency is reported without a promotion threshold.
- Triton local training fusion: output, feature-gradient, position-gradient, and
  common parameter-gradient maximum relative L2 error must be at most `1e-3`;
  maximum absolute error must be at most `5e-3`; parameter-gradient name sets
  must match; and scalar/vector/tensor/directional fused dispatch must all be
  observed. Latency and memory are descriptive unless these numerical gates
  pass.
- Private compiled core: no fallback or fallback warning; at least one cold
  compiler graph; no extra graph for steady warm execution or same-shape changed
  topology; initial, changed-topology, and changed-shape maximum absolute error
  each at most `1e-3`; warm compiled inference median no slower than the same
  public prepared eager path (`ratio <= 1.0`).

## Frozen real-data screens

Every arm is one-seed exploratory/process evidence. Success means the exact
updates complete with finite losses and metrics, paired static arms have equal
initial state/schema/output, recorded split and access boundaries match, and no
test row is indexed or evaluated. There is no preregistered full-arm accuracy
win threshold. Null or negative lane effects are retained.

- QM9 gap: float32 CUDA, width 64, depth 3, batch 64, cutoff 2.5 Angstrom,
  100 updates per arm, learning rate `3e-4`, weight decay `0.01`, clipping 1,
  model/order/split seed 42. Static arms are `full`, `no-cg12`, and
  `no-multiscale`; the separate `stagewise` functionality arm is included. Its
  coordinate parameters must change and its final evaluation displacement must
  be finite and nonzero.
  Train/validation/test sizes are 110,000/10,000/10,000 from the first 130,000
  rows, and metrics use the first fixed 1,000 validation rows. Primary reported
  metric is validation MAE in eV, with RMSE secondary. The monolithic PyG cache
  is admitted as containing test-label storage, while test indices remain unused.
- LBA 16-complex overfit compares normalized MSE before and after training on
  the same complete 16-complex set; minibatch step losses are only diagnostics.
- LBA ID30 must fail before training unless the split is exactly 3,507 train and
  466 validation complexes with 32,302,952 directed candidate edges. Train and
  validation identity/topology receipts are recorded separately and combined by
  a digest over those split receipts. Historical local test materialization is
  disclosed separately from this runner's structurally blocked test access.
- LBA train-only capacity: pinned ID30 train shards, first 16 complexes, float32
  CUDA, width 64, depth 3, batch 2, cutoff 6 Angstrom, 250 updates per arm,
  learning rate `1e-3`, zero weight decay, clipping 1, model/order seed 20260723.
  Arms are `full`, `no-relation`, `no-cg12`, and `no-multiscale`. Primary metric
  is train MAE in pK, with RMSE secondary; improvement is descriptive capacity
  evidence only.
- LBA ID30 validation: all allowlisted train and validation rows, float32 CUDA,
  width 64, depth 3, batch 16, cutoff 6 Angstrom, 220 updates per arm, learning
  rate `3e-4`, weight decay `0.01`, clipping 1, model/order seed 42. Arms are the
  same four static arms. Primary metric is validation RMSE in pK, with MAE
  secondary. The test resolver remains fail-closed and the contaminated local
  test holdout is never opened.

## Risks

- Large fused CUDA work can regress double backward or numerical equivariance.
- Safe-default public cache validation remains O(E) for explicit edges because
  DLPack-safe automatic trust is impossible. The explicit immutable-storage
  opt-in avoids that scan by reusing the packed carrier; its caller promise is
  part of the performance contract.
- Direct CSR ordering can silently change floating-point reduction order.
- Multi-scale edges can multiply memory unless lanes share one candidate CSR.
- Accuracy ablations are noisy; decisions require fixed splits, seeds, budgets,
  and no test-label selection.
