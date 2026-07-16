# PROJECT.md

## Status

- State: confirmed for local code and test changes by the user's 2026-07-16
  instruction to incorporate the two additional review answers
- Direction confirmed: 2026-07-16
- Evidence basis: repository tests and experiment ledger, the external review
  shared by the user, and independent read-only mathematical, implementation,
  and experiment audits of the current source tree.
- Latest scope confirmation: the user's 2026-07-16 instruction to proceed from
  the feedback at
  `https://chatgpt.com/share/6a58dc3e-cff4-83ee-85dc-29a9d9cbe18f` authorizes
  a bounded Stage-0 HEMM activation diagnostic before further architecture or
  QM9 expansion.

## Project

- Name: equivariant-linear-attention
- Type: PyTorch ML research prototype
- Goal: develop one mixed-connectivity, multi-memory factorized
  moment-attention architecture for 3D structural graphs, with narrow and
  testable O(3), locality, degeneration, and evaluation contracts.
- Users/workflow: research and development from Python scripts and tests.
- Non-goals: multiple public attention families, production training
  infrastructure, chirality-sensitive prediction, or unvalidated
  protein/ligand segment semantics.

## One-Architecture Contract

- Public implementation: only `EquivariantAttention` with
  `EquivariantAttentionConfig`; local, global, and hybrid behavior are routing
  and memory-cardinality settings of the same transport block rather than
  separate model classes.
- Internal state: persistent scalar (`0e`) and polar-vector (`1o`) channels;
  symmetric-traceless rank-2 (`2e`) moments and an optional radial-trace
  scalar are transient within each block.
- Outputs: tensor-only dictionary with plural keys `node_scalars`,
  `node_vectors`, `node_tensors`, `graph_scalars`, `graph_vectors`, and
  `graph_tensors`.
- Batch IDs: integer, nonnegative, contiguous IDs starting at zero. Empty
  graphs are not represented.
- Symmetry: O(3), including reflections; translation invariant and
  permutation consistent. Scalar outputs cannot distinguish enantiomers.
- Numerical policy: geometry squares, angular/ST feature construction,
  moment reductions, and invariant normalization use at least float32;
  float64 remains float64. Coordinates must be float32 or float64. Feature
  tensors may use model precision, and coordinates are never downcast to it.
- Global geometry preprocessing is scale-first. Each graph is divided by its
  maximum absolute coordinate before its centroid and RMS are reduced, and
  physical log-radius/log-scale features are formed without directly
  multiplying values that can overflow. On ordinary float64 inputs this must
  agree with the direct formula to the declared tolerance.
- Configuration policy: dimensions and switches have exact runtime types;
  every real-valued control and its float32 representation is finite. Kernel
  floor/init/max controls and init/max ratios must be normal float32 values so
  declared positive initialization cannot underflow to zero; strict ordering
  and the derived kernel upper bound remain representable in float32.
  Scale-first local/memory geometry controls retain separately tested support
  for positive subnormal values.

## Local/Global Architecture

- The comparison architecture has three identical-schema transport blocks
  with the same query/key/value projections, bounded degree-2 kernel,
  updater, and FFN. `local_head_counts` changes only head connectivity.
- Registered routing presets at four heads are:
  - `ggg = (0, 0, 0)`: global-only incumbent.
  - `lgl = (4, 0, 4)`: local encoder, global context, local refinement.
  - `lll = (4, 4, 4)`: mechanistic local-only control.
  Mixed local/global head counts remain legal but are not a promoted default
  without a separate registered comparison.
- Until the preregistered validation comparison passes, the public default
  remains `ggg`. Implementing a capability is not evidence for promoting it.
- Initial scalar embedding uses invariant `node_feats` only and initial vector
  state is zero. Graph centroid, RMS scale, and normalized radius may enter
  only when the first global head executes, so `lll` cannot leak graph-global
  geometry through preprocessing.
- Global heads use centered/RMS-normalized coordinates and the existing exact
  graph-wise factorized moments. At fixed width, heads, and depth they do not
  materialize an `N x N` attention tensor.
- Local heads use directed same-graph edges from raw coordinates. An edge
  `i <- j` exists when `||p_j-p_i||^2 < R_c^2`; self edges are always present.
  The registered QM9 cutoff is `R_c = 2.5 Angstrom` with 16 Gaussian RBFs.
- With `d=(p_j-p_i)/R_c` and `u=||d||^2`, the cutoff is
  `f_c(u)=0.5(1+cos(pi*u))` for `u<1` and zero otherwise. This makes the
  message value and first coordinate derivative vanish at the cutoff and
  avoids a square root at coincident coordinates. No second-derivative
  continuity claim is made.
- The positive local radial gate has a fixed positive mixture floor, so a self
  edge and the kernel floor keep each local denominator positive in finite
  precision. Routing studies freeze learned radial-gate parameters; learning
  them is a later, separate ablation.
- A core-only fallback may build candidate pairs per graph and use O(E)
  transport storage, but its candidate search can remain O(N^2) for bounded
  QM9 graphs. The retained edge indices, normalized displacement, cutoff
  distance, and RBF basis are built once per forward and reused across local
  stages, reducing repeated geometry work from `L*N^2` to `N^2 + L*E` without
  changing outputs or gradients. End-to-end O(E) or production sparse-neighbor
  performance is not claimed without an explicit edge API or radius backend.

## Multi-Memory Global Transport

- Version 1 is a memory-gated extension of the existing query-dependent
  factorized kernel, not a separate write/read model. A naive single-memory
  state would not reduce to the incumbent attention and is excluded.
- Invariant local scalar states and deterministic slot codes produce bounded
  logits and soft assignments `pi_i in simplex(M)`. One explicit refinement
  may use a bounded squared-distance penalty to preliminary weighted
  centroids; no implicit assignment/centroid fixed point is used.
- Memory centers divide by their exact positive occupancy, without adding
  `eps` to the denominator. Adding `eps` there would scale the translation
  term and violate exact translation equivariance. Bounded logits and bounded
  radial penalties prevent finite-precision slot starvation.
- The fixed nonnegative coupling is a cosine cutoff of squared center distance,
  with `0 <= C_mn <= 1`, `C_mm=1`, and zero value/first derivative at the
  cutoff. There are no free Cartesian memory parameters or learned couplings.
- The effective pair gate is
  `G_ij = sum_mn pi_im C_mn pi_jn`, and the attention kernel is `K_ij G_ij`
  followed by the registered row normalization or exact one-cycle balancing.
  Per-memory structured summaries evaluate this without materializing an
  `N x N` matrix.
- With memory interaction disabled, `C` is all ones and the partitioned
  summaries reduce algebraically to the incumbent for every `M`; memory count
  alone therefore carries no expressivity claim. Interaction is a separate
  switch.
- `memory_count=1` and interaction off are the public defaults. For `M=1`,
  `pi=1`, `C=1`, and `G=1`, so
  the multi-memory function must reduce exactly to the incumbent structured
  attention for forward values and every differentiable input, with and
  without balancing.
- Multi-memory transport is initially enabled only for the middle global stage
  of `lgl`. `M=4` and `M=8` are registered experimental arms; they are not
  defaults. Fixed `M` costs `O(NM + M^2)` at fixed head width.
- Memory-slot permutation must leave node outputs unchanged when assignments
  and `C` are permuted consistently. Node permutation, batch isolation, O(3),
  translation, and coordinate-gradient contracts also apply.
- Soft assignments may collapse for identical or highly symmetric nodes.
  Occupancy and assignment entropy are recorded; no occupancy regularizer is
  introduced before a collapse is observed. Multi-level memories, semantic
  banks, learned interaction kernels, hard top-k, persistent tensor memories,
  and higher angular degree are deferred.

## Kernel and Moment Baseline

- Queries/keys use positive unit-normalized scalar content and finite-precision
  unit-ball polar vectors. With `t=q_i.k_j`, the registered kernel is
  `K=c+a_i.b_j+beta+beta*t+gamma*t^2`.
- The alignment ablation removes only `beta*t`; it retains the same `beta`
  constant in both arms. The switch is named
  `use_alignment_linear_term`; no result may attribute an effect to alignment
  if the constant baseline also changes.
- Linear vector and quadratic 3x3 PSD summaries compute global masses and
  denominators without graph-summed signed flattened outer features. Value
  numerators use analogous signed summaries and are not clamped.
- Key balancing is exactly one cycle when enabled. Private balance exponents,
  extra Sinkhorn iterations, and unregistered normalization paths are absent.
- One-cycle balancing can erase a pure key-side aligned/anti-aligned
  preference when all queries are identical. This is an executable
  expressivity counterexample, and balancing is selected empirically rather
  than justified by positivity.
- The kernel-baseline modes are `fixed` and `inverse_graph_size`. With alignment
  switch `delta` in `{0,1}`, the latter uses
  `K=a.b+(c+beta+delta*beta*t)/N_g+gamma*t^2`: it scales the complete
  nonnegative shifted-alignment baseline, not only `c`, while leaving content
  and quadratic selectivity unscaled. It is initially a global row-only
  experimental lane. `inverse_graph_size` with key balancing is rejected
  rather than assigned an unsupported denominator claim, and a local head
  never substitutes graph size for its receiver degree. A fixed positive
  baseline is not described as sparse attention because its maximum normalized
  weight is forced toward `O(1/N)` as graph size grows.
- The exact radial trace is available behind a flag and reserves the same
  scalar slot when disabled. Global heads reconstruct it in O(N) as
  `S2_i + ||x_i||^2 m_i - 2 x_i^T p_i`; local heads compute it directly on
  edges. The disabled slot is exactly zero, and the public rank-2 output
  remains symmetric traceless.
- The public defaults for key balancing, the linear angular term, radial
  trace, and `lgl` routing may change only through the registered validation
  decision rules below.

## Verification

- Maximum float64 O(3), translation, and permutation error: below `1e-6`.
- Factorized/global and sparse/local references, including gradients with
  respect to all query/key/value/kernel inputs and coordinates: `atol=1e-10`,
  `rtol=1e-9` in float64.
- Required local cases: coincident coordinates, self and isolated nodes,
  cutoff value/first-gradient continuity, cross-graph isolation, local-only
  separated-fragment invariance, and far-node communication through `lgl`.
- Required global cases: independent and exact antiparallel kernel-bound
  probes, balanced/unbalanced dense equivalence, moment collision, reflection,
  coordinate-gradient equivariance, alignment-constant isolation, balancing
  erasure, fixed-versus-inverse-size floor references, extreme finite
  scale-first coordinates, and multi-memory degeneration/permutation.
- Inference wrappers preserve eval state and public metadata. CPU, CUDA auto,
  bf16/fp16, compile, benchmark, and state-schema claims are tested where the
  corresponding runtime is available.
- Required project check: `scripts/check.sh fast` with at least 80% coverage;
  `scripts/check.sh gpu` before CUDA precision claims.

## Preregistered Evaluation Boundary

### Stage 0: memory-mechanism activation

- Before an interacting M=4 or M=8 arm is treated as a memory experiment, the
  selected actual runtime layer must report the effective pair gate
  `G_ijh = pi_ih^T C_h pi_jh` separately for every global head rather than infer
  activation from assignment entropy or coupling quantiles alone. Metrics are
  never pooled across graph or head normalization domains, and the worst head
  determines the activation decision.
- The JSON-safe summary reports min, p01, median, p99, max, population
  coefficient of variation, centered-Frobenius ratio, and the fraction whose
  absolute deviation from the mean exceeds `1e-3` times the positive mean.
  An all-zero gate is invalid.
- For a positive constant gate, CV, centered-Frobenius ratio, and nonconstant
  fraction are exactly zero and row normalization cancels the gate. Such a run
  is not evidence for multi-memory transport.
- On the fixed bounded heterogeneous synthetic probe, both M=4 and M=8 must
  have normalized assignment entropy in `[0.05, 0.995]`, minimum occupancy
  fraction at least `1e-4`, coupling q00 at most `0.99`, pair-gate
  centered-Frobenius ratio at least `1e-2`, nonconstant fraction at least
  `0.10`, and relative RMS output difference from the matched M=1 bypass at
  least `1e-5`. All values must be finite. CV is reported but is not a second
  independent pass condition because `D=CV/sqrt(1+CV^2)` exactly.
- Failure of any frozen threshold blocks the later interacting M=4/M=8 memory
  arms and triggers a separately preregistered router/coupling redesign;
  thresholds are not moved after observing the probe. It does not block the
  independent Stage 1--3 kernel or local/global studies.
- This Stage-0 change does not add a learned router, separate read/write or
  typed memories, raw-distance coupling, segment semantics, an external sparse
  edge API, higher-order channels, or a new default. Those remain contingent
  on architecture-lock evidence.

- Dataset/target: QM9 `gap` in eV; random-row split seed 42 with
  train/validation/test sizes 110k/10k/10k. This is not scaffold or
  cold-molecule generalization evidence.
- Adaptive runs use model seeds 41, 42, and 43, FP32, batch size 64, hidden
  width 64, three blocks, four heads, and otherwise identical optimizer and
  schedule settings. Common initialization hashes, total parameters,
  nonzero-gradient parameters, synchronized latency, and peak CUDA memory are
  recorded.
- Test metrics are opt-in through `--evaluate-test` and remain disabled during
  all adaptive work. Historical test access means the split is not a pristine
  confirmatory holdout.
- Stage 1 compares the 2x2 alignment-term/key-balance variants on `ggg` with
  radial trace off and the same `beta` constant. A 500-step pass is a numerical
  screen; surviving decisions are based on paired 2,000-step validation runs,
  not a single seed.
- Stage 2 fixes the selected alignment/balancing setting and compares the fixed
  kernel against graph-size scaling of `(c+beta+delta*beta*t)` under row
  normalization, leaving `gamma*t^2` unchanged. It also measures synthetic
  graph sizes 16, 32, 64, 128, 512, and 2,048 for maximum weight,
  entropy/log-N, gradients, and runtime.
- Stage 3 fixes the selected kernel normalization and compares `ggg` against
  `lgl`, with radial trace and memory interaction off. `lll` is a seed-42
  mechanistic control, not a performance candidate.
- Stage 4 compares `lgl` with interacting `M=1,4,8` under the selected
  preceding settings and radial trace off.
  The multi-memory claim is falsified for this benchmark if it fails the
  registered collision test or the validation promotion rule.
- Stage 5 changes only radial trace off/on for the selected routing and memory
  setting. Learned local radial gates are a later isolated ablation.
- A mechanism is promoted only when paired mean validation MAE improves by at
  least 0.01 eV, at least two of three seeds improve, the worst seed regresses
  by at most 0.02 eV, and median latency and peak-memory increases are each at
  most 20%. A joint-only gain is recorded as an interaction, not as evidence
  for either mechanism alone.
- Diagnostics include bounded kernel scales, mass and denominator quantiles,
  condition proxies, attention entropy normalized by log node count, column
  marginal CV, singular-spectrum effective rank, gradient and residual norms,
  runtime, peak memory, memory occupancy/assignment entropy, and node-count
  strata. Positive row/column scaling preserves exact matrix rank, so no
  balancing claim is based on exact rank change.
- A final 5-seed/10,000-step candidate-incumbent test evaluation is a separate
  approval gate after architecture lock; it is not part of this draft's
  adaptive budget.
- Existing runs in `docs/EXPERIMENTS.jsonl` are historical context only. In
  particular, a radial trace improved one 500-step seed but lost at 2,000
  steps, so it is not presently a justified default.

## Claim Boundaries

- A nonzero global route prevents cluster decomposition and extensive size
  consistency claims. The mean graph readout is not an additive energy model.
- Finite RBFs and degree-2 moments retain representation collisions.
- Whole-system reflection invariance prevents chirality-sensitive scalar
  predictions.
- Protein/ligand semantic memory banks are deferred: the current QM9
  schema cannot validate entity IDs, typed intra/inter-segment edges, separate
  cutoffs, residue/fragment assignments, segment pooling, or complex-frame
  semantics.

## Deferred Compute Gate

- Local code and test changes are authorized. Before adaptive QM9 execution,
  request one bounded approval for `uv sync --locked --extra qm9` and at most
  30 GPU-minutes. Any final 10k-step/test study is a later, separate gate after
  architecture lock.

## Commands

- Setup: `uv sync --locked`
- QM9 setup after approval: `uv sync --locked --extra qm9`
- Fast verification: `scripts/check.sh fast`
- GPU smoke: `scripts/check.sh gpu`
- Training probe: `uv run python scripts/train_compare.py --dataset synthetic`
- Benchmark: `uv run python scripts/bench_attention.py`

## Paths

- Data: `data/` (untracked)
- Outputs/logs: `outputs/` (untracked)
- Scientific run records: `artifacts/` (untracked control/evidence bundles)
- Source: `src/equivariant_attention/`
- Tests: `tests/`
