# PROJECT.md

## Status

- Reproducibility hardening confirmed on 2026-07-23. Before another accuracy
  architecture experiment, the matched harness must expose a legacy-compatible
  `seeded` lane and an opt-in `strict` lane that requests deterministic PyTorch
  algorithms, deterministic cuDNN behavior, disabled cuDNN benchmarking, and a
  valid cuBLAS workspace configuration before model construction or CUDA work.
  Every result records the effective runtime state. A same-source/config/data/
  split/initial-state/model-seed gate consumes five independent process results,
  reports validation-MAE mean/sample standard deviation/range, and admits a
  near-`0.01 eV` effect only when the same-seed range is at most `0.005 eV`.
  The strict lane additionally requires identical final-state hashes and metric
  values. Unsupported deterministic CUDA operators are a recorded failure, not
  permission to silently fall back. This packet implements and CPU-verifies the
  gate only; it authorizes no CUDA training or architecture promotion.
- PDBBind/ATOM3D-LBA extension confirmed on 2026-07-23 by the user's
  instruction to download and proceed:
  use the public `vector-institute/atom3d-lba` Parquet conversion of the
  PDBBind 2019 refined-set ATOM3D LBA task for a bounded train-only overfit
  sanity check. The source has 4,463 complexes and an ID30 train/validation/test
  split, but the initial packet uses only 16 deterministic train rows. It keeps
  only the provided pocket (`token_type_id=1`) and ligand
  (`token_type_id=2`) copies, excludes the duplicated full-protein copy, uses
  the pinned upstream atom-token category plus pocket/ligand identity as
  invariant node input, and pools
  the affinity prediction over ligand nodes after pocket-to-ligand transport.
  The pK label is used exactly as supplied and train-target normalization is
  fitted only on those 16 rows. The data remains under ignored `data/`, is not
  redistributed or committed, and may be used here only after the user
  confirms non-commercial research use under the upstream
  CC-BY-NC-ND-4.0 terms.
- The confirmed architecture intervention is limited to an opt-in persistent
  symmetric-traceless `2e` hidden state with invariant gated residual/FFN
  interaction, plus the already implemented edge-free multiscale global
  spatial kernel. The data path adds an optional permutation-consistent
  per-node readout mask; existing QM9/default calls remain unchanged.
  General `l>2`, `2o`, chirality-sensitive parity channels, e3nn/spherical
  harmonics, learned coordinate updates, and a new public model family are
  excluded from this packet.
- The confirmed experiment compares the edge-free attention candidate and the
  private static EGNN control on the identical 16 complexes and features.
  Overfit admission is finite training with train MAE at most `0.10 pK` within
  3,000 updates; it is a wiring/capacity check, not an accuracy or
  generalization claim. Once both arms reach the threshold, compare
  time-to-threshold, synchronized step latency, and peak CUDA memory. The
  packet uses no validation/test labels, caps cumulative GPU wall time at
  1,800 seconds, and stops when both arms pass or the bound is exhausted. A
  later full ID30 validation study requires a separate registered hypothesis
  and approval.
- The confirmed material envelope adds the optional `datasets` dependency,
  download/cache about 473 MB from Hugging Face at a recorded immutable
  revision, accept the optional graph/readout schema extension, and authorize
  at most 30 GPU-minutes on one local GPU.
- The 2026-07-23 implementation and train-only CPU fallback completed on the
  frozen 16 rows. Both 3,000-step CPU arms missed the registered `0.10 pK`
  numerical threshold: attention finished at `0.199863 pK` and private static
  EGNN at `0.163732 pK`. This rejects the CPU fallback instantiation, while the
  exact preregistered CUDA C3 remains not verified. Their best observed
  50-step evaluations were
  `0.151798 pK` (attention, step 2,950) and `0.163536 pK` (EGNN, step 2,850);
  that post-outcome comparison is descriptive and does not rescue the failed
  gate. On CPU, attention's median measured train step was `0.117964 s` versus
  EGNN's `0.252309 s` over 382,530 supplied radius candidates, a descriptive
  `2.139x` ratio. This single sequential timing excludes radius construction,
  batch collation, and periodic full-train evaluation. CUDA C2 execution and
  C3 overfit remain unverified; C4 is unfulfilled/rejected as registered, with
  its underlying CUDA evidence not verified, because
  the sandbox lacked driver access and the approved external execution was
  blocked by the current Codex usage limit. No validation or test row was
  evaluated. Evidence lives under
  `artifacts/pdbbind-overfit-persistent2e-20260723/`.
- Edge-free spatial-linear extension confirmed on 2026-07-23. The user
  authorized an opt-in, no-edge global transport mode plus bounded local GPU
  forward latency/peak-memory measurement. The registered intervention adds a
  fixed-rank, head-wise multi-scale Euclidean feature kernel to the existing
  factorized global attention; it materializes neither `edge_index` nor an
  `N x N` pair tensor and retains the existing bounded centroid-preserving
  coordinate updater. The default and state schema remain unchanged when the
  option is off. Dense equivalence, rigid-transform/permutation/batch tests,
  finite coordinate/train gradients, 396 fast tests, and bf16/fp32 CUDA smoke
  pass. On the recorded RTX PRO 6000 at `N=8192`, 100-repeat static spatial
  attention was 1.03x/2.24x faster and used 6.19x/12.39x less measured
  working-plus-edge memory than private static EGNN at `k=64/128`. The dynamic
  spatial path remained 3.6% slower at `k=64`, crossed by exploratory `k=80`,
  and was 2.10x faster at `k=128`. Small/sparse EGNN remains faster. This is a
  forward-only synthetic systems result, excludes neighbor construction and
  training/accuracy, and cannot establish graph-topology, molecule, protein,
  force, or point-cloud task superiority. The frozen contract and evidence
  live under `artifacts/edge-free-spatial-linear-20260723/`.
- Exact-edge-multiplier scaling extension completed on 2026-07-23. The
  benchmark now generates deterministic receiver-regular directed graphs with
  exactly `E=kN` candidates, one self edge and exactly `k` incoming candidates
  per node, then gives the same edge tensor to EC-LGL and private static EGNN.
  All 24 CUDA cells for `N={128,512,2048,8192}` and
  `k={4,8,16,32,64,128}` completed. At `N=8192`, 31-repeat confirmation over
  three topology seeds with model seed 20260723 fixed found EC-LGL/EGNN
  latency ratios of 1.593--1.597 at
  `k=32`, 0.985--0.988 at `k=64`, and 0.673--0.678 at `k=128`; the first
  measured same-edge crossover is therefore `k=64`, but the margin there is
  small. This is forward-only, excludes graph construction, and does not alter
  the failed QM9 accuracy decision or any default. The frozen/amended contract
  and evidence are under `artifacts/edge-multiplier-scaling-20260723/`.
- The first exact-edge generator used one flattened affine traversal. A
  quality-control RED test found receiver-degree skew despite valid counts and
  uniqueness, so those outputs are retained as `affine-exploratory` and the
  authoritative grid was rerun with exact receiver degree. This amendment is
  disclosed because it followed inspection of the initial timings.
- Scaling-aware EC-LGL extension: the user's 2026-07-22 instruction to improve
  the model while accounting for the small/low-edge-count QM9 regime confirms
  local implementation, sparse edge plumbing, synthetic scaling/crossover
  measurement, and a validation-only QM9 screen/conditional confirmation on
  one local GPU. The packet caps cumulative GPU wall time at 1,200 seconds,
  keeps test evaluation disabled, adds no dependency, and does not claim that
  preprocessing a radius/kNN graph is linear. The frozen contract is
  `artifacts/ec-lgl-sparse-scaling-20260722/scope.md`.
- The scaling-aware packet completed on 2026-07-22. Sparse edges now traverse
  loading, collation, device transfer, training, public attention, and the
  private EGNN control. Collation validates edge contents once; the trusted
  forward path avoids repeated sorting/uniqueness work, while direct callers
  remain validated by default. The exact factorized kernel first beat its materialized
  dense form at 4096 nodes while using 3.38 MB versus 671.09 MB peak CUDA delta.
  Degree-16 EC-LGL first crossed complete-edge EGNN at 512 nodes, but remained
  slower on the same sparse edges. Its repeated 500-step QM9 mean was 0.802194
  eV versus static LGL at 0.712178 eV, failing the +0.020 eV admission gate; no
  confirmation or test evaluation ran and the EC switch remains off by default.
- State: confirmed 2026-07-19 evidence-first strengthening extension; all
  earlier confirmed contracts remain active.
- State: confirmed for local code, bounded QM9 dependency setup, and at most
  30 GPU-minutes by the user's 2026-07-17 instruction to implement the latest
  review and measure performance properly
- Direction confirmed: 2026-07-17
- Evidence basis: repository tests and experiment ledger, the external review
  shared by the user, and independent read-only mathematical, implementation,
  and experiment audits of the current source tree.
- Latest scope confirmation: the user's 2026-07-17 instruction to proceed from
  `https://chatgpt.com/share/6a59db49-8b50-83ee-9691-07bb73a472f4`
  authorizes the counterfactual HEMM diagnosis, the smallest repair admitted by
  the unchanged Stage-0 thresholds, the independent `ggg`/`lgl` comparison,
  and synchronized CUDA/QM9 measurement. The exact frozen plan is
  `artifacts/hemm-coupling-repair-performance-20260717/scope.md`.
- Current scope: the user's 2026-07-18 instruction to develop the review at
  `https://chatgpt.com/share/6a5b0cd2-25a4-83e8-a54a-d4c43df5fa09`
  admits exact learned/uniform/none global-transport controls, `lgg`/`ggl`
  route decomposition, lazy global geometry, local-head diagnostics, an
  explicit HEMM Stage-0 warning, and a private same-harness static EGNN
  baseline. The frozen scope and experiment boundary are in
  `artifacts/egnn-matched-baseline-development-20260718/`.
- Current scope extension: the user's 2026-07-19 selection and post-update
  confirmation of option A admit a
  backward-compatible optional precomputed local `edge_index`, diagnostics for
  every active local stage over a deterministic bounded validation sample, and
  the already registered validation-only QM9 transport/route/internal-EGNN
  study on one local GPU with a cumulative ceiling of 25 GPU-minutes. No new
  dependency, edge feature, size-conditioned routing, HEMM redesign, final
  test evaluation, or 10,000-step claim is admitted. An evidence-conditioned
  default or simplification may be documented only when its frozen promotion
  rule passes. The exact frozen scope is
  `artifacts/evidence-first-strengthening-20260719/scope.md`.
- Dynamic-coordinate extension: the user's 2026-07-19 confirmation admits an
  opt-in, bounded, graph-centroid-preserving coordinate update in the public
  attention model; a private dynamic-coordinate EGNN control alongside the
  existing static control; and a validation-only QM9 study on one local GPU
  with a fresh cumulative ceiling of 25 GPU-minutes. Static defaults and
  checkpoint/output behavior remain unchanged. The exact frozen scope is
  `artifacts/dynamic-coordinate-egnn-20260719/scope.md`.
- The registered dynamic-coordinate packet completed on 2026-07-19: all six
  screen arms and twenty confirmation arms ran in 944.3 GPU-wall seconds with
  test evaluation disabled. Neither family passed every frozen promotion
  condition, so coordinate updates remain opt-in and no default changed.
- Competitiveness assessment: the current private static EGNN is the accuracy
  reference at `0.408932 eV` five-seed mean validation MAE. Static GGG trails it
  by `0.174014 eV`; the stronger historical static LGL result trails it
  descriptively by `0.106756 eV`. The next proposed hypothesis is that learned
  receiver/sender/distance-conditioned local edge content plus explicit
  neighborhood mass closes at least `0.050 eV` of the LGL gap. This proposal is
  recorded in `docs/EVALUATION.md`.
- EGNN-parity extension: the user's 2026-07-20 confirmation authorizes the
  proposed static-coordinate LGL work, at most three sequential architecture
  iterations, and at most 60 cumulative GPU-minutes on one local GPU. The
  ordered interventions are learned local radial gating, a parameter-bounded
  receiver/sender/RBF edge-content branch with explicit neighborhood
  mass/degree invariants, and one evidence-selected topology or optimization
  repair if needed. Test evaluation, new dependencies, multi-memory,
  coordinate updates, checkpoint publication, and 10,000-step claims remain
  excluded. The exact frozen packet is
  `artifacts/egnn-parity-20260720/scope.md`.
- The EGNN-parity packet completed all three permitted architecture iterations
  in 850.7 cumulative GPU-wall seconds with no test evaluation. Radial-only
  reached 0.499508 eV five-seed mean versus its rerun static EGNN at 0.421199
  eV. Pairwise content at residual scale 0.1 failed its screen; exact-baseline
  zero initialization passed the screen but reached 0.509008 eV versus rerun
  EGNN at 0.438268 eV. Neither confirmation won any paired seed, so no model or
  default is promoted. Further architecture training requires a new confirmed
  packet; seeded CUDA repeat drift must be addressed first.

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
  symmetric-traceless rank-2 (`2e`) moments are transient by default and may
  also be carried as opt-in persistent hidden channels. The optional radial
  trace remains transient within each block.
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
  - `lgg = (4, 0, 0)`: local encoder followed by two global stages.
  - `ggl = (0, 0, 4)`: two global stages followed by local refinement.
  - `lgl = (4, 0, 4)`: local encoder, global context, local refinement.
  - `lll = (4, 4, 4)`: mechanistic local-only control.
  Mixed local/global head counts remain legal but are not a promoted default
  without a separate registered comparison.
- Until the preregistered validation comparison passes, the public default
  remains `ggg`. Implementing a capability is not evidence for promoting it.
- Initial scalar embedding uses invariant `node_feats` only and initial vector
  state is zero. Graph centroid, RMS scale, and normalized radius are computed
  once, immediately before the first active learned or uniform global
  transport. `lll` and global-transport `none` do not execute or receive this
  preprocessing.
- Global transport has three state-schema-identical controls. `learned` is the
  incumbent factorized kernel. `uniform` broadcasts exact graph means of the
  same value and relative-moment sufficient statistics, giving
  `A_ij=1/N_g` in O(N). `none` removes global message and attention-updater
  residuals; an all-global block retains only its pointwise equivariant FFN.
  Memory interaction is legal only with learned global transport.
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
- The core-only fallback vectorizes the same-graph Cartesian candidates across
  the batch and uses O(E) retained transport storage, but its candidate search
  remains `O(sum_g N_g^2)` for bounded QM9 graphs. A keyword-only precomputed
  `edge_index` bypasses that discovery and uses O(E) candidate/retained storage;
  supplied candidates still pass through the same strict cutoff. The retained
  edge indices, normalized displacement, cutoff distance, and RBF basis are
  built once per forward and reused across local stages. Thus fallback work is
  `N^2 + L*E`; after validation, supplied-edge geometry/transport is `L*E` at
  fixed width. A production radius-neighbor backend or end-to-end
  neighbor-construction claim is not included.
- Coordinate updates are opt-in and disabled by default. When enabled, each
  nonfinal transport block forms an O(3)-equivariant displacement from its
  invariant scalar state and polar-vector state, bounds the per-node step by
  `0.25 Angstrom`, removes the exact graph-wise mean displacement, and updates
  the working coordinates before the next block. Omitting a dead post-readout
  updater ensures every coordinate parameter affects the scalar training loss.
  This preserves graph centroids and
  translation/O(3)/permutation behavior while preventing unbounded latent
  geometry drift. The coordinate-enabled state schema is intentionally
  distinct; the disabled state schema remains byte-for-byte compatible.
- With coordinate updates enabled, local cutoff/RBF geometry and scale-first
  global geometry are recomputed from the current positions before every
  applicable block. A supplied `edge_index` remains a fixed candidate topology
  but its candidates are cutoff-filtered again at each local stage; omitted
  edges cannot enter later solely because coordinates move. The complete-graph
  fallback can admit every same-graph candidate at each stage.
- A coordinate-enabled forward adds `node_positions` to the tensor-only output
  dictionary. The disabled default retains exactly the existing six output
  keys. Updated coordinates are latent task features, not claims of optimized
  physical geometries, conservative forces, molecular dynamics, or a learned
  potential-energy surface.

## Multi-Memory Global Transport

- Version 1 is a memory-gated extension of the existing query-dependent
  factorized kernel, not a separate write/read model. A naive single-memory
  state would not reduce to the incumbent attention and is excluded.
- Invariant local scalar states and fixed invariant slot codes produce bounded
  logits and soft assignments `pi_i in simplex(M)`. The incumbent deterministic
  one-dimensional router is retained only as a counterfactual control. If its
  identity-coupling gate is functionally constant under the frozen Stage-0
  threshold, the registered repair is one shared multidimensional invariant
  MLP whose parameters and state schema do not depend on `M`. One explicit
  refinement may use a bounded squared-distance penalty to preliminary weighted
  centroids; no implicit assignment/centroid fixed point is used.
- Memory centers divide by their exact positive occupancy, without adding
  `eps` to the denominator. Adding `eps` there would scale the translation
  term and violate exact translation equivariance. Bounded logits and bounded
  radial penalties prevent finite-precision slot starvation.
- The radial coupling is a cosine cutoff of squared center distance, with
  `0 <= C_mn <= 1`, `C_mm=1`, and zero value/first derivative at the cutoff.
  The only registered activation repair is the fixed nonparametric mixture
  `C_lambda=(1-lambda)C_radial+lambda I`, choosing the smallest of
  `{0.10,0.25,0.50}` that passes the unchanged Stage-0 gate. There are no free
  Cartesian memory parameters or learned couplings.
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
  defaults. Constructing an interacting `M>1` arm emits an explicit Stage-0
  blocked warning. Fixed `M` costs `O(NM + M^2)` at fixed head width.
- Memory-slot permutation must leave node outputs unchanged when assignments
  and `C` are permuted consistently. Node permutation, batch isolation, O(3),
  translation, and coordinate-gradient contracts also apply.
- Soft assignments may collapse for identical or highly symmetric nodes.
  Occupancy, marginal/conditional assignment entropy, and their normalized
  mutual information are recorded; no occupancy regularizer is introduced
  before a training collapse is observed. Multi-level memories, semantic
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
- The existing private `internal_static_egnn_baseline` remains unchanged. A
  separate private `internal_dynamic_egnn_baseline` follows the EGNN pattern:
  invariant edge messages depend on node states and squared distance, and an
  invariant scalar edge projection weights relative-coordinate vectors before
  a node update. Its aggregate displacement is bounded to `0.25 Angstrom` and
  graph-mean-centered under the same stability contract as the attention arm.
  This is an equation-level same-harness control, not an official repository
  reproduction or a second public architecture.

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
- Required coordinate-update cases: raw-boundary float64 O(3), reflection,
  translation, node permutation, graph centroid preservation, singleton and
  mixed-batch isolation, finite coincident coordinates, the per-layer `0.25
  Angstrom` bound, nonzero coordinate-parameter gradients, updated-position
  coordinate-gradient covariance, and exact disabled-mode output/state
  compatibility. Dynamic local routes must prove that later stages receive
  recomputed geometry rather than the initial geometry object.
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
- On the feature-spatial aligned probe, both M=4 and M=8 must have conditional
  assignment entropy in `[0.05, 0.995]`, marginal entropy at least `0.05`,
  normalized assignment mutual information at least `1e-3`, minimum occupancy
  fraction at least `1e-4`, coupling q00 at most `0.99`, pair-gate
  centered-Frobenius ratio at least `1e-2`, nonconstant fraction at least
  `0.10`, middle-message and post-middle symmetric relative RMS at least
  `1e-5`, scalar/vector/position gradient symmetric relative RMS at least
  `1e-5` each, and full-output relative RMS from the matched M=1 bypass at
  least `1e-5`. All values must be finite. CV is reported but is not a second
  independent pass condition because `D=CV/sqrt(1+CV^2)` exactly.
- Failure of any frozen threshold blocks the later interacting M=4/M=8 memory
  arms and triggers a separately preregistered router/coupling redesign;
  thresholds are not moved after observing the probe. It does not block the
  independent Stage 1--3 kernel or local/global studies.
- Stage-0 is run at widths 16 and 64, seeds 401--403, and on aligned, crossed,
  spatial-only, and semantic-only graphs with distinct registered roles. The
  aligned graph is the all-seed admission gate; the other graphs diagnose
  robustness and limitations. Separate read/write or typed memories,
  raw-distance coupling, segment semantics, an external sparse edge API,
  higher-order channels, occupancy loss, and a new default remain deferred.

- Dataset/target: QM9 `gap` in eV; random-row split seed 42 with
  train/validation/test sizes 110k/10k/10k. This is not scaffold or
  cold-molecule generalization evidence.
- Stage 1--3 adaptive confirmations use model seeds 41, 42, and 43. Stage 3a
  and the subsequent EGNN comparison extend the registered confirmation set to
  seeds 41--45 as specified below. All use FP32, batch size 64, three blocks,
  and otherwise identical optimizer and schedule settings; attention width 64
  is matched by EGNN width 91. Common initialization hashes, total parameters,
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
- Stage 3a first screens `ggg/lgg/ggl/lgl` learned transport plus `lgl`
  uniform/none at seed 42 and 500 steps. It then compares `lgl`
  learned/uniform/none at seeds 41--45 and 2,000 steps. A learned-kernel or
  global-transport claim requires mean paired validation improvement at least
  0.010 eV, at least three of five improving seeds, worst regression no more
  than 0.020 eV, and median latency/peak-memory increases no more than 20%.
- The completed 2026-07-19 Stage-3a confirmation found learned/uniform/none
  mean validation MAEs of 0.515688/0.534776/0.691821 eV. Accuracy criteria
  passed for learned versus uniform and for both global arms versus none, but
  learned exceeded the memory ceiling versus uniform and learned/uniform
  exceeded at least one latency/memory ceiling versus none. The joint
  promotion gate therefore failed, no transport mode was locked, defaults did
  not change, and the conditional EGNN comparison did not run.
- After the transport mechanism is locked, a private width-91,
  static-coordinate, three-layer EGNN baseline may be trained in the same
  harness. It uses the identical PyG features, split, target-only train-fitted
  normalization, MSE/AdamW/cyclic update budget, and mean readout. It is labeled
  `internal_static_egnn_baseline`, not an official EGNN reproduction or public
  model family.
- The dynamic-coordinate study is independent of the failed transport
  efficiency lock and changes only the coordinate-update switch within each
  paired family. Its seed-42, 500-step screen contains static/dynamic `ggg`,
  static/dynamic `lgl`, and static/dynamic private EGNN. An attention route is
  eligible only if both arms are finite, the dynamic displacement is active,
  and dynamic validation MAE is no more than `0.020 eV` worse than its static
  pair; the eligible route with lower dynamic validation MAE advances.
- Confirmation reruns the selected attention route static/dynamic and private
  EGNN static/dynamic at seeds 41--45 for 2,000 updates. For either family,
  coordinate updating is promoted only when mean paired validation MAE improves
  by at least `0.010 eV`, at least three of five seeds improve, the worst seed
  regresses by no more than `0.020 eV`, and median elapsed-time and peak-memory
  ratios are each at most `1.20`. A failed gate leaves the public default off;
  cross-family EGNN/attention numbers are descriptive because their update
  parameterizations differ.
- The completed coordinate confirmation selected `ggg` in the screen. Attention
  static/dynamic mean validation MAE was `0.582946/0.585535 eV`, a mean paired
  improvement of `-0.002589 eV`; three seeds improved and both resource and
  worst-seed gates passed, but the mean-accuracy gate failed. Private EGNN
  static/dynamic mean validation MAE was `0.408932/0.410428 eV`, a mean paired
  improvement of `-0.001496 eV`; three seeds improved, but the mean-accuracy,
  worst-seed (`-0.052560 eV`), and elapsed-ratio (`1.456`) gates failed. Both
  promotion decisions are therefore false. All ten dynamic confirmation arms
  had active nonzero coordinate gradients, maximum observed per-layer step
  `0.25000003 Angstrom`, and maximum graph-centroid drift
  `4.92e-7 Angstrom`.
- Stage 4 compares `lgl` with interacting `M=1,4,8` under the selected
  preceding settings and radial trace off.
  The multi-memory claim is falsified for this benchmark if it fails the
  registered collision test or the validation promotion rule.
- Stage 5 changes only radial trace off/on for the selected routing and memory
  setting. Learned local radial gates are a later isolated ablation.
- Unless a stage explicitly registers a five-seed rule, a mechanism is promoted
  only when paired mean validation MAE improves by at least 0.01 eV, at least
  two of three seeds improve, the worst seed regresses by at most 0.02 eV, and
  median latency and peak-memory increases are each at most 20%. Stage 3a uses
  its explicit three-of-five rule. A joint-only gain is recorded as an
  interaction, not as evidence for either mechanism alone.
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

- The confirmed 2026-07-19 option-A packet was consumed by the registered
  validation-only transport study: 21 runs completed in 819.2 GPU-wall seconds
  with test evaluation disabled. The efficiency gate failed, so the conditional
  EGNN arms did not run despite remaining time.
- Any EGNN continuation, final 10,000-step run, or test evaluation requires a
  new frozen hypothesis and compute approval after the transport-efficiency
  failure is addressed. The existing result does not authorize using the
  remaining nominal budget for a different experiment.
- The user's 2026-07-19 dynamic-coordinate confirmation supplies a new, separate
  ceiling of 25 cumulative GPU-minutes for the frozen screen and conditional
  confirmation above. That packet is now consumed: 26 runs completed in 944.3
  seconds and both promotion decisions failed. It authorizes no test
  evaluation, checkpoint publication, dependency change, 10,000-step run, or
  additional post-hoc arm. Further GPU training requires a new frozen
  hypothesis and compute approval.
- The user's 2026-07-20 confirmation supplies a fresh 3,600-second cumulative
  GPU ceiling for the EGNN-parity packet only. Each architecture iteration
  begins with a seed-42/500-update validation screen and advances to matched
  seeds 41--45/2,000 updates only when finite and nonregressing. The packet
  stops immediately when the candidate reaches mean validation MAE at most
  `0.398932 eV`, wins at least three of five paired seeds against a rerun private
  static EGNN, and has worst paired regression no lower than `-0.020 eV`; or
  when three iterations or 3,600 GPU-seconds are consumed. No test labels may
  be opened, and failed/null iterations must remain in the ledger.

## Commands

- Setup: `uv sync --locked`
- QM9 setup after approval: `uv sync --locked --extra qm9`
- Fast verification: `scripts/check.sh fast`
- GPU smoke: `scripts/check.sh gpu`
- Training probe: `uv run python scripts/train_compare.py --dataset synthetic`
- Benchmark: `uv run python scripts/bench_attention.py`
- Coordinate-study inspection: `uv run --locked python
  scripts/run_registered_coordinate_study.py artifacts/coordinate-study-reproduction
  --dry-run`

## Paths

- Data: `data/` (untracked)
- Outputs/logs: `outputs/` (untracked)
- Scientific run records: `artifacts/` (untracked control/evidence bundles)
- Source: `src/equivariant_attention/`
- Tests: `tests/`
