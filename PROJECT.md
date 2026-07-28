# PROJECT.md

## Status

- The recorded cross-run topology reproducibility defect is repaired and the
  candidate-list contract is frozen. `torch.cdist` defaults to the
  matrix-multiplication Euclidean identity above 25 points; in float32 its error
  grows with coordinate magnitude (up to `0.25 Angstrom` at a `300 Angstrom`
  offset) and its bytes change with the BLAS thread count. That made the LBA
  topology neither translation invariant nor reproducible, which is the
  mechanism behind the 32,303,245 versus 32,303,244 edge drift.
  `segment_balanced_knn_edge_index` now retains an edge exactly when the float64
  squared displacement is below the squared cutoff, keeps every exact tie at the
  kth boundary, and is chunked without an `N x N` distance matrix; it is also
  1.2--4.4x faster at 500--5,000 nodes. `topology_sha256` is one shared
  definition. The official ID30 identity is now `32,302,952` edges /
  `57f40fb157e6416558db5507d95c3a5e4f828881e0bc92e142e1b85de802dc6c`, verified
  equal in fresh 1-thread and 4-thread processes; the legacy `344158...` and
  `1eea0af8...` hashes are 293 and 292 edges larger and must not be reused as
  expected values. No model equation, default, or checkpoint schema changed, and
  every historical packet stays internally valid because its arms shared one
  in-memory list. See `docs/TOPOLOGY_CONTRACT_20260727.md`.
- Sparse geometry-aware local attention with selectable `O(3)`/`SE(3)`
  symmetry is implemented and **not promoted**. The useful EquiFlex transfer
  is an opt-in sparse `0e/1o/2e` score refinement, with a mathematically
  explicit axial `2e x 2e -> l=1` value admitted only under `SE3`; dense pair
  state, triangle enumeration, and `N x N` attention remain excluded. A
  matched seed-41, 20-epoch ATOM3D-LBA ID30 screen gave candidate/O(3)/SE(3)
  validation RMSEs of `1.602722/1.606076/1.610371 pK`. End-to-end median
  step ratios were `1.000/1.180/1.202x` and peak-allocation ratios were
  `1.000/1.157/1.164x`. The axial gate received a finite nonzero gradient but
  did not add measurable fitting or validation benefit. Public defaults remain
  unchanged; see `docs/GEOMETRY_AWARE_SE3_20260727.md`.
- The static Cartesian CTP-LGL packet is complete and is **not promoted**.
  Mathematical/software contracts and the isolated real-batch resource gate
  passed, but the fixed seeds 41--43 ATOM3D-LBA ID30 comparison rejected the
  accuracy hypothesis. Candidate/persistent-`2e`/CTP mean best validation
  RMSEs were `1.580164/1.601467/1.590217 pK`. CTP recovered
  `0.011251 pK` on average from the persistent-tensor-only control, yet
  regressed the current candidate by `0.010053 pK` and lost all three paired
  seeds. Its parameter/step/peak ratios were
  `1.00157/1.13421/1.15046x`, within the frozen ceilings. The code remains an
  opt-in, statically compiled `0e+1o+2e` research capability; public defaults
  and the arbitrary-irrep boundary are unchanged. Test labels were not
  evaluated. See `docs/CTP_LGL_20260727.md`.
- Static-compiled CTP-LGL extension confirmed on 2026-07-27 after the user
  selected the combined design: a configurable irrep specification at model
  construction, but a fixed and optimized tensor-product execution graph at
  runtime. The first implementation packet is deliberately narrower than
  arbitrary irreps. It keeps the existing Cartesian `0e/1o/2e` backend,
  retains the exact fixed-rank `O(N)` global transport, and adds an opt-in
  persistent-`2e` local tensor-product branch to LGL. No dependency, public
  default, coordinate policy, or incumbent checkpoint schema change is
  authorized. The frozen contract is in
  `Static-Compiled Irreps and CTP-LGL Extension` below.
- A frozen seed-44 ATOM3D-LBA clipping screen found that the remaining
  optimization issue is not local-message explosion. At 20 matched epochs,
  clip-1 / clip-10 / unclipped last validation RMSEs were
  `1.628645/1.611120/1.600802 pK`. No clipping improved the primary metric by
  `0.027843 pK`, passed every one-seed screen criterion, and stayed resource
  neutral (`1.0101x` latency, `0.9988x` peak allocation). Clip 10 improved
  `0.017524 pK` but missed the registered `0.020 pK` threshold. The clip-1
  path scaled gradients by only `0.1719x` on average, while squared-norm share
  was spread over FFN/global/input/readout/local
  (`36.25/24.64/19.26/15.81/4.04%`). The default remains `grad_clip=1.0`:
  this is one-seed exploratory evidence and needs a clean paired multi-seed
  confirmation. The runner now accepts `--grad-clip none` and records scale,
  norm dispersion, threshold exceedance, and path shares. See
  `docs/LBA_GRADIENT_CLIPPING_20260727.md`.
- That packet also found a cross-run topology reproducibility defect. Its one
  shared precomputed list had 32,303,245 edges/hash `344158...`, while the
  preceding three-seed packet had 32,303,244/hash `1eea0a...`; a fresh seed-41
  rebuild returned `344158...`, so seed is not the cause. The paired clipping
  conclusion is unaffected because every arm consumed the same list, but a
  permutation-safe deterministic distance/tie contract must be frozen before
  multi-seed confirmation. An explicit squared-distance probe is only a
  candidate contract and has not replaced the current topology.
- A real-batch operator profile explains the accepted LBA candidate's resource
  behavior. On 16 cached train complexes (7,378 nodes / 153,029 edges), the
  candidate used fewer indexing/scatter launches and more edge-MLP matrix
  multiplies; this profile is consistent with its `0.7364x` synchronized step
  time and `1.3698x` peak allocation relative to the incumbent. Checkpointing
  the latter gated edge-MLP
  segment reduced candidate peak allocation by `20.29%` at a `22.03%`
  step-latency cost. The LBA runner now exposes this existing
  equation-preserving mechanism as opt-in
  `--checkpoint-gated-local-mlp`; default speed behavior is unchanged. See
  `docs/LBA_OPERATOR_PROFILE_20260727.md`.
- The 2026-07-27 matched-budget ATOM3D-LBA ID30 study gives the first
  multi-seed real-affinity support for the current squared-RBF
  gated-plus-grouped LGL. At exactly 35 epochs / 7,700 updates per arm and
  seeds 41--43, candidate/incumbent mean validation RMSEs were
  `1.598765/1.619865 pK`; paired improvement was `0.021099 pK` with `3/3`
  wins. The candidate was also faster (`0.9347x` median step latency) while
  using more memory (`1.3723x` peak allocation). All frozen numeric/resource
  gates passed and test stayed structurally inaccessible. The evidence remains
  exploratory because the matched-epoch repair was chosen after partial
  seed-41 curves were observed; most of the mean gain came from seed 41.
  Pathwise diagnostics show the candidate reduced shared pre-clip norms, but
  both arms still clipped about 99% of updates. See
  `docs/LBA_MULTISEED_CONFIRMATION_20260727.md`.
- The 2026-07-27 strict-CUDA confirmation closes the radial-spacing question.
  Five paired 2,000-update QM9 `gap` runs at model seeds 41--45 gave mean
  validation MAE `0.371793 +/- 0.020792 eV` for the current gated-plus-grouped
  LGL with the incumbent squared-distance RBF and
  `0.383284 +/- 0.019450 eV` for distance spacing. Distance spacing improved
  only seed 45, regressed the paired mean by `0.011491 eV`, and had a worst
  regression of `0.036583 eV`; it failed three of the five frozen promotion
  criteria. Its median train-step latency and peak allocation were effectively
  unchanged (`0.994x` and `1.000x`), so the rejection is an accuracy result,
  not a resource tradeoff. The 500-update seed-42 gain of `0.029220 eV` was an
  early-training false positive. The option remains an experimental,
  schema-compatible radial basis for reproducibility and non-QM9 work, but the
  public default remains `local_rbf_spacing="squared"`. The conditional LBA
  run was correctly not executed and test labels remained closed. See
  `docs/DISTANCE_RBF_CONFIRMATION_20260727.md`.
- The same packet materially revises the old same-harness EGNN narrative. The
  current squared-RBF LGL averaged `0.371793 eV`, versus `0.418467 eV` for the
  private complete-graph static EGNN and `0.417983 eV` for the private
  topology-matched 2.5-Angstrom static EGNN. LGL won three of five complete
  pairs and all five topology-matched pairs, with mean advantages
  `0.046674/0.046191 eV`. This is validation-only evidence against internal
  controls, not an official EGNN or published-model superiority claim. The
  cost boundary is equally important: LGL was `5.53x/5.56x` slower per
  train step than the complete/matched controls; it used `0.708x` the complete
  EGNN peak allocation but `1.427x` the matched EGNN allocation. Accuracy is
  no longer the immediate same-harness blocker; small-graph constant factors,
  affinity seed uncertainty, and an official external baseline are.
- Production cleanup followed the negative evidence rather than preserving
  every screened mechanism. Differential global attention, the global output
  gain, and local angular conditioning were removed from the production model,
  CLI, and construction path after their registered negative/null screens.
  Their immutable results remain in the experiment ledger and local artifacts.
  The retained implementation delta is the small opt-in radial-spacing control
  plus the corrected private-EGNN topology-matching harness. The full local
  gate after cleanup passed 572 tests at 88.68% coverage, and direct CUDA BF16
  and FP32 smokes plus real-QM9 strict-CUDA two-update smokes passed.
- The local angular-conditioning packet was implemented and strict-CUDA screened
  on 2026-07-26 in 140.7 of 900 authorized GPU-wall seconds. The defect it
  targeted is recorded as an executable fact rather than an argument: because the
  hidden vector state is zero at the first block, the gated local edge MLP sees
  only distances from the receiver, and for two three-node configurations with
  equal distances from the centre and different neighbour angles the incumbent's
  central-edge hidden activations agree to `atol=1e-12`. Angular information
  therefore reaches the model only after aggregation, where it cannot modulate an
  individual message. The opt-in intervention conditions each gated edge message
  on two parity-even invariants of the receiver's cutoff-weighted first and
  second direction moments contracted with the edge direction, normalized by the
  soft-normalization v2 convention. It costs one extra fused reduction and two
  scalars per edge, so `O(E)` with no triplet enumeration, and adds no irrep, no
  square root, and 128 parameters; a separate zero-initialized projection keeps
  every incumbent weight draw intact so the enabled model reproduces the
  incumbent exactly at initialization while still receiving gradient from the
  first update. Screened against both radial bases, incumbent, angular,
  squared-incumbent, and squared-angular validation MAEs were
  `0.617117/0.626126/0.646338/0.643612 eV`. Nothing was admitted and no default
  changed. Evidence is in `artifacts/local-angular-20260726/`.
- That result is a null, not a refutation. The two paired effects straddle zero
  (`-0.009008 eV` on the distance basis, `+0.002726 eV` on the squared basis),
  both are far under the `0.020 eV` threshold, and the sign flips with the radial
  basis, so the effect sits below the resolution of a one-seed 500-update screen.
  This differs from the differential-attention packet, whose regressions were
  about six times the threshold. Cost was negligible (`1.059x` latency, `1.001x`
  peak allocation, `1.0008x` parameters), so the rejection concerns only the
  absent gain, and the mechanism was verified active beforehand, so the null is
  not an inert-wiring artifact. Two limitations are recorded: 500 updates is an
  early-training screen and this project has already seen the radial trace
  improve at 500 updates and lose at 2,000, and whether the zero-initialized
  angular weights grew materially within 500 updates was not measured, so "too
  few updates for a newly initialized path" is not excluded. The
  `squared_incumbent` arm returned `0.6463378731`, bit-matching the
  architecture-v3 screen's incumbent, so cross-packet strict determinism is
  reproduced and effects of this size can be read at all.
- The differential-attention packet was implemented and strict-CUDA screened on
  2026-07-26 in 144.9 of 900 authorized GPU-wall seconds, and it closes a
  direction. The motivating measurement, taken with the repository's own bounded
  diagnostics on a 20-atom cached QM9 graph, is that the exact factorized global
  kernel is numerically uniform: normalized entropy over `log N` is `0.999759`,
  mean maximum weight is `1.05x` the uniform weight, column CV is `0.00253`, and
  the selectivity-bearing alignment and quadratic terms span `4.9e-3` and
  `1.6e-4` inside a kernel of `1.51..2.01`, so they are `0.3%` of it. Neither
  the never-before-screened `inverse_graph_size` floor nor a tenfold larger
  alignment initialization repairs this. Two opt-in interventions from the recent
  attention literature were screened as a 2x2 factorial: differential attention,
  which subtracts a second independently parameterized normalized global output
  with a bounded per-head `lambda`, and a bounded per-head multiplicative gain
  `2 sigmoid(W x)` on the global transport output. Both keep the exact
  factorization, stay `O(N)` with no pair tensor, keep full `O(3)`, and are
  initialized to the incumbent function exactly. Incumbent, differential,
  output-gain, and combined validation MAEs were
  `0.617117/0.741417/0.741077/0.650565 eV`. The hypothesis is falsified: the two
  isolated interventions regressed by `0.124300` and `0.123960 eV`, about six
  times the admission threshold, and the combination regressed `0.033448 eV`
  with a large positive factorial interaction of `0.214811 eV` that must be read
  as an interaction and not as evidence for either mechanism. Every resource
  ceiling passed, so the rejection is on accuracy alone. Nothing is promoted and
  no default changed. Evidence is in `artifacts/differential-attention-20260726/`.
- The interventions were verified active before the screen rather than assumed:
  at `lambda = 0.5` the trained per-head `lambda` held near `0.48..0.51` and the
  across-node dispersion of the global scalar message rose from `0.0324` to
  `0.0958`, a `2.96x` increase, while the exactly-incumbent `lambda = 0`
  initialization left it inert at `+2%`. Making the global message node
  dependent is therefore precisely what cost accuracy. Read together with the
  registered transport study, where learned transport (`0.515688 eV`) barely
  beat an exact uniform mean broadcast (`0.534776 eV`) and both far beat no
  transport (`0.691821 eV`), the conclusion is that for QM9 `gap` the useful
  content of the global path is the near-uniform graph mean: its uniformity is
  desirable, not defective. This closes the "make global attention selective"
  direction, which was the last untested structural explanation for the
  private-EGNN gap, and it retrospectively explains the architecture-v2 and v3
  rejections, since adding polynomial order to terms worth `0.3%` of a kernel
  whose uniformity is wanted could not have helped. One oddity is recorded and
  not interpreted: two structurally different perturbations landed within
  `0.00034 eV` of each other, which the recorded pathwise pre-clip gradient
  norms do not explain. Local attention, by contrast, is measurably selective
  (entropy over log degree, mean `0.85`, minimum `0.67`), and every accepted gain
  in this project's history came from the local path, so that is where the next
  packet should spend capacity.
- The receptive-field/radial-resolution packet was implemented and strict-CUDA
  screened on 2026-07-25 in 171.6 of 900 authorized GPU-wall seconds. It split
  two previously confounded factors of the local route at fixed features,
  split, seed, optimizer, and one shared initial state: how much of the molecule
  the local heads may see (`local_cutoff`), and how the radial basis distributes
  resolution at that cutoff (the new opt-in `local_rbf_spacing="distance"`,
  whose Gaussian centers are uniform in normalized radius instead of normalized
  squared distance, still evaluated as a square-root-free polynomial of the
  coordinates with an unchanged parameter schema). Incumbent, distance-spacing,
  5.0-Angstrom, and 5.0-Angstrom-plus-distance validation MAEs were
  `0.646338/0.617117/0.796026/0.804809 eV`. Only distance spacing at the
  incumbent 2.5-Angstrom cutoff passed the frozen gate, improving `0.029220 eV`
  at `0.969x` median train-step latency and `1.000x` peak allocation. The
  receptive-field half of the hypothesis is falsified: raising the cutoff lifted
  same-molecule pair coverage from `0.418` to `0.941` but cost about `0.15 eV`
  and `1.445x` peak allocation, so local coverage is not the binding constraint
  at this budget. The incumbent value reproduced the architecture-v3 screen's
  incumbent exactly. The gain mechanism is not the predicted one: at
  `R_c = 2.5` the distance basis is slightly coarser near `1.5 Angstrom`
  (radius-space FWHM `0.418` versus `0.342`), so it is not finer covalent
  resolution and remains unexplained. Clipping stayed above `0.90` in every arm.
  No test labels were evaluated, no default changed, and the arm remains opt-in
  pending multi-seed confirmation under a separate packet. Evidence is in
  `artifacts/receptive-field-20260725/`.
- The same screen fixed a control defect rather than an architecture: the
  private static EGNN had been consuming the complete same-graph edge list while
  the attention arms' local heads saw only candidates inside `local_cutoff`. The
  harness now accepts a matched cutoff for the EGNN arm together with
  `--precompute-local-edges` and rejects a cutoff that would be inert. At 500
  updates the EGNN reached `0.718055 eV` on complete edges versus
  `0.740313 eV` on the matched 5.0-Angstrom candidates, so earlier QM9
  EGNN-versus-attention numbers included a topology advantage of about
  `0.022258 eV`. Both EGNN arms ran at 500 updates and are therefore not
  comparable to the historical five-seed 2,000-update EGNN mean of
  `0.408932 eV`; this revises how those numbers may be read and is not an
  accuracy claim for either family.
- Architecture-v3 was confirmed, implemented, and strict-CUDA screened on
  2026-07-25. It deliberately does not repeat the rejected
  bounded-content/persistent-`2e` architecture-v2 package. The opt-in v3
  candidate adds: (1) a fixed-width exact quartic angular feature map whose dot
  product is `(q_i . k_j)^4`, (2) two direct-summed polar-`1o` query/key axes
  while retaining exact factorization, (3) public polar-`1o` vector and
  symmetric-traceless-`2e` tensor inputs, (4) invariant RMS normalization for
  non-scalar irreps, and (5) optional activation recomputation for the gated
  local edge MLP. The quadratic factorization is compressed from `D^2` to
  `D(D+1)/2` components without changing the kernel. Defaults and existing
  checkpoints remain compatible. Focused mathematical/software contracts
  passed, but the registered seed-42/500-update QM9 screen rejected every v3
  arm. Incumbent/irrep-normalized/quartic/rank-two/combined validation MAEs
  were `0.646338/0.675463/0.673753/0.703972/0.642692 eV`. Combined improved by
  only `0.003646 eV`, below the required `0.010 eV`, and used `1.297x` peak
  allocation, above the `1.25x` ceiling. Its clipping fraction also rose from
  `0.906` to `0.950`. No v3 option is promoted and, by the frozen gate, the
  conditional v3 LBA run did not execute. Test labels were not evaluated.
- The v3 mathematical contract is full `O(3)`, translation invariance of scalar
  predictions, node permutation equivariance, edge-order invariance,
  graph-batch isolation, and no materialized `N x N` pair tensor. At fixed
  angular rank, the new global path must remain `O(N)` in node count. The
  quartic map must agree with an explicit dense kernel and expose fourth-order
  angular moments that collide under the incumbent degree-two map. External
  `1o` and `2e` inputs must transform covariantly and receive finite nonzero
  gradients.
- Training repair is part of v3 rather than an architecture-specific advantage.
  The common harness will record pathwise pre-clip norms, support a matched
  clipping policy for every compared arm, use a fixed evaluation/epoch budget
  for the primary comparison, and report both fixed-budget and best-checkpoint
  metrics. A clipping intervention is admitted only if it materially reduces
  clipping without degrading the matched validation metric; merely raising a
  threshold is reported as a policy change, not an architecture gain.
- Functional comparison will cover the incumbent moment attention, the private
  same-feature EGNN control, SE(3)-Transformer/Equiformer-style local
  equivariant attention, and published global equivariant linear/subquadratic
  attention along explicit axes: `O(3)` versus `SO(3)`, permutation contract,
  local/global receptive field, asymptotic pair materialization, accepted
  input/output irreps, coordinate updates, sparse-edge dependence, and
  train-step rather than forward-only cost. "Functionally superior" is allowed
  only for the conjunction actually implemented and verified; v3 does not
  claim arbitrary `l`, parity-complete `0o/1e/2o`, softmax equivalence,
  universal approximation, or published-model accuracy superiority.
- The confirmed real-data packet uses only cached QM9 `gap` and official
  ATOM3D-LBA ID30 train/validation data, never test labels. It first runs a
  strict seed-42/500-update QM9 ablation of quartic, rank-two, and normalization
  interventions against the current gated-plus-grouped LGL. Only a finite
  candidate improving validation MAE by at least `0.010 eV` and staying within
  `1.25x` train-step latency and peak allocation advances. The admitted
  candidate then receives a fixed-budget LBA validation comparison against the
  incumbent and private EGNN. The proposed ceiling is one local GPU, no new
  dependency or download, and at most 1,800 cumulative GPU-wall seconds; OOM,
  nonfinite values, or exhaustion is a recorded terminal result.
- Soft-normalization v2 was implemented and screened on 2026-07-24. The
  previous divisor `sqrt(sum_j f_c^2 + eps)` cancelled radial attenuation for
  a singleton edge. Normalized edge-conditioned, gated, pairwise, and
  interaction aggregations now use
  `sum_j f_c message / sqrt(1 + sum_j f_c)`; the squared-cutoff statistic is
  retained only as an explicit concentration feature where applicable. Exact
  cutoff zero-edge reduction and direct BF16 interaction-readout dtype handling
  are also repaired. Singleton radial sweeps, `d/sqrt(1+d)` scaling, finite
  coordinate gradients, affected symmetry suites, and direct CUDA BF16
  forward/backward pass.
- The strict seed-42/500-update QM9 2x2 attribution screen gave validation MAE
  `0.709287/0.737526/0.683842/0.647637 eV` for
  incumbent/grouped-only/gated-only/gated-plus-grouped. Grouped-only regressed,
  corrected gated-only improved by `0.025445 eV`, and the combined arm improved
  by `0.061650 eV` at `1.04745x` parameters. Adding grouped normalization on
  the gated path improved another `0.036205 eV`, with a favorable factorial
  interaction of `0.064444 eV`; therefore the earlier grouped-only alternative
  is rejected for this screen. The combined arm remains opt-in pending
  multi-seed confirmation. Test labels were not evaluated. Evidence is in
  `artifacts/hybrid-local-global-20260724/soft-normalization-v2/`.
- The current v2 combined path also passed the frozen train-only
  ATOM3D-LBA/PDBBind capacity check on 2026-07-24. With identical cached train
  rows 0--15, raw features, coordinates, labels, batches, and 153,029 directed
  candidates, it first reached the evaluated `0.10 pK` threshold at update
  950/25.03 s, versus 1,050/27.60 s for the preceding cutoff-squared version
  and 1,800/51.15 s for the current incumbent. The private static EGNN ended at
  `0.116225 pK` after 3,000 updates and missed the threshold. Dataset, sample,
  node/edge, candidate-initialization, and control final-state identities all
  match the historical run. V2 reduced threshold updates by `9.52%` versus v1
  with effectively unchanged median step latency, but raised peak CUDA
  allocation by `5.35%`; it was still `5.54x` slower per step and used `1.30x`
  the peak memory of EGNN. This is one-seed memorization evidence only. No
  validation or test labels were read, and the combined path remains opt-in.
  Evidence is in
  `artifacts/hybrid-local-global-20260724/soft-normalization-v2/pdbbind-overfit/`.
- The first model-feedback follow-up was implemented on 2026-07-24. Automatic
  GitHub Actions
  triggers are disabled; the CPU workflow is available only through manual
  `workflow_dispatch`, while `scripts/check.sh fast` remains the local release
  gate. That version normalized gated, normalized edge-conditioned, and
  pairwise local paths with smooth effective degree `sum_j f_c(u_ij)^2`; this
  divisor is superseded by soft-normalization v2 above. Its edge contributions
  were packed into one receiver reduction per stage. Against base commit
  `626275a`, the matched CUDA profiler reduced
  `aten::index_add` calls from 75 to 45 over three train steps; forward device
  time and peak allocation changed by `+2.3%/+3.3%`, so this is a launch-count
  reduction, not a demonstrated latency or memory win.
- Dynamic coordinates now require an explicit external-neighbor contract:
  default `error`, approximate `fixed`, or exact complete-candidate `rebuild`.
  An opt-in ligand/pocket/cross-interface readout adds parity-even products of
  learned pseudoscalars while preserving full O(3). It is zero initialized and
  initially matches ligand mean pooling exactly. On the frozen cached
  ATOM3D-LBA train rows 0--15, 1,000 updates gave final train MAE
  `0.088952 pK` for mean and `0.183435 pK` for interaction readout; median
  steps were `24.01/26.99 ms`. The interaction head is therefore retained only
  as an experimental task-specific path and is not promoted. No validation or
  test labels were read. Full parity-complete hidden irreps and a production
  Verlet/cell-list neighbor backend remain future work. Evidence is in
  `artifacts/model-feedback-followup-20260724/`.
- Same-feature gated local/global packet confirmed on 2026-07-24. Raw node
  features, coordinates, splits, targets, and matched sparse candidates remain
  identical across compared models; only the architecture may change. The two
  opt-in interventions are a width-matched gated equivariant local edge
  transport and grouped invariant-family pre-normalization before the incumbent
  update. The exact edge-free `O(N)` global channel remains unchanged. The
  frozen QM9 screen, conditional train-only ATOM3D-LBA/PDBBind capacity check,
  feature-parity rules, thresholds, no-test boundary, and 900-second local-GPU
  ceiling are recorded in
  `artifacts/hybrid-local-global-20260724/scope.md`.
- The packet completed on strict CUDA without validation/test access outside
  its frozen boundaries. On the seed-42 500-update QM9 `gap` screen, incumbent,
  gated-only, and gated-plus-grouped validation MAEs were
  `0.709287/0.749135/0.683609 eV`. Thus gated transport alone regressed, while
  the combined package improved over the incumbent by `0.025678 eV` at
  `1.04745x` parameters. Mean pre-clip norm fell by `30.61%`, satisfying the
  optimization diagnostic, but clipping remained `455/500` for both incumbent
  and selected package. The result supports the combined interaction, not an
  isolated gated-local or grouped-normalization effect.
- On the identical 16 cached ATOM3D-LBA train complexes, identical raw
  features, and identical 153,029 directed candidates, the combined package
  reached the frozen `0.10 pK` overfit threshold in 1,050 updates/27.60 s,
  versus 1,800/49.82 s for the incumbent. The near-parameter private static
  EGNN ended at `0.116225 pK` after 3,000 updates and missed the threshold.
  This is capacity/convergence evidence only: the selected package used
  `1.50x` incumbent peak CUDA allocation and its median step was `5.55x`
  slower than EGNN. It establishes neither affinity generalization nor
  small-graph systems superiority. The feature remains opt-in pending
  multi-seed QM9 and leakage-controlled affinity validation.
- Architecture-v2 packet confirmed on 2026-07-23 by the user's instruction to
  implement substantial equivariant-linear-attention improvements and validate
  them on real data. The two opt-in interventions are: (1) bounded-magnitude
  positive scalar content, which retains a learned content-norm signal while
  preserving a finite positive kernel bound, and (2) a shifted persistent-`2e`
  Frobenius kernel
  `eta_h * (1 + <Q2_ih, K2_jh>_F)`, factorized by augmenting the scalar feature
  map with `sqrt(eta_h) * [1, vec(T)]`. The latter requires persistent `2e`
  hidden channels. Neither intervention may allocate parameters or change
  outputs/state when disabled; no `N x N` pair tensor, spherical harmonics,
  parity-odd state, new dependency, or new public model family is admitted.
  Persistent-`2e` bounding evaluates the mathematically identical denominator
  `sqrt(1 + ||T||_F^2 / 5)` directly; this avoids the undefined `sqrt(0)`
  backward derivative exposed by the new multi-update smoke.
- The frozen QM9 screen reruns strict-CUDA, FP32, test-disabled, 500-update
  static LGL arms on the pinned 110k/10k random-row split: the exact incumbent,
  bounded scalar content alone, shifted `2e` kernel alone with `4x2e` hidden
  state, and their combination. A candidate is screen-safe when finite and no
  more than `0.020 eV` worse than the rerun incumbent; only a candidate at
  least `0.010 eV` better advances. The lowest-MAE admitted candidate is then
  compared with the incumbent and private static EGNN at seeds 41--45 for
  2,000 updates. Architecture promotion requires at least `0.020 eV` mean
  improvement over the incumbent, at least four improving attention pairs, and
  worst attention regression no larger than `0.020 eV`. EGNN competitiveness
  is reported separately and requires lower mean plus at least three paired
  wins; no test labels are opened.
- The independent large-graph capacity lane uses the cached immutable
  ATOM3D-LBA/PDBBind train rows 0--15 only. It compares the existing edge-free
  GGG spatial/persistent-`4x2e` incumbent, the combined architecture-v2
  candidate, and the near-parameter private static EGNN under the same
  deterministic batches, target normalization, constant AdamW settings,
  3,000-update cap, and ligand-only readout. Reaching train MAE at most
  `0.10 pK` is a wiring/capacity result, not validation or generalization
  evidence. The complete packet uses cached data, one local GPU, no dependency
  or network change, no validation/test labels, and at most 1,800 cumulative
  GPU-wall seconds. Failures and null results remain authoritative.
- Receiver-degree normalization packet confirmed on 2026-07-23. The only model
  intervention is an opt-in division of every edge-conditioned local scalar,
  vector, relative-vector, and symmetric-traceless receiver sum by the square
  root of its non-self incoming candidate count. The existing unnormalized
  sum remains the default, parameter schema and initialization must remain
  unchanged, and the option is invalid unless edge-conditioned local transport
  is enabled. Float64 tests must cover an explicit reference, zero-degree
  receivers, disabled-state compatibility, O(3), translation, permutation,
  edge-order invariance, graph isolation, finite gradients, and CLI/config
  provenance.
- The frozen screen compares strict-CUDA EC-LGL `sum` and
  `sum/sqrt(receiver_degree)` with identical source, initial state, cached QM9
  data, random-row split, precomputed 2.5-Angstrom candidates, FP32, 500
  updates, model/split seed 42, and test evaluation disabled. The candidate
  passes the diagnostic gate only if clipping fraction decreases by at least
  `0.05` absolute and validation MAE is no more than `0.020 eV` worse than the
  matched EC-LGL baseline. Mean pre-clip gradient norm, peak norm, runtime, and
  peak CUDA allocation are descriptive. One candidate smoke precedes the two
  full runs; no confirmation, default change, EGNN superiority claim, or
  accuracy promotion is authorized by this screen.
- The receiver-degree implementation passed its exact-reference,
  compatibility, O(3)/translation, permutation/edge-order, graph-isolation,
  gradient, CLI, and repository gates. The registered CUDA diagnostic failed:
  clipping changed only from `0.920` to `0.916`, below the required `0.05`
  reduction, while mean/maximum pre-clip norms increased from
  `6.154/44.101` to `6.726/53.507`. The candidate validation MAE improved
  descriptively from `0.744964` to `0.715997 eV`, satisfying the accuracy
  guard but not rescuing the failed primary gate. The option remains off by
  default; no confirmation or test evaluation ran.
- QM9 strict-CUDA repeatability packet confirmed on 2026-07-23. Before another
  accuracy architecture comparison, run five independent processes of the
  current static LGL attention control on the pinned QM9 `gap` random-row split
  with `num_samples=130000`, train/validation sizes `110000/10000`, batch size
  64, 500 updates, FP32, model/split seed 42, train-only target normalization,
  and test evaluation disabled. Each process requests strict deterministic
  PyTorch/cuDNN/cuBLAS behavior before CUDA work and records effective runtime,
  source/data/split/config/initial/final-state identities. Admission requires
  exactly equal validation MAE values and one final-state hash across all five
  processes; the registered `0.005 eV` range is additionally reported but
  cannot weaken the strict bitwise gate. A two-step strict CUDA smoke runs
  first. An unsupported deterministic operator is a terminal recorded failure,
  not permission to fall back to seeded execution. The packet uses one local
  GPU, no new download or dependency, no test labels, and at most 900 seconds
  of cumulative GPU wall time.
- The strict-CUDA packet passed on the recorded RTX PRO 6000. Five fresh
  500-update processes produced exactly the same validation MAE
  (`0.6988662063 eV`) and canonical final-state SHA-256, with zero metric span
  and no deterministic-operator failure. Test evaluation remained disabled.
  This removes same-seed runtime variance as a confounder for this exact
  source/configuration/hardware lane; it is not an accuracy or cross-hardware
  claim. Each run still clipped 456/500 updates (`91.2%`), so the next
  architecture packet should address local-message scaling and clipping before
  interpreting small validation changes.
- Full train-step scaling packet confirmed on 2026-07-23. The registered
  comparison measures eager FP32
  `zero_grad -> forward -> synthetic MSE -> backward -> AdamW.step` for the
  edge-free static/dynamic spatial-linear candidates and the private static
  EGNN control at `N={512,2048,8192}` and exact receiver degree
  `k={16,64,128}`. It uses one local GPU, five warmups, twenty synchronized
  repeats, fixed model/topology seeds, and a 1,200-second cumulative wall
  ceiling. Model/input/optimizer construction is excluded. Exact graph
  construction and host-to-device transfer are reported separately from the
  model step; absolute peak CUDA allocation includes the isolated model,
  AdamW state, inputs, activations, gradients, and EGNN edge tensor. OOM,
  nonfinite loss/gradient, or the wall ceiling is a recorded result. This is a
  synthetic systems comparison only: no dataset, validation/test label,
  accuracy, topology-preservation, or domain-generalization inference is
  authorized.
- The registered train-step grid completed all nine cells in 33.46 seconds
  without OOM or nonfinite gradients, but initially falsified the latency
  hypothesis: at `N=8192,k=128`, static spatial attention took `109.884 ms`
  versus EGNN `62.601 ms`, despite using only `0.141x` peak memory. A disclosed
  post-outcome profiler found duplicate-index `IndexBackward` dominating the
  single-graph attention path. Replacing only one-graph `summary[batch]`
  expansion with an equivalent stride-zero broadcast removed that operator
  from the diagnostic profile and reduced the static step to `25.471 ms`
  (`76.8%`). In the complete optimized rerun, static attention crossed EGNN at
  `N=8192,k=64` (`0.849x` latency, `0.274x` memory) and strengthened at
  `k=128` (`0.407x` latency, `0.138x` memory); the coordinate-updating path
  crossed only at `k=128` (`0.580x`, `0.141x`). At smaller/sparser cells EGNN
  remained substantially faster, and at `k=16` attention peak memory was
  slightly higher. Both the preregistered failure and post-outcome optimized
  rerun are retained under `artifacts/train-step-scaling-20260723/`.
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
- The user's 2026-07-24 request for a proper LBA training comparison authorizes
  a validation-only ID30 study under
  `artifacts/hybrid-local-global-20260724/lba-id30-validation/scope.md`. It uses
  all 3,507 official train complexes and 466 validation complexes, keeps the
  490-row test split structurally inadmissible, and compares the current
  gated-plus-grouped LGL, the previous LGL, and a near-parameter private static
  EGNN with identical raw inputs, sparse candidates, train-only target
  normalization, batches, optimizer, and checkpoint selection. The primary
  metric is best validation RMSE in pK; the registered candidate gate is at
  least 0.02 pK improvement over the same-harness incumbent. The published
  ATOM3D ID30 GNN RMSE is descriptive context only. The local CUDA envelope is
  capped at two GPU-hours with no network, download, test access, or default
  architecture promotion.
- The study completed in 755.0 seconds of runner wall time. Candidate,
  incumbent, and private-EGNN validation RMSEs were
  `1.550035/1.592008/1.692812 pK`, so the candidate passed the registered
  one-seed point-estimate gate by `0.041973 pK`. The paired 10,000-resample
  candidate-versus-incumbent interval was
  `[-0.130138, +0.043411] pK`, however, and every arm clipped more than 99% of
  updates. The candidate therefore remains opt-in pending multi-seed
  confirmation and optimization repair. Test remained unopened; the complete
  analysis is in `docs/LBA_ID30_VALIDATION_20260724.md`.
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

## Static-Compiled Irreps and CTP-LGL Extension

- Decision state: confirmed by the user's 2026-07-27 post-draft instruction to
  proceed. The user's architectural direction is to combine a generalizable
  irrep contract with statically compiled execution rather than build a
  runtime-dynamic irrep interpreter or replace this project with a separate
  Equiformer family.
- Phase-1 representation scope remains full-`O(3)` Cartesian
  `0e + polar-1o + symmetric-traceless-2e`. The existing parser continues to
  reject `0o`, `1e`, `2o`, and `l > 2` in this packet. A parity-complete
  `IrrepsLayout` and optional general-CG/eSCN backend are later model-class
  decisions, admitted only if the phase-1 tensor-product path is useful on
  real data. No `e3nn`, spherical-harmonic, or other dependency is added.
- "Static compiled" means that the enabled tensor-product paths, input/output
  degrees, parities, multiplicities, and tensor storage slices are resolved
  once in `__init__`. Forward execution contains no input-dependent irrep
  discovery or architecture routing. The internal path description may be
  reusable by a future general irrep parser, but phase 1 implements only the
  three paths below with native Cartesian operations.
- The public model remains `EquivariantAttention`. The new opt-in switch is
  `use_cartesian_tensor_product_local_transport=False`. It is valid only with
  gated local transport and at least one persistent `2e` hidden channel.
  Disabled construction must retain the incumbent parameters, outputs, state
  dictionary, RNG consumption, and checkpoints exactly. The current defaults
  therefore remain `hidden_tensor_dim=0` and CTP off.
- The first LBA candidate uses
  `64x0e + 4x1o + 4x2e`, four heads, the existing `(4,0,4)` LGL route,
  6-Angstrom local cutoff, static coordinates, squared-distance RBFs, gated
  local transport, and grouped invariant normalization. The global kernel,
  global moment summaries, readout, feature schema, topology, optimizer, and
  training policy are held fixed.
- The admitted efficient execution uses `use_static_tensor_carrier=True` with
  `C2 == num_heads`. It stores the compact `2e` state directly in head layout,
  updates and consumes it only in local stages, carries it unchanged across
  the middle global stage, and reuses the incumbent moment tensor as its
  bounded residual update. The generic persistent-tensor FFN/mixing path
  remains available when this opt-in is false. Full CTP is statically scheduled
  only in the final local refinement (`cartesian_tensor_product_local_layers =
  (2,)`); the first local stage seeds the carrier with the incumbent tensor
  message. The CTP gate reuses the incumbent edge latent and adds only the five
  tensor invariants before its separate zero-initialized output projection.
- Persistent tensor state is projected from its `C2` channels to the active
  local heads once per node before sender/receiver edge indexing. With
  `d_ij=(r_j-r_i)/R_c` and
  `Q_ij=ST(d_ij outer d_ij)`, the separate CTP gate branch receives the
  incumbent invariant edge features plus

  ```text
  chi_ij = [
      <T_i,T_j>_F,
      <T_i,Q_ij>_F,
      <T_j,Q_ij>_F,
      ||T_i||_F^2,
      ||T_j||_F^2,
  ].
  ```

  The five terms are parity-even invariants. They do not directly become
  equivariant values or Cartesian learned parameters.
- The statically admitted equivariant bases are exactly

  ```text
  2e x 1o -> 1o:  T_j d_ij
  2e x 0e -> 2e:  T_j
  1o x 1o -> 2e:  ST(v_j outer d_ij)
  ```

  The CTP branch emits invariant bounded gates for these bases and adds them
  to the incumbent vector/tensor local messages. It uses the incumbent
  cutoff and soft receiver normalization
  `sum_j f_c message / sqrt(1 + sum_j f_c)`. A separate final gate projection
  is zero initialized, so the enabled CTP model initially equals its
  persistent-`2e`-only control while the new projection receives gradient.
  The incumbent local MLP is not widened by the five tensor invariants.
- The first packet does not add tensor-conditioned scalar content,
  cross-product/Levi-Civita paths, parity-odd state, higher `l`, dynamic
  coordinates, a tensor term in the global attention kernel, or new
  ligand/pocket features. These are separate ablations rather than hidden
  components of the CTP claim.
- For `R in O(3)`, the contract is
  `v,d -> Rv,Rd`, `T,Q -> R T R^T,R Q R^T`. Consequently
  `<T,Q>_F` is `0e`, `Td` is `1o`, and `ST(v outer d)` is `2e`.
  Relative coordinates preserve translation behavior; shared edge functions
  and receiver sums preserve node permutation and edge-order behavior; the
  existing same-graph validation preserves batch isolation. Reflections are
  required, not only proper rotations.
- At fixed channels and paths, node projections cost `O(N)`, local
  contractions and aggregation cost `O(E)`, and the global route remains
  fixed-rank `O(N)`. The packet may not materialize an `N x N` tensor or
  enumerate neighbor triplets. Candidate parameters must be at most `1.10x`,
  synchronized real-LBA train-step latency at most `1.25x`, and peak CUDA
  allocation at most `1.25x` the current gated-plus-grouped LBA candidate.
- The implementation is rejected before training unless all of the following
  hold: the `{+x,-x}` versus `{+y,-y}` quadrupole witness collides under the
  incumbent and separates under CTP; changing only sender `T_j` changes an
  admitted message; new parameters receive finite nonzero gradients; and
  float64 rotation, reflection, translation, node permutation, edge-order,
  batch-isolation, cutoff, and coordinate-gradient contracts pass.
- Real-data attribution uses three arms with identical raw features and
  training information: the current `64x0e+4x1o` candidate,
  `64x0e+4x1o+4x2e` with persistent tensors but CTP off, and the same
  persistent-tensor model with CTP on. QM9 is only a bounded regression smoke.
  The deciding task is validation-only ATOM3D-LBA ID30; test remains
  structurally inaccessible. Before a cross-run or multi-seed claim, the
  recorded one-edge topology reproducibility defect must be repaired or every
  arm must consume one serialized, hashed candidate list.
- After the implementation/resource gates, a fixed seed-44 screen may advance
  only if CTP improves validation RMSE over both controls without a regression
  larger than `0.05 pK`. Confirmation uses seeds 41--43, fixed 35 epochs and
  identical batches. Architectural promotion requires mean improvement of at
  least `0.020 pK` over the current candidate, positive mean improvement over
  the persistent-`2e`-only control, at least two of three paired wins, worst
  paired regression no larger than `0.050 pK`, and the resource ceilings
  above. These thresholds are fixed before outcome inspection.
- The conditional seeds 41--43 confirmation keeps 35 epochs and the frozen
  optimizer/schedule but uses batch size 24 for every arm so the full
  three-arm comparison remains inside the approved compute envelope. One
  process materializes and hashes the topology once; all nine arm/seed runs
  consume that same in-memory candidate list.
- The approved execution envelope is the existing locked environment and
  cached QM9/LBA data, no network or dependency changes, one local GPU,
  validation only, and at most 3,600 cumulative GPU-wall seconds. Run
  artifacts remain under ignored
  `artifacts/`; results, including rejection or null results, are appended to
  `docs/EXPERIMENTS.jsonl`. Automatic GitHub Actions remain disabled.

## Deterministic Sparse Topology Contract

- Decision state: implemented and verified on 2026-07-27 as the repair the
  preceding LBA packets recorded as a precondition for any cross-run or
  multi-seed claim. It is a data-contract repair, not an architecture change.
- A cached candidate list retains `i <- j` exactly when the float64 squared
  displacement is below the squared cutoff. The float64 promotion is mandatory
  whatever the storage dtype, so the topology cannot depend on model precision,
  coordinate magnitude, translation, node permutation, or BLAS thread count. No
  matrix-multiplication Euclidean distance may decide retention.
- Self edges are always present. Each relation keeps the `k` smallest squared
  distances and retains every exact tie at the boundary, so a receiver degree may
  exceed its budget; that is required for permutation equivariance. Candidate
  order is receiver major, then self, intra-segment, cross-segment, then
  ascending sender index.
- `equivariant_attention.pdbbind.topology_sha256` is the only admitted candidate
  identity. Runners delegate to it rather than hashing edges themselves, and
  `scripts/verify_lba_topology.py` must report one edge count and one hash across
  fresh processes before a multi-seed LBA claim.
- The frozen official ID30 identity at revision
  `f93dd2d150a47c270f624620f84e07451a158705`, `R_c = 6.0 Angstrom`, and
  `intra_k = cross_k = 16` is `32,302,952` edges and
  `57f40fb157e6416558db5507d95c3a5e4f828881e0bc92e142e1b85de802dc6c`. Legacy
  hashes remain valid only as records of past runs.
- New LBA numbers are not bit-comparable with pre-repair numbers. Any
  cross-packet comparison must rerun both arms under this contract.

## Registered LBA Clipping Confirmation

- Decision state: preregistered and implemented, **awaiting compute approval**.
  No GPU run is authorized by this section alone.
- Motivation: the frozen seed-44 screen improved last-epoch validation RMSE from
  `1.628645` to `1.600802 pK` by removing global gradient clipping, above its
  registered `0.020 pK` threshold, at `1.0101x` latency and `0.9988x` peak
  allocation. It is one-seed evidence and ran on the drifting candidate list, so
  the public default remains `grad_clip=1.0`.
- The runner is `scripts/run_lba_clipping_confirmation.py`. It compares clip 1
  against no clipping at model/order seeds 41--43 in one process, on one shared
  and hashed topology, with the current squared-RBF gated-plus-grouped LGL,
  matched 20 epochs, batch 16, AdamW `3e-4`, weight decay `0.01`, five warmup
  epochs, cosine decay, strict deterministic FP32, and no test path. It aborts
  before training when `--expect-topology` does not match.
- The frozen promotion rule, fixed before any outcome is inspected: mean paired
  last-epoch improvement at least `0.020 pK`, at least two of three paired wins,
  worst paired regression at most `0.050 pK`, median step-latency and
  peak-allocation ratios at most `1.05`, identical update counts and initial
  state hashes within each pair, and every arm finite and completed. Only that
  conjunction authorizes changing the public `grad_clip` default.
- Registering the protocol is not evidence. A failed gate leaves the default at
  `1.0` and is recorded in `docs/EXPERIMENTS.jsonl` as an authoritative null.

## Sparse Geometry-Aware O(3)/SE(3) Local Attention

- Decision state: confirmed by the user's 2026-07-27 instruction after review
  of the supplied EquiFlex architecture deck. The useful transfer is the order
  of computation--pair-conditioned transport first creates geometric
  `1o/2e` moments, then the same local block uses them in a refined score--rather
  than EquiFlex's dense pair tensor, triangle update, or quadratic global
  attention.
- The public class remains `EquivariantAttention`. Three appended,
  backward-compatible configuration controls are admitted:
  `symmetry_group="O3"`,
  `use_geometry_aware_local_attention=False`, and
  `use_se3_axial_tensor_product=False`, with the optional static schedule
  `geometry_aware_local_layers=None`. The geometry-aware path requires gated
  local transport. `None` resolves to every local stage; the initial LBA
  resource/accuracy screen uses only the first local stage `(0,)` so its
  geometry reaches the middle global stage without paying for a duplicate
  final-stage refinement. The axial path additionally requires the
  geometry-aware path and `symmetry_group="SE3"`. Disabled construction must
  preserve the incumbent state schema, outputs, and common seeded parameters.
- For each nonself retained edge `i <- j`, the incumbent factorized edge MLP
  supplies a hidden pair latent `h_ijh`. Its existing pair-conditioned,
  cutoff-weighted and degree-normalized aggregation supplies the bootstrap
  states without a second edge MLP or an extra softmax:

  ```text
  v_boot_ih = m_vector_ih + m_relative_ih,
  T_boot_ih = m_tensor_ih.
  ```

  Empty neighborhoods produce exact zero moments. Self edges remain excluded,
  and no dense pair state is created.
- Channel gates of the invariant node state form bounded query/key versions of
  the current-plus-bootstrap polar vectors and, when present, the
  persistent-plus-bootstrap symmetric-traceless tensors. The refined sparse
  score is

  ```text
  ell_ijh =
      a0_h b_pair(h_ijh)
    + a1_h <q1_ih,k1_jh>
    + a2_h <q2_ih,k2_jh>_F,
  alpha_geom_ijh =
      softmax_j(softclip(ell_ijh) + log f_c(u_ij)).
  ```

  The three score lanes are invariant under full `O(3)`. Geometry-attended
  scalar, sender-vector, relative-vector, and symmetric-traceless values are
  added as a small learned residual to the incumbent gated local messages.
  This creates and consumes `2e` inside one local block without requiring a
  persistent hidden tensor.
- In `O3` mode all vector values are polar and every tensor value is
  reflection-even. In optional `SE3` mode, the extra value basis is

  ```text
  a_ijh = vee(T_boot_jh Q_ij - Q_ij T_boot_jh),
  Q_ij = ST(d_ij outer d_ij).
  ```

  This is the `l=1` component of `2 x 2`: it transforms as a vector for proper
  rotations but as an axial `1e` object under reflection. The implementation
  may mix it into the vector carrier only when the declared contract is
  `SE3`. It must not be called polar `1o` under the `O3` contract. Proper
  rotations, translations, node permutations, edge order, and batch
  isolation remain required in both modes; reflection covariance is required
  only in `O3` mode.
- Q/K normalization is implemented with the incumbent bounded vector and true
  symmetric-traceless Frobenius maps. `softclip(x)=L tanh(x/L)` has a fixed,
  documented finite `L`; receiver softmax is evaluated stably on sparse edges.
  No reflected structure is used as generic augmentation for chiral
  biomolecules.
- At fixed channels, the incumbent aggregation plus refined sparse attention
  cost `O(E)` and store only
  fixed-width edge/node intermediates. The existing global factorized moment
  route remains exactly `O(N)` and unchanged. Dense `p_ij`, dense triangle
  updates, pairwise global distance bias, flow matching, physics losses, and
  inference guidance are non-goals of this packet.
- RED verification must cover disabled compatibility, receiver normalization,
  an equal-distance angular witness, finite nonzero new-path gradients,
  float64 full-`O(3)` behavior for the common path, float64 proper-rotation
  behavior plus a reflection-separation witness for the axial path,
  translation, permutation, edge order, batch isolation, and construction
  errors. The public builder and cached LBA runner must expose the new options
  without changing existing defaults.
- The smallest real-data check is validation-only cached ATOM3D-LBA ID30 with
  identical features, topology, seed, batches, optimizer, and update budget
  for current, geometry-`O3`, and geometry-`SE3` arms. It is a rejection
  screen, not a test or superiority result. Record parameters, synchronized
  train-step latency, peak allocation, training/validation metric, and whether
  the axial parameters receive gradient. No held-out test label is accessed.

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
  applicable block. An external `edge_index` is rejected by default. The
  explicit `fixed` policy re-filters the same candidates at each local stage,
  so omitted edges cannot enter later; `rebuild` ignores the supplied topology
  and reconstructs complete same-graph candidates so cutoff-crossing pairs can
  enter. The exact fallback is quadratic without a production neighbor backend.
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
