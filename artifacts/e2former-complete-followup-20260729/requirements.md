# Shared-feedback completion matrix

This file freezes the consolidated scope of the public ChatGPT share supplied
by the user, the user's subsequent in-thread directions, and the repository
audit performed while implementing them:

- initial design-feedback URL:
  `https://chatgpt.com/share/6a68ca40-9730-83ee-8c49-1a122068d2af?ogimg=plain`
- cached response SHA-256:
  `ebc484a186dc2c850a126c22f321c12dcaa9f7c08589d4086cebe87b76683c5f`
- completion-scope URL recorded by the native Codex Science goal:
  `https://chatgpt.com/share/6a698a40-3598-83ee-a972-bd2b8298c4d0`
- recorded retrieved-HTML SHA-256:
  `86a30c552c7c689e9dc7138769cab28d21db3292d869520bfa8627ec9179af59`
- extraction date: 2026-07-29

The shares are design/scope inputs, not empirical evidence. Rows below are a
requirement-by-requirement synthesis, not a claim that every row is a verbatim
statement from either share. Follow-up user directions and defects found
by source/test review are included so the completion boundary is auditable.

Status vocabulary:

- `proved`: current code and named evidence already establish the requirement.
- `partial`: a useful implementation exists, but one or more named contracts are
  missing.
- `todo`: no sufficient implementation/evidence exists yet.
- `experiment`: executable support may exist, but the scientific claim requires a
  preregistered matched experiment.
- `excluded`: the shared feedback explicitly rejects this interpretation; preserving
  the exclusion is part of completion.

The matrix is intentionally stricter than `PROJECT.md`: absence of a known defect is
not proof. A row becomes `proved` only when its listed evidence exists.

Audit snapshot (2026-07-29, dirty worktree): statuses below describe the
implemented reference/PyTorch paths and named tests, not a merged release.
`proved` is a software-contract statement only. It does not imply a CUDA
speed/memory gain, downstream accuracy, a production neighbor builder, or a
complete SBDD pipeline. Rows left `partial`, `todo`, or `experiment` name the
remaining evidence gap explicitly.

## A. Architectural identity and public boundary

| ID | Requirement | Current status | Completion evidence |
|---|---|---:|---|
| A01 | Keep one public `EquivariantAttention` model; operator choices are config, not competing public classes. | proved | public exports and API test |
| A02 | Preserve exact finite-feature global transport, augmented numerator/denominator accumulation, and no `N x N` hot-path tensor. | proved | dense/factorized forward+gradient tests and allocation-contract test |
| A03 | State complexity honestly: global branch fixed-width `O(N)`; sparse hybrid `O(N+E)`; neighbor discovery separately costed. | proved | `README.md`, `docs/LAYER_MATH.md`, provider capability receipts, and missing-neighbor/complete-fallback tests explicitly separate model execution from discovery |
| A04 | Use homogeneous blocks: every block keeps global heads and selected blocks add low-rank sparse residuals. | proved | zero-init equivalence and stride/schedule tests |
| A05 | Support every-block, periodic refresh, and local-stem/global-trunk schedules without persistent edge hidden state. | proved | `residual_layers=None`, deterministic stride compilation, explicit layer tuples, state-schema tests, and the real-LBA smoke receipt with local stem `(0,)` |
| A06 | Preserve legacy LGL implementation/config/checkpoint loading as a baseline, but not as the vNext public abstraction. | partial | legacy route and exact structured config round-trip remain; an explicit old-checkpoint strict-load fixture through the structured builder is still missing |
| A07 | Keep Cartesian `0e/1o/2e` as the optimized default backend; do not relabel metadata-only planning as arbitrary-`l` numerical execution. | proved | API boundary tests and docs |
| A08 | Default symmetry is O(3); SE(3)/chiral axial paths are explicit opt-ins with reflection contract stated. | partial | O(3) improper-reflection suites and SE(3) reference-path/geometry tests exist; a single profile-parametrized end-to-end chiral-model suite is still missing |
| A09 | Decouple persistent irrep multiplicity from attention-head multiplicity through projections. | proved | channel projections map carrier copies to/from heads; generic/high-order execution tests include unequal multiplicities, and the real-LBA smoke runs `6x1o + 2x2e` with four heads (specialized legacy lanes retain explicit narrower guards) |
| A10 | Public config exposes width, depth, input/output irreps, symmetry/profile, angular bandwidth, local stride/rank; raw TP instruction lists remain expert-only. | proved | `tests/test_structured_config.py` profile snapshots, strict JSON, invalid matrix, stride compilation, and expert-only rejection |

## B. Sparse residual mathematics

| ID | Requirement | Current status | Completion evidence |
|---|---|---:|---|
| B01 | Cutoff may not cancel under receiver normalization for singleton or low degree. Use `numerator/(1+mass)` or equivalent mass damping. | proved | `tests/test_sparse_low_rank_residual.py`, sparse-v2 receipt |
| B02 | Smooth cutoff behavior at `Rc-eps`, `Rc`, and `Rc+eps` for output and first coordinate derivative. | proved | float64 boundary/gradient test |
| B03 | When one of two edges crosses cutoff, the remaining contribution must not be renormalized into a discontinuous full-strength message. | proved | two-edge crossing regression |
| B04 | Fuse receiver statistics for mass, squared mass, and message families in one logical reduction. | proved | one-call receiver-sum contract |
| B05 | Expose `log1p(mass)` and `log1p(mass_square)` as invariant scalar update features. | proved | degree witness and zero-init projection |
| B06 | Replace contrast-limited `sigmoid(tanh(score))` with a positive, bounded-temperature score map. | proved | monotonicity/contrast/finiteness test |
| B07 | Score includes multiplicative scalar query/key content plus radial bias, not only an additive weak gate. | proved | source equation and gradient suite |
| B08 | Score may include invariant vector inner product and receiver/sender edge-projection product using safe bounded displacement. | proved | active O(3)/reflection/coincident tests |
| B09 | Value families receive independent learned radial modulation from one compact RBF projection. | proved | family isolation and gradient test |
| B10 | Values cover sender scalar, sender polar vector, relative displacement, and `ST(d)`; persistent `2e` transport remains opt-in. | proved | sparse-residual dense/reference, family-isolation, tensor-transport, and symmetry tests; persistent tensor defaults remain off |
| B11 | Local normalization supports positive mass-damped and receiver softmax as explicit local-only ablations; never alter exact global factorization. | proved | explicit singleton reference tests for both modes |
| B12 | `sparse_residual_balancing="receiver"` is explicit; conflicting legacy local-balancing flags error instead of becoming silent no-ops. | proved | config conflict tests for true/false |
| B13 | Zero/small initialization exactly recovers the all-global incumbent while allowing the local lane to wake up. | proved | forward equivalence and first/second-step gradient tests |
| B14 | Score, cutoff, and normalization remain invariant; scalar/vector/tensor outputs obey O(3), translation, permutation, edge-order, and graph-isolation contracts. | proved | consolidated symmetry suite |
| B15 | Isolated nodes, empty edge sets, coincident coordinates, and finite parameter/coordinate gradients are supported. | proved | degree-zero/coincident/boundary/gradient suite |

## C. Neighbor and packed sparse execution

| ID | Requirement | Current status | Completion evidence |
|---|---|---:|---|
| C01 | `PackedNeighborGraph.to(same_device)` returns self; cross-device transfer uses a trusted constructor and does not rerun semantic validation. | proved | `tests/test_packed_neighbors.py::test_to_preserves_index_dtype_and_round_trip_contract` and `test_cross_device_to_uses_trusted_constructor_without_revalidation` |
| C02 | Split receiver CSR and reverse CSR construction; sparse forward does not build reverse metadata unless backward/key balancing needs it. | proved | independent-builder and receiver-only geometry construction tests in `tests/test_packed_neighbors.py` |
| C03 | Preserve compact int32 indices when safe and explicit int64 overflow fallback. | proved | packed-neighbor boundary tests |
| C04 | Add static degree, max-degree, histogram/skew, and degree-bucket metadata. | proved | exact metadata and `8/16/32/64/128` boundary tests in `tests/test_packed_neighbors.py` |
| C05 | Add optional ELL neighbor/degrees layout and lossless COO↔CSR↔ELL round trip with stable receiver order. | proved | ELL round-trip, zero-degree, and excessive-padding guard tests |
| C06 | Receiver-owned reduction avoids materializing an expanded receiver `E`-vector or unconditional int64 cast. | proved | receiver-CSR reduction monkeypatch/operator contract and int32 geometry-gradient equivalence tests |
| C07 | Reverse sender plan gives deterministic sender-side reduction without atomic scatter. | proved | reverse CSR sender-reduction forward/gradient and edge-permutation tests |
| C08 | Provide `NeighborProvider` with precomputed, deterministic reference-radius, and external-adapter implementations. | proved | provider protocol/equivalence, canonical external ordering, and model integration tests in `tests/test_neighbor_providers.py` |
| C09 | No supplied neighbors under sparse residual obeys explicit `require` or `complete_fallback`; production default cannot silently build `O(N^2)` edges. | proved | missing-neighbor error, equivalence, and pre-allocation size guard |
| C10 | Force/moving-coordinate policy distinguishes fixed graph, radius+skin/Verlet, and disallowed hard learned top-k. | proved | fixed/rebuild model policy tests, Verlet crossing/rebuild/invalidate tests, and capability rejection for learned top-k |
| C11 | Relation IDs are immutable scalar metadata, not persistent edge hidden state; reverse relation semantics are explicit. | proved | `RelationTable` involution plus packed round-trip/reverse-view/model numerical tests in `tests/test_typed_relations.py` |
| C12 | Distance-band execution is optional additive residual (near/medium/global), never claimed as an exact partition of unity. | proved | overlap/non-partition, zero-init, gradient, and one-edge-list tests in `tests/test_distance_bands.py`; docs state additive semantics |
| C13 | Production cell-list, Verlet rebuild, and PBC are separated from the reference provider and are not claimed before implemented/profiled. | proved | provider capability receipts mark reference/Verlet non-production and PBC/cell-list false; production cell-list/PBC remain explicit gaps |

## D. Geometry and activation policy

| ID | Requirement | Current status | Completion evidence |
|---|---|---:|---|
| D01 | Compare squared distance to squared cutoff; avoid unnecessary `sqrt` for edge admission. | proved | squared-admission source guard, exact boundary, and float32-overflow tests in `tests/test_packed_neighbors.py` |
| D02 | RBF centers/widths and repeated triangular indices are registered/cached buffers. | partial | model RBF parameters and quartic indices/coefficients are buffers; a consolidated state/device/dtype migration test for every repeated index family is still missing |
| D03 | Geometry activation modes `full`, `compact`, `recompute`, and deterministic `auto` exist. | proved | storage policy plus bitwise/FP64 forward, input, parameter, coordinate-gradient and gradgrad tests in `tests/test_geometry_cache_modes.py` |
| D04 | Geometry caches are invalidated for coordinate updates; multiple local refreshes cannot consume stale geometry. | proved | dynamic rebuild crossing and per-layer geometry call tests in `tests/test_coordinate_updates.py`/provider integration |
| D05 | Higher-order/force-loss or `create_graph=True` automatically uses a PyTorch/recompute fallback when a custom kernel lacks double backward. | partial | streamed selector and execution receipt record gradgrad fallback; model/force runner does not yet infer `create_graph=True` automatically for a future custom backend |
| D06 | Mixed precision keeps coordinates, sensitive geometry, denominators, reductions, tensor contractions, and coordinate gradients in FP32; BF16 projections are optional. | partial | BF16 streamed, cache, reference-irrep and transient-l3 forward/backward tests pass; no end-to-end CUDA autocast matrix covers every global/tensor/force combination |

## E. Exact global execution

| ID | Requirement | Current status | Completion evidence |
|---|---|---:|---|
| E01 | `feature_gemm` matches outer/scatter for all scalar, vector, tensor, quartic, adaptive-spatial, distinct-QK, and balancing combinations. | proved | FP64 forward/gradient parameter matrix and wide registered payload tests in `tests/test_global_feature_gemm.py` |
| E02 | Add deterministic `global_reduction_backend="auto"`. | proved | flat/structured config validation, deterministic selector tests, and full numerical dispatch tests |
| E03 | Auto dispatch distinguishes single GEMM, similar-size padded/bucketed BMM, ragged grouped GEMM, and extreme-ragged fallback. | proved | `tests/test_graph_layout.py` structure matrix and `tests/test_execution_metadata.py` effective-lane receipt |
| E04 | Zero-pad feature widths to hardware-friendly multiples without changing the exact kernel. | proved | exact width-policy tests and FP64 forward/gradient padded/unpadded equality |
| E05 | Cache/precompute graph feature layout in batch metadata; no CUDA `.item()`, `tolist()`, or repeated sort/host synchronization per forward. | proved | `GraphBatch.graph_layout` collation plus cached-forward monkeypatch forbidding sort/metadata/`.item()`/`.tolist()`; layout construction is still caller cost when omitted |
| E06 | Persistent global `2e` value transport supports none/value/kernel/both ablation and remains opt-in until utility evidence. | partial | value transport and tensor-kernel switches compose and mechanism tests pass; no registered four-arm none/value/kernel/both matched utility result exists |
| E07 | Whitened global read remains opt-in/rejected unless new matched evidence overturns its multi-seed result. | proved | current experiment ledger |

## F. Streamed/local kernel system work

| ID | Requirement | Current status | Completion evidence |
|---|---|---:|---|
| F01 | Receiver-centric streamed positive forward keeps only accumulators and does not write normalized `[E,H]` weights or `E×H×D` messages to global memory. | partial | PyTorch CSR/ELL row references match materialized forward/gradients and have a row-local saved-intermediate audit; no fused custom forward or device-memory trace exists |
| F02 | Kernel shares geometry across heads and loads receiver query/state once per receiver. | todo | no custom kernel/source or profiler counter artifact; PyTorch reference alone does not prove load behavior |
| F03 | Accumulators are FP32 and block sizes are measured for the project’s actual `H≈4,D≈16`, not copied from 128-head E2Former settings. | partial | low-precision reference accumulation is FP32; no named target-GPU block-size tuning artifact exists |
| F04 | Reverse-CSR backward separates receiver and sender contributions and supports balancing-off one pass / balancing-on two passes. | todo | deterministic reverse-CSR reductions exist, but there is no custom separated backward implementation/profile |
| F05 | Optional receiver-wise online softmax streams max/sum/value; it is a local `O(E)` ablation, not the project’s global linear-attention identity. | proved | streamed softmax cutoff/zero-degree/reference tests and docs explicitly label it local-only |
| F06 | PyTorch/CSR/ELL/custom paths have deterministic auto dispatch with safe small-degree fallback. | partial | deterministic capability/degree selector covers materialized/CSR/ELL and custom fallback receipts; custom path and registered `<10%` low-degree performance gate are absent |
| F07 | Gated edge-MLP fusion is profiler-gated and not attempted before lightweight streamed paths establish the bottleneck. | excluded | design/docs audit |
| F08 | EAAS/Wigner-6j/Triton E2Former backend, if added, is an optional backend with its own semantics; current moment attention is never renamed E2Former. | excluded | no EAAS/Triton backend is shipped; README/project docs call this an E2Former-informed systems packet, not E2Former |
| F09 | Any stochastic local-frame gauge must pass repeat/rotation/translation/force-gradient-noise tests or be replaced with deterministic framing. | excluded | no stochastic local frame or EAAS backend exists; adding one reopens this gate |
| F10 | Paper-vs-release cutoff and activation-materialization semantics are recorded rather than conflated. | todo | no EAAS backend receipt currently exists, so there is no implemented paper/release semantic comparison to record |

## G. Irreps, parity, and transient high order

| ID | Requirement | Current status | Completion evidence |
|---|---|---:|---|
| G01 | `IrrepLayout` handles arbitrary nonnegative degree, parity, multiplicity, deterministic packing, and offsets. | proved | current planner tests |
| G02 | Replace/expand the current plan signature with executable TP instructions: multiplicities, connection mode, sharing, CG convention/normalization, offsets, output multiplicity, dtype/device/order. | proved | deterministic `ExecutableTensorProductPlan` schema/round-trip/validation tests in `tests/test_tensor_product_executor.py` |
| G03 | Compile reachable paths using triangle and parity rules, prune unreachable/duplicate paths, and preserve deterministic order. | proved | planner/compiler reachable-path order plus duplicate/selection validation in irrep and executable-plan tests |
| G04 | Implement same-irrep channel mixing, scalar-gated nonlinearities, and irrep-wise RMS normalization. | proved | numerical, shape, parameter-count and O(3)/SE(3) equivariance tests in `tests/test_reference_irreps.py` |
| G05 | Freeze a real spherical-harmonic and Clebsch–Gordan convention with cached coefficients. | proved | real-tesseral golden values, cache, orthogonality/completeness, swap, addition/product identities through `l=4` |
| G06 | Optimized Cartesian `l<=2` agrees with the generic reference in forward and gradients. | proved | `tests/test_tensor_product_executor.py::test_generic_low_degree_paths_match_cartesian_forward_and_gradients` plus Cartesian bridge tests |
| G07 | Implement a bounded transient `l=3` workspace: lift carrier → reachable TP → project back; do not persist `l=3` across global layers. | proved | aggregate-before-project witness, full O(3)/translation/permutation, gradgrad, BF16, live profile, and state-lifetime tests in `tests/test_transient_l3.py` |
| G08 | Profiles: `minimal` (`0e+1o`, transient `2e`), `standard` (`0e+1o+2e`), `chiral`, and `high_order` (low-l carrier, transient `l=3`). | proved | frozen profile snapshots/construction in `tests/test_structured_config.py`; high-order model execution in `tests/test_transient_l3.py` |
| G09 | O(3) tests proper/improper transforms with parity; SE(3)/chiral tests proper transforms without falsely requiring reflection. | partial | O(3) reference/high-order tests include improper transforms and SE(3) parity-mixed path tests require only proper rotations; one end-to-end profile-parametrized chiral suite is missing |
| G10 | BF16 numerical support and required gradgrad fallback are explicit per backend. | partial | reference irrep/transient/streamed BF16 and gradgrad tests plus capability receipts exist; no complete optimized/custom backend dtype-capability matrix exists |
| G11 | Persistent `2e` and higher persistent irreps remain opt-in research arms until matched evidence supports promotion. | proved | defaults and docs |

## H. Structured configuration and compatibility

| ID | Requirement | Current status | Completion evidence |
|---|---|---:|---|
| H01 | Introduce structured `ArchitectureConfig`, `RepresentationConfig`, `GlobalTransportConfig`, `LocalResidualConfig`, and `NeighborConfig`. | proved | immutable dataclasses, strict versioned JSON, duplicate/unknown rejection and builder tests in `tests/test_structured_config.py` |
| H02 | Preserve legacy flat config loading and strict checkpoint compatibility. | partial | every flat field round-trips exactly and the structured builder emits the same legacy config; an archived checkpoint strict-load migration fixture is still missing |
| H03 | Validate depth/rank/stride/profile/symmetry/cache/backend combinations early. | proved | invalid profile/depth/rank/stride/backend/cache/expert matrix and cross-group `replace` tests |
| H04 | Record requested and effective backends, neighbor policy, cache mode, symmetry, and fallback decisions in run metadata. | proved | immutable receipt resolver/JSON/fallback tests plus `scripts/smoke_generic_3d_real_data.py` and three persisted real-LBA receipts |
| H05 | New options disabled preserve parameters, RNG state, state schema, and outputs as required by their compatibility class. | partial | sparse residual and transient-l3 disabled-state tests cover RNG/schema/output; no single compatibility matrix spans every new role/relation/band/cache/backend option |

## I. Generic core / SBDD adapter boundary

| ID | Requirement | Current status | Completion evidence |
|---|---|---:|---|
| I01 | Generic core accepts configurable scalar node features, integer role/relation IDs, masks, and hierarchy assignment without biological vocabulary. | proved | role model tests, typed relations, `GraphSample`/`GraphBatch` collation/move/hierarchy tests, and generic primitive API tests |
| I02 | Biological atom/residue/ligand/water/metal/cofactor vocabularies live only in an SBDD adapter; the core readout uses generic selected/context masks. | proved | SBDD vocabularies stay under `equivariant_attention.sbdd` and are not top-level-exported; `tests/test_sbdd_features.py`, vocabulary-isolation tests, and `tests/test_readout_mask.py::test_legacy_interaction_alias_matches_generic_bipartite_readout` |
| I03 | Typed relation radial bias/cutoff uses one sparse list and scalar relation metadata, not relation-specific equivariant kernels or large edge state. | proved | one-list packed/COO numerical and relation-cutoff/isolation tests in `tests/test_typed_relations.py` |
| I04 | Adapter supports atom features: atomic number, charge, donor/acceptor, aromaticity, hybridization, role; protein residue/atom/backbone fields; ligand bond/stereo/protonation fields. | proved | deterministic schema/validation/bond tests in `tests/test_sbdd_features.py` |
| I05 | Generic multiscale API supports atom-level and coarse/residue graphs; the adapter constructs atom→coarse hierarchy IDs while relation/domain cutoffs remain explicit model configuration. | proved | rich SBDD hierarchy and preencoded hierarchy bridge tests in `tests/test_sbdd_adapter.py`, generic hierarchy pool/broadcast tests, and explicit relation-cutoff tests in `tests/test_typed_relations.py` |
| I06 | Generic masked invariant pooling returns interface and global representations separately; the adapter constructs an interface mask from typed cross-role edges. | proved | generic mask/empty/isolation tests plus SBDD interface-mask construction, permutation identity, and annotation-collation tests in `tests/test_sbdd_adapter.py` |
| I07 | Generic `l=1` vector/coordinate head supports update masks; adapter chooses ligand/flexible-side-chain nodes. | partial | generic equivariance/exact-mask tests and SBDD `PoseRefinementRequest.update_mask` exist; no raw-structure flexible-side-chain selection policy exists |
| I08 | Generic scalar energy plus conservative force primitive and optional direct vector-force primitive are separate and labeled honestly. | proved | finite-difference, O(3), mask, graph-isolation, and gradgrad tests in `tests/test_generic_3d_primitives.py` |

## J. SBDD heads, losses, data contracts, and metrics

| ID | Requirement | Current status | Completion evidence |
|---|---|---:|---|
| J01 | Affinity head separates interface/global representation, same-bound-geometry interaction residual, and optional strain; docking score is not an affinity label. | proved | affinity component tests plus schema rejection of docking-score-as-affinity in `tests/test_sbdd_losses_heads.py`/`test_sbdd_schema.py` |
| J02 | Pose ranking supports native/near-native/wrong-pose groups, within-pair listwise or RMSD-aware loss, and clash/strain/contact auxiliaries. | proved | grouped isolation, both ranking losses, and auxiliary component tests |
| J03 | Pose refinement returns ligand-atom equivariant displacement/velocity/score with explicit protein/flexible-mask contract and optional time conditioning. | proved | output-kind API and O(3)/mask/time tests for `PoseRefinementHead` |
| J04 | Virtual screening is a separate ranking/enrichment task with inactive/non-binder/decoy provenance and shortcut audit. | proved | separate screening outcome/negative schema plus PR-AUC/enrichment/property-shortcut metric tests; `screening_metrics` requires one explicit screen identity and rejects silently flattened multi-campaign inputs |
| J05 | MD/potential production force defaults to `-grad(E)`; direct force is auxiliary/screening unless independently justified. | proved | `ForcePredictionPolicy`, conservative finite-difference tests, and separately named direct-force API |
| J06 | Dataset contract freezes prediction unit, ligand/protein/complex/pose IDs, construct/state/assembly, pocket, cofactors/waters/ions, assay/species/method, apo/holo, and bound/generated provenance. | proved | immutable `SBDDSampleContract`, `EntityIdentifiers`, `StructureIntake`, and assay/schema tests |
| J07 | Structure intake records alt-loc, insertion, nonstandard/missing residues, chain breaks, assembly, sequence alignment, confidence, and conformation provenance. | partial | immutable `StructureIntake` validates every receipt field; no PDB/mmCIF parser fixture populates it from real input |
| J08 | Deduplication and split membership hashes prevent alternate poses, replicates, mutants, homologs, and analogs from leaking across splits. | partial | pose/replicate/derived/ligand/protein cluster grouping, deterministic membership hash and manual leakage audit exist; `SplitResult` rejects duplicate IDs, conflicting assignments, malformed hashes, and assignment/digest mismatch; mutant/homology/analogy cluster generation from raw data is absent |
| J09 | Split matrix supports warm-pair, cold-drug, cold-target, cold-both, protein/ligand clusters, pose-group isolation, and optional temporal/campaign holdout. | proved | deterministic label-blind split matrix, cold-both, campaign, temporal and leakage tests in `tests/test_sbdd_splits.py` |
| J10 | Negative provenance distinguishes measured inactive, verified non-binder, wrong pose, property-matched decoy, random pair, and unknown; missing edges are not negatives. | proved | enum/outcome contracts and missing/wrong-pose nonbinder rejection tests |
| J11 | Labels preserve units, direction, qualifiers/censoring, assay conditions, and replicate type; transforms fit train-only and metrics report original units. | proved | schema/transform/metric tests preserve original units and train membership; `AffinityHead` declares higher-is-stronger and `censored_affinity_loss` requires labels to match the declared prediction direction while handling exact, lower/upper-bound, and interval labels; adapter tests reject flattening censored labels into exact graph targets |
| J12 | Affinity metrics include MAE/RMSE/Pearson/Spearman, within-target ranking, cold slices, uncertainty, and calibration. | proved | affinity and slice metric fixtures in `tests/test_sbdd_metrics.py` |
| J13 | Pose metrics include top-k success under declared criterion, within-pair ranking, clash/strain/contact, and flexibility/similarity/source slices. | partial | core pose metrics and generic slice function exist; dedicated fixtures for every flexibility/similarity/source slice are missing |
| J14 | Screening metrics include PR-AUC, EF, BEDROC, hit rate, and property-only audit; ROC-AUC alone is insufficient. | proved | tie-aware enrichment, BEDROC stability, property shortcut and class-validation tests |
| J15 | MD evaluation includes force finite difference, NVE drift, ligand/site stability, rotamers, waters, coordination, multiple systems/seeds, and distributions before production claims. | experiment | preregistration and trajectory artifacts |

## K. Scientific and resource validation

| ID | Requirement | Current status | Completion evidence |
|---|---|---:|---|
| K01 | Resource-matched architecture arms: legacy LGL, deep-global, H3-R4, H4-R2, H6-R2, and workspace-l2/l3 candidates. | partial | `scripts/benchmark_architecture_matrix.py` executes all named arms except a distinct workspace-l2 arm and performs bounded parameter-width search. The representative CUDA artifact completed 72/72 rows, but discrete widths missed the 1% parameter gate for five non-reference arms, so it is a resource diagnostic rather than a matched comparison. |
| K02 | Match parameters within 1%, actual median optimizer-inclusive CUDA step within 5%, same edges/features/splits/loss/precision/updates. | experiment | preregistration and resource receipt |
| K03 | Report best and last, 3–5 seeds, uncertainty, peak allocated/reserved memory, saved tensors, launches, and graph construction included/excluded. | experiment | result bundle |
| K04 | Backend grid covers `N={128,512,2048,8192}`, `k={4,8,16,32,64,128}`, uniform/skew/ragged, forward and full train step. | partial | `scripts/benchmark_architecture_matrix.py` exposes the full grid/variant CLI and records forward plus optimizer-inclusive train-step resources. `architecture-matrix-cuda-representative.json` completed 72/72 rows for `N={128,512}`, `k={4,32}`, all three topologies and six arms on the target GPU; the full grid remains unexecuted. |
| K05 | Promotion target for streamed system: ≥25% train-memory reduction, ≥20% train-step reduction, ≥10–15% faster than same-edge EGNN at `N=8192,k=64`, <10% low-degree fallback regression. | experiment | preregistered benchmark result |
| K06 | Accuracy experiments compare local balancing, positive/softmax, schedules, gated LGL, homogeneous hybrid, and bounded generic 3D tasks before domain-specific claims. | experiment | experiment packet |
| K07 | General architecture evidence spans scalar property, force/coordinate-sensitive, and protein–ligand or point-cloud task families. | experiment | one real ATOM3D-LBA train-row wiring/capacity smoke exercises the generic high-order core plus generic `readout_mode="bipartite"` selected/context pooling; scalar-property and force/coordinate-sensitive matched confirmations remain absent |
| K08 | SBDD baselines include ligand-only, protein-only, nearest pair, docking score, concatenation, EGNN, Equiformer, MACE-like local model, incumbent, and any actual E2Former backend/checkpoint. | experiment | baseline matrix |
| K09 | PDBBind/LBA overfit is only a wiring/capacity check, never generalization or affinity-SOTA evidence. | proved | documentation/ledger |
| K10 | QM9/LBA/PDBBind test labels remain inaccessible until frozen promotion; random-row results are not called cold-target evidence. | proved | runner contracts and ledger |
| K11 | Real target GPU is named; H20 results are not extrapolated to unrelated GPUs; low-degree launch overhead and explicit-solvent scale are reported separately. | partial | real-LBA and representative matrix receipts name RTX PRO 6000 Blackwell Max-Q and report allocations/timing; the representative grid includes `k=4` but is not a repeated launch-overhead study, and explicit-solvent scale is untested |

Current real-data evidence is deliberately narrow. The CPU, seeded CUDA BF16,
and strict-CUDA receipts under this packet consume one ATOM3D-LBA ID30 **train**
row (331 nodes / 9,306 typed edges) through roles/masks, four relation IDs,
distance bands, packed streamed CSR, global auto, transient `l=3`, and the
generic bipartite selected/context readout. The SBDD adapter maps the
task-specific roles into that vocabulary-free mask contract. The
five-update BF16 smoke changed training loss `34.9281 -> 33.4549`, reported
zero discrepancy under a determinant-`-1` O(3) transform against a hard
`1e-4` FP32 tolerance, and peaked at 93,915,136 allocated / 117,440,512
reserved bytes on the named RTX PRO 6000. The strict lane completed one update
with deterministic algorithms enabled. Per-update samples and their actual
median are stored. These receipts prove wiring, finite optimization, symmetry
smoke behavior, and reporting only; one train complex cannot establish
generalization, cold-target performance, affinity utility, resource
superiority, or architecture superiority.

The representative CUDA mechanics matrix uses twelve exact simple directed
candidate topologies (`N={128,512}`, `k={4,32}`, uniform/skew/ragged) and six
architecture arms, for 72 completed rows. It times model-only forward and an
optimizer-inclusive train step, records peak allocation, and excludes graph
construction explicitly. One untuned repeat per row, imperfect parameter
matching, and the PyTorch streamed reference make the numbers diagnostic only;
no architecture or backend is promoted from this matrix.

## L. Explicit exclusions and claim guardrails

| ID | Requirement | Current status | Completion evidence |
|---|---|---:|---|
| L01 | Do not replace exact global transport with sparse softmax. | excluded | architecture/source audit |
| L02 | Do not make literal LGL the vNext default or remove global heads in “local” vNext blocks. | excluded | construction tests |
| L03 | Do not call local sparse softmax linear global attention. | excluded | public docs/API names |
| L04 | Do not introduce Wigner/EAAS into optimized `l<=2` merely for fashion; require a justified optional backend. | excluded | backend boundary |
| L05 | Do not expose `hidden_irreps` as arbitrary numerical support before an executor exists. | excluded | a separate reference TP executor now exists, but persistent model `hidden_irreps/output_irreps` still reject non-Cartesian or `l>2`; transient `l=3` is explicitly nonpersistent |
| L06 | Do not copy E2Former tile sizes or claim K-independent activation without measurement of this model’s materialized tensors. | excluded | benchmark/docs audit |
| L07 | Do not use unsafe unguarded unit displacement at coincident coordinates. | excluded | coincident-coordinate test |
| L08 | Do not claim general potential training from a custom first-backward-only kernel. | excluded | no custom kernel is shipped; gradgrad capability/fallback receipts and conservative/direct-force naming preserve the boundary |
| L09 | Do not hardcode biological semantics into the generic core. | excluded | annotations, primitives, and `readout_mode="bipartite"` are vocabulary-free; SBDD owns the biological mapping, while legacy `"interaction"` is bitwise/state-key-compatible alias spelling only |
| L10 | Do not infer SBDD utility from DFT energy/force, mix affinity/docking labels, merge wrong poses with non-binders, or use whole-protein mean pooling. | excluded | SBDD task contracts separate labels/outcomes, reject docking-as-affinity and wrong-pose-as-nonbinder, and affinity head consumes separate interface/global representations |
| L11 | Do not infer production MD from one short trajectory or architecture superiority from one seed. | excluded | reporting guard |

## Implementation order frozen by this run

1. Close B01–B15 and C09 first because current mathematics can erase the cutoff
   and can silently create a quadratic graph.
2. Close C01–C08, D01–D06, and E01–E05 before making efficiency claims.
3. Implement the PyTorch reference forms of F01–F06 before profiler-gated custom
   kernels; implement F08–F10 only as a clearly separate optional backend.
4. Close G01–G11 and H01–H05 with a bounded transient `l=3` reference path.
5. Implement I/J adapters and contracts without changing the generic core’s
   vocabulary.
6. Run K experiments only after their hypotheses, baselines, thresholds, and
   no-test boundaries are preregistered.
7. Update every row with code/test/artifact evidence, run independent review, and
   perform a final row-by-row completion audit.
