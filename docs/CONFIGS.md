# Configuration

## Structured architecture boundary

`ArchitectureConfig` is a strict, versioned wrapper around the flat
`EquivariantAttentionConfig`. It owns four nested immutable groups:

| Group | Scope |
|---|---|
| `RepresentationConfig` | persistent input/hidden/output irreps, role vocabulary size, angular bandwidth, transient-workspace policy, reference/expert TP declaration |
| `GlobalTransportConfig` | exact finite-feature kernel, balancing, memory, tensor-value lane, and requested reduction backend |
| `LocalResidualConfig` | legacy local route, homogeneous sparse rank/schedule, normalization, distance bands, and requested sparse backend |
| `NeighborConfig` | moving-coordinate policy, missing-neighbor policy, cache mode, provider kind, relation count, and relation cutoffs |

`ArchitectureConfig.for_profile(...)` freezes these representation presets:

| Profile | Persistent carrier | Symmetry | Transient workspace |
|---|---|---|---|
| `minimal` | `0e + 1o` | O(3) | degree at most 2 |
| `standard` | `0e + 1o + 2e` | O(3) | degree at most 2 |
| `chiral` | `0e + 1o + 2e` | SE(3) | degree at most 2 |
| `high_order` | low-`l` `0e + 1o + 2e` | O(3) | nonpersistent `l=3` |
| `expert` | explicit compatibility layout | explicit | raw TP declarations are accepted only here and may be deferred |

`from_legacy`/`to_legacy` is the compatibility route. It rejects a structured
feature that the Cartesian model cannot execute instead of dropping it.
Serialization rejects unknown/duplicate fields and unsupported schema
versions. `ExecutionMetadata` is a separate immutable receipt for requested
and effective global/local/cache/provider decisions. A deterministic receipt
is not a performance result; the auto thresholds still require calibration on
the named target GPU.

## Model

| Field | Default | Constraint |
|---|---:|---|
| `node_dim` | required | positive |
| `input_irreps` (structured representation) | `None` | optional explicit Cartesian declaration that must exactly match `node_dim`, `input_vector_dim`, and `input_tensor_dim` |
| `hidden_irreps` | `64x0e + 4x1o` | positive scalar/vector channels, no persistent tensors |
| `output_irreps` | `1x0e` | at least one supported `0e/1o/2e` term |
| `num_node_roles` | 0 | when positive, require integer role IDs in `[0, num_node_roles)` and add an invariant scalar embedding |
| `num_layers` | 3 | positive |
| `num_heads` | 4 | positive; divides scalar channels |
| `linear_kernel_init` | 0.05 | normal float32; ratio to max is normal and below one |
| `linear_kernel_max` | 1.0 | finite normal positive linear-scale bound |
| `vector_kernel_init` | 0.05 | normal float32; ratio to max is normal and below one |
| `vector_kernel_max` | 1.0 | finite normal positive quadratic-scale bound |
| `kernel_floor` | 1.0 | normal float32 strictly positive pair-kernel floor |
| `kernel_floor_mode` | `fixed` | fixed or graph-size-scaled positive kernel baseline |
| `use_alignment_linear_term` | true | false removes only `beta * (q dot k)` and retains the `beta` constant |
| `use_key_balancing` | true | exactly one key-balancing cycle when true |
| `use_local_key_balancing` | `None` | optional local-only override; `None` inherits `use_key_balancing` |
| `use_global_key_balancing` | `None` | optional global-only override; `None` inherits `use_key_balancing` |
| `local_head_counts` | `None` | `None` means all-global; otherwise a length-`num_layers` tuple with each entry in `[0, num_heads]` |
| `global_transport_mode` | `learned` | `learned`, exact graph-mean `uniform`, or attention-residual `none` |
| `global_reduction_backend` | `outer_scatter` | `outer_scatter`, exact explicit-feature `feature_gemm`, or deterministic `auto`; GEMM lanes require learned transport |
| `local_reduction_backend` | `index_add` | compatibility `index_add` or receiver-CSR `segment_csr` |
| `use_sparse_low_rank_local_residual` | false | add an edge-state-free separable sparse residual without removing any global head; requires an all-global base route and active global transport |
| `local_residual_rank` | 4 | positive invariant edge rank for the sparse residual |
| `local_residual_layers` | `None` | `None` refreshes every block; otherwise a nonempty unique tuple of block indices |
| `sparse_residual_normalization` | `positive` | mass-damped positive transport or local-only post-cutoff `softmax` ablation |
| `sparse_residual_score_limit` | 3.0 | finite bounded-exponential log-gate limit in `[0.5, 4.0]` |
| `sparse_residual_balancing` | `receiver` | explicit receiver-owned sparse normalization; legacy local key-balancing overrides are rejected |
| `sparse_residual_neighbor_policy` | `require` | require supplied sparse candidates or explicitly allow a bounded `complete_fallback` |
| `sparse_residual_complete_fallback_max_nodes` | 256 | positive total-node guard evaluated before constructing a complete fallback |
| `sparse_residual_backend` | `materialized` | PyTorch `materialized`, `streamed_csr`, `ell`, or deterministic `auto`; `custom` is capability-gated and not implemented here |
| `sparse_residual_stream_chunk_size` | 64 | positive receiver-row chunk bound for the PyTorch streamed reference |
| `geometry_cache_mode` | `full` | `full`, `compact`, `recompute`, or deterministic `auto` |
| `num_edge_relations` | 0 | nonnegative number of invariant integer relation IDs |
| `relation_cutoffs` | `None` | one positive cutoff per relation when relations are enabled |
| `distance_band_cutoffs` | `()` | strictly increasing positive cutoffs for overlapping additive sparse-residual bands |
| `use_multiscale_spatial_kernel` | false | opt-in ten-feature/head spatial kernel; requires all-global learned transport and no memory interaction |
| `use_adaptive_multiscale_spatial_kernel` | false | opt-in content-adaptive four-scale spatial kernel in the middle global stage of exact three-layer LGL |
| `use_global_tensor_value_transport` | false | opt-in exact global transport of persistent `2e` values; requires hidden `2e`, an active global mode, and at least one global head |
| `use_cartesian_tensor_product_local_transport` | false | opt-in native Cartesian `2e x 1o -> 1o`, `2e x 0e -> 2e`, and `1o x 1o -> 2e` local paths; requires gated local transport and persistent `2e` |
| `use_static_tensor_carrier` | false | compact local-only `2e` carrier; requires hidden `2e` multiplicity equal to `num_heads` |
| `cartesian_tensor_product_local_layers` | `None` | tuple of local-stage indices; `None` selects every local stage when CTP is enabled |
| `symmetry_group` | `"O3"` | `"O3"` or proper-rotation-only `"SE3"` |
| `use_geometry_aware_local_attention` | false | sparse `0e/1o/2e` local score refinement; requires gated local transport |
| `use_se3_axial_tensor_product` | false | optional axial `2e x 2e -> l=1` value; requires geometry attention and `symmetry_group="SE3"` |
| `geometry_aware_local_layers` | `None` | tuple of local-stage indices; `None` selects every local stage when geometry attention is enabled |
| `use_transient_l3_workspace` | false | bounded aggregate-then-project `1o x Y2 -> 3o`, `3o x 2e -> 1o` local reference path |
| `transient_l3_channels` | 1 | positive transient `3o` multiplicity |
| `transient_l3_layers` | `None` | every layer when enabled, or an explicit unique layer tuple |
| `transient_l3_residual_scale_init` | 0.05 | nonnegative residual scale before depth normalization |
| `readout_mode` | `"mean"` | `"mean"`, `"sum"`, or generic selected/context `"bipartite"`; legacy `"interaction"` is an exact alias with identical module/state keys |
| `local_cutoff` | 2.5 | positive raw-coordinate local cutoff |
| `num_rbf` | 16 | positive number of local Gaussian RBFs |
| `learn_local_radial_gate` | false | frozen radial gate for registered routing studies |
| `global_memory_count` | 1 | positive; CLI registers 1, 4, and 8 |
| `use_memory_interaction` | false | registered only for the middle global stage of three-layer `lgl` |
| `memory_assignment_temperature` | 1.0 | positive bounded-assignment temperature |
| `memory_assignment_scale` | 2.5 | positive centroid-refinement distance scale |
| `memory_interaction_cutoff` | 2.5 | positive fixed center-coupling cutoff |
| `use_radial_trace` | false | reserve and populate the exact relative radial second-moment scalar |
| `residual_scale_init` | 0.1 | nonnegative |
| `eps` | `1e-12` | positive; normalization occurs in float32+ |

The ratio-2 equivariant FFN remains fixed. All settings configure the same
`EquivariantAttention` class. `local_head_counts=None`, one memory, memory
interaction off, and radial trace off are the public defaults. The training
presets map `ggg` to `(0,0,0)`, `lgg` to `(H,0,0)`, `ggl` to `(0,0,H)`,
`lgl` to `(H,0,H)`, and `lll` to `(H,H,H)` for three layers. `uniform`
broadcasts the exact graph mean of the same moment sufficient statistics.
`none` bypasses the attention updater in all-global blocks and keeps only the
pointwise FFN; it does not execute global geometry preprocessing.

The sparse low-rank residual defines the homogeneous hybrid route. Every block
keeps `num_heads` global heads; the selected residual layers independently add
an `O(E R D_head)` transported scalar payload plus fixed 3/5-coordinate
equivariant payloads. At fixed head width this is `O(E R)`. Node projections
are computed before edge gathering, the edge latent has width
`local_residual_rank`, and the default positive lane uses
`sum(raw * value) / (1 + sum(raw))`. The stabilizing one prevents a singleton
receiver from cancelling the smooth cutoff. One logical receiver reduction
accumulates raw mass, squared mass, and every value family; their logarithms
are available to a zero-initialized invariant scalar projection. One compact
RBF projection independently modulates scalar, vector, relative, tensor, and
radial values. No edge state persists across blocks. Separate
rank-to-head output maps are zero-initialized inside a forked RNG context, so
enabled construction preserves all common initialization and initially
computes the incumbent function. It cannot be combined with nonzero
`local_head_counts`; legacy LGL remains a separate compatibility/evidence
route. Sparse candidates are required by default; the explicit complete
fallback is node-bounded and intended only for small diagnostics.

`global_reduction_backend="feature_gemm"` is an equation-preserving execution
choice for ordinary learned global transport. It concatenates the content,
constant, linear vector, isometric quadratic, and optional spatial feature
blocks and computes graph-wise `Phi_K^T V` / `Phi_Q S` products. It supports
one-cycle global key balancing, fixed and unbalanced inverse-graph-size
baselines, distinct adaptive spatial query/key features, and widened
persistent-`2e` value payloads. Interacting multi-memory transport remains on
its existing backend.

`local_reduction_backend="segment_csr"` sorts retained COO edges
receiver-major once per geometry build, stores int32 offsets when safe, and
supplies reverse sender rows for generic local key balancing. A supplied
`PackedNeighborGraph` instead consumes its receiver and optional reverse plan
directly; cutoff filtering restricts those plans in linear time without
resorting. Gated, edge-conditioned, pairwise-content, generic local, and
sparse-residual receiver sums consume the CSR metadata. `packed_neighbors=`
and `edge_index=` are mutually exclusive.

`PackedNeighborGraph` may additionally store receiver degree, maximum degree,
fixed bucket IDs and histogram/skew, an optional lossless ELL view, immutable
relation IDs, and a separately constructed reverse sender plan. Same-device
`.to()` is identity; cross-device transfer trusts already validated metadata.
Reverse CSR reorders the same directed edges for reduction and does not apply
the semantic reverse-relation involution.

`sparse_residual_backend="streamed_csr"` and `"ell"` execute row-owned
PyTorch references for positive mass-damped or local softmax reduction. The
references keep row intermediates local to a receiver chunk and accumulate
sensitive sums in FP32 for BF16/FP16 inputs. `"auto"` uses static degree/layout
metadata and capability flags; requesting gradgrad may produce an explicit
PyTorch fallback receipt. There is no fused custom CUDA/Triton implementation,
custom reverse-CSR backward, or target-GPU tuning result in this repository,
so these settings must not be read as a speed or memory guarantee.

`geometry_cache_mode` changes only activation storage. `full` keeps derived
RBF/direction terms, `compact` keeps a smaller base, `recompute` rebuilds
geometry under autograd, and `auto` uses deterministic edge-count thresholds.
Coordinate-updating models rebuild layer geometry after each coordinate step.
Reference-radius discovery is deterministic but quadratic; precomputed and
external providers bypass discovery; the reference Verlet provider implements
skin/rebuild semantics but does not claim a production cell list or PBC.

`num_edge_relations>0` requires one relation ID per sparse edge and one cutoff
per relation. Relation IDs select invariant scalar bias/cutoff parameters
inside one sparse list; they do not create relation-specific equivariant
kernels or persistent edge state. `distance_band_cutoffs` similarly enables
overlapping additive score residuals. The gates are deliberately not an exact
near/medium/global partition.

`use_transient_l3_workspace=True` requires explicit sparse neighbors. Just
before each selected layer's normal transport, it filters the supplied
same-graph edges by the local and relation-specific cutoffs, lifts sender
`1o` with edge `Y2` to `3o`, aggregates the
`3o` and `2e` summaries independently, projects their receiver-level tensor
product back to `1o`, bounds the vector residual, and discards the `l=3`
workspace. The path supports O(3), FP32-sensitive accumulation, BF16
projections, and PyTorch gradgrad. It does not permit persistent `l=3` in
`hidden_irreps` or `output_irreps`.

`use_multiscale_spatial_kernel=True` adds a fixed-rank positive spatial
dot-product kernel to each global head. The four-head default uses log-spaced
scales `[0.125, 0.25, 0.5, 1.0]` and ten degree-two Gaussian-Taylor features
per head. The features are built from graph-centered/RMS-normalized positions
and enter the exact graph-segmented sufficient statistics; no edge list,
node-pair tensor, or new learned parameter is created. Coordinate-updating
models recompute the spatial features after every coordinate step. This first
route is rejected with local heads, uniform/disabled global transport, or
memory interaction. The option remains off by default.

`use_adaptive_multiscale_spatial_kernel=True` is a separate LGL-only research
path. The fully global middle layer uses scales `[0.125, 0.25, 0.5, 1.0]` for
every head. An invariant scalar projection produces separate query/key scale
logits. With `p=softmax(logits)` and `epsilon=finfo(dtype).eps`,
`sqrt((p+epsilon)/(1+4*epsilon))` multiplies all ten components at a scale
before the four blocks are concatenated. The epsilon floor prevents
softmax-underflow NaNs while preserving unit squared profile mass and logit
shift invariance. The resulting query/key spatial features retain the existing
exact segmented factorization and add a nonnegative term at fixed `O(4N)` rank
overhead; the incumbent kernel floor keeps the full denominator positive. The
flag requires exact three-layer
`local_head_counts=(H,0,H)`, learned global transport, and excludes the fixed
spatial kernel, interacting memory, and whitened global read.

`use_global_tensor_value_transport=True` appends the bounded head-space
persistent tensor as five additional value coordinates in every active global
head. The existing factorized numerator and denominator therefore compute
`sum_j A_ijh W_to H_j^(2e)` without a pair tensor. The generic carrier consumes
the result through its existing head-to-`2e` gated residual map; the static
carrier uses its identity head layout and bounded residual directly. It adds no
parameter or checkpoint key and supports learned and uniform global transport.
The option is rejected without hidden `2e`, with
`global_transport_mode="none"`, or on an all-local route. It remains off by
default.

`inverse_graph_size` keeps the compatibility-oriented option name but scales
the full shifted positive global baseline `(c + beta + delta*beta*t)` by
`1/N_g`; content and `gamma*t^2` stay unscaled. It is rejected with key
balancing and is never used as a proxy for local receiver degree. Enabling
memory interaction requires learned transport and the registered `lgl` route;
merely raising the
memory count with interaction off is algebraically the incumbent. The M=4/8
CLI values are implemented diagnostic arms, not promoted configurations: the
frozen Stage-0 pair-gate probe currently blocks broader interacting-memory
experiments and constructing one emits a Stage-0-blocked warning. The internal
M-shared invariant router is present with the same
parameter schema for every memory count, but no residual-coupling value is a
public configuration because none of the registered candidates passed the
unchanged full Stage-0 thresholds.

Kernel controls deliberately reject positive float32 subnormals: inverse-logit
initialization and `c/N_g` can otherwise round them to zero and invalidate the
strictly positive denominator contract. This does not apply to the scale-first
local cutoff and memory geometry controls, whose subnormal behavior has
separate finite-value and gradient tests.

The generic `IrrepLayout`/`TensorProductPlan` planner accepts arbitrary
nonnegative `l` and `e/o` labels, canonicalizes multiplicities/slices, and
applies O(3)/SE(3) selection rules at construction.
`ExecutableTensorProductPlan` freezes reachable reference instructions with
multiplicities, offsets, `uvw` connection mode, shared weights, real-tesseral
CG convention, component normalization, coefficient dtype/device, and stable
order. `ReferenceTensorProduct` numerically executes those instructions; the
reference irrep package also supplies same-irrep mixing, per-copy RMS
normalization, scalar gates, and real spherical harmonics/CG coefficients.
This does not widen persistent model storage: `CartesianIrreps` still accepts
only scalar `0e`, polar-vector `1o`, and reflection-even
symmetric-traceless `2e`; parity-odd or persistent `l>2` model channels remain
rejected.
`use_static_tensor_carrier=True` is an execution specialization for
`C2 == num_heads`; it carries the compact tensor state unchanged through
global-only stages and updates it in local stages by default. Enabling global
tensor value transport additionally lets an active global stage update this
static carrier through the same five-coordinate factorized read.

The geometry-aware fields are also statically compiled. `O3` retains
reflection covariance. `SE3` permits the axial `l=1` component of
`2e x 2e`, whose parity differs from the polar vector carrier under reflection.
The LBA runner exposes `geometry_o3` and `geometry_se3` research arms using
only local layer 0; neither is a promoted default.

## Generic annotations and task adapters

`GraphSample`/`GraphBatch` fields `node_role_id`, `edge_relation_id`,
`node_masks`, and `hierarchy_id` are optional generic metadata. Every sample
in a collated batch must either supply a field or omit it; named masks must use
the same sorted keys. Hierarchy IDs are made batch-global by an offset, while
roles and relation IDs retain caller-defined integer semantics.
`MaskedInvariantPooling` and `HierarchyAssignment` consume these generic
annotations. Coordinate/update heads consume an explicit Boolean update mask.

The `equivariant_attention.sbdd` package owns biological vocabularies and task
contracts. Its current scope is deterministic tensor featurization, immutable
schema/split receipts, leakage auditing, affinity/pose heads and losses, and
affinity/pose/screening metrics. Its label-blind graph adapters emit stable
role/relation IDs, rich hierarchy IDs, role masks, a cross-edge interface
mask, and a generic `readout_mask`; `censored_affinity_loss` preserves exact,
lower/upper-bound, and interval labels without flattening them into exact
targets and requires label direction to match the declared prediction
direction (`AffinityHead` is higher-is-stronger). Screening metrics require one
explicit screen identity and reject silently flattened multi-campaign inputs.
It does not yet convert raw PDB/mmCIF/RDKit objects into
`GraphSample`, build production atom/coarse neighbor lists, generate
protein/ligand clusters, or wire an end-to-end SBDD runner. The older
`pdbbind.py` loader is a separate dataset adapter and remains outside this new
structured adapter. Core `readout_mode="bipartite"` consumes only a selected
mask and its context complement; `"interaction"` is an exact legacy alias, not
a biological contract.

## Training probe

The CLI exposes dataset/split/model seeds, width, depth, heads, AdamW learning
rate and weight decay, gradient clipping, device, bf16 autocast, target
normalization, alignment/balancing/floor controls, routing, local cutoff/RBF,
memory, radial-trace, and test-evaluation policy. The explicit public flags are
`--routing ggg/lgg/ggl/lgl/lll`,
`--global-transport-mode learned/uniform/none`, `--memory-count 1/4/8`,
`--memory-interaction`, `--radial-trace`,
`--edge-conditioned-local-transport`, `--precompute-local-edges`, and opt-in
`--evaluate-test`. Edge-conditioned local transport requires all local heads,
matching hidden vector/head counts, and cannot be combined with
`--pairwise-local-content` or `--learn-local-radial-gate`. Precomputed QM9
radius candidates avoid forward-time pair discovery but do not make the data
loader asymptotically linear. `GraphBatch` collation validates supplied edge
contents and enables the model's trusted linear hot path; direct model calls
validate by default, and callers must not assert `edge_index_is_validated` for
unchecked external tensors.
`--bounded-diagnostics` recomputes every active local layer/head outside timed
training; `--diagnostic-sample-count` defaults to 32 deterministic validation
graphs selected after sorting by `(node_count, dataset_index)`.
`--benchmark-model internal_static_egnn_baseline` selects only the private
same-harness comparison baseline; it does not add a public model API. Its
model-specific default is width 91 (the factorized default remains 64), and
nondefault factorized-only controls are rejected instead of silently ignored.
`metrics.run_config` records every run-defining argument and marks the EGNN arm
`official_reproduction=false`. It also records whether global transport and
global geometry actually executed; local-only routes report
`not_applicable_no_global_heads` regardless of the configured transport label.

## Scaling benchmark

`scripts/benchmark_sparse_scaling.py --edge-multiplier-grid` selects the exact
same-edge model comparison. `--sizes` sets node counts and
`--edge-multipliers` sets positive `k` values. The generator emits exactly
`min(kN,N^2)` duplicate-free directed candidates with one self edge and exact
receiver degree `min(k,N)`. `--seed` controls graph topology while
`--model-seed` independently controls model initialization; both model-state
hashes are recorded. `--warmup`, `--repeats`, and `--max-wall-seconds` bound timing.
The JSON records edge hashes, execution order, receiver degree, synchronized
median latency, CUDA peak allocation delta, per-node-count fits, failures and
interpretation boundaries. Edge construction is deliberately outside the model
timing. Re-run multiple `--seed` values with one fixed `--model-seed` when a
crossover margin is small.

`--edge-free-spatial-grid` instead measures current static GGG, opt-in static
spatial GGG, coordinate-updating spatial GGG, and private static EGNN. The
three attention variants receive no `edge_index`; EGNN receives the
deterministic prebuilt `E=kN` controls. The JSON reports parameter/state
hashes, synchronized median latency, peak CUDA allocation delta, edge-index
bytes, and explicit topology/accuracy boundaries. This is a different-model,
different-topology systems comparison, not a same-computation benchmark.

`scripts/run_registered_transport_study.py` freezes the 2026-07-19 six-arm
screen, fifteen-arm five-seed transport confirmation, conditional five-arm
private EGNN comparison, exact QM9 data/split hashes, test-disabled policy, and
1,500-second GPU ceiling. A scientific threshold failure exits successfully
but records `transport_locked=false`; infrastructure, provenance, nonfinite, or
budget failures stop the runner.
