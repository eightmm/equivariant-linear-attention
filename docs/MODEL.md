# Model Contract

## Public API

```python
model = EquivariantAttention(EquivariantAttentionConfig(...))
out = model(
    node_feats,
    pos,
    batch=None,
    edge_index=None,
    node_vectors=None,
    node_tensors=None,
)
```

There is one public attention implementation, `EquivariantAttention`. Local,
global, hybrid, memory-gated, and radial-trace behavior are configuration
settings rather than separate model classes. `model.attention_kind` is
`"factorized_moment"` and `model.symmetry` is `"O3"`.

## Inputs

| Name | Shape | Contract |
|---|---:|---|
| `node_feats` | `(N, node_dim)` | finite O(3)-invariant scalar (`0e`) channels |
| `pos` | `(N, 3)` | finite float32+ coordinates on the same device |
| `batch` | `(N,)` | optional integer IDs `0..G-1` |
| `edge_index` | `(2, E)` | optional keyword-only integer local candidates: receiver row then sender row |
| `readout_mask` | `(N,)` | optional keyword-only boolean mask selecting graph-pooling nodes |
| `node_vectors` | `(N, input_vector_dim, 3)` | optional keyword-only polar-vector (`1o`) channels |
| `node_tensors` | `(N, input_tensor_dim, 3, 3)` | optional keyword-only symmetric-traceless reflection-even (`2e`) channels |

At least one node is required. Graph IDs must be nonnegative, contiguous, and
start at zero; empty graphs are not encoded. A missing batch means one graph.
Supplied edges are legal only when at least one layer has local heads. They must
be on the input device, nonnegative and in range, unique as directed pairs,
same-graph, and contain `i <- i` for every node. Candidates outside the strict
local cutoff are discarded. Supplying edges bypasses complete-pair discovery;
omitting them preserves the existing fallback.

When supplied, `readout_mask` must select at least one node in every graph.
Transport still uses every node; only the final node-to-graph mean is masked.
Omitting the mask, or supplying an all-true mask, preserves the default
all-node mean exactly.

`node_vectors` and `node_tensors` are required exactly when their configured
input dimensions are positive. Tensor inputs must be symmetric and traceless
within the dtype-specific tolerance, and `input_tensor_dim > 0` requires hidden
`2e` channels. Channel-only projections inject these inputs into the hidden
equivariant states; disabled input paths allocate no parameters.

## Outputs

The forward result is a tensor-only dictionary:

| Key | Shape | Transformation |
|---|---:|---|
| `node_scalars` | `(N, Cs)` | O(3)-invariant |
| `node_vectors` | `(N, Cv, 3)` | polar-vector equivariant |
| `node_tensors` | `(N, Ct, 3, 3)` | symmetric-traceless rank-2 equivariant |
| `graph_scalars` | `(G, Cs)` | O(3)-invariant |
| `graph_vectors` | `(G, Cv, 3)` | polar-vector equivariant |
| `graph_tensors` | `(G, Ct, 3, 3)` | symmetric-traceless rank-2 equivariant |
| `node_positions` | `(N, 3)` | point-coordinate equivariant; only with `coordinate_updates=True` |

Representation metadata belongs to the model rather than the output dictionary
so compiled/tensor consumers do not receive strings.

Without hidden `2e` channels, `node_tensors` and `graph_tensors` read out the
last attention block's transient symmetric-traceless moment as before. When
`hidden_irreps` contains `Cx2e`, the model instead carries `C` persistent
five-component symmetric-traceless channels through every block and reads the
requested output tensors from that final hidden state.

The default `coordinate_updates=False` returns exactly the original six keys
and allocates no coordinate parameters. With the option enabled, every
nonfinal block produces an invariant-gated polar displacement, subtracts its
graph-wise mean, applies one graph-wise scale so every node step is at most
0.25 Angstrom, and passes the updated positions to the next block. Omitting a
post-readout update ensures every allocated coordinate parameter can affect the
scalar training objective. `node_positions` is the resulting latent geometry.

## Architecture choices

- unit-normalized positive scalar content plus a bounded degree-2 vector
  kernel with linear and quadratic angular terms;
- optional exact positive quartic angular term and a two-`1o`-axis direct-sum
  kernel, both retaining fixed-width graph-summary factorization;
- finite-precision unit-ball vector queries/keys and learned angular scales in
  configured closed bounds;
- compressed `D(D+1)/2` quadratic summaries for masses, denominators, and
  signed value transport instead of redundant `D x D` outer products;
- row normalization, with exactly one key-mass balancing cycle enabled by
  default and a no-balancing lane restricted to controlled experiments;
- exact factorized relative vector and rank-2 moment transport;
- scalar/vector residual updates and optional persistent Cartesian `2e`
  residual/FFN updates;
- optional invariant shared-RMS normalization of vector and `2e` states;
- ratio-2 pointwise equivariant FFN in every block.

The persistent `2e` path is opt-in through `hidden_irreps`, for example
`"64x0e + 4x1o + 4x2e"`. Each block mixes its bounded hidden tensor state into
the transient head moments, feeds Frobenius contractions back to the scalar
path, and updates the tensor state with invariant scalar gates. Channel mixing
acts only over multiplicities, so the state transforms as `T -> R T R^T`.
This is a Cartesian rank-2 implementation; it does not add `2o`, spherical
harmonics, Clebsch--Gordan products, or arbitrary `l>2` input support.

`angular_feature_rank=2` means two learned polar-vector axes per head, not an
`l=2` irrep. Their six-dimensional direct sum drives the linear and quadratic
angular kernel. `use_quartic_kernel=True` separately adds the exact
`kappa*(q_primary dot k_primary)^4` term through a 15-component symmetric
degree-four feature map. Both are opt-in and preserve `O(N)` global scaling at
fixed width, although their train-step constants are larger.

Turning off `use_alignment_linear_term` removes only the `beta * (q dot k)`
term while retaining the same `beta` constant. `kernel_floor_mode="fixed"`
uses the configured baseline for every pair. The experimental global row-only
`"inverse_graph_size"` mode scales `(c + beta + delta*beta*(q dot k))` by
`1/N_g`, leaves content and the quadratic angular term unscaled, and is
rejected with balancing.

## Routing and geometry

`local_head_counts=None` is the public all-global default. For the registered
three-block CLI presets, `ggg=(0,0,0)`, `lgg=(H,0,0)`, `ggl=(0,0,H)`,
`lgl=(H,0,H)`, and `lll=(H,H,H)`. An initial scalar embedding uses
`node_feats` only and the initial vector state is zero.

`global_transport_mode="learned"` is the public default. `"uniform"` replaces
only the global attention weights with exact `1/N_g` graph means of the same
value/moment sufficient statistics. `"none"` removes global messages; in an
all-global block it bypasses the attention updater residual and retains only
the pointwise equivariant FFN. All modes allocate the same parameter schema.
With static coordinates, centroid/RMS-normalized global geometry is computed
once immediately before the first learned or uniform global block, preventing
preprocessing from leaking into `lll` or a no-global-transport control. With
coordinate updates enabled, scale-first global geometry is recomputed before
each active global block and local cutoff/RBF geometry is recomputed before each
active local block. A supplied `edge_index` remains fixed candidate topology;
its candidates are re-filtered against the current positions at every local
stage.

Global heads use structured graph summaries and do not construct an `N x N`
attention tensor. Local heads construct directed same-graph raw-coordinate
edges with self edges, a default 2.5-Angstrom cutoff, 16 Gaussian RBFs, and a
positive cosine gate of squared scaled distance. The core fallback stores O(E)
transport and reuses one geometry/RBF construction across all local stages. It
vectorizes all same-graph Cartesian candidates across the batch, but still
searches `O(sum_g N_g^2)` pairs. Precomputed `edge_index` bypasses that search
and uses O(E) candidate/retained storage, while the caller remains responsible
for neighbor construction. Thus fallback work is `N^2 + L*E` and supplied-edge
geometry/transport after validation is `L*E` at fixed width; no production
radius-backend claim is made.

Global preprocessing is scale-first: coordinates are scaled by each graph's
maximum absolute coordinate before centroid and RMS reductions, and physical
log features are formed without an overflow-prone direct product. Coordinates
must be float32 or float64 and are never downcast through feature precision.
Kernel floors, initial values, maxima, and init/max ratios must be normal
float32 values. Positive subnormals are rejected for these coefficients because
sigmoid initialization or inverse-size scaling can otherwise erase the positive
denominator floor; scale-first local/memory geometry controls remain separate.

## Multi-memory and radial trace

The public default is one global memory with interaction disabled. The HEMM
extension uses invariant soft assignments, exact-occupancy equivariant centers,
and an optional fixed nonnegative squared-distance cosine coupling. For `M=1`,
the implementation takes the incumbent path exactly. With interaction off,
the all-ones coupling reduces the partitioned summaries algebraically to the
incumbent for any `M`; memory count alone has no expressivity claim.

The CLI registers interacting `M=4` and `M=8` only for the middle global block
of `lgl`. Slot permutations leave outputs unchanged when assignments and the
coupling are permuted together. Soft slot collapse remains possible, and no
performance benefit is claimed without the registered validation study.

This implementation is a low-rank pair gate, not a persistent memory state.
It uses an M-independent two-layer invariant router with fixed unit DCT slot
codes, the same assignment for reads and writes, and a symmetric coupling
derived from graph-normalized center coordinates. Thus the coupling is
shape-relative rather than a raw physical-distance decay. In a mixed
local/global block, local heads also read the shared scalar/vector state after
global geometry has been injected; mixed heads are legal but are not a
promoted independent route.

The frozen Stage-0 suite uses the same state dictionary for M=1, M=4, and M=8
at widths 16/64, seeds 401--403, and four graph roles. It checks assignment
marginal/conditional entropy and mutual information, centers, radial and
effective coupling, actual middle transport, post-state, input gradients, and
full output. Radial, identity, and fixed residual-coupling counterfactuals all
failed the all-aligned-lane admission rule. Therefore M=4/8 flags remain
available for mechanism diagnostics, but broader memory accuracy/performance
arms are blocked pending another separately preregistered redesign.

`use_radial_trace` controls an exact relative squared-distance moment. Its
state slot exists in both arms and is exactly zero when disabled, while public
rank-2 output remains symmetric traceless.

## Applicability

The current contract is O(3), not chirality-sensitive SE(3). It is appropriate
for properties that should be unchanged by reflection. Enantiomer-sensitive
scalar prediction is outside the current model class.

Any route with a global head does not enforce cluster decomposition or
extensive size consistency. Mean graph readout is not an additive energy
model. Finite-degree moments/RBFs retain representation collisions even with
the quartic option, and soft memory assignments can collapse. The model is
therefore scoped to property probes or global context within a local/global
encoder, not validated interatomic potentials or energy-conserving force
fields.

Coordinate-enabled positions are latent representations trained only through
the property objective. They are not validated relaxed structures, forces,
trajectories, or a conservative potential-energy surface.
