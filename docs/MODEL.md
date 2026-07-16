# Model Contract

## Public API

```python
model = EquivariantAttention(EquivariantAttentionConfig(...))
out = model(node_feats, pos, batch=None)
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

At least one node is required. Graph IDs must be nonnegative, contiguous, and
start at zero; empty graphs are not encoded. A missing batch means one graph.

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

Representation metadata belongs to the model rather than the output dictionary
so compiled/tensor consumers do not receive strings.

`node_tensors` and `graph_tensors` read out the last attention block's
transient symmetric-traceless moment. They are not a persistent tensor hidden
state computed after the final scalar/vector FFN.

## Architecture choices

- unit-normalized positive scalar content plus a bounded degree-2 vector
  kernel with linear and quadratic angular terms;
- finite-precision unit-ball vector queries/keys and learned angular scales in
  configured closed bounds;
- structured vector and 3x3 PSD summaries for masses and denominators, plus
  analogous signed value summaries that are not clamped;
- row normalization, with exactly one key-mass balancing cycle enabled by
  default and a no-balancing lane restricted to controlled experiments;
- exact factorized relative vector and rank-2 moment transport;
- scalar/vector residual updates;
- ratio-2 pointwise equivariant FFN in every block.

Turning off `use_alignment_linear_term` removes only the `beta * (q dot k)`
term while retaining the same `beta` constant. `kernel_floor_mode="fixed"`
uses the configured baseline for every pair. The experimental global row-only
`"inverse_graph_size"` mode scales `(c + beta + delta*beta*(q dot k))` by
`1/N_g`, leaves content and the quadratic angular term unscaled, and is
rejected with balancing.

## Routing and geometry

`local_head_counts=None` is the public all-global default. For the registered
three-block CLI presets, `ggg=(0,0,0)`, `lgl=(H,0,H)`, and `lll=(H,H,H)`. An
initial scalar embedding uses `node_feats` only and the initial vector state is
zero. Centroid/RMS-normalized global geometry enters immediately before the
first block that has a global head, preventing global preprocessing from
leaking into `lll`.

Global heads use structured graph summaries and do not construct an `N x N`
attention tensor. Local heads construct directed same-graph raw-coordinate
edges with self edges, a default 2.5-Angstrom cutoff, 16 Gaussian RBFs, and a
positive cosine gate of squared scaled distance. The core fallback stores
`O(E)` transport and reuses one geometry/RBF construction across all local
stages, but may search `O(N^2)` candidate pairs. Thus the repeated work is
`N^2 + L*E`, while no production end-to-end `O(E)` claim is made.

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
It uses a deterministic one-dimensional invariant router, the same assignment
for reads and writes, and a symmetric coupling derived from graph-normalized
center coordinates. Thus the coupling is shape-relative rather than a raw
physical-distance decay. In a mixed local/global block, local heads also read
the shared scalar/vector state after global geometry has been injected; mixed
heads are legal but are not a promoted independent route.

The frozen 16-node Stage-0 probe uses the same state dictionary for M=1, M=4,
and M=8 and checks every global head separately. On the current implementation,
M=4 and M=8 have occupied, non-collapsed assignments but coupling matrices and
effective pair gates numerically indistinguishable from all ones. The complete
output changes by only about `1e-12` to `1e-11` in relative RMS. Therefore the
implemented M=4/8 flags remain available for mechanism diagnostics, but broader
memory accuracy/performance arms are blocked pending a separately
preregistered coupling/router redesign.

`use_radial_trace` controls an exact relative squared-distance moment. Its
state slot exists in both arms and is exactly zero when disabled, while public
rank-2 output remains symmetric traceless.

## Applicability

The current contract is O(3), not chirality-sensitive SE(3). It is appropriate
for properties that should be unchanged by reflection. Enantiomer-sensitive
scalar prediction is outside the current model class.

Any route with a global head does not enforce cluster decomposition or
extensive size consistency. Mean graph readout is not an additive energy
model. Finite degree-2 moments/RBFs retain representation collisions, and soft
memory assignments can collapse. The model is therefore scoped to bounded-size
property probes or global context within a local/global encoder, not validated
interatomic potentials or energy-conserving force fields.
