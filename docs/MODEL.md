# Model Contract

## Public API

```python
model = EquivariantAttention(EquivariantAttentionConfig(...))
out = model(node_feats, pos, batch=None)
```

There is one attention implementation. `model.attention_kind` is
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

## Fixed architecture choices

- positive scalar content plus a bounded quadratic vector kernel;
- finite-precision unit-ball vector queries/keys and a learned angular scale in
  `[0, vector_kernel_max]` (mathematically open before floating-point
  saturation);
- structured 3x3 PSD summaries for quadratic masses and denominators, plus
  analogous signed matrix-valued value summaries that are not clamped;
- one key-mass balancing cycle followed by row normalization;
- exact factorized relative vector and rank-2 moment transport;
- scalar/vector residual updates;
- ratio-2 pointwise equivariant FFN in every block.

Optional architecture flags were removed. A proposed mechanism must first add a
dense or symbolic reference test, preserve O(3)/translation/permutation tests,
and be evaluated as one isolated change.

## Applicability

The current contract is O(3), not chirality-sensitive SE(3). It is appropriate
for properties that should be unchanged by reflection. Enantiomer-sensitive
scalar prediction is outside the current model class.

Graph-wide centroid/RMS normalization and global attention do not enforce
cluster decomposition or extensive size consistency. The standalone model is
therefore scoped to bounded-size intensive-property probes or use as a global
context block above a local equivariant encoder, not interatomic potentials or
energy-conserving force fields.
