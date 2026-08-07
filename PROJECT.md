# Project contract

ELA exposes exactly one public workflow:

```text
ELAGraph -> ELA -> ELAGraph
```

The architecture is completely edge-free. `ELAGraph` contains node irreps,
coordinates, sample IDs, optional interaction-component IDs, optional invariant
context, targets, and an optional coordinate mask. It contains no topology.

The canonical layer is fixed:

```text
adaptive exact relative moments through order four
+ self-adjoint content/Mercer/atlas relation
+ three-term invariantly orthogonal Krylov filter
+ parity-complete l<=2 closure with transient l=3/l=4 carriers
+ equivariant feed-forward update
```

Coordinate refinement uses an SPD atlas metric and an `SE(3)` quotient
translation/rotation/shape decomposition.

## Non-negotiable invariants

- no explicit or inferred edges;
- no dense `N x N` matrix;
- no pair or tuple hidden state;
- no sparse gather/scatter execution path;
- no topology cache or checkpoint migration layer;
- O(3), translation, and permutation equivariance;
- exact interaction-component isolation;
- node-linear memory at fixed representation order;
- one PyTorch implementation without PyG, DGL, Triton, or custom kernels.
