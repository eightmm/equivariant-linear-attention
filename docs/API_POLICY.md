# ELA public API policy

The repository exposes one public architecture, one architecture layer, and one
graph container:

```text
ELA
ELALayer
ELABatch
```

No public legacy, implicit, AttnRes, conditioned, padded, scalar-only, or
coordinate-updating model class may be added. Optional capabilities remain
facades around the same `ELA` and `ELALayer` implementation.

## 1. Representation policy

Input and output representations are declared only with irreps.

```python
model = ELA(
    input_irreps="32x0e + 4x1o",
    output_irreps="1x0e + 1x1o",
    width=128,
    depth=8,
    cutoff=6.0,
)
```

Scalar-only data use scalar irreps:

```text
32 scalar input channels  -> "32x0e"
1 scalar output channel   -> "1x0e"
```

The public model must not introduce parallel representation arguments such as
`node_dim`, `output_dim`, or a separate scalar model factory. One representation
language prevents ambiguous checkpoints and keeps scalar, vector, and tensor
models on the same API.

The lower-level configuration form remains valid:

```python
model = ELA(ELAConfig(...))
```

A config and direct constructor fields are mutually exclusive.

Task names do not create additional model classes. A flow-matching velocity,
coordinate score field, or learned coordinate displacement is a node-wise `1o`
output and is read from `output["node"]`. `output["delta"]` has a narrower
meaning: it is the bounded coordinate-refinement residual and is not a generic
vector prediction.

## 2. Model input policy

`ELA.forward` accepts one `ELABatch`:

```python
batch = ELABatch(
    node_irreps=x,
    positions=pos,
    ptr=ptr,
    edge_index=edge_index,
)

output = model(batch)
```

Convenience constructors normalize external layouts before model execution:

```python
batch = ELA.batch(x, pos, batch=batch_index, edge_index=edge_index)
batch = ELA.padded(x_padded, pos_padded, mask=node_mask)
batch = ELA.collate(samples)
```

The numerical core always receives packed nodes and one receiver-major sparse
execution graph. There is no separate padded model, dense-adjacency model, or
framework-specific graph architecture.

## 3. Graph policy

`ELABatch.edge_index` uses receiver/sender COO:

```text
edge_index[0] = receiver
edge_index[1] = sender
```

If edges are supplied, ELA packs them to receiver-major CSR. If edges are absent,
ELA constructs exact radius candidates using the model cutoff.

Automatic candidates represent geometric proximity only. Bond, mesh, temporal,
periodic, or typed-relation semantics must be supplied explicitly.

The public graph boundary is `ELABatch`. Internal types such as
`Prepared3DGraph`, `PackedNeighborGraph`, graph-layout schedules, and radius
builders are not package-root API.

## 4. Prepared execution policy

For fixed topology:

```python
batch = model.prepare(batch)
output = model.forward_prepared(batch)
```

Preparation may include radius discovery, COO validation, receiver sorting, CSR
construction, and graph-layout planning. Performance claims must distinguish
this work from the prepared layer/stack execution.

Prepared metadata is an execution cache, not a second graph container.

## 5. Configuration policy

Canonical public architecture choices are:

```text
input_irreps
output_irreps
width
depth
geometry
features
```

The direct constructor may expose common geometry and feature fields such as
`cutoff`, `num_rbf`, `condition_dim`, and `order_dim`; they compile into the same
`ELAConfig` objects.

The following remain derived, fixed, or execution-only:

```text
num_heads
local_rank
hidden_irreps
residual scale
normalization epsilon
tensor-closure paths
chirality construction
kernel backend
Triton launch policy
```

Execution kernels never enter checkpoint schemas.

## 6. Runtime optional features

Condition, semantic order, and coordinate refinement are optional fields of the
same `ELABatch`:

```python
batch = ELABatch(
    node_irreps=x,
    positions=pos,
    condition=condition,
    order=order,
    refinement=refinement,
)
```

If a field is absent, its path is bypassed rather than evaluated with a learned
zero-input bias.

## 7. Semantic-order policy

Order PE uses node-attached semantic coordinates, never the current tensor row
index.

\[
F(PX,Px,PGP^T,Po,Pm)=PF(X,x,G,o,m).
\]

Valid examples include residue rank, polymer-backbone rank, trajectory time,
grid coordinates, and stable topology coordinates. Arbitrary atom serialization
or DataLoader row order is not semantic order.

## 8. Conditioning policy

Condition is invariant `0e` information and may be shared, graph-level, or
node-level. Even scalars receive bounded affine modulation; non-scalar sectors
receive invariant copy-wise scale only.

Vector and tensor conditions belong in `input_irreps`.

## 9. Coordinate policy

Coordinate refinement is a bounded outer loop activated by a
`RefinementRequest` in the batch. It owns step count, maximum displacement,
update mask, centering policy, and optional graph rebuild callback.

Direct refinement is not a conservative integrator. Conservative forces use

\[
F_i=-\nabla_{x_i}E.
\]

Likewise, a direct `1o` vector output is not automatically conservative. It is
appropriate for velocity/score/displacement supervision; force conservation
requires the scalar-potential construction above.

## 10. Dependency policy

The core package and data interface depend only on PyTorch. Framework-specific
dataset loaders may exist only as optional integrations.

Core graph packing, batching, edge offsetting, masks, radius candidates, and
model execution must remain usable without PyG or DGL.

Plain mapping samples plus `ELA.collate` are the canonical dataset boundary.

## 11. Kernel policy

The PyTorch implementation is the numerical reference. `torch.compile`, Triton,
CUDA, or other kernels may accelerate the prepared path but must not add:

- another architecture class;
- a model hyperparameter;
- a state-dict key;
- a mathematical branch;
- a different output contract.

Backend selection is automatic or controlled by an execution/debug environment
variable. Every custom kernel requires a PyTorch fallback and forward, gradient,
equivariance, graph-isolation, dtype, and performance tests.

## 12. Root exports

The package root may expose:

- `ELA`, `ELAConfig`, `ELAFeatures`, `ELALayer`;
- `ELABatch` and optional context/geometry data types;
- irrep construction helpers;
- task-level physics heads.

It must not expose a second backbone, a second architecture layer, or internal
packed graph/runtime types. The API-policy test guards historical names and the
irreps-only representation contract.
