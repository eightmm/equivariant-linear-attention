# API policy

The repository has one public architecture and one public architecture layer:

```text
ELA
ELALayer
```

No public legacy, implicit, AttnRes, conditioned, padded, or coordinate-updating
model class may be added. Input convenience and optional capabilities remain
facades around the same `ELA` and `ELALayer` implementation.

## 1. Default user surface

The shortest scalar path is:

```python
model = ELA.scalar(node_dim=32, output_dim=1, cutoff=6.0)
out = model(x, pos)
```

Generic irreps use:

```python
model = ELA(
    input_irreps="32x0e + 4x1o",
    output_irreps="1x0e + 1x1o",
    width=128,
    depth=8,
    cutoff=6.0,
)
```

The direct constructor is a facade that builds `ELAConfig`, `SparseGeometry`,
and `ELAFeatures`. The lower-level form remains valid:

```python
model = ELA(ELAConfig(...))
```

The two construction forms are mutually exclusive.

## 2. Public input layouts

One `ELA.forward` accepts:

### Flat single graph

```python
model(x, pos)
```

### Flat packed batch

```python
model(x, pos, batch=batch)
```

### Flat packed batch with supplied candidates

```python
model(x, pos, batch=batch, edge_index=edge_index)
```

### Padded batch

```python
model(x_padded, pos_padded, mask=node_mask)
```

### Collated plain mapping

```python
loader = DataLoader(dataset, collate_fn=ELA.collate)
for batch in loader:
    output = model(batch)
```

All layouts normalize to one packed ragged node axis and one
`Prepared3DGraph`. There is no separate padded model, graph model, or dataset
adapter architecture.

## 3. Graph policy

If `graph` is supplied, it is used directly and no edge input may also be
supplied.

If `edge_index` or `adjacency` is supplied, ELA packs it to receiver-major CSR.

If neither is supplied, ELA constructs exact radius candidates using the model
cutoff. This dependency-free path is a correctness and convenience reference;
it has quadratic pair-test arithmetic within each graph. Repeated or large
workloads should call:

```python
graph = model.prepare_graph(...)
output = model(x, pos, graph)
```

Automatic geometric candidates do not infer bond, relation, mesh, or temporal
semantics. Those must be supplied explicitly when required.

## 4. Configuration policy

Canonical public architecture choices are:

```text
input_irreps
output_irreps
width
depth
geometry
features
```

The direct facade also exposes common fields such as `cutoff`, `condition_dim`,
and `order_dim`, but they compile into the same config objects.

The following remain derived or fixed:

```text
num_heads
local_rank
hidden_irreps
residual_scale_init
norm_eps
residual_dropout
drop_path_rate
implicit schedule
AttnRes block schedule
kernel backend
```

Execution kernels are never model hyperparameters and do not enter checkpoints.

## 5. Runtime optional features

The direct keyword API is preferred for one-off use:

```python
output = model(
    x,
    pos,
    graph,
    condition=condition,
    order=semantic_order,
    order_group=component_id,
    order_mask=ordered_nodes,
    refine_steps=4,
    max_coordinate_step=0.2,
    update_mask=movable_nodes,
)
```

`ELAContext` remains the reusable advanced representation:

```python
output = model(x, pos, graph, context=ELAContext(...))
```

An explicit `ELAContext` and shortcut context keywords are mutually exclusive.
If a field is absent, its path is bypassed rather than evaluated with a learned
zero-input bias.

## 6. Semantic order policy

Order PE uses node-attached semantic coordinates, never the current tensor row
index.

\[
F(PX,Px,PGP^T,Po,Pm)=PF(X,x,G,o,m).
\]

Valid examples include residue rank, polymer backbone rank, trajectory time,
grid coordinates, and stable topology coordinates. Arbitrary atom serialization
or dataloader row order is not semantic order.

## 7. Conditioning policy

Condition is an invariant `0e` tensor and may be shared, graph-level, packed
node-level, or padded node-level. Even scalars receive bounded shift and scale;
non-scalar sectors receive invariant copy-wise scale only.

Vector and tensor conditions belong in `input_irreps`.

## 8. Coordinate policy

Coordinate refinement is a bounded outer loop inside the same `ELA.forward`
entry point. It is activated per call and owns step count, maximum displacement,
update mask, centering policy, and optional graph rebuild callback.

Direct refinement is not a conservative integrator. Conservative forces use:

\[
F_i=-\nabla_{x_i}E.
\]

## 9. Dependency policy

The core package and data interface depend only on PyTorch.

Framework-specific dataset loaders may exist only as optional integrations.
Core batching, edge offsetting, masks, adjacency conversion, radius candidates,
and model execution must remain usable without PyG or DGL.

Plain mapping samples and `ELA.collate` are the canonical dataset boundary.

## 10. Kernel policy

The PyTorch implementation is the numerical reference. `torch.compile`, Triton,
CUDA, or other kernels may accelerate the prepared flat path, but must not add a
new architecture class, config option, state-dict key, or mathematical branch.

Backend selection is automatic or an execution/debug environment control. Every
custom kernel must provide a PyTorch fallback and pass forward, gradient,
equivariance, graph-isolation, dtype, and performance gates.

## 11. Root exports

The package root may expose:

- `ELA`, `ELAConfig`, `ELALayer`;
- context and geometry data types;
- dependency-free graph/data helpers;
- irrep packing helpers;
- physics heads and neighbor providers.

It must not expose a second backbone or architecture layer. The existing API
policy test guards forbidden historical names.
