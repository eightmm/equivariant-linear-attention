# Dependency-free graph and batch API

ELA uses plain PyTorch tensors. PyG, DGL, and framework-specific graph objects
are not required.

The one public graph container is `ELABatch`. Its canonical representation is a
packed ragged node axis plus optional receiver/sender COO:

```text
node_irreps:      [N_total, D]
positions:        [N_total, 3]
ptr:              [B + 1]
edge_index:       [2, E] or None
edge_relation_id: [E] or None
```

The model itself accepts only this container:

```python
output = model(batch)
```

## 1. Single graph

```python
from equivariant_attention import ELA, ELABatch

model = ELA(
    input_irreps="16x0e",
    output_irreps="1x0e",
    cutoff=6.0,
)

batch = ELABatch(
    node_irreps=x,
    positions=pos,
)

output = model(batch)
```

If `ptr` is omitted, the batch contains one graph. If `edge_index` is omitted,
ELA constructs exact geometric radius candidates.

## 2. Packed mini-batch

```python
batch = ELABatch(
    node_irreps=x,
    positions=pos,
    ptr=torch.tensor([0, n0, n0 + n1, n0 + n1 + n2]),
    edge_index=edge_index,
)
```

`ptr[g]:ptr[g+1]` is the packed node interval of graph `g`.
Every `ELABatch` contains at least one node, and packed batches do not admit
empty graph segments.

Existing graph-major batch IDs can be normalized through:

```python
batch = ELA.batch(
    x,
    pos,
    batch=batch_index,
    edge_index=edge_index,
)
```

Batch IDs must be contiguous and graph-major. `ptr` is the stored canonical
membership representation.

## 3. Edge convention

```text
edge_index[0] = receiver
edge_index[1] = sender
```

Edges may not cross `ptr` graph boundaries. Relation IDs have shape `[E]`.

```python
batch = ELABatch(
    node_irreps=x,
    positions=pos,
    ptr=ptr,
    edge_index=edge_index,
    edge_relation_id=edge_type,
)
```

Typed relations require relation capacity in the model configuration.

## 4. Padded source tensors

Padded tensors are converted once at the data boundary:

```python
batch = ELA.padded(
    x_padded,       # [B, M, D]
    pos_padded,     # [B, M, 3]
    mask=node_mask, # bool [B, M]
)
```

The numerical core receives only valid packed nodes. Restore a packed node output
when required:

```python
padded_output = batch.restore_nodes(output["node"])
```

The lower-level constructor `ELABatch.from_padded` additionally accepts padded
COO, ragged per-graph COO, or boolean adjacency. These layouts are ingestion
formats, not separate model modes.

## 5. Plain dictionary datasets

A sample may be a normal mapping:

```python
{
    "x": x,
    "pos": pos,
    "edge_index": edge_index, # optional
    "edge_type": relation,    # optional
    "condition": condition,   # optional
    "order": order,           # optional
    "target": y,              # aliases: y, label, labels
    "id": sample_id,
}
```

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    pin_memory=True,
    collate_fn=ELA.collate,
)

for batch in loader:
    batch = batch.to("cuda", dtype=torch.bfloat16, non_blocking=True)
    output = model(batch)
```

The collator:

- concatenates variable-size nodes;
- builds `ptr`;
- offsets graph-local edge indices;
- concatenates relation IDs and node-level annotations;
- stacks graph-level targets or conditions;
- preserves sample IDs.

All samples in one collate call must consistently provide or omit each optional
field whose shape must be aligned across the batch.

## 6. Device and dtype transfer

```python
batch = batch.to("cuda")
```

moves tensors without changing floating dtypes.

```python
batch = batch.to("cuda", dtype=torch.bfloat16)
```

moves node features, targets, and conditions to BF16 while retaining FP32
geometry by default.

```python
batch = batch.to(
    "cuda",
    dtype=torch.float32,
    geometry_dtype=torch.float64,
)
```

controls representation and coordinate precision independently.

Integer indices, relation IDs, groups, and masks retain their semantic dtypes.
`batch.pin_memory()` is available for CPU DataLoader output.

## 7. Prepared execution

For fixed topology:

```python
batch = model.prepare(batch)

for _ in range(training_steps):
    output = model.forward_prepared(batch)
```

Preparation includes radius discovery when needed, edge validation, receiver
sorting, CSR construction, and graph-layout planning. The prepared graph is a
private execution cache inside `ELABatch`.

Device transfer preserves compatible explicit prepared metadata. If topology or
membership changes, construct or prepare a new batch.

## 8. Automatic radius candidates

The built-in builder is exact:

- small graphs use chunked all-pairs distance tests;
- larger graphs use a 3D cell list and exact final distance filtering.

Both paths use float64 geometry for float64 coordinates and float32 geometry
otherwise. This makes float16/bfloat16 topology independent of the dense/cell
dispatch threshold.

An optional `max_neighbors=k` keeps the nearest distance shells. A tie at the
k-th distance is retained in full, so the realized degree can exceed `k` only
for an exact boundary tie. This avoids an index-based choice that would violate
node-permutation consistency.

It never connects different `ptr` segments. Under bounded spatial density and
fixed cutoff, the cell-list path has expected `O(N+E)` work. Its worst case can
still be quadratic.

Automatic candidates do not infer:

- bonds;
- mesh connectivity;
- temporal transitions;
- periodic minimum-image relations;
- semantic edge types.

Supply explicit edges when those meanings matter.

## 9. Conditions and semantic order

Conditions may be shared, graph-level, or packed node-level according to the
configured condition width.

`OrderContext` carries node-attached semantic coordinates, optional group IDs,
periods, and an enable mask. Order labels must follow node permutations. Tensor
row order is never interpreted as semantic order automatically.

## 10. Output and targets

The model ignores training metadata stored in the batch. Targets remain
available as:

```python
batch.target
```

Model output uses packed node tensors:

```text
output["node"]       packed node output
output["graph"]      graph-wise mean
output["graph_sum"]  graph-wise sum
output["pos"]        final positions
output["delta"]      coordinate displacement
```

Structured sectors are recovered from either packed readout with
`model.split_output(...)`. For example, a model with
`output_irreps="1x1o"` returns a flow-matching velocity as
`model.split_output(output["node"])["1o"].squeeze(-2)`, with shape `[N,3]`.
This node `1o` value is independent of `output["delta"]`: the latter is only the
optional coordinate-refinement displacement and is zero when no refinement
request is present.
