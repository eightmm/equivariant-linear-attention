# Dependency-free graph data API

ELA uses plain PyTorch tensors. PyG, DGL, and framework-specific graph objects
are not required.

The internal execution representation is a packed ragged node axis plus a
receiver-major sparse graph. The public API accepts convenient input layouts and
normalizes them to that representation.

## 1. Single graph

```python
model = ELA.scalar(node_dim=32, cutoff=6.0)
out = model(x, pos)
```

Shapes:

```text
x:   [N, D]
pos: [N, 3]
```

When neither `graph` nor `edge_index` is supplied, ELA builds exact radius
candidates from `pos` using the model cutoff.

## 2. Flat packed mini-batch

```python
out = model(
    x,
    pos,
    batch=batch,
)
```

Shapes:

```text
x:     [N_total, D]
pos:   [N_total, 3]
batch: [N_total]
```

`batch` values must be contiguous graph IDs `0 ... B-1`. Radius discovery and
global reductions never cross graph boundaries.

With supplied edges:

```python
out = model(
    x,
    pos,
    batch=batch,
    edge_index=edge_index,
)
```

`edge_index` has shape `[2,E]`. Row 0 is the receiver and row 1 is the sender.
Edges must already use indices in the packed node axis.

## 3. Padded batch

```python
out = model(
    x_padded,
    pos_padded,
    mask=node_mask,
)
```

Shapes:

```text
x_padded:   [B, M, D]
pos_padded: [B, M, 3]
node_mask:  bool [B, M]
```

Every graph must contain at least one valid node. Valid nodes are packed in
stable graph-major, row-major order. Output node tensors are restored to padded
shape.

```text
out["node_irreps"]:     [B, M, D_out]
out["positions"]:       [B, M, 3]
out["coordinate_delta"]:[B, M, 3]
out["node_mask"]:        [B, M]
out["graph_irreps"]:     [B, D_out]
```

Invalid node outputs and coordinate deltas are zero. Invalid positions are
copied from the input template.

### Padded COO

```python
out = model(
    x_padded,
    pos_padded,
    mask=node_mask,
    edge_index=edge_index,
    edge_mask=edge_mask,
)
```

Accepted edge shapes:

```text
[B, 2, E_max]
[B, E_max, 2]
```

`edge_mask` is bool `[B,E_max]`. Without `edge_mask`, an edge is valid when both
endpoints are nonnegative. Negative endpoints are padding.

### Ragged edge list

```python
out = model(
    x_padded,
    pos_padded,
    mask=node_mask,
    edge_index=[edge_0, edge_1, ...],
)
```

Each item has shape `[2,E_b]` and uses graph-local padded node indices.

### Dense adjacency

```python
out = model(
    x_padded,
    pos_padded,
    mask=node_mask,
    adjacency=adjacency,
)
```

`adjacency` is bool `[B,M,M]`. This is convenient for small dense fixtures but
not recommended for large graphs.

## 4. Plain dictionary datasets

A dataset item may be a mapping:

```python
{
    "node_irreps": x,       # aliases: x, node_features
    "pos": pos,             # alias: positions
    "edge_index": edges,    # optional
    "edge_relation_id": relation, # optional
    "condition": condition, # optional
    "order": order,         # optional
    "order_group": group,   # optional
    "order_mask": enabled,  # optional
    "target": y,            # retained for the training loop
    "sample_id": sample_id,
}
```

Collate with:

```python
loader = DataLoader(
    dataset,
    batch_size=16,
    collate_fn=ELA.collate,
)

for batch in loader:
    output = model(batch)
    loss = criterion(output["graph_irreps"], batch["target"])
```

The collator:

- concatenates variable-size node tensors;
- creates the packed `batch` vector;
- offsets per-graph edges;
- concatenates relation IDs and node-level annotations;
- stacks graph-level targets and conditions;
- preserves sample IDs.

If every sample omits `edge_index`, the collated mapping also omits it and ELA
builds radius candidates for the batch.

All samples in one collate call must either provide edges or omit them. The same
all-or-none rule applies to relation IDs, targets, conditions, and order fields.

## 5. Prepared graph hot path

For a fixed candidate graph:

```python
graph = model.prepare_graph(
    pos,
    batch=batch,
    edge_index=edge_index,
)

for _ in range(training_steps):
    output = model(x, pos, graph)
```

This avoids repeated radius discovery, COO sorting, CSR construction, and graph
layout planning. It is the preferred path for training and benchmarking.

A prepared graph is immutable runtime metadata. It can be moved with:

```python
graph = graph.to("cuda")
```

## 6. Automatic radius graph

The public convenience path uses a chunked PyTorch implementation:

```python
from equivariant_attention import radius_graph

edge_index = radius_graph(
    pos,
    batch=batch,
    cutoff=6.0,
    max_neighbors=64,
    include_self=True,
)
```

It performs exact distance tests but has quadratic arithmetic inside each graph:

\[
O\left(\sum_g N_g^2\right).
\]

Chunking bounds temporary memory; it does not change that arithmetic order.
Topology discovery is detached from coordinates, while the ELA geometry and
messages remain differentiable for the selected candidate edges.

Use this path for:

- examples and small graphs;
- one-off inference;
- tests and reference checks;
- coordinate batches where simplicity is more important than neighbor-search
  throughput.

For repeated or large workloads:

- precompute `edge_index`;
- cache `Prepared3DGraph`;
- use a cell-list or Verlet provider;
- rebuild candidates only when coordinate motion exceeds the chosen skin.

## 7. Conditions and order in batches

Conditions may be:

```text
[C]       shared by the full batch
[1,C]     shared by the full batch
[B,C]     graph-level
[N,C]     packed node-level
[B,M,C]   padded node-level
```

For `condition_dim=1`, padded scalar node conditions may also use `[B,M]`.

Semantic order may use packed `[N]`/`[N,K]` or padded `[B,M]`/`[B,M,K]` tensors.
Order masks and groups follow the node layout. Order coordinates are node labels
and must be permuted together with node tensors.

## 8. What ELA cannot infer

Automatic radius candidates express geometric proximity only. ELA cannot infer
from coordinates alone whether an edge means:

- a covalent bond;
- a protein--ligand contact;
- a mesh connection;
- a temporal transition;
- a typed relation with a special cutoff.

Supply `edge_index` and `edge_relation_id` when those semantics matter. The
automatic graph is a convenient geometric candidate graph, not a replacement
for task-specific topology.
