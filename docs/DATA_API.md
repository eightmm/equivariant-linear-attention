# Data API

ELA uses one public data object: `ELAGraph`.

## Single graph

```python
import torch
from equivariant_linear_attention import ELAGraph

graph = ELAGraph(
    x=torch.randn(24, 32),
    pos=torch.randn(24, 3),
)
```

Omitting `batch` means all nodes belong to one graph. Omitting `edge_index` means
the model will construct a radius graph from `pos` and its configured cutoff.

## Explicit edges

```python
graph = ELAGraph(
    x=x,
    pos=pos,
    edge_index=edge_index,
)
```

Public edges use source-to-target order:

```text
edge_index[0] = sender
edge_index[1] = receiver
```

Edges must stay inside graph boundaries and, when `group` is supplied, inside
interaction-component boundaries.

## Mini-batch

A packed mini-batch is still an `ELAGraph`:

```python
graph = ELAGraph(
    x=x,
    pos=pos,
    batch=batch_index,
    edge_index=edge_index,
)
```

`batch` must be graph-major, nondecreasing, and contiguous from zero. For a
normal dataset, use the built-in collator instead of constructing packed indices
by hand:

```python
loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=ELAGraph.collate,
)
```

Each dataset item must contain exactly one graph.

## Edge relations

Declare the number of relation types on the model and provide matching IDs on the
graph:

```python
model = ELA(
    "32x0e",
    "1x0e",
    edge_types=3,
)

graph = ELAGraph(
    x=x,
    pos=pos,
    edge_index=edge_index,
    edge_type=edge_type,
)
```

`edge_type` has shape `[E]` and values in `[0, edge_types)`. Relation-conditioned
radial scores and value gates use these IDs. Multiple semantic relation types
require explicit edges because geometry alone cannot infer their meaning.

## Disconnected interaction components

`group` separates interaction components inside one sample:

```python
graph = ELAGraph(
    x=x,
    pos=pos,
    batch=batch_index,
    group=component_index,
)
```

Graph-level readouts still use `batch`, while global interaction and automatic
edges are restricted by `(batch, group)`. This is useful for fragment additivity
and for preventing unrelated components from communicating.

## Conditions and semantic order

Invariant conditions are attached directly to the graph:

```python
graph = ELAGraph(
    x=x,
    pos=pos,
    condition=time_or_context,
)
```

A graph-level condition has shape `[B, C]`; a node-level condition has shape
`[N, C]`. The model must be declared with the matching `condition_dim`.

Semantic order uses `OrderContext` from the advanced namespace:

```python
from equivariant_linear_attention.advanced import OrderContext

graph = ELAGraph(
    x=x,
    pos=pos,
    order=OrderContext(coordinates=order_coordinates),
)
```

## Coordinate-update mask

A coordinate-updating model updates every node unless a mask is supplied:

```python
graph = ELAGraph(
    x=x,
    pos=pos,
    update_mask=movable_nodes,
)
out = moving_model(graph)
```

The mask is boolean with shape `[N]`. The result keeps the same topology and
metadata fields and places final geometry in `out.pos`.

## Targets and IDs

Targets and sample IDs can travel with the graph:

```python
graph = ELAGraph(
    x=x,
    pos=pos,
    y=target,
    ids=(sample_id,),
)
```

After collation, `y` is stacked graph-wise and `ids` contains one item per graph.

## Device transfer

```python
graph = graph.to("cuda", non_blocking=True)
out = model(graph)
```

Floating node values follow the requested dtype. Geometry remains `float32` by
default, or `float64` when explicitly requested, so lower-precision model compute
does not silently alter radius membership.
