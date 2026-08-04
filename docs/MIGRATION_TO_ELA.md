# Migration to the unified ELA API

The canonical public interface is now:

```python
from equivariant_linear_attention import ELA, ELAGraph

model = ELA("32x0e", "1x0e")
graph = ELAGraph(x=x, pos=pos, edge_index=edge_index, batch=batch)
out = model(graph)
```

## Imports

Replace broad root imports with:

```python
from equivariant_linear_attention import ELA, ELAGraph
```

Reproducibility configuration and utilities are under:

```python
from equivariant_linear_attention.advanced import ELAConfig, SparseGeometry
```

## Input conversion

Replace model-specific batch construction:

```python
# old: build a separate packed batch through a helper
# new:
graph = ELAGraph(
    x=node_features,
    pos=positions,
    edge_index=edge_index,
    batch=batch_index,
)
out = model(graph)
```

Public edges are source-to-target. Code that used the historical internal
receiver-sender order must flip the two rows once during migration.

## Output conversion

Replace dictionaries or dedicated output wrappers:

```python
node_prediction = out.x
graph_prediction = out.graph_x
extensive_prediction = out.graph_sum
positions = out.pos
displacement = out.delta
```

The returned object is an `ELAGraph`, not a dictionary and not a separate output
type.

## DataLoader conversion

```python
loader = DataLoader(
    dataset,
    batch_size=16,
    collate_fn=ELAGraph.collate,
)
```

Dataset samples should return one `ELAGraph` each. Padded tensors should be packed
inside the dataset or preprocessing pipeline; the model has no separate padded
execution API.

## Coordinate-update conversion

Replace runtime refinement requests and refiner objects with a model declaration:

```python
model = ELA(
    "32x0e",
    "1x0e",
    update_positions=True,
    max_coordinate_step=0.2,
)
out = model(graph)
```

Advanced multi-step updates use:

```python
config = ELAConfig(
    input_irreps="32x0e",
    output_irreps="1x0e",
    coordinate_updates=4,
    max_coordinate_step=0.2,
)
model = ELA.from_config(config)
```

`coordinate_updates=K` chooses K distinct, deterministic layer boundaries
spread across the stack, always including the final layer; K cannot exceed the
model depth. The hidden state is carried across every boundary instead of
re-embedding the original node features. The public `update_positions=True`
mode chooses K equal to the depth. Historical outer-loop coordinate-update
checkpoints cannot be mapped exactly; load the ordinary ELA weights and retrain
or calibrate the coordinate head.

## Advanced constructor conversion

Replace direct config positional construction:

```python
model = ELA.from_config(config)
```

The normal direct constructor remains preferred for new code.

## Checkpoints

Supported checkpoints contain a plain configuration dictionary and a
`state_dict`. Pickling an entire model object is not a supported interchange
format. Historical branch-router parameters cannot be represented by the
canonical fixed global-plus-local fusion. Dropping them therefore requires an
explicit lossy-conversion acknowledgement:

```python
from equivariant_linear_attention.migration import load_advanced_ela_state

receipt = load_advanced_ela_state(
    model,
    checkpoint,
    allow_drop_learned_fusion=True,
)
assert receipt.dropped_keys
```

Without the flag migration fails closed. The receipt lists each discarded key
and records that canonical parameters retained their initialization.
