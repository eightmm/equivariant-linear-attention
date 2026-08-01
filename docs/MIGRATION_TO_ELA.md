# Migrating to the single ELA API

The package exposes one model, one architecture layer, and one graph container:

```text
ELA
ELALayer
ELABatch
```

Historical checkpoint helpers remain internal utilities; they do not reintroduce
historical models as package-root choices.

## 1. Representation migration

Current ELA uses irreps for every input and output representation.

```python
model = ELA(
    input_irreps="32x0e + 4x1o",
    output_irreps="1x0e",
    width=128,
    depth=8,
    cutoff=6.0,
)
```

Scalar-only historical dimensions map directly:

```text
node_dim=32   -> input_irreps="32x0e"
output_dim=1  -> output_irreps="1x0e"
```

There is no separate scalar model or dimension-based constructor.

## 2. Configuration mapping

Historical refined configuration:

```python
EquivariantLinearAttentionConfig(
    input_irreps="32x0e + 4x1o",
    output_irreps="1x0e",
    hidden_dim=128,
    num_layers=8,
    num_heads=8,
    local_rank=4,
    local_cutoff=6.0,
    num_rbf=16,
    condition_dim=256,
)
```

Current configuration:

```python
from equivariant_attention import ELAConfig, ELAFeatures, SparseGeometry

config = ELAConfig(
    input_irreps="32x0e + 4x1o",
    output_irreps="1x0e",
    width=128,
    depth=8,
    geometry=SparseGeometry(
        cutoff=6.0,
        num_rbf=16,
    ),
    features=ELAFeatures(condition_dim=256),
)
```

Mapping:

| Historical field | Current location |
|---|---|
| `input_irreps` | `input_irreps` |
| `output_irreps` | `output_irreps` |
| `hidden_dim` | `width` |
| `num_layers` | `depth` |
| `local_cutoff` | `geometry.cutoff` |
| `num_rbf` | `geometry.num_rbf` |
| `relation_cutoffs` | `geometry.relation_cutoffs` |
| `condition_dim` | `features.condition_dim` |
| `num_heads` | derived from `width` |
| `local_rank` | derived from `width` |
| `coordinate_updates` | manual migration to outer refinement |
| `residual_dropout` | training policy, not canonical config |
| `drop_path_rate` | training policy, not canonical config |

## 3. Config migration helper

```python
from equivariant_attention.migration import canonical_config_from_advanced

config = canonical_config_from_advanced(historical_config)
```

The helper converts invariant conditioning automatically. It fails when:

- historical head/local rank differs from width-derived canonical values;
- node-role embedding cannot be represented by input irreps;
- dropout or DropPath changes the contract;
- historical per-layer coordinate mutation is requested.

Per-layer coordinate mutation is not silently mapped because current refinement
uses a different outer-loop execution and head schema.

## 4. Checkpoint migration

```python
from equivariant_attention import ELA
from equivariant_attention.migration import load_advanced_ela_state

model = ELA(config)
receipt = load_advanced_ela_state(model, historical_state_dict)
```

The canonical layer adds one branch-fusion module per depth. The helper permits
missing keys only under

```text
core.blocks.<layer>.branch_fusion.*
```

and rejects unexpected keys, missing shared keys, partial router state, and shape
mismatches.

A newly initialized router satisfies

\[
w_G=w_L=1,
\qquad
\beta=0,
\]

so shared historical weights begin at the historical `G + L` function.

## 5. Data-call migration

Historical tensor calls:

```python
output = model(node_irreps, positions, graph)
```

become:

```python
from equivariant_attention import ELABatch

batch = ELABatch(
    node_irreps=node_irreps,
    positions=positions,
    edge_index=edge_index,
)

output = model(batch)
```

For repeated fixed topology:

```python
batch = model.prepare(batch)
output = model.forward_prepared(batch)
```

## 6. Runtime conditioning

```python
batch = ELABatch(
    node_irreps=node_irreps,
    positions=positions,
    edge_index=edge_index,
    condition=condition,
)

output = model(batch)
```

Omitting `condition` bypasses trained conditioner weights entirely. No separate
conditioned model exists.

## 7. Semantic order

Historical checkpoints have no semantic-order encoder. Allocate it explicitly:

```python
config = ELAConfig(
    ...,
    features=ELAFeatures(
        condition_dim=256,
        order_dim=1,
    ),
)
```

Then attach `OrderContext` to `ELABatch.order`. Loading a historical checkpoint
into a model with a new order encoder requires an explicit initialization
receipt; the generic migration helper rejects missing non-router keys.

## 8. Coordinate refinement

Current refinement is requested through the same model and batch:

```python
from equivariant_attention import ELAFeatures, RefinementRequest

config = ELAConfig(
    ...,
    features=ELAFeatures(coordinate_refinement=True),
)
model = ELA(config)

batch = ELABatch(
    node_irreps=node_irreps,
    positions=positions,
    edge_index=edge_index,
    refinement=RefinementRequest(
        steps=4,
        max_step=0.2,
        centering="selected",
        update_mask=movable_nodes,
        graph_rebuilder=optional_rebuilder,
    ),
)

output = model(batch)
```

Historical per-layer coordinate-head weights are not automatically compatible
with this outer-loop head.

## 9. Removed public mechanisms

Do not migrate the following into `ELAConfig`:

```text
implicit_every
implicit full-state transport
attention_residual_blocks
alternative unified/legacy backbone selection
node_dim/output_dim representation aliases
```

They are not public architecture options.

## 10. Checkpoint receipt

Store:

- `ELAConfig` and `ELAFeatures`;
- state-dict hash;
- migration receipt, when applicable;
- git SHA;
- data and split revision;
- exact geometry-provider contract;
- runtime context required by the task.
