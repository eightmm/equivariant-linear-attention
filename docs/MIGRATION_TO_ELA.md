# Migrating to the single ELA API

The package now exposes one model and one architecture layer:

```text
ELA
ELALayer
```

Historical checkpoint helpers remain internal utilities; they do not reintroduce
historical models as public package-root choices.

## 1. Configuration mapping

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
from equivariant_attention import (
    ELAConfig,
    ELAFeatures,
    SparseGeometry,
)

config = ELAConfig(
    input_irreps="32x0e + 4x1o",
    output_irreps="1x0e",
    width=128,
    depth=8,
    geometry=SparseGeometry(
        cutoff=6.0,
        num_rbf=16,
    ),
    features=ELAFeatures(
        condition_dim=256,
    ),
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
| `coordinate_updates` | manual migration to `features.coordinate_refinement` plus `ELAContext.refinement` |
| `residual_dropout` | training policy, not canonical config |
| `drop_path_rate` | training policy, not canonical config |

## 2. Config migration helper

```python
from equivariant_attention.migration import canonical_config_from_advanced

config = canonical_config_from_advanced(historical_config)
```

The helper converts invariant conditioning automatically. It fails when:

- historical head/local rank differs from width-derived canonical values;
- node-role embedding cannot be represented by input irreps;
- dropout or DropPath changes the architecture/training contract;
- historical per-layer coordinate mutation is requested.

Per-layer coordinate mutation is not silently mapped because current refinement
uses a different outer-loop execution and head schema.

## 3. Checkpoint migration

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

Conditioner parameters are schema-compatible when the historical and current
condition dimensions match.

## 4. Runtime conditioning

```python
from equivariant_attention import ELAContext

output = model(
    node_irreps,
    positions,
    graph,
    context=ELAContext(condition=condition),
)
```

Omitting `context` or `condition` bypasses trained conditioner weights entirely.
No separate conditioned model exists.

## 5. Semantic order

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

Then provide `OrderContext` at runtime. Loading a checkpoint into a model with a
new order encoder requires an explicit initialization receipt; the generic
historical migration helper intentionally rejects missing non-router keys.

## 6. Coordinate refinement

Current refinement is requested through the same model:

```python
from equivariant_attention import (
    ELAContext,
    ELAFeatures,
    RefinementRequest,
)

config = ELAConfig(
    ...,
    features=ELAFeatures(coordinate_refinement=True),
)
model = ELA(config)

output = model(
    node_irreps,
    positions,
    graph,
    context=ELAContext(
        refinement=RefinementRequest(
            steps=4,
            max_step=0.2,
            centering="selected",
            update_mask=movable_nodes,
            graph_rebuilder=optional_rebuilder,
        )
    ),
)
```

Historical per-layer coordinate-head weights are not automatically compatible
with this outer-loop head.

## 7. Removed public mechanisms

Do not migrate the following into `ELAConfig`:

```text
implicit_every
implicit full-state transport
attention_residual_blocks
alternative unified/legacy backbone selection
```

They are not public architecture options.

## 8. Checkpoint receipt

Store:

- `ELAConfig` and `ELAFeatures`;
- state-dict hash;
- migration receipt, when applicable;
- git SHA;
- data and split revision;
- exact geometry-provider contract;
- runtime context contract required by the task.
