# Migrating to canonical ELA

Canonical ELA reduces architecture configuration while keeping historical model
classes available for reproducibility.

## 1. Configuration mapping

Previous refined configuration:

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
)
```

Canonical configuration:

```python
ELAConfig(
    input_irreps="32x0e + 4x1o",
    output_irreps="1x0e",
    width=128,
    depth=8,
    geometry=SparseGeometry(
        cutoff=6.0,
        num_rbf=16,
    ),
)
```

Mapping:

| Advanced field | Canonical location |
|---|---|
| `input_irreps` | `input_irreps` |
| `output_irreps` | `output_irreps` |
| `hidden_dim` | `width` |
| `num_layers` | `depth` |
| `local_cutoff` | `geometry.cutoff` |
| `num_rbf` | `geometry.num_rbf` |
| `relation_cutoffs` | `geometry.relation_cutoffs` |
| `num_heads` | derived from `width` |
| `local_rank` | derived from `width` |
| `condition_dim` | `ConditionedELA` wrapper |
| `coordinate_updates` | `ELACoordinateRefiner` wrapper |
| `residual_dropout` | training policy or advanced compatibility model |
| `drop_path_rate` | training policy or advanced compatibility model |

## 2. Config migration helper

```python
from equivariant_attention.migration import canonical_config_from_advanced

minimal = canonical_config_from_advanced(advanced_config)
```

The conversion fails when an advanced head/local rank differs from the canonical
width-derived values. It also fails when condition, coordinate mutation,
node-role embedding, dropout, or DropPath must be moved to a wrapper. It never
silently changes parameter shapes.

## 3. Checkpoint migration

Canonical ELA adds one branch-fusion module per layer. Existing refined ELA
weights are otherwise schema-compatible when configuration shapes match.

```python
from equivariant_attention import ELA
from equivariant_attention.migration import load_advanced_ela_state

model = ELA(minimal)
receipt = load_advanced_ela_state(model, old_state_dict)
```

The helper allows missing keys only under

```text
core.blocks.<layer>.branch_fusion.*
```

and rejects every unexpected key or missing common key. The new router remains
at its identity initialization:

\[
w_G=w_L=1,
\qquad
\beta=0.
\]

Therefore shared old weights reproduce the old `G + L` function before the
router learns.

Store the migration receipt with the new checkpoint.

## 4. Conditioning migration

```python
from equivariant_attention.conditioning import (
    ConditionedELA,
    InvariantConditioningConfig,
)

model = ConditionedELA(
    minimal,
    InvariantConditioningConfig(condition_dim=256),
)
```

Conditioner output projections are zero initialized. Load shared base weights
with `strict=False` only when the missing keys are all conditioner parameters;
record that receipt explicitly.

## 5. Coordinate migration

Previous in-layer coordinate mutation:

```python
coordinate_updates=True
max_coordinate_step=0.2
```

Canonical composition:

```python
from equivariant_attention import (
    CoordinateRefinementConfig,
    ELACoordinateRefiner,
)

model = ELACoordinateRefiner(
    backbone,
    CoordinateRefinementConfig(
        steps=4,
        max_step=0.2,
        centering="selected",
    ),
)
```

The caller now owns neighbor reuse or rebuild through `graph_rebuilder`.

## 6. Implicit and AttnRes migration

Do not copy these settings into `ELAConfig`:

```text
implicit_every
implicit scales/order/normalization
attention_residual_blocks
```

Use the experimental namespace only for a preregistered comparison:

```python
from equivariant_attention.experimental import ...
```

The tracked architecture decision keeps exact sparse local geometry canonical.

## 7. Historical checkpoints

Historical `EquivariantAttention`, `UnifiedEquivariantAttention`, and advanced
`EquivariantLinearAttention` checkpoints should continue to load into their
original classes. Do not rewrite archival receipts merely to use the new API.

Canonical ELA has a distinct model identity and should be saved with:

- canonical config;
- state-dict hash;
- migration receipt, when applicable;
- git SHA;
- data/split receipt;
- exact geometry-provider contract.
