# API policy

The repository contains one canonical model and several retained research or
compatibility surfaces. This document defines which APIs new code should use.

## 1. Canonical public surface

New applications should import:

```python
from equivariant_attention import (
    ELA,
    ELAConfig,
    ELALayer,
    SparseGeometry,
    pack_irreps,
    split_irreps,
)
```

Optional task wrappers:

```python
from equivariant_attention import (
    CoordinateRefinementConfig,
    ELACoordinateRefiner,
    ELARegressionModel,
)
```

Invariant conditioning is an explicit wrapper module:

```python
from equivariant_attention.conditioning import (
    ConditionedELA,
    InvariantConditioningConfig,
)
```

`ELAConfig` exposes only representation, capacity, and geometry:

```python
ELAConfig(
    input_irreps="...",
    output_irreps="...",
    width=128,
    depth=8,
    geometry=SparseGeometry(cutoff=6.0),
)
```

The public model config does not expose:

```text
num_heads
local_rank
hidden_irreps
residual_scale_init
norm_eps
implicit_every
attention_residual_blocks
condition_dim
coordinate_updates
residual_dropout
drop_path_rate
```

Those values are either derived, fixed, assigned to a wrapper, or excluded from
the canonical architecture.

## 2. Geometry policy

`SparseGeometry` owns cutoff, radial basis count, and relation-specific cutoff
narrowing. It accepts a candidate graph and packs it into `Prepared3DGraph`.

Neighbor discovery remains a provider concern. A model layer must not silently
switch between complete graphs, kNN, radius graphs, or approximate implicit
kernels.

This separation makes the complexity claim auditable:

\[
T_{\rm layer}=O(L(N+E))
\]

for a supplied graph, while provider construction is measured separately.

## 3. Advanced compatibility surface

The following classes remain at package root to avoid breaking existing scripts
and checkpoints:

```python
EquivariantLinearAttention
EquivariantLinearAttentionConfig
UnifiedEquivariantAttention
Unified3DConfig
EquivariantAttention
EquivariantAttentionConfig
```

They are not the recommended starting point for new architecture work.

Use the compatibility namespace when intent should be explicit:

```python
from equivariant_attention.legacy import (
    EquivariantAttention,
    UnifiedEquivariantAttention,
)
```

`EquivariantLinearAttentionConfig` remains the advanced route for existing
experiments needing copy dropout, DropPath, historical state schemas, or direct
access to the older conditioned configuration.

## 4. Experimental surface

Import noncanonical mechanisms through:

```python
from equivariant_attention.experimental import (
    EquivariantAttentionResiduals,
    ImplicitGaussianSpatialKernel,
    ImplicitSpatialResidual,
    SpatialOperatorAblationModel,
)
```

Experimental means:

- transformation and numerical contracts may be tested;
- tracked experiments may exist;
- the mechanism is not part of the canonical architecture;
- no downstream or efficiency superiority is implied;
- configuration is allowed to be more verbose because it is a research surface.

## 5. Coordinate policy

The canonical ELA layer never mutates coordinates. Direct coordinate refinement
is composed through `ELACoordinateRefiner`.

This keeps separate:

- state propagation;
- displacement prediction;
- update masking and centroid policy;
- neighbor-list reuse or rebuild;
- conservative force computation.

A static property model therefore does not carry refinement-specific branches or
booleans.

## 6. Conditioning policy

Invariant DiT-style conditioning is composed through `ConditionedELA`, not a
field in `ELAConfig`.

```python
model = ConditionedELA(
    base_config,
    InvariantConditioningConfig(condition_dim=256),
)
output = model(node_irreps, positions, graph, condition=time_embedding)
```

The condition is an ordinary invariant `0e` feature and may be shared,
graph-level, or node-level. Even scalars receive bounded shift and scale;
non-scalar sectors receive invariant copy-wise scale only. Attention and FFN
residual gates are separate. Conditioner projections are zero initialized, so
shared ELA weights reproduce the unconditioned function at initialization.

Vector or tensor conditions are ordinary `input_irreps` blocks, not invariant
condition vectors.

## 7. Compatibility and removal policy

Historical code is retained when at least one of the following holds:

- a tracked artifact or checkpoint depends on it;
- it is a numerical reference implementation;
- it is required for an explicit ablation;
- removing it would invalidate provenance.

Retention does not keep an option in canonical documentation.

A historical implementation may be deleted when:

1. no tracked receipt imports it;
2. a migration path exists;
3. its mathematical reference is covered elsewhere;
4. repository-wide tests pass after removal.

## 8. Naming policy

Use:

```text
ELA                         canonical model
ELALayer                    canonical reusable layer
ELAConfig                   minimal config
SparseGeometry              exact sparse geometry contract
ConditionedELA              invariant-condition wrapper
ELACoordinateRefiner        coordinate-refinement wrapper
```

Use descriptive full names for experimental mechanisms. Avoid adding new
boolean `use_*` flags to canonical config. A genuinely different mathematical
operator should be a separate experimental class until evidence justifies
integration.

## 9. Promotion policy

A mechanism may enter canonical ELA only when it satisfies all relevant gates:

- precise mathematical role not already covered by another branch;
- O(3), translation, permutation, batching, and gradient contracts;
- same-schema and paired-initialization comparisons;
- stable multi-seed downstream value;
- acceptable latency and memory on intended hardware;
- no task-family failure inconsistent with the claimed generality;
- reduced or unchanged public option count.

One-seed wins, training-set capacity, or isolated microbenchmarks are not enough.
