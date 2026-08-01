# API policy

The repository has one public architecture and one public architecture layer:

```text
ELA
ELALayer
```

There is no public model-selection namespace for legacy, implicit, AttnRes,
conditioned, or coordinate-updating variants.

## 1. Public surface

New applications import:

```python
from equivariant_attention import (
    ELA,
    ELAConfig,
    ELAContext,
    ELAFeatures,
    ELALayer,
    OrderContext,
    RefinementRequest,
    SparseGeometry,
    pack_irreps,
    split_irreps,
)
```

The package root may also expose graph, irrep, neighbor-provider, and physics-head
utilities. It must not expose a second backbone or architecture layer.

`tests/test_api_policy.py` fails when any of the following names return to the
package root:

```text
CanonicalEquivariantLinearAttention
ConditionedELA
ELACoordinateRefiner
EquivariantAttention
EquivariantAttentionResiduals
EquivariantLinearAttention
ImplicitGaussianSpatialKernel
SpatialOperatorAblationModel
UnifiedEquivariantAttention
UnifiedEquivariantLayer
```

## 2. Configuration policy

```python
ELAConfig(
    input_irreps="...",
    output_irreps="...",
    width=128,
    depth=8,
    geometry=SparseGeometry(cutoff=6.0),
    features=ELAFeatures(...),
)
```

The core public choices are:

```text
input_irreps
output_irreps
width
depth
geometry
features
```

The following stay derived or fixed:

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
```

`ELAFeatures` allocates optional capability modules:

```python
ELAFeatures(
    condition_dim=0,
    order_dim=0,
    coordinate_refinement=False,
)
```

These values do not choose another architecture. The spatial equation and layer
class remain unchanged.

## 3. Runtime feature switching

`ELAContext` activates optional functionality for one call:

```python
output = model(
    node_irreps,
    positions,
    graph,
    context=ELAContext(
        condition=condition,
        order=order,
        refinement=refinement,
    ),
)
```

Each field is optional. If a field is absent, the corresponding path is bypassed
rather than evaluated with a learned zero-input bias.

This permits one trained model to run with or without a configured context while
keeping the architecture class fixed.

## 4. Semantic order policy

Order PE is based on node-attached semantic coordinates, never the current row
index.

The permutation contract is

\[
F(PX,Px,PGP^T,Po,Pm)=PF(X,x,G,o,m).
\]

Examples of valid order coordinates:

- protein residue rank within a chain;
- polymer backbone rank;
- trajectory time;
- grid or lattice coordinates;
- stable topology coordinates.

An arbitrary ligand atom serialization or dataloader row order is not a valid
semantic order.

`OrderContext.enabled` selects ordered node types inside mixed systems. Disabled
nodes do not contribute to order statistics and receive no order PE.

## 5. Conditioning policy

The invariant condition is an ordinary `0e` tensor. It may be shared,
graph-level, or node-level.

Even scalars receive bounded shift and scale. Non-scalar sectors receive only
invariant copy-wise scale. Vector or tensor conditions belong in
`input_irreps`.

Conditioner outputs are zero initialized. Omitting `ELAContext.condition`
bypasses the conditioner entirely.

## 6. Coordinate policy

Coordinate refinement is an outer execution loop inside the same `ELA.forward`
entry point, activated by `ELAContext.refinement`.

`RefinementRequest` owns:

- outer step count;
- maximum displacement per step;
- update mask;
- centroid policy;
- optional graph rebuild callback.

The canonical `ELALayer` itself remains a state-propagation layer and does not
contain a second coordinate-updating layer class.

For conservative force fields, compute

\[
F_i=-\nabla_{x_i}E
\]

from a scalar energy rather than interpreting direct refinement as conservative
dynamics.

## 7. Geometry and complexity policy

`SparseGeometry` owns cutoff, radial basis count, and relation-specific cutoff
narrowing. Neighbor discovery remains a provider concern.

For a supplied candidate graph,

\[
T=O(L(N+E))
\]

at fixed widths and ranks. A node-linear claim additionally requires
`E = O(N)`. Neighbor discovery or rebuild cost must be reported separately.

## 8. Internal implementation policy

Historical numerical references may remain as private implementation modules
when canonical ELA depends on them or tracked provenance requires them. They are
not exported from the package root and must not be documented as selectable
architectures.

New mathematical mechanisms enter ELA only by improving the one layer equation
without adding another public backbone class. A mechanism must satisfy:

- clear role not already covered by global or local branches;
- O(3), translation, permutation, batching, and gradient contracts;
- stable downstream value across paired seeds;
- acceptable latency and memory;
- no increase in public architecture count.

## 9. Removal policy

A duplicate wrapper or namespace should be deleted when its behavior is
available through `ELAFeatures` and `ELAContext`.

A historical internal implementation may be deleted after:

1. canonical ELA no longer imports it;
2. tracked checkpoint migration is preserved or intentionally retired;
3. numerical reference coverage exists elsewhere;
4. repository-wide tests pass after removal.
