# Architecture decision: one ELA model and one ELA layer

Date: 2026-07-31

Status: implemented on `main`.

## Decision

The repository has one public architecture:

\[
\boxed{
\text{exact global equivariant linear attention}
+
\text{exact sparse short-range local residual}
+
\text{invariant branch-aware fusion}
}
\]

The public model and layer are:

```text
ELA
ELALayer
```

Semantic order, invariant conditioning, and coordinate refinement are optional
capabilities of this same model. They do not create wrapper models or alternate
layer classes.

## Why one architecture

Earlier development exposed several overlapping surfaces:

- configurable historical ELA variants;
- unified wrappers;
- conditioned wrappers;
- coordinate-refinement wrappers;
- implicit full-state spatial transport;
- block Attention Residuals;
- explicit/implicit/hybrid ablation models.

This made the repository look like a model menu rather than one reusable layer.
It also encouraged configuration search over mechanisms whose mathematical
roles were different.

The final policy is:

```text
public backbone: ELA
public layer:    ELALayer
optional input: ELAContext
```

Historical numerical reference modules may remain internal while canonical ELA
or checkpoint migration depends on them. They are not package-root models.

## Spatial equation

The global branch provides graph-wide exact finite-feature linear attention. The
local branch provides compact-support, edge-axis-sensitive, relation-aware
short-range interaction.

The branches are not interchangeable. They remain separate until an invariant
router combines them sector by sector.

The router is identity initialized:

\[
(w_G^\tau,w_L^\tau)=(1,1),
\]

so the initial function is the established additive model

\[
M^\tau=G^\tau+L^\tau.
\]

Implicit Gaussian--Taylor full-state transport is not part of the public
architecture. It overlaps the global sufficient-statistic role, smooths full
irrep state, adds scale/schedule options, and does not reproduce compact local
semantics at fixed rank.

Block AttnRes is also not part of the public architecture. It introduces a depth
cache and block-count axis that is not required by the base stability contract.

## Optional capabilities

`ELAFeatures` allocates optional modules:

```python
ELAFeatures(
    condition_dim=0,
    order_dim=0,
    coordinate_refinement=False,
)
```

`ELAContext` activates them for one call:

```python
ELAContext(
    condition=None,
    order=None,
    refinement=None,
)
```

Omitting a field bypasses the corresponding path entirely, including after the
optional module has trained.

### Invariant condition

A `0e` condition modulates the same `ELALayer` through bounded DiT-style shift,
scale, and residual gates. Vector and tensor conditions remain input irreps.

### Semantic order

Order is a node-attached semantic coordinate, not the tensor row index. It is
encoded as invariant Fourier PE and supplied through the same condition path.
An enable mask supports mixed ordered/unordered node types.

### Coordinate refinement

A zero-initialized `1o` head is allocated only when requested. A
`RefinementRequest` runs the same ELA stack in an explicit outer loop and owns
step size, masking, centering, and optional graph reconstruction.

No second coordinate-updating backbone or layer class exists.

## Public options

The public architecture config contains:

```text
input_irreps
output_irreps
width
depth
geometry
features
```

Derived or fixed:

```text
num_heads
local_rank
hidden irreps
normalization
residual scales
tensor closure
chirality construction
```

Runtime context contains task inputs and execution requests, not alternate
architecture selection.

## Complexity

For `N` nodes, `E` directed candidates, and `L` layers, fixed widths and ranks
give

\[
T=O(L(N+E)).
\]

Node-linear arithmetic additionally requires `E = O(N)`. Neighbor discovery is
outside this bound.

With `S` coordinate-refinement steps, ELA is evaluated approximately `S+1`
times:

\[
O((S+1)L(N+E)),
\]

excluding graph rebuild.

## Required validation

The canonical gate must cover:

- package root exposes only `ELA` and `ELALayer` as backbone/layer;
- context-free forward bypasses trained conditioner weights;
- semantic-order permutation equivariance;
- disabled order labels have no effect;
- condition and order projections receive gradients;
- coordinate refinement is identity at initialization;
- activated displacement is bounded, masked, and equivariant;
- global/local fusion starts at exact `G + L`;
- proper/improper O(3), translation, node permutation, graph isolation, and
  edge-order contracts;
- input and coordinate gradients, including double backward where required;
- CUDA BF16 forward/backward;
- latency and memory measured without overstating neighbor costs.

## Superseded public names

The following are no longer package-root architecture choices:

```text
CanonicalEquivariantLinearAttention
ConditionedELA
ELACoordinateRefiner
EquivariantAttention
EquivariantAttentionResiduals
EquivariantLinearAttention
SpatialOperatorAblationModel
UnifiedEquivariantAttention
UnifiedEquivariantLayer
```

The API test fails if these names return to the root namespace.
