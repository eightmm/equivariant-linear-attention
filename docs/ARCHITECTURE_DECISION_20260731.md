# Architecture decision: one ELA model, layer, and graph container

Date: 2026-07-31

Status: implemented on `main`.

## Decision

The repository has one public architecture:

$$
\boxed{
\text{exact global equivariant linear attention}
+
\text{exact sparse short-range local residual}
+
\text{invariant branch-aware fusion}
}
$$

The public computational objects are:

```text
ELA
ELALayer
ELABatch
```

Semantic order, invariant conditioning, and coordinate refinement are optional
capabilities of this same model and batch. They do not create wrapper models or
alternate layer classes.

Input and output representations are declared only with irreps. Scalar-only data
use scalar irreps such as `"32x0e"`; no separate `node_dim`, `output_dim`, or
scalar-model API exists.

## Why one architecture

Earlier development exposed several overlapping surfaces:

- configurable historical ELA variants;
- unified wrappers;
- conditioned wrappers;
- coordinate-refinement wrappers;
- implicit full-state spatial transport;
- block Attention Residuals;
- explicit/implicit/hybrid ablation models;
- multiple raw, padded, and prepared graph call signatures.

This made the repository look like a model menu rather than one reusable layer.
The final policy is:

```text
public backbone:        ELA
public layer:           ELALayer
public graph container: ELABatch
representation API:     input_irreps / output_irreps
```

Historical numerical modules may remain private while canonical ELA or
checkpoint migration depends on them. They are not package-root models.

## Spatial equation

The global branch provides graph-wide exact finite-feature linear attention. The
local branch provides compact-support, edge-axis-sensitive, relation-aware
short-range interaction.

The branches are not interchangeable. They remain separate until an invariant
router combines them sector by sector.

The router is identity initialized:

$$
(w_G^\tau,w_L^\tau)=(1,1),
$$

so the initial function is

$$
M^\tau=G^\tau+L^\tau.
$$

Implicit Gaussian--Taylor full-state transport is not part of the public
architecture. It overlaps the global sufficient-statistic role, smooths the full
irrep state, adds scale/schedule options, and does not reproduce compact local
semantics at fixed rank.

Block AttnRes is also not public. It adds a depth cache and block-count axis not
required by the base stability contract.

## Data and graph decision

The numerical core uses packed nodes and a receiver-major sparse graph.
`ELABatch` is the only public graph boundary:

```python
batch = ELABatch(
    node_irreps=x,
    positions=pos,
    ptr=ptr,
    edge_index=edge_index,
)

output = model(batch)
```

Flat batch IDs, padded tensors, and mapping datasets are normalized by
`ELA.batch`, `ELA.padded`, and `ELA.collate` before model execution.

If edge topology is absent, preparation constructs exact geometric radius
candidates. Small graphs use chunked dense distance tests; larger graphs use a
3D cell list plus exact filtering. Semantic topology must still be supplied
explicitly.

## Optional capabilities

`ELAFeatures` allocates optional modules:

```python
ELAFeatures(
    condition_dim=0,
    order_dim=0,
    coordinate_refinement=False,
)
```

`ELABatch` activates them with optional fields:

```python
ELABatch(
    node_irreps=x,
    positions=pos,
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

Order is a node-attached semantic coordinate, not tensor row index. It is encoded
as invariant Fourier PE. An enable mask supports mixed ordered/unordered nodes.

### Coordinate refinement

A zero-initialized `1o` head is allocated only when configured. A
`RefinementRequest` runs the same stack in an explicit outer loop and owns step
size, masking, centering, and optional graph reconstruction.

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
kernel backend
```

## Kernel decision

The PyTorch prepared path is the numerical reference. Triton is an optional
execution backend for supported receiver-major reductions and must not alter the
model, checkpoint, or equations.

Backend selection remains execution policy, not architecture configuration.

## Complexity

For `N` nodes, `E` directed candidates, and `L` layers, fixed widths and ranks
give

$$
T=O(L(N+E)).
$$

Node-linear arithmetic additionally requires `E = O(N)`. Neighbor discovery is
reported separately.

With `S` coordinate-refinement steps, ELA is evaluated approximately `S+1`
times:

$$
O((S+1)L(N+E)),
$$

excluding graph reconstruction.

## Required validation

The canonical gate covers:

- one package-root architecture, layer, and graph container;
- irreps-only representation configuration;
- context-free bypass of trained optional modules;
- semantic-order permutation equivariance;
- coordinate-refinement identity, bounds, masking, centering, and equivariance;
- exact zero-initialized `G + L` fusion;
- proper/improper O(3), translation, node permutation, graph isolation, and
  edge-order contracts;
- input and coordinate gradients, including required double backward;
- dense/cell-list radius equivalence;
- CUDA FP32/BF16 and PyTorch/Triton agreement;
- latency and memory measured separately from graph construction.

## Superseded public names

The following are not package-root architecture choices:

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

The API test fails if these names, dimension-based representation arguments, or a
scalar-only model factory return to the public surface.
