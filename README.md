# Equivariant Linear Attention

A general-purpose PyTorch layer for parity-aware 3D data.

The repository exposes **one architecture** and **one layer**:

```text
ELA
ELALayer
```

Its spatial equation is fixed:

\[
\boxed{
\text{exact global equivariant linear attention}
+
\text{exact sparse short-range residual}
}
\]

Global and local messages remain separate until an invariant,
identity-initialized router combines them. There is no dense `N x N` attention
tensor and no persistent edge hidden state.

## Install

```bash
uv sync --locked
```

Automated GitHub Actions are disabled. Validation is explicit:

```bash
scripts/check.sh fast
scripts/check.sh gpu
```

## Basic use

```python
from equivariant_attention import ELA, ELAConfig, SparseGeometry

config = ELAConfig(
    input_irreps="32x0e + 4x1o + 1x1e",
    output_irreps="1x0e + 1x1o",
    width=128,
    depth=8,
    geometry=SparseGeometry(
        cutoff=6.0,
        num_rbf=16,
    ),
)
model = ELA(config)

graph = config.geometry.prepare(
    batch,
    edge_index,  # edge_index[0] receives from edge_index[1]
)

output = model(node_irreps, positions, graph)
node_output = output["node_irreps"]
graph_output = output["graph_irreps"]
```

The public architecture choices are deliberately limited to:

```text
input_irreps
output_irreps
width
depth
geometry
features
```

Head count, local rank, hidden irreps, normalization, residual scales, tensor
closure, and chirality construction are derived internally.

## One layer

For hidden state \(h^\ell\),

\[
\bar h^\ell
=
\operatorname{EqRMSNorm}_{\rm attn}(h^\ell),
\]

\[
G^\ell
=
\operatorname{ExactGlobalELA}_{l\le2}(\bar h^\ell),
\qquad
L^\ell
=
\operatorname{ExactSparseLocal}_{l\le2}
(\bar h^\ell,x,\mathcal E).
\]

For each sector

\[
\tau\in\{0e,0o,1o,1e,2e,2o\},
\]

an invariant router computes positive branch weights:

\[
(w_{G,i}^{\tau},w_{L,i}^{\tau})
=
2\operatorname{softmax}
\left[
R_\tau
\left(
\bar h_i^{0e},
\log\operatorname{RMS}(G_i^\tau),
\log\operatorname{RMS}(L_i^\tau)
\right)
\right].
\]

The router is zero initialized, so initially

\[
w_G^\tau=w_L^\tau=1,
\qquad
M_i^\tau=G_i^\tau+L_i^\tau.
\]

Then

\[
\begin{aligned}
M^\ell &= \operatorname{Fuse}(G^\ell,L^\ell),\\
\widetilde h^\ell
&=h^\ell+
\operatorname{AttnResidual}
\left[
\operatorname{ParityUpdate}(M^\ell)
+
\operatorname{TPClosure}_{l\le2}
\right],\\
h^{\ell+1}
&=\widetilde h^\ell+
\operatorname{EqFFN}
\left(
\operatorname{EqRMSNorm}_{\rm ffn}
(\widetilde h^\ell)
\right).
\end{aligned}
\]

`ELALayer` is the layer class used at every depth. `model.layers` exposes the
stack for checkpointing, intermediate losses, or custom execution.

## Input and output irreps

The optimized path supports arbitrary multiplicities of

```text
0e  ordinary scalar
0o  pseudoscalar
1o  polar vector
1e  axial vector
2e  even symmetric-traceless tensor
2o  odd symmetric-traceless tensor
```

with `l <= 2`.

Cartesian bases:

```text
l=1: [x, y, z]
l=2: [xx, yy, xy, xz, yz], where zz = -xx - yy
```

Helpers:

```python
from equivariant_attention import (
    matrix_to_st5,
    pack_irreps,
    split_irreps,
    st5_to_matrix,
)
```

Positions remain a separate affine geometry input:

\[
x_i\mapsto Rx_i+t.
\]

They must not be packed as an ordinary `1o` feature, which transforms only as
`v -> Rv`.

## Optional functionality in the same model

Optional capabilities are allocated once with `ELAFeatures` and activated per
call through `ELAContext`. They do not create another model class.

```python
from equivariant_attention import (
    ELA,
    ELAConfig,
    ELAContext,
    ELAFeatures,
    OrderContext,
    RefinementRequest,
    SparseGeometry,
)

config = ELAConfig(
    input_irreps="32x0e + 4x1o",
    output_irreps="1x0e + 1x1o",
    width=128,
    depth=8,
    geometry=SparseGeometry(cutoff=6.0),
    features=ELAFeatures(
        condition_dim=256,
        order_dim=1,
        coordinate_refinement=True,
    ),
)
model = ELA(config)

context = ELAContext(
    condition=time_or_class_embedding,
    order=OrderContext.sequence(
        residue_rank,
        segment_id=chain_id,
        enabled=is_ordered_node,
    ),
    refinement=RefinementRequest(
        steps=4,
        max_step=0.2,
        centering="selected",
        update_mask=movable_nodes,
        graph_rebuilder=optional_neighbor_rebuilder,
    ),
)

output = model(
    node_irreps,
    positions,
    graph,
    context=context,
)
```

Each context field is independently optional:

- no `condition`: invariant DiT modulation is bypassed;
- no `order`: semantic-order PE is bypassed;
- no `refinement`: positions are not mutated.

A configured feature can therefore be switched on or off without changing the
model class. Conditioner and coordinate-head outputs are zero initialized, so
the initial function matches context-free ELA.

### Invariant condition

`condition` is an ordinary `0e` feature with shape:

```text
[D]
[1, D]
[G, D]
[N, D]
```

Even scalars receive bounded shift and scale. Non-scalar sectors receive only
invariant copy-wise scale. Vector or tensor conditions belong in
`input_irreps`, not in this tensor.

### Semantic order PE

Order coordinates are semantic labels, not current tensor row indices.
Permuting nodes requires permuting features, positions, graph references, and
order labels together.

```python
order = OrderContext.sequence(
    residue_rank,
    segment_id=chain_id,
    enabled=is_protein_atom,
)
```

The `enabled` mask supports mixed ordered/unordered systems, for example a
protein with residue order and a ligand whose atom serialization order has no
meaning.

For an ordering list of node IDs, first construct inverse ranks:

```python
rank = torch.empty_like(node_order)
rank[node_order] = torch.arange(node_order.numel())
order = OrderContext.permutation_rank(rank)
```

Grid, lattice, time, and cyclic coordinates are supported through
`OrderContext.grid(..., periods=...)`.

### Coordinate refinement

Refinement predicts a bounded polar displacement in an outer loop:

\[
x_i^{t+1}=x_i^t+\Delta x_i^t.
\]

A caller-provided graph rebuilder makes topology policy explicit. Without one,
the prepared candidate topology is reused while continuous geometry is
recomputed. For conservative force fields, derive forces from scalar energy:

\[
F_i=-\nabla_{x_i}E.
\]

## Complexity

For `N` nodes, `E` directed candidate edges, and `L` layers, with fixed widths
and ranks,

\[
T=O\left(L(N+E)\right).
\]

The branch router and optional node-level conditioning add `O(LN)` work.
Coordinate refinement with `S` outer steps costs approximately
`O(SL(N+E))`, excluding neighbor-list reconstruction.

The model is linear in node count only when `E = O(N)`. Neighbor discovery is
outside the layer and must be measured separately.

## Public API policy

The package root exports only one architecture and one architecture layer:

```text
ELA
ELALayer
```

Graph, irrep, physics-head, and context utilities remain public. Historical
implementation modules are not model-selection APIs.

## Focused validation

```bash
uv run pytest -q \
  tests/test_api_policy.py \
  tests/test_ela_context.py \
  tests/test_branch_fusion.py \
  tests/test_canonical_api.py
```

Full local gates:

```bash
scripts/check.sh fast
scripts/check.sh gpu
```

## Design documents

- `docs/CANONICAL_ELA.md`
- `docs/API_POLICY.md`
- `docs/ARCHITECTURE_DECISION_20260731.md`
- `docs/SCALING.md`

Unit tests establish shape, symmetry, initialization, and finite-gradient
contracts. They do not by themselves establish downstream accuracy or a
hardware speedup.
