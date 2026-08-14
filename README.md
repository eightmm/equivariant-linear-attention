# Equivariant Linear Attention

ELA is an entirely **edge-free, parity-complete O(3)-equivariant neural layer**
for molecules, protein structures, point clouds, and other long 3D sequences.
Its public API is deliberately singular:

```text
ELAGraph -> ELA -> ELAGraph
```

There is no neighbor search, edge list, pair representation, sparse message
passing path, topology cache, or backward-compatibility execution layer.

## Install

```bash
git clone https://github.com/eightmm/equivariant-linear-attention.git
cd equivariant-linear-attention
uv sync
```

Python 3.12 or newer is required.

## Basic use

```python
import torch
from equivariant_linear_attention import ELA, ELAGraph

model = ELA(
    "32x0e",
    "1x0e",
    width=128,
    depth=8,
)

graph = ELAGraph(
    x=torch.randn(128, 32),
    pos=torch.randn(128, 3),
    batch=torch.arange(8).repeat_interleave(16),
)

out = model(graph)
node_prediction = out.x
mean_graph_prediction = out.graph_x
sum_graph_prediction = out.graph_sum
```

`batch` defaults to one sample. `group` optionally separates independent
interaction components within a sample; no operation mixes different
`(batch, group)` components.

## Representation

Inputs and outputs are declared with real O(3) irreps:

```python
model = ELA(
    "16x0e + 4x1o + 2x2e",
    "2x0e + 1x1o + 1x2e",
)
```

The persistent carrier contains arbitrary multiplicities of

```text
0e  scalar
0o  pseudoscalar
1o  polar vector
1e  axial vector
2e  even symmetric-traceless tensor
2o  odd symmetric-traceless tensor
```

Every layer also constructs transient `3o` and `4e` coordinate moments and
contracts them back into the persistent `l <= 2` carrier.

## Geometry without pairs

For learned positive source weights `w_j`, ELA computes exact receiver-centred
moments

$$
M_i^{(k)} = \frac{\sum_j w_j(x_j-x_i)^{\otimes k}}{\sum_j w_j},
\qquad k=1,2,3,4,
$$

from graphwise raw sums. Individual pairs, triangles, and higher tuples are
never materialized. The moments provide directional, anisotropic, chiral,
third-order, and fourth-order geometry in node-linear memory.

## Self-adjoint relation operator

A layer builds one invariant positive-semidefinite relation as a convex mixture
of:

1. an all-irrep content Gram operator;
2. an isotropic Gaussian Mercer feature expansion through Cartesian order four;
3. a learned SPD manifold atlas.

The atlas predicts a node-to-chart partition `A`; its induced relation is

$$
S=A D^{-1} W A^\top.
$$

This is model-predicted soft connectivity of bounded rank, applied as
node-to-chart-to-node contractions rather than as explicit edges.

The same operator is reused at three orders,

$$
RV,\qquad R^2V,\qquad R^3V,
$$

then graph/head-wise invariant Gram-Schmidt forms a conditioned Krylov basis
before spectral mixing.

## Coordinate refinement

```python
model = ELA(
    "32x0e",
    "1x0e",
    update_positions=True,
    max_coordinate_step=0.25,
)
```

The learned atlas metric preconditions the predicted polar field as a natural
step. The update is decomposed into rigid translation, rigid rotation, and an
internal shape tangent orthogonal to the global `SE(3)` orbit. Fully movable
components remove rigid gauge modes; partially movable components can learn
pose motion relative to fixed context. `update_mask` keeps unselected nodes
exactly fixed.

## Complexity

For `N` nodes, hidden width `C`, fixed moment order four, three Krylov terms,
and `K` latent charts, a layer costs

$$
O(NC^2 + NKC)
$$

with

$$
O(NC+C^2+NK)
$$

working memory. No stored object scales as `N^2`.

## Validation

```bash
uv run scripts/check.sh
```

The suite covers proper and improper O(3) actions, translations, permutations,
component isolation, exact moments through order four, PSD/self-adjoint
relations, orthogonal Krylov construction, atlas metrics, quotient coordinate
updates, second-order autograd, mixed-irrep I/O, and graph collation.

See [the mathematical architecture](docs/ARCHITECTURE.md).

Current real-data evidence, including the ATOM3D PSR comparison and its
limitations, is recorded in [PSR results](docs/PSR_RESULTS.md).

How contact-scale locality entered the edge-free core, and what is still open,
is recorded in [the locality track](docs/LOCALITY_TRACK.md).

`local_points` enables a non-canonical pointwise local-jet branch that builds a
transient kNN support. It is off by default, it is not edge-free, and it exists
only to measure the hard-cutoff upper bound; `describe()` reports
`canonical_edge_free_path` so such a model cannot be mistaken for the canonical
one.
