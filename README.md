# Equivariant Linear Attention

A domain-agnostic PyTorch layer for unordered 3D data. The canonical path is
SE(3)-equivariant, keeps explicit parity bookkeeping internally, combines exact
linear-time global attention with one receiver-normalized sparse local operator,
and exposes only the final `output_irreps` as a representation-level user
choice.

The same core can be used for molecules, proteins, protein--ligand complexes,
particles, meshes, point clouds, and other sparse 3D systems. Chemical or
biological vocabulary is not part of the layer; domain adapters provide scalar
features, roles, relations, masks, hierarchy, and task heads.

## Canonical architecture

Every block evaluates

\[
G^\ell
=
\operatorname{ExactGlobal}_{l\le2}(h^\ell),
\]

\[
S^\ell
=
\operatorname{SparseLocal}_{l\le2}(h^\ell,x,\mathcal E),
\]

\[
h^{\ell+1}
=
\operatorname{EquivariantUpdateFFN}
\left(h^\ell,G^\ell+S^\ell\right).
\]

The persistent hidden carrier is fixed automatically to

\[
C_0\times0e
\oplus H\times0o
\oplus H\times1o
\oplus H\times1e
\oplus H\times2e
\oplus H\times2o.
\]

Users do not select hidden parity, angular degree, local/global routing,
normalization, backend fallback, or layer schedule.

The canonical implementation includes:

- exact positive finite-feature global attention with one balancing cycle;
- one sparse rank-`R` local score, weight, receiver mass, and value transport;
- static receiver-centered radial multipoles corresponding to Cartesian
  `l=0,1,2` spherical-harmonic moments;
- active `0o`, `1o`, `1e`, `2e`, and `2o` routing;
- low-rank Cartesian tensor-product closure through `l<=2`;
- aggregate cross/triple-product chirality without explicit edge triplets;
- irrep-sector RMS pre-normalization and per-copy LayerScale;
- a compact C2 cutoff envelope;
- graph scale and local density context;
- no `N x N` attention tensor and no persistent edge hidden state.

At fixed channel width, head count, radial rank, and local rank,

\[
\operatorname{time}=O(L(N+E)),
\qquad
\operatorname{persistent\ state}=O(N).
\]

Neighbor discovery is intentionally outside the layer and must be costed
separately.

## Install

```bash
uv sync --locked
```

Optional evaluation dependencies:

```bash
uv sync --locked --extra qm9
uv sync --locked --extra pdbbind
```

Automated GitHub Actions are intentionally disabled for push and pull-request
events. Repository checks are run explicitly when needed:

```bash
scripts/check.sh fast
scripts/check.sh gpu
```

## Basic use

```python
import torch

from equivariant_attention import (
    Unified3DConfig,
    UnifiedEquivariantAttention,
    prepare_3d_graph,
)

config = Unified3DConfig(
    input_irreps="32x0e",
    output_irreps="1x0e",
    hidden_dim=128,
    num_layers=6,
    num_heads=8,
    local_rank=4,
    local_cutoff=6.0,
    num_rbf=16,
)
model = UnifiedEquivariantAttention(config)

# Directed candidate edges j -> i are stored as edge_index[0] = receiver,
# edge_index[1] = sender.
graph = prepare_3d_graph(
    batch,
    edge_index,
    edge_relation_id=edge_relation_id,
)

output = model(node_irreps, positions, graph)

node_output = output["node_irreps"]
graph_mean_diagnostic = output["graph_irreps"]
blocks = model.split_output(node_output)
```

`prepare_3d_graph` validates graph isolation and packs receiver-major CSR once,
outside the model hot path. The canonical model does not silently construct a
complete graph and has no runtime neighbor-provider fallback.

## Choosing output irreps

The representation-level public control is `output_irreps`.

Parity-even scalar, such as an energy or a conventional affinity head input:

```python
output_irreps="1x0e"
```

Pseudoscalar, such as signed chirality or an optical observable:

```python
output_irreps="1x0o"
```

Polar vector, such as a displacement or force-like vector:

```python
output_irreps="1x1o"
```

Axial vector:

```python
output_irreps="1x1e"
```

Even or odd symmetric-traceless tensors:

```python
output_irreps="1x2e"
output_irreps="1x2o"
```

Mixed outputs are allowed:

```python
output_irreps="2x0e + 1x0o + 3x1o + 1x1e + 1x2e"
```

The optimized canonical output currently supports `l<=2`. A request for a
higher output degree is rejected rather than silently dispatched to a slower or
semantically different backend.

## Inputs

The canonical path accepts one flattened `l<=2` irrep tensor and separate
positions:

```text
node_irreps: (N, input_layout.dim)
positions:   (N, 3)
```

`input_irreps` may contain any multiplicity of `0e`, `0o`, `1e`, `1o`, `2e`,
and `2o`. The `l=1` basis is Cartesian xyz. The compact `l=2` basis is
`[xx, yy, xy, xz, yz]`, with `zz=-xx-yy`. `pack_irreps`, `split_irreps`,
`matrix_to_st5`, and `st5_to_matrix` are provided at the package root.
Positions are geometry, not an input feature sector. `input_irreps="0"` is the
geometry-only path.

Integer node roles and edge relations are invariant metadata. Relation-specific
cutoffs may narrow one shared local domain but do not create relation-specific
equivariant kernels or persistent edge states.

## Node multipoles

For radial shell `n`, the static receiver-centered moments are

\[
P_{in}^{1o}
=
\frac{
\sum_{j\to i}a_{ijn}\hat d_{ij}
}{1+\sum_{j\to i}a_{ijn}},
\]

\[
Q_{in}^{2e}
=
\frac{
\sum_{j\to i}a_{ijn}\operatorname{ST}(\hat d_{ij})
}{1+\sum_{j\to i}a_{ijn}}.
\]

Three radial-shell polar moments generate axial, pseudoscalar, and odd-tensor
features after aggregation:

\[
A^{1e}=P_1^{1o}\times P_2^{1o},
\]

\[
\chi^{0o}=A^{1e}\cdot P_3^{1o},
\]

\[
U^{2o}=\operatorname{ST}(P_3^{1o},A^{1e}).
\]

This gives chirality-sensitive node state in `O(E)` time without materializing
neighbor triplets.

## Exact global attention

The global feature map contains positive scalar, polar, axial, even-tensor, and
odd-tensor sectors. Same-irrep contractions are parity-even. A positive constant
block bounds the complete kernel away from zero.

For each graph and head, one-cycle balancing and augmented transport are

\[
Q_{gh}=\sum_{i\in g}\Phi^Q_{ih},
\]

\[
\widetilde\Phi^K_{jh}
=
\frac{\Phi^K_{jh}}{(\Phi^K_{jh})^TQ_{g(j)h}},
\]

\[
A_{gh}
=
\sum_{j\in g}\widetilde\Phi^K_{jh}\otimes[z_{jh},1],
\]

\[
G_{ih}
=
\frac{[(\Phi^Q_{ih})^TA_{g(i)h}]_{1:-1}}
{[(\Phi^Q_{ih})^TA_{g(i)h}]_{-1}}.
\]

No pairwise global attention matrix is formed. Single-graph, padded, bucketed,
and ragged matrix schedules evaluate the same equation.

## Unified sparse local operator

All local routing terms contribute to one invariant score `a_ijr`, including
scalar, polar, axial, pseudoscalar, even-tensor, odd-tensor, and tensor-axis
contractions. The single positive edge weight is

\[
w_{ijr}
=
f_c(\tilde u_{ij})
\exp\left(3\tanh(a_{ijr}/3)\right).
\]

Every local value family uses the same receiver mass:

\[
S_{ir}^{f}
=
\frac{\sum_jw_{ijr}\rho_{ijr}^{f}z_{jr}^{f}}
{1+\sum_jw_{ijr}}.
\]

Thus tensor routing does not create a second sparse operator, second mass, or
second value lane.

## Task heads

`graph_irreps` is a fixed mean-pooled diagnostic, not a universal task head.
Task semantics remain outside the core:

- extensive energy: sum an invariant scalar node head;
- conservative force: differentiate scalar energy with respect to positions;
- pose refinement: project a `1o` node head and apply an update mask;
- interface or selected-node property: apply a task-specific mask and pooling;
- hierarchy: pool/broadcast using an external assignment;
- chirality-sensitive observable: consume `0o`, `1e`, or mixed sectors as needed.

Generic pooling, coordinate/vector heads, energy/force primitives, hierarchy
operations, and the SBDD adapter remain available as separate modules.

## Compatibility and research API

`EquivariantAttention` and `EquivariantAttentionConfig` remain available for
reproducing legacy experiments and testing explicit architecture ablations:
local/global routes, memory transport, alternative normalization, transient
high-order workspaces, and backend studies.

They are not the canonical public path. New downstream integrations should use
`UnifiedEquivariantAttention` unless an experiment explicitly requires a legacy
or research switch.

## Mathematical and implementation notes

- `docs/UNIFIED_3D_CORE.md`
- `docs/UNIFIED_3D_INITIALIZATION.md`
- `docs/UNIFIED_3D_MULTIPOLES.md`
- `docs/MATHEMATICAL_SPEC.md`
- `docs/INVARIANCE.md`
- `docs/SCALING.md`
- `docs/EVALUATION.md`

## Evidence boundary

Unit and smoke tests can establish algebraic transformation laws, numerical
finiteness, gradient paths, and implementation equivalence. They do not by
themselves establish downstream accuracy, architecture superiority, production
neighbor-search performance, or a fused sparse-kernel speedup. Those claims
require resource-matched and leakage-controlled experiments.
