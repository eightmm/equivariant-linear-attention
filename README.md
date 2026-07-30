# Equivariant Linear Attention

A domain-agnostic PyTorch layer for unordered 3D data. The canonical path is
SE(3)-equivariant, keeps explicit O(3) parity bookkeeping internally, combines
exact linear-time global attention with one receiver-normalized sparse local
operator, and exposes `input_irreps` and `output_irreps` as its representation
contract.

The same layer can be used for molecules, proteins, protein--ligand complexes,
particles, meshes, point clouds, and other sparse 3D systems. Domain vocabulary
and task heads remain outside the core.

## Canonical architecture

The fixed hidden carrier is

\[
C_0\times0e
\oplus H\times0o
\oplus H\times1o
\oplus H\times1e
\oplus H\times2e
\oplus H\times2o.
\]

Each layer evaluates

\[
G^\ell=\operatorname{ExactGlobal}_{l\le2}(h^\ell),
\]

\[
S^\ell=\operatorname{SparseLocal}_{l\le2}(h^\ell,x^\ell,\mathcal E),
\]

followed by an attention/tensor-closure residual and an equivariant FFN
residual:

\[
\widetilde h^\ell
=h^\ell+\Delta h^\ell_{\rm attn}+\Delta h^\ell_{\rm closure},
\]

\[
h^{\ell+1}=\widetilde h^\ell+\Delta h^\ell_{\rm ffn}.
\]

The implementation includes:

- exact positive finite-feature global attention with one balancing cycle;
- one sparse rank-`R` score, edge weight, receiver mass, and value transport;
- receiver-centered Cartesian `l=0,1,2` node multipoles;
- active `0o`, `1o`, `1e`, `2e`, and `2o` routing;
- low-rank Cartesian tensor-product closure through `l<=2`;
- chirality through aggregate cross/triple products without edge triplets;
- irrep-sector RMS pre-normalization and per-copy LayerScale;
- a compact C2 cutoff;
- optional invariant DiT-style conditioning;
- optional bounded coordinate refinement;
- no `N x N` attention tensor and no persistent edge hidden state.

At fixed widths and ranks,

\[
\operatorname{time}=O(L(N+E)),
\qquad
\operatorname{persistent\ state}=O(N).
\]

Neighbor discovery is outside the layer and must be costed separately.

## Install

```bash
uv sync --locked
```

Optional evaluation dependencies:

```bash
uv sync --locked --extra qm9
uv sync --locked --extra pdbbind
```

Automated GitHub Actions are disabled for push and pull-request events. Run
checks explicitly:

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
    pack_irreps,
    prepare_3d_graph,
)

config = Unified3DConfig(
    input_irreps="32x0e + 4x1o",
    output_irreps="1x0e",
    hidden_dim=128,
    num_layers=6,
    num_heads=8,
    local_rank=4,
    local_cutoff=6.0,
    num_rbf=16,
)
model = UnifiedEquivariantAttention(config)

node_irreps = pack_irreps(
    config.input_layout,
    {
        "0e": scalar_features[..., None],
        "1o": vector_features,
    },
)

# edge_index[0] is receiver i and edge_index[1] is sender j for j -> i.
graph = prepare_3d_graph(batch, edge_index)
output = model(node_irreps, positions, graph)

node_output = output["node_irreps"]
graph_mean_diagnostic = output["graph_irreps"]
```

`prepare_3d_graph` validates graph isolation and packs receiver-major CSR once.
The canonical model does not silently construct a complete graph.

## Input and output irreps

The optimized canonical path supports any multiplicity of

```text
0e  ordinary scalar
0o  pseudoscalar
1o  polar vector
1e  axial vector
2e  even symmetric-traceless tensor
2o  odd symmetric-traceless tensor
```

for both input and output. The Cartesian `l=1` basis is xyz. The compact `l=2`
basis is `[xx, yy, xy, xz, yz]` with `zz=-xx-yy`.

Helpers are available at package root:

```python
pack_irreps(...)
split_irreps(...)
matrix_to_st5(...)
st5_to_matrix(...)
```

Raw positions are not an `1o` feature. They transform affinely as
`x -> R x + t` and remain a separate argument. `input_irreps="0"` is the
geometry-only path.

## Layer-level API

The full stack is a composition of public `UnifiedEquivariantLayer` instances:

```python
state, context = model.embed_input(node_irreps, positions, graph)

for layer in model.layers:
    after_attention = layer.attention_residual(state, context)
    state = layer.ffn_residual(after_attention, context)

node_output = model.project_state(state)
```

The equivalent one-call layer form is:

```python
layer_output = layer(state, context)
state = layer_output.state
```

This separation allows custom residual schedules, layer sharing, activation
checkpointing, external control logic, or integration into diffusion/flow
backbones without redefining the equivariant operator.

## DiT-style invariant conditioning

Set `condition_dim` and pass an ordinary invariant condition per graph or per
node:

```python
config = Unified3DConfig(
    input_irreps="32x0e",
    output_irreps="1x1o",
    condition_dim=256,
)
model = UnifiedEquivariantAttention(config)

# condition: [G, 256], [N, 256], or [256]
output = model(node_irreps, positions, graph, condition=condition)
```

The condition projection is zero initialized. It produces separate adaptive
modulation and residual gates for the attention and FFN branches. Only even
scalars receive additive shifts; non-scalar sectors receive copy-wise scales,
which preserves equivariance.

Typical conditions include diffusion time, noise level, class, temperature, or
another graph-level state. Conditions are `0e`; arbitrary vector conditions must
instead be declared in `input_irreps`.

## Coordinate refinement

A `1o` output is a displacement-like polar vector, not an absolute coordinate.
For an affine coordinate output, enable layer-wise coordinate residuals:

```python
config = Unified3DConfig(
    input_irreps="32x0e",
    output_irreps="1x0e",
    coordinate_updates=True,
    max_coordinate_step=0.2,
)
model = UnifiedEquivariantAttention(config)
output = model(node_irreps, positions, graph)

refined_positions = output["positions"]
total_displacement = output["coordinate_delta"]
```

Each layer predicts a bounded polar displacement

\[
\Delta x_i
=
\Delta_{\max}\sigma(a_i)
\frac{W_xV_i^{1o}}{\sqrt{1+\|W_xV_i^{1o}\|^2}},
\]

and applies `x <- x + delta_x`. The coordinate projection is zero initialized,
so enabling the path does not alter the initial function.

Geometry, RBFs, cutoffs, normalized positions, and node multipoles are recomputed
after each applied update. Receiver/sender candidates remain fixed. Dynamic
systems must therefore provide a candidate graph with sufficient skin or rebuild
it in an outer loop.

Coordinate refinement is meaningful for diffusion/flow denoising, pose or
conformation refinement, learned relaxation, point-cloud registration, and
coarse-to-fine generation. It is usually unnecessary for a fixed-geometry
scalar property model.

## Task heads

`graph_irreps` is mean-pooled diagnostic output, not a universal task head.
Typical downstream semantics are:

- extensive energy: sum invariant scalar node contributions;
- conservative force: differentiate scalar energy with respect to positions;
- pose refinement: use refined positions or a task-specific `1o` head;
- selected-node property: apply a mask and task-specific pooling;
- chirality-sensitive observable: consume `0o`, `1e`, or mixed sectors.

## Compatibility and research API

`EquivariantAttention` and `EquivariantAttentionConfig` remain available for
legacy experiments and explicit architecture ablations. New integrations should
use `UnifiedEquivariantAttention` unless a study specifically requires a legacy
switch.

## Mathematical and implementation notes

- `docs/UNIFIED_3D_CORE.md`
- `docs/UNIFIED_3D_INITIALIZATION.md`
- `docs/UNIFIED_3D_MULTIPOLES.md`
- `docs/LAYERED_SE3_API.md`
- `docs/MATHEMATICAL_SPEC.md`
- `docs/INVARIANCE.md`
- `docs/SCALING.md`
- `docs/EVALUATION.md`

## Evidence boundary

Unit and smoke tests establish transformation laws, numerical finiteness,
gradient paths, and implementation equivalence. They do not establish downstream
accuracy, architecture superiority, production neighbor-search performance, or
a fused sparse-kernel speedup. Those claims require resource-matched and
leakage-controlled experiments.
