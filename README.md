# Equivariant Linear Attention

A domain-agnostic PyTorch implementation of **equivariant linear attention** for
unordered 3D data. The canonical model combines:

- exact positive-feature global linear attention;
- one receiver-normalized sparse local residual;
- parity-aware `0e/0o/1o/1e/2e/2o` hidden state;
- receiver-centered Cartesian `l<=2` multipoles;
- low-rank tensor closure and chirality without explicit triplets;
- reusable attention and FFN residual layers;
- optional invariant DiT-style conditioning and coordinate refinement.

The same layer can be used for molecules, proteins, protein--ligand complexes,
particles, meshes, point clouds, and other sparse 3D systems. Domain vocabulary
and task semantics remain outside the core.

## What defines the architecture

For layer `l`,

\[
\widehat h^l=\operatorname{EqRMSNorm}_{\rm attn}(h^l),
\]

\[
G^l=\operatorname{ExactLinearAttention}_{l\le2}(\widehat h^l),
\]

\[
S^l=\operatorname{SparseLocalResidual}_{l\le2}
(\widehat h^l,x^l,\mathcal E),
\]

\[
\widetilde h^l
=h^l+\operatorname{AttnResidual}(G^l+S^l),
\]

\[
h^{l+1}
=\widetilde h^l+
\operatorname{EquivariantFFN}
(\operatorname{EqRMSNorm}_{\rm ffn}(\widetilde h^l)).
\]

The global operator is still the defining operation. Node multipoles and the
sparse local path augment the linear-attention layer; they do not replace it
with a graph convolution or dense pair attention.

The persistent hidden carrier is fixed automatically to

\[
C_0\times0e
\oplus H\times0o
\oplus H\times1o
\oplus H\times1e
\oplus H\times2e
\oplus H\times2o.
\]

Users declare `input_irreps` and `output_irreps`; hidden parity and angular
degree are fixed.

## Install

```bash
uv sync --locked
```

Optional evaluation dependencies:

```bash
uv sync --locked --extra qm9
uv sync --locked --extra pdbbind
```

Automated Actions are not run on push or pull request. Local checks are explicit:

```bash
scripts/check.sh fast
scripts/check.sh gpu
```

## Canonical use

```python
import torch

from equivariant_attention import (
    EquivariantLinearAttention,
    EquivariantLinearAttentionConfig,
    pack_irreps,
    prepare_3d_graph,
)

config = EquivariantLinearAttentionConfig(
    input_irreps="32x0e + 4x1o + 1x1e",
    output_irreps="1x0e + 1x1o",
    hidden_dim=128,
    num_layers=6,
    num_heads=8,
    local_rank=4,
    local_cutoff=6.0,
    num_rbf=16,
    residual_dropout=0.05,
    drop_path_rate=0.10,
)
model = EquivariantLinearAttention(config)

node_irreps = pack_irreps(
    config.input_layout,
    {
        "0e": scalar_features[:, :, None],
        "1o": polar_vectors,
        "1e": axial_vectors,
    },
)

# edge_index[0] is receiver i; edge_index[1] is sender j.
graph = prepare_3d_graph(batch, edge_index)

output = model(node_irreps, positions, graph)
node_output = output["node_irreps"]
final_positions = output["positions"]
```

`positions` are a separate affine geometry input. They are not packed as a
`1o` feature because

\[
x_i\mapsto Rx_i+t
\]

contains translation, whereas an ordinary polar feature transforms as
`v -> Rv`.

## Input and output irreps

The optimized canonical path supports arbitrary multiplicities of

```text
0e  ordinary scalar
0o  pseudoscalar
1o  polar vector
1e  axial vector
2e  even symmetric-traceless tensor
2o  odd symmetric-traceless tensor
```

The Cartesian bases are:

```text
l=1: [x, y, z]
l=2: [xx, yy, xy, xz, yz], with zz = -xx - yy
```

Helpers are available at package root:

```python
pack_irreps(...)
split_irreps(...)
matrix_to_st5(...)
st5_to_matrix(...)
```

`l>2` input or output is rejected rather than silently dispatched to a slower or
semantically different path.

## Linear-attention stabilization

The scalar query and key are RMS-normalized per head before the positive feature
map:

\[
\bar q_{ih}
=\gamma_h^Q
\frac{q_{ih}}
{\sqrt{\operatorname{mean}_d(q_{ihd}^2)+\epsilon}},
\]

with the same equation for keys. The positive map remains

\[
\phi(z)=\frac{\operatorname{ELU}(z)+1}{\sqrt D}.
\]

The feature kernel and one-cycle balancing remain exactly factorized, so no
`N x N` attention matrix is formed.

## Equivariant normalization and activation

Every branch has its own all-sector RMS pre-normalization. Non-scalar residuals
use a direction-preserving gate derived only from invariant magnitudes:

\[
\Delta X^{l,p}\leftarrow
2\sigma(\operatorname{MLP}(\text{sector norms}))\Delta X^{l,p}.
\]

The last gate projection is zero initialized, giving an exact identity gate at
initialization.

Residual dropout samples one mask per irrep copy and broadcasts it over the
irrep components. Stochastic depth samples one mask per graph. Both preserve the
transformation law of every retained residual.

## DiT-style conditioning

Set `condition_dim` to use invariant conditioning:

```python
config = EquivariantLinearAttentionConfig(
    input_irreps="32x0e",
    output_irreps="1x1o",
    condition_dim=256,
)

output = model(
    node_irreps,
    positions,
    graph,
    condition=time_embedding,
)
```

Condition shape may be `[D]`, `[1,D]`, `[G,D]`, or `[N,D]`. The condition is
`0e`. Scalar channels receive bounded shift and scale; non-scalar channels
receive bounded copy-wise scale only. Attention and FFN residuals are gated
independently.

Vector or tensor conditions belong in `input_irreps`, not in the invariant
condition tensor.

## Reusable layer API

The stack exposes its layers:

```python
state, context = model.embed_input(node_irreps, positions, graph)

for layer in model.layers:
    state = layer.attention_residual(state, context, condition)
    state = layer.ffn_residual(state, context, condition)

node_output = model.project_state(state)
```

The preferred public layer class is `EquivariantLinearAttentionLayer`. The
lower-level `UnifiedEquivariantLayer` remains available as a compatibility and
ablation boundary.

## Coordinate refinement

Set

```python
coordinate_updates=True
max_coordinate_step=0.2
```

to apply a bounded polar displacement after each layer:

\[
x_i^{l+1}=x_i^l+\Delta x_i^l.
\]

Geometry and multipoles are refreshed after each layer on the same prepared
candidate topology. For large motion, the caller should construct candidates
with a skin or rebuild the graph in an outer loop.

Direct coordinate refinement is useful for denoising, pose refinement,
registration, and learned relaxation. Conservative forces should instead be
obtained from a scalar energy through `-grad_x E`.

## Complexity

At fixed channel width, head count, local rank, and multipole rank,

\[
\operatorname{time}=O(L(N+E)),
\qquad
\operatorname{persistent\ state}=O(N).
\]

The implementation has no dense pair-attention tensor and no persistent edge
hidden state. Neighbor discovery is outside the layer and must be costed
separately.

## Compatibility and research APIs

- `EquivariantLinearAttention`: preferred refined linear-attention stack.
- `UnifiedEquivariantAttention`: canonical compatibility stack without the new
  regularization wrapper.
- `EquivariantAttention`: legacy research path for explicit architecture and
  backend ablations.

Mathematical details:

- `docs/EQUIVARIANT_LINEAR_ATTENTION.md`
- `docs/LAYERED_SE3_API.md`
- `docs/UNIFIED_3D_CORE.md`
- `docs/UNIFIED_3D_MULTIPOLES.md`

Unit tests establish transformation laws, numerical finiteness, and gradient
paths. They do not establish downstream accuracy, production neighbor-search
performance, or fused-kernel speedup.
