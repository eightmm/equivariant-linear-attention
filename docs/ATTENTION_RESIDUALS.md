# Equivariant Attention Residuals

This module adapts Moonshot/Kimi Attention Residuals to the parity-complete
`EquivariantLinearAttention` stack.

The mechanism is opt-in because the published evidence is for large language
models, not yet for molecular, point-cloud, or physical 3D workloads. It is most
plausible for deep pre-normalized stacks where additive residual accumulation
causes hidden-state growth and weakens the relative contribution of individual
layers.

## 1. Why ordinary AttnRes cannot be copied directly

For a scalar token representation, Attention Residuals computes

\[
h_l=\sum_{a\in\mathcal S_l}\alpha_{a\to l}v_a,
\]

\[
\alpha_{a\to l}
=
\operatorname{softmax}_a
\left[
 w_l^T\operatorname{RMSNorm}(v_a)
\right].
\]

The hidden state in this project is not one unconstrained vector. It is

\[
h_i=
(s_i^{0e},s_i^{0o},V_i^{1o},A_i^{1e},T_i^{2e},U_i^{2o}).
\]

A depth weight may multiply every sector, but the weight itself must be an
invariant scalar. A pseudo-query must therefore not dot directly with oriented
vector or tensor components.

## 2. Invariant depth key

For source state `a`, node `i`, define

\[
d_{ai}=
\left[
 s_{ai}^{0e},
 |s_{ai}^{0o}|,
 \operatorname{RMS}_m(V_{ai}^{1o}),
 \operatorname{RMS}_m(A_{ai}^{1e}),
 \lVert T_{ai}^{2e}\rVert_F/\sqrt5,
 \lVert U_{ai}^{2o}\rVert_F/\sqrt5
\right].
\]

The complete descriptor is RMS-normalized:

\[
\widehat d_{ai}
=
\frac{d_{ai}}
{\sqrt{\operatorname{mean}(d_{ai}^2)+\epsilon}}.
\]

One learned pseudo-query is used for each attention and FFN sublayer:

\[
\ell_{ai}^{(l)}
=
\frac{(w_l)^T\widehat d_{ai}}{\sqrt{D_d}},
\]

\[
\alpha_{ai}^{(l)}
=
\operatorname{softmax}_a(\ell_{ai}^{(l)}).
\]

All pseudo-queries are initialized to zero, so the initial source distribution
is uniform, matching the stabilization prescription of Attention Residuals.

## 3. Shared equivariant mixture

The same scalar weight is applied to every irrep sector:

\[
\bar s_i^{0e}=\sum_a\alpha_{ai}s_{ai}^{0e},
\qquad
\bar s_i^{0o}=\sum_a\alpha_{ai}s_{ai}^{0o},
\]

\[
\bar V_i^{1o}=\sum_a\alpha_{ai}V_{ai}^{1o},
\qquad
\bar A_i^{1e}=\sum_a\alpha_{ai}A_{ai}^{1e},
\]

\[
\bar T_i^{2e}=\sum_a\alpha_{ai}T_{ai}^{2e},
\qquad
\bar U_i^{2o}=\sum_a\alpha_{ai}U_{ai}^{2o}.
\]

Because `alpha` is invariant and shared across irrep components, this operation
commutes with the complete tracked O(3) action. Per-head or per-component depth
weights are intentionally not used.

## 4. Block recurrence

The embedding state is retained as the first completed source. Layers are split
into `B=attention_residual_blocks` contiguous blocks. Each block accumulates a
partial residual state `p`.

Before the attention sublayer,

\[
h_{l,\mathrm{attn}}
=
\operatorname{DepthAttn}
(\{b_0,\ldots,b_{n-1},p\};w_l^{\mathrm{attn}}).
\]

The linear-attention branch returns a residual contribution

\[
r_l^{\mathrm{attn}}
=
\operatorname{ELA}_{\mathrm{attn}}(h_{l,\mathrm{attn}})
-h_{l,\mathrm{attn}},
\]

and updates the partial block sum:

\[
p\leftarrow p+r_l^{\mathrm{attn}},
\]

with `p=r` at the start of a new block.

Before the FFN sublayer,

\[
h_{l,\mathrm{ffn}}
=
\operatorname{DepthAttn}
(\{b_0,\ldots,b_{n-1},p\};w_l^{\mathrm{ffn}}),
\]

\[
r_l^{\mathrm{ffn}}
=
\operatorname{EqFFN}(h_{l,\mathrm{ffn}})
-h_{l,\mathrm{ffn}},
\]

\[
p\leftarrow p+r_l^{\mathrm{ffn}}.
\]

At a block boundary, the completed partial state is cached as a new block source
and a fresh partial state is started.

## 5. Interaction with sequence/graph linear attention

Depth attention and node attention are different axes.

The global 3D operator remains exact positive-feature linear attention over
nodes:

\[
G_{ih}
=
\frac{[(\Phi_{ih}^Q)^TA_{g(i)h}]_{1:-1}}
{[(\Phi_{ih}^Q)^TA_{g(i)h}]_{-1}}.
\]

AttnRes only changes which prior-depth representation is presented to this
operator. It does not form a dense node-pair matrix and does not replace the
sparse local residual.

For `B` depth blocks, the persistent depth cache is

\[
O(BN D_h),
\]

while the spatial architecture remains

\[
O(L(N+E)).
\]

## 6. Coordinates

Coordinate refinement is not local-only. The displacement head consumes the
polar hidden state after:

1. depth routing;
2. exact global linear attention;
3. sparse local residual;
4. tensor closure;
5. equivariant FFN.

Hence

\[
\Delta x_i^l
=
\operatorname{CoordinateHead}
(h_i^{l+1})
\]

contains both global and local information.

Geometry-dependent operations are split as follows:

- direct edge displacement, cutoff, RBF, and node multipoles are local;
- normalized positions and exact relative moments also enter the global linear
  attention value transport;
- after a coordinate update, both local geometry and global position moments are
  recomputed on the same candidate topology.

The old depth states were produced at earlier coordinates. Their weighted sum is
still equivariant, but for aggressive coordinate motion a shallow depth cache or
an outer-loop neighbor rebuild is preferable.

## 7. API

```python
from equivariant_attention import (
    EquivariantAttentionResidualConfig,
    EquivariantAttentionResiduals,
)

config = EquivariantAttentionResidualConfig(
    input_irreps="32x0e + 4x1o",
    output_irreps="1x0e + 1x1o",
    hidden_dim=128,
    num_layers=16,
    num_heads=8,
    local_rank=4,
    attention_residual_blocks=8,
)

model = EquivariantAttentionResiduals(config)
output = model(node_irreps, positions, graph)
```

For shallow stacks, the ordinary `EquivariantLinearAttention` remains the safer
baseline. Promotion of AttnRes to the default requires depth-matched experiments
that record validation quality, hidden-state RMS versus depth, gradient norms,
latency, and peak memory.
