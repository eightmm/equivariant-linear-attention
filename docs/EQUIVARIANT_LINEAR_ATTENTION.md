# Equivariant linear-attention layer

This document is normative for `EquivariantLinearAttention` and
`EquivariantLinearAttentionLayer`.

The defining operation of this project is exact finite-feature **linear
attention**. Receiver-centered multipoles, the sparse local residual, tensor
closure, conditioning, and coordinate refinement augment that operator; they do
not replace it with a tensor-product graph convolution or a dense pair-attention
matrix.

## 1. State and geometry

The persistent hidden state is fixed to

\[
h_i=
\left(
 s_i^{0e},
 s_i^{0o},
 V_i^{1o},
 A_i^{1e},
 T_i^{2e},
 U_i^{2o}
\right).
\]

Input and output layouts are public:

```python
EquivariantLinearAttentionConfig(
    input_irreps="32x0e + 4x1o + 1x1e",
    output_irreps="1x0e + 1x1o",
)
```

Raw positions remain a separate affine geometry input:

\[
x_i\mapsto Rx_i+t.
\]

They are not packed into `input_irreps`.

## 2. Layer recurrence

One layer is

\[
\widehat h_i^\ell
=
\operatorname{EqRMSNorm}_{\rm attn}(h_i^\ell),
\]

\[
G_i^\ell
=
\operatorname{ExactLinearAttention}_{l\le2}
(\widehat h^\ell)_i,
\]

\[
S_i^\ell
=
\operatorname{SparseLocalResidual}_{l\le2}
(\widehat h^\ell,x^\ell,\mathcal E)_i,
\]

\[
\Delta h_{i,\rm attn}^\ell
=
\operatorname{NormGate}_{\rm attn}
\left[
\operatorname{ParityUpdate}(G_i^\ell+S_i^\ell)
+
\operatorname{TPClosure}_{l\le2}(h_i^\ell)
\right],
\]

\[
\widetilde h_i^\ell
=
h_i^\ell+
\operatorname{DropPath}
\left(
\operatorname{EqDropout}(\Delta h_{i,\rm attn}^\ell)
\right),
\]

\[
\widehat{\widetilde h}_i^\ell
=
\operatorname{EqRMSNorm}_{\rm ffn}(\widetilde h_i^\ell),
\]

\[
h_i^{\ell+1}
=
\widetilde h_i^\ell+
\operatorname{DropPath}
\left(
\operatorname{EqDropout}
\left[
\operatorname{NormGate}_{\rm ffn}
\operatorname{EqFFN}(\widehat{\widetilde h}_i^\ell)
\right]
\right).
\]

Attention and FFN have separate normalization, norm-gated activation, dropout,
and residual branches.

## 3. Equivariant RMS normalization

For even scalars,

\[
\widehat s_i^{0e}
=
\gamma^{0e}\odot
\frac{s_i^{0e}}
{\sqrt{\operatorname{mean}_c[(s_{ic}^{0e})^2]+\epsilon}}.
\]

Odd scalars use the same equation and therefore preserve odd parity.

For `1o` or `1e` states,

\[
\widehat X_i
=
\gamma^X\odot
\frac{X_i}
{\sqrt{\operatorname{mean}_{c,m}[X_{icm}^2]+\epsilon}}.
\]

For compact symmetric-traceless `2e` or `2o` states,

\[
\widehat T_i
=
\gamma^T\odot
\frac{T_i}
{\sqrt{
\frac{1}{5C_T}
\sum_c\lVert T_{ic}\rVert_F^2+\epsilon
}}.
\]

The learned gains act only on copy indices. No vector or tensor component gets a
separate gain.

FP16 and BF16 norms accumulate in FP32; FP64 inputs remain FP64.

## 4. Query/key normalization for linear attention

The scalar query and key projection is normalized per head before the positive
feature map:

\[
\bar q_{ih}
=
\gamma_h^Q
\frac{q_{ih}}
{\sqrt{\operatorname{mean}_d(q_{ihd}^2)+\epsilon}},
\]

\[
\bar k_{ih}
=
\gamma_h^K
\frac{k_{ih}}
{\sqrt{\operatorname{mean}_d(k_{ihd}^2)+\epsilon}}.
\]

The positive scalar feature remains

\[
\phi(z)=\frac{\operatorname{ELU}(z)+1}{\sqrt D}.
\]

The tensor/vector feature blocks and the positive constant block are unchanged,
so the global attention still has an exact feature-map factorization:

\[
K_{ijh}=(\Phi_{ih}^{Q})^T\Phi_{jh}^{K},
\]

\[
A_{gh}
=
\sum_{j\in g}
\widetilde\Phi_{jh}^{K}\otimes[z_{jh},1],
\]

\[
G_{ih}
=
\frac{[(\Phi_{ih}^{Q})^TA_{g(i)h}]_{1:-1}}
{[(\Phi_{ih}^{Q})^TA_{g(i)h}]_{-1}}.
\]

No `N x N` tensor is formed.

The rank-`R` local scalar query/key uses a bounded RMS map,

\[
\bar q_{ir}^{\rm local}
=
\gamma^Q
\frac{q_{ir}}
{\sqrt{1+\operatorname{mean}_r(q_{ir}^2)+\epsilon}},
\]

so the `R=1` case retains content magnitude instead of collapsing to a sign.

## 5. Norm-gated equivariant activation

The residual activation receives only invariant magnitudes:

\[
z_i=
\left[
\widehat s_i^{0e},
|\widehat s_i^{0o}|,
\lVert\widehat V_i^{1o}\rVert,
\lVert\widehat A_i^{1e}\rVert,
\lVert\widehat T_i^{2e}\rVert_F,
\lVert\widehat U_i^{2o}\rVert_F
\right].
\]

It predicts one even scalar gate per non-even-scalar copy:

\[
g_i=2\sigma(\operatorname{MLP}(z_i)).
\]

Then

\[
\Delta X_i^{l,p}
\leftarrow
g_i^{l,p}\Delta X_i^{l,p}.
\]

The final gate projection is zero initialized, hence `g=1` exactly at
initialization. Directions and parity are preserved in every forward pass.

## 6. Equivariant dropout and stochastic depth

Dropout samples one mask per irrep copy. A vector or tensor is retained or
dropped as a whole:

\[
X_{icm}^{l,p}
\leftarrow
\frac{M_{ic}}{1-p}X_{icm}^{l,p}.
\]

The mask does not depend on the irrep component `m`.

Stochastic depth samples one scalar mask per graph and applies it to every node
and every sector in that graph. It is scheduled linearly from zero in the first
layer to `drop_path_rate` in the final layer.

Both mechanisms default to zero and therefore have no inference cost.

## 7. Bounded DiT-style conditioning

The condition is an invariant `0e` feature. Even scalars receive shift and
scale, while all other sectors receive copy-wise scale only:

\[
\operatorname{Ada}(s^{0e};c)
=
[1+0.1\tanh\gamma^{0e}(c)]s^{0e}
+0.1\tanh\beta^{0e}(c),
\]

\[
\operatorname{Ada}(X^{l,p};c)
=
[1+0.1\tanh\gamma^{l,p}(c)]X^{l,p}.
\]

Attention and FFN residual gates remain separately conditioned. The conditioner
output is zero initialized, so the conditioned model initially equals the
unconditioned model.

## 8. Coordinates

When coordinate refinement is enabled,

\[
\Delta x_i^\ell
=
\Delta_{\max}
\sigma(a_i^\ell)
\frac{W_xV_i^{1o,\ell}}
{\sqrt{1+\lVert W_xV_i^{1o,\ell}\rVert^2}},
\]

\[
x_i^{\ell+1}=x_i^\ell+\Delta x_i^\ell.
\]

Geometry and multipoles are recomputed after each layer on the same prepared
candidate topology. The path is an equivariant learned refinement field, not a
guarantee of conservative dynamics.

## 9. Complexity and identity

At fixed hidden width, heads, local rank, and multipole rank,

\[
\operatorname{time}=O(L(N+E)),
\qquad
\operatorname{persistent\ state}=O(N).
\]

The global operator is exact linear attention, the sparse local path has no
persistent edge state, and no dense pair matrix is materialized.
