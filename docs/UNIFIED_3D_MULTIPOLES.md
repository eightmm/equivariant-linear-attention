# Canonical node-multipole SE(3) core

This document is normative for `UnifiedEquivariantAttention`.

The user chooses only final `output_irreps`. Internal parity, angular degree,
normalization, local/global composition, and execution semantics are fixed.
The persistent carrier is

\[
C_0\times0e
\oplus H\times0o
\oplus H\times1o
\oplus H\times1e
\oplus H\times2e
\oplus H\times2o.
\]

Persistent degree is bounded by \(l\le2\). No hidden architecture switch or
fallback changes these equations.

## 1. Prepared sparse geometry

For receiver-major candidate edge \(j\to i\),

\[
d_{ij}=\frac{x_j-x_i}{R_c},
\qquad
u_{ij}=\|d_{ij}\|^2.
\]

The pure angular direction is

\[
\hat d_{ij}=
\begin{cases}
d_{ij}/\sqrt{\nu_{ij}},&\nu_{ij}>0,\\
0,&\nu_{ij}=0.
\end{cases}
\]

A typed relation may only narrow the shared local domain. With relation cutoff
\(0<R_\rho\le R_c\),

\[
\tilde\nu_{ij}
=
\nu_{ij}\left(\frac{R_c}{R_{\rho_{ij}}}\right)^2.
\]

Untyped edges use \(\tilde\nu_{ij}=\nu_{ij}\). Relations never create a
separate equivariant basis, edge hidden state, or local operator.

The compact cutoff is

\[
f_c(u)=
\begin{cases}
1-10u^3+15u^4-6u^5,&0\le u<1,\\
0,&u\ge1.
\end{cases}
\]

It satisfies

\[
f_c(1)=f_c'(1)=f_c''(1)=0.
\]

## 2. Static node multipoles

For deterministic radial shell \(n\), define

\[
a_{ijn}=f_c(\tilde\nu_{ij})R_n(\nu_{ij}).
\]

Receiver mass and squared mass are

\[
m_{in}=\sum_{j\to i}a_{ijn},
\qquad
m_{in}^{(2)}=\sum_{j\to i}a_{ijn}^2.
\]

The Cartesian \(l=1\) and \(l=2\) node multipoles are

\[
P_{in}^{1o}
=
\frac{\sum_{j\to i}a_{ijn}\hat d_{ij}}
{1+m_{in}},
\]

\[
Q_{in}^{2e}
=
\frac{\sum_{j\to i}a_{ijn}\operatorname{ST}(\hat d_{ij})}
{1+m_{in}}.
\]

These are the real Cartesian equivalents of radial \(Y_1\) and \(Y_2\)
coefficients. They are computed once per forward and reused by every block.

Three cyclic radial-shell polar moments generate parity-complete node geometry:

\[
A_{in}^{1e}
=
P_{in}^{1o}\times P_{i,n-1}^{1o},
\]

\[
\chi_{in}^{0o}
=
A_{in}^{1e}\cdot P_{i,n-2}^{1o},
\]

\[
U_{in}^{2o}
=
\operatorname{ST}
\left(P_{i,n-2}^{1o},A_{in}^{1e}\right).
\]

No edge triplet is enumerated or retained.

## 3. Scale and density initialization

For graph \(g\),

\[
\mu_g=\frac1{N_g}\sum_{i\in g}x_i,
\]

\[
r_g=
\sqrt{
\frac1{N_g}\sum_{i\in g}\|x_i-\mu_g\|^2+\epsilon
}.
\]

Let \(\bar x_i=(x_i-\mu_g)/r_g\). The invariant scalar context is

\[
c_i=
\left[
\log(1+r_g),
\log(1+\|x_i-\mu_g\|),
\log(1+\overline m_i),
\log(1+\overline m_i^{(2)})
\right],
\]

where bars average over radial shells. Initialization is

\[
s_i^{0e,0}=W_f f_i+e_{\rm role(i)}+W_c c_i,
\]

\[
V_i^{1o,0}
=
\tanh(G_Vs_i^{0e,0})\odot\bar x_i
+W_PP_i^{1o}
+W_vv_i^{\rm in},
\]

\[
A_i^{1e,0}=W_AA_i^{1e},
\]

\[
T_i^{2e,0}
=
\tanh(G_Ts_i^{0e,0})\odot\operatorname{ST}(\bar x_i)
+W_QQ_i^{2e}
+W_TT_i^{\rm in},
\]

\[
s_i^{0o,0}=W_\chi\chi_i^{0o},
\qquad
U_i^{2o,0}=W_UU_i^{2o}.
\]

## 4. Irrep-sector pre-normalization

Even scalars use scalar LayerNorm. All non-even-scalar sectors use invariant
sector RMS normalization before query/key/value construction and before the
pointwise equivariant FFN.

For odd scalars,

\[
\widehat s_i^{0o}
=
g^{0o}
\frac{s_i^{0o}}
{\sqrt{\operatorname{mean}_c[(s_{ic}^{0o})^2]+\epsilon_{\rm norm}}}.
\]

For \(X\in\{1o,1e\}\),

\[
\widehat X_i
=
g^X
\frac{X_i}
{\sqrt{\operatorname{mean}_{c,m}[X_{icm}^2]+\epsilon_{\rm norm}}}.
\]

For \(T\in\{2e,2o\}\),

\[
\widehat T_i
=
g^T
\frac{T_i}
{\sqrt{
\frac1{5C_T}\sum_c\|T_{ic}\|_F^2+\epsilon_{\rm norm}
}}.
\]

The normalization floor is

\[
\epsilon_{\rm norm}=\max(\epsilon,10^{-8}),
\]

so a nearly zero odd sector is not numerically over-amplified.

## 5. One exact global feature map through l=2

The global path evaluates exactly one finite-feature attention per block.
Normalized same-irrep query/key representatives are

\[
q^{1o},k^{1o},
\quad
q^{1e},k^{1e},
\quad
q^{2e},k^{2e},
\quad
q^{2o},k^{2o}.
\]

Compact ST5 tensors are mapped to an orthonormal Frobenius basis. The query map
is

\[
\Phi_i^Q
=
\operatorname{concat}
\left[
\phi(q_i^{0e}),
c_h,
\sqrt{\alpha_{1o,h}}q_i^{1o},
\sqrt{\alpha_{1e,h}}q_i^{1e},
\sqrt{\alpha_{2e,h}}q_i^{2e},
\sqrt{\alpha_{2o,h}}q_i^{2o}
\right],
\]

with an analogous key map and

\[
c_h^2
=
1+\alpha_{1o,h}+\alpha_{1e,h}
+\alpha_{2e,h}+\alpha_{2o,h}.
\]

Because each bounded same-irrep inner product lies in \([-1,1]\),

\[
(\Phi_i^Q)^T\Phi_j^K\ge1.
\]

One-cycle balancing and augmented transport are

\[
Q_{gh}=\sum_{i\in g}\Phi_{ih}^Q,
\]

\[
\widetilde\Phi_{jh}^K
=
\frac{\Phi_{jh}^K}
{(\Phi_{jh}^K)^TQ_{g(j)h}},
\]

\[
A_{gh}
=
\sum_{j\in g}
\widetilde\Phi_{jh}^K\otimes[z_{jh},1],
\]

\[
G_{ih}
=
\frac{[(\Phi_{ih}^Q)^TA_{g(i)h}]_{1:-1}}
{[(\Phi_{ih}^Q)^TA_{g(i)h}]_{-1}}.
\]

Numerator and denominator share one contraction. No \(N\times N\) tensor is
formed.

## 6. One tensor-aware sparse local operator

Every parity sector contributes to one invariant rank-\(R\) score. In addition
to scalar, polar, and axial terms, the score includes

\[
q_{ir}^{0o}k_{jr}^{0o},
\]

\[
Q_{ir}^{2e}:K_{jr}^{2e},
\qquad
U_{ir}^{2o}:V_{jr}^{2o},
\]

\[
(\hat d_{ij}^TQ_{ir}^{2e}\hat d_{ij})
(\hat d_{ij}^TK_{jr}^{2e}\hat d_{ij}),
\]

\[
(\hat d_{ij}^TU_{ir}^{2o}\hat d_{ij})
(\hat d_{ij}^TV_{jr}^{2o}\hat d_{ij}).
\]

All are parity-even scalars. The complete score is denoted \(a_{ijr}\), and the
single positive edge weight is

\[
w_{ijr}
=
f_c(\tilde\nu_{ij})
\exp\left(3\tanh(a_{ijr}/3)\right).
\]

There is one receiver mass

\[
M_{ir}=\sum_{j\to i}w_{ijr}
\]

and every local value family uses it:

\[
S_{ir}^{f}
=
\frac{
\sum_{j\to i}w_{ijr}\rho_{ijr}^{f}z_{jr}^{f}
}{1+M_{ir}}.
\]

Tensor routing does not create a second score, second weight, second mass, or
second value lane. Tensor-specific score coefficients are neutral initialized,
so the initial local equation is the parity-complete scalar/vector reference;
the new score terms receive gradients through the existing local output path.

## 7. Low-order parity-complete tensor closure

After global/local message update, a low-rank node self-interaction realizes all
persistent output sectors reachable from selected \(l\le2\) Cartesian products.

For polar vectors \(p,q\), axial vectors \(a,b\), even tensors \(E,F\), and odd
tensors \(U,V\):

\[
\Delta s^{0e}
\supset
p\cdot q+a\cdot b+E:F+U:V,
\]

\[
\Delta s^{0o}
\supset
p\cdot a+E:U,
\]

\[
\Delta A^{1e}
\supset
p\times q+\operatorname{vex}([E,F])+\operatorname{vex}([U,V]),
\]

\[
\Delta V^{1o}
\supset
a\times p+\operatorname{vex}([E,U]),
\]

\[
\Delta T^{2e}
\supset
\operatorname{ST}(p,q)
+\operatorname{ST}(a,b)
+\operatorname{ST}(EF+FE)
+\operatorname{ST}(UV+VU),
\]

\[
\Delta U^{2o}
\supset
\operatorname{ST}(p,a)
+\operatorname{ST}(EU+UE).
\]

Thus the fast path covers selected components of

\[
1\otimes1\to0\oplus1\oplus2
\]

and

\[
2\otimes2\to0\oplus1\oplus2
\]

without a generic edge tensor-product executor. Geometry multipoles enter the
same low-rank closure as static operands. Closure output projections are neutral
initialized.

## 8. Per-copy LayerScale and FFN

Attention update, tensor closure, and equivariant FFN each use separate
per-copy residual scales:

\[
\alpha^{0e}\in\mathbb R^{C_0},
\qquad
\alpha^{0o}\in\mathbb R^H,
\]

\[
\alpha^{1o},\alpha^{1e},\alpha^{2e},\alpha^{2o}
\in\mathbb R^H.
\]

Vector and tensor scales broadcast only over representation components. The
pointwise FFN combines parity-valid scalar gates and channel-only equivariant
maps after a second irrep-sector pre-normalization.

## 9. Complete recurrence

The homogeneous canonical recurrence is

\[
A=\operatorname{NodeMultipoles}_{l\le2}(x,\mathcal E),
\]

\[
h^0=\operatorname{Embed}(f,x,A),
\]

\[
\widehat h^\ell=\operatorname{IrrepPreNorm}(h^\ell),
\]

\[
G^\ell
=
\operatorname{ExactGlobal}_{l\le2}(\widehat h^\ell),
\]

\[
S^\ell
=
\operatorname{SingleSparseLocal}_{l\le2}
(\widehat h^\ell,x,\mathcal E),
\]

\[
\widetilde h^{\ell+1}
=
h^\ell
+
\operatorname{LayerScale}
\left[
\operatorname{Update}(G^\ell+S^\ell)
+
\operatorname{TPClosure}_{l\le2}(h^\ell,A)
\right],
\]

\[
h^{\ell+1}
=
\widetilde h^{\ell+1}
+
\operatorname{LayerScale}_{\rm ffn}
\operatorname{EquivariantFFN}
\left(
\operatorname{IrrepPreNorm}(\widetilde h^{\ell+1})
\right).
\]

## 10. Complexity and boundary

At fixed hidden width, head count, RBF count, local rank, and multipole rank,

\[
\operatorname{time}=O(L(N+E)),
\]

\[
\operatorname{persistent\ state}=O(N),
\]

\[
\operatorname{neighbor\ metadata}=O(N+E).
\]

The static multipole bank costs \(O(E)\) once per forward and is reused by all
blocks. Tensor sectors are fused into the existing global feature map, so there
is one global attention contraction per block. Tensor terms augment the existing
local score, so there is one sparse receiver mass and one family of local value
reductions per block.

Persistent \(l>2\) is intentionally excluded. A future transient \(l=3\) path
must project back to the persistent \(l\le2\) carrier in the same block and must
be justified by matched resource and utility evidence.
