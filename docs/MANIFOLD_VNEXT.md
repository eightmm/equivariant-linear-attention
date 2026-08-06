# Manifold-aware edge-free ELA vNext

This document specifies the mathematical additions layered on top of the
edge-free Krylov ELA. The persistent hidden carrier remains
`0e + 0o + 1o + 1e + 2e + 2o`; no pair representation, dense attention matrix,
or persistent edge state is introduced.

## 1. Third relative coordinate moment

For one source lane with invariant scalar weight $w_j$, define the receiver
centred third moment

$$
M_i^{(3)}=
\sum_j w_j(x_j-x_i)^{\otimes 3}.
$$

Writing graphwise raw moments as

$$
S_a=\sum_j w_j x_j^{\otimes a},
$$

$M_i^{(3)}$ follows from the symmetric binomial expansion using only
$S_0,S_1,S_2,S_3$. No tuple enumeration is required. The tensor is projected to
its symmetric trace-free part,

$$
T_{abc}=M_{abc}^{(3)}-
\frac{1}{5}
\left(
\delta_{ab}t_c+\delta_{ac}t_b+\delta_{bc}t_a
\right),
$$

where $t_c=M_{aac}^{(3)}$. This is the seven-dimensional Cartesian realization
of the `3o` irrep.

The `3o` carrier is transient. It is contracted into the persistent carrier by

$$
3o\otimes 2e\rightarrow 1o,
\qquad
3o\otimes 2o\rightarrow 1e,
$$

and

$$
3o\otimes 1o\rightarrow 2e,
\qquad
3o\otimes 1e\rightarrow 2o.
$$

The output projections are zero initialized, so historical first-order behavior
is preserved until the new lane is trained.

## 2. Orthogonal Krylov relation basis

One invariant factorized relation operator $R$ is reused inside a layer to form

$$
Z_1=RV,
\qquad
Z_2=R^2V,
\qquad
Z_3=R^3V.
$$

Directly mixing these monomials can become poorly conditioned because repeated
averaging tends to align them with the dominant eigenspace. vNext therefore
uses graph/head-wise modified Gram-Schmidt under the invariant irrep inner
product

$$
\langle A,B\rangle=
\langle A^{0e},B^{0e}\rangle+
\langle A^{0o},B^{0o}\rangle+
\langle A^{1o},B^{1o}\rangle+
\langle A^{1e},B^{1e}\rangle+
\langle A^{2e},B^{2e}\rangle+
\langle A^{2o},B^{2o}\rangle.
$$

This produces

$$
B_1=Z_1,
\qquad
B_2=\text{orth}(Z_2;B_1),
\qquad
B_3=\text{orth}(Z_3;B_1,B_2).
$$

All projection coefficients are `0e` scalars, so the operation commutes with
O(3). Higher-order gates remain zero initialized.

## 3. Learned soft edges as a latent atlas

Predicting an arbitrary discrete edge list requires either an $N^2$ score matrix
or a candidate-generation rule. vNext instead predicts a bounded-rank soft
incidence matrix

$$
A\in\mathbb R^{N\times K},
$$

where each column is a learned chart or latent interaction hub. The induced
node relation is

$$
S=A D^{-1}A^\top,
\qquad
D_{kk}=\sum_i A_{ik}.
$$

$S$ is symmetric positive semidefinite and has rank at most $K$. Its action is
computed as node-to-chart-to-node transport,

$$
SV=A\left[D^{-1}(A^\top V)\right],
$$

with $O(NKC)$ work and no $N\times N$ object.

Assignments are refined geometrically. Initial invariant logits produce chart
centres and chart covariances. The covariance defines an SPD Mahalanobis metric,
and the logits are corrected by the corresponding chart distance. Under an
orthogonal transformation $Q$,

$$
c_k\mapsto Qc_k,
\qquad
C_k\mapsto QC_kQ^\top,
$$

so the Mahalanobis distance and chart assignment remain invariant.

The atlas message is orthogonalized against $B_1,B_2,B_3$ and enters through a
zero-initialized gate. It is the architecture's learned-connectivity path: the
model predicts soft relations internally without storing explicit edges.

## 4. Coordinate configuration manifold

A raw polar velocity $v_i$ is decomposed over each movable component into rigid
translation, rigid rotation, and internal shape motion. Let

$$
c=\frac{1}{M}\sum_i m_i x_i,
\qquad
t=\frac{1}{M}\sum_i m_i v_i,
\qquad
r_i=x_i-c.
$$

Define

$$
I=\sum_i m_i
\left(
\lVert r_i\rVert^2 I_3-r_ir_i^\top
\right),
$$

$$
L=\sum_i m_i r_i\times(v_i-t),
\qquad
\omega=(I+\epsilon I_3)^{-1}L.
$$

The internal shape tangent is

$$
u_i=v_i-t-\omega\times r_i.
$$

It satisfies, up to numerical regularization,

$$
\sum_i m_i u_i=0,
\qquad
\sum_i m_i r_i\times u_i=0.
$$

For a fully movable interaction component, global translation and rotation are
gauge directions and only $u_i$ is retained. For a partially movable component,
rigid translation and rotation are genuine pose degrees of freedom relative to
fixed context, so all three terms are retained with learned invariant gates.
This gives an automatic

$$
SE(3)\times\text{shape-space}
$$

coordinate policy without adding a public mode switch.

## 5. Role of explicit edges

Explicit edges are no longer the default geometric substrate. They are retained
only when topology is part of the observation or supervision, for example:

- covalent bonds;
- mesh connectivity;
- temporal transitions;
- metal coordination;
- curated contact candidates.

The canonical model can operate without them using relative moments, content
Krylov transport, and the latent atlas relation. A future discrete-edge decoder
may recover top-k edges from atlas or relation factors as a post-processing or
auxiliary-supervision head. Such decoding is intentionally outside the core
forward path because materializing arbitrary pair scores would violate the
node-linear memory contract.

## 6. Complexity

For $N$ nodes, hidden width $C$, $K$ latent charts, and fixed three Krylov
orders, the added terms have

$$
O(NC^2)+O(NKC)
$$

work and

$$
O(NC+C^2+NK)
$$

working memory. The third coordinate moment has a fixed Cartesian size and is
therefore node-linear. Explicit input topology, when supplied, adds the existing
sparse residual cost.

## 7. Validation targets

The implementation is required to preserve:

1. exact third-moment agreement with explicit tuple enumeration;
2. trace-free rank-three structure;
3. graphwise invariant orthogonality of the Krylov basis;
4. PSD and O(3)-equivariance of the latent atlas relation;
5. O(3) and parity correctness of transient `l=3` contractions;
6. removal of rigid gauge modes for fully movable components;
7. preservation of rigid pose motion for partially movable components;
8. the `ELAGraph -> ELA -> ELAGraph` public contract.
