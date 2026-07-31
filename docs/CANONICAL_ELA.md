# Canonical equivariant linear attention

This document is normative for `ELA`, `ELAConfig`, `ELALayer`, and `ELACore`.

## 1. Scope

The canonical layer is one homogeneous 3D operator:

\[
\boxed{
\operatorname{ELA}
=
\operatorname{ExactGlobalLinearAttention}
+
\operatorname{ExactSparseLocalResidual}
}
\]

It is not selected by dataset type. Molecules, proteins, particles, point
clouds, and meshes use the same equations and differ only in input features,
geometry construction, and task heads.

The persistent hidden carrier is

\[
C_0\times0e
\oplus C_{0o}\times0o
\oplus C_{1o}\times1o
\oplus C_{1e}\times1e
\oplus C_{2e}\times2e
\oplus C_{2o}\times2o.
\]

The optimized implementation currently derives all non-scalar multiplicities
from model width. Users do not choose hidden irreps.

## 2. Inputs

The model receives

\[
(X,x,\mathcal G),
\]

where

- `X` is one flattened `input_irreps` tensor;
- `x` is an affine position tensor of shape `[N,3]`;
- `G` is one prepacked sparse candidate graph.

Positions remain separate because

\[
x_i\mapsto Rx_i+t,
\]

while an irrep feature transforms linearly.

## 3. Pre-normalized branches

For layer `ell`,

\[
\bar h_i^\ell
=
\operatorname{EqRMSNorm}_{\rm attn}(h_i^\ell).
\]

The same normalized state enters two branches.

### Exact global branch

\[
G_i^\ell
=
\operatorname{ExactGlobalELA}_{l\le2}(\bar h^\ell)_i.
\]

The global operator uses a finite positive feature map, graph sufficient
statistics, and an augmented numerator/denominator contraction. It does not form
a dense pair-attention matrix.

### Exact sparse local branch

\[
L_i^\ell
=
\operatorname{ExactSparseLocal}_{l\le2}
(\bar h^\ell,x,\mathcal E)_i.
\]

The local branch retains compact cutoff, radial basis, edge direction, relation
metadata, tensor contractions, receiver normalization, and aggregate chirality.
It has no persistent edge hidden state.

## 4. Branch-aware fusion

Raw addition hides branch identity and makes its relative scale an accidental
property of projection widths, degree, and graph size. Canonical ELA keeps the
branches separate until an invariant router.

For sector

\[
\tau\in\{0e,0o,1o,1e,2e,2o\},
\]

define the scalar/vector RMS with the ordinary Euclidean component metric:

\[
r_{G,i}^{\tau}
=
\sqrt{
\operatorname{mean}_{c,m}
\left[(G_{i,c,m}^{\tau})^2\right]
+\epsilon
},
\]

\[
r_{L,i}^{\tau}
=
\sqrt{
\operatorname{mean}_{c,m}
\left[(L_{i,c,m}^{\tau})^2\right]
+\epsilon
}.
\]

For the compact Cartesian \(l=2\) storage
\(T=[xx,yy,xy,xz,yz]\), the five stored coordinates are not an
orthonormal basis. The router therefore uses the O(3)-invariant Frobenius
metric

\[
\lVert T\rVert_F^2
=
xx^2+yy^2+(xx+yy)^2
+2(xy^2+xz^2+yz^2),
\]

followed by the same channel mean and division by five. This distinction is
required: a plain mean of the five stored coordinates changes under a generic
rotation even though the represented traceless tensor does not.

The router input is invariant:

\[
z_i
=
\left[
\bar h_i^{0e},
\log r_{G,i}^{0e},\ldots,\log r_{G,i}^{2o},
\log r_{L,i}^{0e},\ldots,\log r_{L,i}^{2o}
\right].
\]

One positive two-way weight is produced per node and sector:

\[
(w_{G,i}^{\tau},w_{L,i}^{\tau})
=
2\operatorname{softmax}
\left[R_\tau(z_i)\right].
\]

The same scalar weight is broadcast over every component of the corresponding
irrep copy. The operation therefore commutes with the tracked O(3) action.

The raw routed message is

\[
M_{i,\rm raw}^{\tau}
=
w_{G,i}^{\tau}G_i^{\tau}
+w_{L,i}^{\tau}L_i^{\tau}.
\]

A variance-balanced candidate is

\[
M_{i,\rm bal}^{\tau}
=
\frac{
\sqrt{\frac12[(r_G^\tau)^2+(r_L^\tau)^2]}
}{
\sqrt{\frac12[(w_G^\tau)^2+(w_L^\tau)^2]+\epsilon}
}
\left[
 w_G^\tau\frac{G_i^\tau}{r_G^\tau}
+w_L^\tau\frac{L_i^\tau}{r_L^\tau}
\right].
\]

The actual fusion is

\[
M_i^\tau
=
M_{i,\rm raw}^{\tau}
+	anh(\beta_\tau)
\left(
M_{i,\rm bal}^{\tau}-M_{i,\rm raw}^{\tau}
\right).
\]

Initialization is

\[
R_\tau=0,
\qquad
\beta_\tau=0.
\]

Therefore

\[
w_G^\tau=w_L^\tau=1,
\qquad
M_i^\tau=G_i^\tau+L_i^\tau.
\]

The canonical model begins at the admitted incumbent equation and can learn a
branch preference. The registered 2026-07-31 packet found active, finite routing
and a one-seed QM9 gain, but rejected empirical promotion because resource and
LBA capacity gates failed. Trainable routing is therefore an experimental
mechanism, not a claimed universal improvement.

The extra local pseudoscalar used by the parity update follows the local `0o`
weight. It is not treated as an independent seventh architecture branch.

## 5. Update and tensor closure

The fused message enters one parity-complete Cartesian update:

\[
\Delta h_{i,\rm msg}^{\ell}
=
\operatorname{ParityUpdate}(M_i^\ell).
\]

Low-order tensor products are closed once per layer:

\[
\Delta h_{i,\rm tp}^{\ell}
=
\operatorname{TPClosure}_{l\le2}
\left(
\widetilde h_i^\ell,
A_i^{\rm multipole}
\right).
\]

The attention residual is

\[
\widetilde h_i^\ell
=
h_i^\ell
+
\operatorname{LayerScale}_{\rm attn}
\operatorname{NormGate}
\left(
\Delta h_{i,\rm msg}^{\ell}
+
\Delta h_{i,\rm tp}^{\ell}
\right).
\]

Each residual may use irrep-copy dropout and graph-wise stochastic depth in the
advanced compatibility class. Canonical `ELAConfig` fixes both to zero so they
belong to training policy rather than architecture selection.

## 6. Equivariant FFN

\[
\widehat h_i^\ell
=
\operatorname{EqRMSNorm}_{\rm ffn}
(\widetilde h_i^\ell),
\]

\[
h_i^{\ell+1}
=
\widetilde h_i^\ell
+
\operatorname{LayerScale}_{\rm ffn}
\operatorname{EqFFN}
(\widehat h_i^\ell).
\]

Even-scalar nonlinearities operate directly. Pseudoscalar, vector, and tensor
updates use invariant scalar gates, preserving parity and orientation laws.

## 7. Geometry and chirality

One coordinate context supplies:

- graph-centered normalized positions;
- C2 cutoff and radial basis;
- receiver-centered `l<=2` multipoles;
- three radial direction moments;
- aggregate axial, pseudoscalar, and odd-tensor chirality carriers.

Chirality is created without explicit edge triplets. It remains a local
high-frequency construction and is not replaced by the smooth implicit kernel.

## 8. Complexity

For `N` nodes, `E` directed candidates, `L` layers, and fixed architecture
widths/ranks,

\[
T=O\left(L(N+E)\right).
\]

The router adds `O(LN)` arithmetic and six scalar branch statistics per node. It
does not alter the asymptotic order or add edge activations.

No unconditional node-linear wall-clock claim is made. Node-linear arithmetic
requires

\[
E=O(N).
\]

Neighbor construction is outside this contract.

## 9. Excluded canonical options

The following mechanisms remain implemented for experiments and reproducibility
but are not canonical choices:

- edge-free Gaussian--Taylor full-state transport;
- periodic implicit schedules;
- always-on explicit/implicit hybrid residuals;
- block Attention Residuals;
- in-layer coordinate mutation;
- historical LGL and broad flag-driven model builders.

Their retained implementations do not imply endorsement. See
`API_POLICY.md` and `ARCHITECTURE_DECISION_20260731.md`.

## 10. Task-specific extensions

Task behavior belongs outside the core:

- coordinate refinement: `ELACoordinateRefiner`;
- conservative forces: scalar energy plus `-grad_x E`;
- invariant conditioning and deep-stack AttnRes: advanced/experimental classes;
- readout masks, ligand/pocket roles, and task heads: adapters;
- neighbor construction and rebuild: geometry providers.

This separation keeps the layer one mathematical object while allowing different
3D applications to compose it explicitly.
