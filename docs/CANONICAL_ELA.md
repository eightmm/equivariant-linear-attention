# Canonical equivariant linear attention

This document is normative for the only public architecture and layer:

```text
ELA
ELALayer
```

## 1. Architecture

Every layer evaluates one homogeneous operator:

\[
\boxed{
\operatorname{ELA}
=
\operatorname{ExactGlobalLinearAttention}
+
\operatorname{ExactSparseLocalResidual}
}
\]

Molecules, proteins, particles, point clouds, and meshes use the same equations.
They differ only in input irreps, sparse geometry, optional context, and task
heads.

The persistent hidden carrier is derived internally:

\[
C_0\times0e
\oplus C_{0o}\times0o
\oplus C_{1o}\times1o
\oplus C_{1e}\times1e
\oplus C_{2e}\times2e
\oplus C_{2o}\times2o.
\]

Users do not select hidden irreps, head count, or local rank.

## 2. Inputs

The base call receives

\[
(X,x,\mathcal G),
\]

where

- `X` is a flattened `input_irreps` tensor;
- `x` is an affine position tensor `[N,3]`;
- `G` is a validated sparse candidate graph.

Positions remain separate because

\[
x_i\mapsto Rx_i+t,
\]

while an irrep feature transforms linearly.

Optional information is supplied by one `ELAContext`:

\[
c=(c_{0e},o,r),
\]

where `condition`, `order`, and `refinement` may each be absent. Their absence
bypasses the corresponding path rather than selecting another model.

## 3. Global and local branches

For layer \(\ell\),

\[
\bar h_i^\ell
=
\operatorname{EqRMSNorm}_{\rm attn}(h_i^\ell).
\]

The exact global branch is

\[
G_i^\ell
=
\operatorname{ExactGlobalELA}_{l\le2}(\bar h^\ell)_i.
\]

It uses a finite positive feature map, graph sufficient statistics, and an
augmented numerator/denominator contraction. It does not form an `N x N`
attention matrix.

The exact sparse local branch is

\[
L_i^\ell
=
\operatorname{ExactSparseLocal}_{l\le2}
(\bar h^\ell,x,\mathcal E)_i.
\]

It retains compact cutoff, radial basis, edge direction, relation metadata,
tensor contractions, receiver normalization, and aggregate chirality. It has no
persistent edge hidden state.

## 4. Branch-aware fusion

For sector

\[
\tau\in\{0e,0o,1o,1e,2e,2o\},
\]

define invariant branch magnitudes

\[
r_{G,i}^{\tau}
=
\sqrt{\operatorname{RMS}(G_i^\tau)^2+\epsilon},
\qquad
r_{L,i}^{\tau}
=
\sqrt{\operatorname{RMS}(L_i^\tau)^2+\epsilon}.
\]

For compact ST5 tensors, the RMS uses the represented Frobenius norm rather than
a plain mean of the five stored coordinates.

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

Positive two-way weights are

\[
(w_{G,i}^{\tau},w_{L,i}^{\tau})
=
2\operatorname{softmax}[R_\tau(z_i)].
\]

The same invariant scalar weight is broadcast across every component of one
irrep sector.

The router and branch-balance parameters are zero initialized:

\[
R_\tau=0,
\qquad
\beta_\tau=0.
\]

Therefore

\[
w_G^\tau=w_L^\tau=1,
\qquad
M_i^\tau=G_i^\tau+L_i^\tau
\]

at initialization. The model starts from the established additive equation and
learns branch preference only when supported by gradients.

## 5. Update and FFN

The fused message enters one parity-complete update and one low-order tensor
closure:

\[
\Delta h_{i,\rm msg}^{\ell}
=
\operatorname{ParityUpdate}(M_i^\ell),
\]

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

Then

\[
\widehat h_i^\ell
=
\operatorname{EqRMSNorm}_{\rm ffn}(\widetilde h_i^\ell),
\]

\[
h_i^{\ell+1}
=
\widetilde h_i^\ell
+
\operatorname{LayerScale}_{\rm ffn}
\operatorname{EqFFN}(\widehat h_i^\ell).
\]

Even-scalar nonlinearities operate directly. Pseudoscalar, vector, and tensor
updates use invariant scalar gates.

## 6. Optional invariant condition

`ELAFeatures.condition_dim > 0` allocates zero-initialized DiT-style modulation.
For a node condition \(c_i\in0e\), even scalar states receive bounded affine
modulation and non-scalar sectors receive copy-wise invariant scale only.

If `ELAContext.condition` is absent, `ELALayer` bypasses its conditioner
entirely. This remains true after conditioner weights and biases have trained.
A configured condition is therefore genuinely switchable per call.

Vector or tensor conditions are regular `input_irreps`, not invariant condition
vectors.

## 7. Optional semantic order

`ELAFeatures.order_dim > 0` allocates an invariant Fourier encoder. `OrderContext`
contains node-attached semantic coordinates, optional segment IDs, periodicity,
and an enable mask.

The contract is

\[
F(PX,Px,PGP^T,Po,Pm)=PF(X,x,G,o,m).
\]

Semantic coordinates may represent residue rank, polymer rank, time, or stable
grid coordinates. The current tensor row index is never inferred as order.

Disabled nodes contribute no order statistics and receive zero order PE. This
supports mixed systems such as an ordered protein and an unordered ligand.

Order PE is an ordinary `0e` condition and uses the same layer modulation path as
other invariant context.

## 8. Optional coordinate refinement

`ELAFeatures.coordinate_refinement=True` allocates one zero-initialized
`1o` displacement head. A `RefinementRequest` activates an outer loop:

\[
h^t=\operatorname{ELA}(X,x^t,\mathcal G^t;c),
\]

\[
\Delta x^t
=
\operatorname{BoundedVectorHead}(h^t),
\]

\[
x^{t+1}=x^t+\Delta x^t.
\]

The request controls step count, maximum displacement, update mask, centering,
and optional graph reconstruction. Without a rebuilder, candidate topology is
reused while continuous geometry is recomputed.

This is still the same `ELA` architecture; refinement is an execution mode, not
a second backbone or layer class.

For conservative forces use

\[
F_i=-\nabla_{x_i}E.
\]

## 9. Complexity

For `N` nodes, `E` directed candidates, `L` layers, and fixed widths/ranks,

\[
T=O\left(L(N+E)\right).
\]

Branch routing and optional node-level conditioning add `O(LN)`. With `S`
refinement steps, the stack is evaluated approximately `S+1` times, so the order
is

\[
O\left((S+1)L(N+E)\right)
\]

before neighbor reconstruction cost.

Node-linear arithmetic additionally requires

\[
E=O(N).
\]

Neighbor discovery is outside the layer contract.

## 10. Public architecture policy

The package root exposes `ELA` and `ELALayer` as the only backbone and
architecture layer. Optional capability does not create names such as
`ConditionedELA`, `OrderConditionedELA`, `ELACoordinateRefiner`, implicit ELA, or
AttnRes ELA.

Historical numerical reference modules may remain internal while canonical ELA
depends on them or migration provenance requires them. They are not selectable
public architectures.
