# Canonical equivariant linear attention

This document is normative for the only public architecture, layer, and graph
container:

```text
ELA
ELALayer
ELABatch
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
They differ only in declared irreps, sparse geometry, optional context, and task
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

## 2. Public input contract

Representations are configured only through

```text
input_irreps
output_irreps
```

A scalar-only width is still an irrep declaration, for example `"32x0e"`.
There is no parallel `node_dim`, `output_dim`, or scalar-only model API.

The public call receives one `ELABatch`:

\[
\mathcal B=(X,x,\operatorname{ptr},\mathcal E,c,o,r),
\]

where

- `X` is the flattened `input_irreps` tensor;
- `x` is an affine position tensor `[N,3]`;
- `ptr` partitions packed nodes into graphs;
- `E` is optional sparse candidate topology;
- `c`, `o`, and `r` are optional condition, semantic order, and refinement data.

Positions remain separate because

\[
x_i\mapsto Rx_i+t,
\]

while an irrep feature transforms linearly.

If no edge topology is supplied, preparation constructs exact radius candidates.
The numerical layer always receives a prepared receiver-major sparse graph.

## 3. Global and local branches

For layer `l`:

\[
\bar h_i^l
=
\operatorname{EqRMSNorm}_{\rm attn}(h_i^l).
\]

The exact global branch is

\[
G_i^l
=
\operatorname{ExactGlobalELA}_{l\le2}(\bar h^l)_i.
\]

It uses a finite positive feature map, graph sufficient statistics, and an
augmented numerator/denominator contraction. It does not form an `N x N`
attention matrix.

The exact sparse local branch is

\[
L_i^l
=
\operatorname{ExactSparseLocal}_{l\le2}
(\bar h^l,x,\mathcal E)_i.
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

For compact ST5 tensors, RMS uses the represented Frobenius norm rather than a
plain mean of the five stored coordinates.

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

The same invariant scalar weight is broadcast over every component of one irrep
sector.

Router and branch-balance parameters are zero initialized:

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
learns branch preference only when gradients support it.

## 5. Update and FFN

The fused message enters one parity-complete update and one low-order tensor
closure:

\[
\Delta h_{i,\rm msg}^{l}
=
\operatorname{ParityUpdate}(M_i^l),
\]

\[
\Delta h_{i,\rm tp}^{l}
=
\operatorname{TPClosure}_{l\le2}
\left(
\widetilde h_i^l,
A_i^{\rm multipole}
\right).
\]

The attention residual is

\[
\widetilde h_i^l
=
h_i^l
+
\operatorname{LayerScale}_{\rm attn}
\operatorname{NormGate}
\left(
\Delta h_{i,\rm msg}^{l}
+
\Delta h_{i,\rm tp}^{l}
\right).
\]

Then

\[
\widehat h_i^l
=
\operatorname{EqRMSNorm}_{\rm ffn}(\widetilde h_i^l),
\]

\[
h_i^{l+1}
=
\widetilde h_i^l
+
\operatorname{LayerScale}_{\rm ffn}
\operatorname{EqFFN}(\widehat h_i^l).
\]

Even-scalar nonlinearities operate directly. Pseudoscalar, vector, and tensor
updates use invariant scalar gates.

## 6. Optional invariant condition

`ELAFeatures.condition_dim > 0` allocates zero-initialized DiT-style modulation.
For condition `c_i in 0e`, even scalars receive bounded affine modulation and
non-scalar sectors receive copy-wise invariant scale only.

If `ELABatch.condition` is absent, every layer bypasses its conditioner entirely,
including after conditioner parameters have trained. A configured condition is
therefore switchable per call.

Vector or tensor conditions are regular `input_irreps`, not invariant condition
vectors.

## 7. Optional semantic order

`ELAFeatures.order_dim > 0` allocates an invariant Fourier encoder.
`ELABatch.order` contains node-attached semantic coordinates, optional segment
IDs, periodicity, and an enable mask.

The contract is

\[
F(PX,Px,PGP^T,Po,Pm)=PF(X,x,G,o,m).
\]

Semantic coordinates may represent residue rank, polymer rank, time, or stable
grid coordinates. Tensor row index is never inferred as order.

Disabled nodes contribute no order statistics and receive zero order PE. This
supports mixed systems such as an ordered protein and unordered ligand.

Order PE is ordinary `0e` context and uses the same layer modulation path as
other invariant condition.

## 8. Optional coordinate refinement

`ELAFeatures.coordinate_refinement=True` allocates one zero-initialized `1o`
displacement head. A `RefinementRequest` in the batch activates an outer loop:

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

Refinement remains an execution mode of the same ELA architecture. For
conservative forces use

\[
F_i=-\nabla_{x_i}E.
\]

## 9. Readouts

The projected output is returned per node. Two graph reductions are provided:

\[
h_g^{\rm sum}=\sum_{i\in g}h_i,
\qquad
h_g^{\rm mean}=\frac{h_g^{\rm sum}}{N_g}.
\]

`graph_irreps` and `graph` denote the mean. `graph_sum` is available for
extensive additive quantities. These reductions preserve the transformation law
of each output irrep sector.

## 10. Complexity

For `N` nodes, `E` directed candidates, `L` layers, and fixed widths/ranks:

\[
T=O\left(L(N+E)\right).
\]

Branch routing and optional node-level condition add `O(LN)`. With `S`
refinement steps, the stack is evaluated approximately `S+1` times:

\[
O\left((S+1)L(N+E)\right)
\]

before neighbor reconstruction cost.

Node-linear arithmetic additionally requires

\[
E=O(N).
\]

## 11. Public architecture policy

The package root exposes `ELA` and `ELALayer` as the only backbone and
architecture layer, and `ELABatch` as the only graph container. Optional
capability does not create names such as `ConditionedELA`,
`OrderConditionedELA`, coordinate-refiner ELA, implicit ELA, or AttnRes ELA.

Historical numerical modules may remain private while the canonical
implementation or checkpoint migration depends on them. They are not selectable
public architectures.
