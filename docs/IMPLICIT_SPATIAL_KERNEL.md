# Edge-free implicit spatial kernel

This document distinguishes three different meanings of “not using edges” and
defines the reference implementation in `implicit_spatial.py`.

## 1. What can and cannot be removed

A hard local neighborhood

\[
\mathcal N(i)=\{j:\lVert x_i-x_j\rVert<R_c\}
\]

cannot generally be evaluated exactly without discovering which point pairs
satisfy the inequality. There are three implementation families:

1. **Materialized sparse graph** — discover neighbors and store receiver/sender
   arrays. This is the current exact local path.
2. **On-the-fly neighborhood kernel** — build a cell list, spatial hash, or
   Verlet structure and enumerate nearby candidates inside a fused kernel. Edge
   tensors are not retained, but neighborhood discovery still occurs.
3. **Implicit spatial kernel** — replace the hard adjacency indicator by a
   low-rank isotropic kernel and evaluate it through sufficient statistics. No
   edge list, neighbor discovery, or node-pair matrix is used. The local
   neighborhood is approximate and soft.

The repository now implements family 3 as a reference operator. Family 2 is a
future fused-backend option and must preserve the same mathematical contract as
the explicit sparse path if it is advertised as exact.

## 2. Gaussian--Taylor feature map

For graph-centered coordinates

\[
z_i=x_i-\mu_{g(i)},
\qquad
\mu_g=\frac1{N_g}\sum_{i\in g}x_i,
\]

and one length scale \(\sigma\), define \(u_i=z_i/\sigma\). The exact Gaussian
kernel is

\[
\kappa_\sigma(i,j)
=
\exp\left[-\frac{\lVert z_i-z_j\rVert^2}{2\sigma^2}\right]
=
\exp(-\lVert u_i\rVert^2/2)
\exp(-\lVert u_j\rVert^2/2)
\exp(u_i^Tu_j).
\]

The implementation truncates the last exponential at degree two:

\[
\exp(t)\approx P_2(t)=1+t+\frac{t^2}{2}.
\]

`P_2(t)` is strictly positive for every real `t`. It admits an explicit
Cartesian feature map

\[
\phi_\sigma(u)
=
\exp(-\lVert u\rVert^2/2)
\left[
1,
 u,
 \frac{\operatorname{ST}_{\rm orth}(uu^T)}{\sqrt2},
 \frac{\lVert u\rVert^2}{\sqrt6}
\right]
\in\mathbb R^{10},
\]

such that

\[
\phi_\sigma(u_i)^T\phi_\sigma(u_j)
=
\exp(-\lVert u_i\rVert^2/2)
\exp(-\lVert u_j\rVert^2/2)
\left[
1+u_i^Tu_j+\frac{(u_i^Tu_j)^2}{2}
\right].
\]

The quadratic identity follows from

\[
\operatorname{ST}(u_i):\operatorname{ST}(u_j)
=
(u_i^Tu_j)^2
-\frac13\lVert u_i\rVert^2\lVert u_j\rVert^2.
\]

For multiple scales, positive mixture weights \(\pi_s\) are represented by
concatenating

\[
\Phi(x)=
\operatorname{concat}_s
\sqrt{\pi_s}\phi_{\sigma_s}(x).
\]

The default weights are uniform. They may be learned through a softmax without
losing positive semidefiniteness or pointwise positivity.

## 3. Edge-free transport

Given node values \(v_j\), the approximate spatial message is

\[
M_i
=
\sum_{j\in g(i)}
\widetilde\kappa(i,j)v_j.
\]

With \(\widetilde\kappa(i,j)=\Phi_i^T\Phi_j\), compute one graph statistic

\[
A_g=\sum_{j\in g}\Phi_j\otimes v_j,
\]

then

\[
M_i=\Phi_i^TA_{g(i)}.
\]

No \(N\times N\) matrix is formed. Self-interaction can be removed exactly by
subtracting

\[
(\Phi_i^T\Phi_i)v_i.
\]

The default local-style normalization is

\[
\widehat M_i
=
\frac{M_i}{1+m_i},
\qquad
m_i=
\sum_{j\ne i}\widetilde\kappa(i,j).
\]

`normalization="mass"` and `"none"` are also available.

## 4. Implicit relative moments

The same sufficient statistics produce receiver-centered geometric moments.
Let

\[
F_i=\sum_j\widetilde\kappa(i,j)z_j.
\]

Then the relative first moment is

\[
R_i^{1o}
=
\frac{F_i-m_i z_i}{1+m_i}.
\]

For \(Q(z)=\operatorname{ST}(zz^T)\), let

\[
H_i=\sum_j\widetilde\kappa(i,j)Q(z_j).
\]

The relative second moment is

\[
R_i^{2e}
=
\frac{
H_i+m_iQ(z_i)-2\operatorname{ST}(F_i,z_i)
}{1+m_i}.
\]

Both are computed without explicit pairs. Translation invariance follows from
graph centering. The scalar kernel is O(3)-invariant, so `1o`, `1e`, `2e`, and
`2o` values transported with the same weights preserve their transformation
laws.

## 5. API

```python
from equivariant_attention import (
    ImplicitGaussianSpatialKernel,
    ImplicitSpatialKernelConfig,
    ImplicitSpatialStateTransport,
)

kernel = ImplicitGaussianSpatialKernel(
    ImplicitSpatialKernelConfig(
        scales=(1.0, 2.0, 4.0),
        order=2,
        exclude_self=True,
        normalization="one_plus_mass",
        chunk_size=2048,
    )
)

message = kernel(values, positions, batch)
node_message = message.output
receiver_mass = message.mass

moments = kernel.moments(positions, batch)
relative_vector = moments.relative_vector
relative_tensor = moments.relative_tensor

state_transport = ImplicitSpatialStateTransport(kernel)
new_state = state_transport(state, positions, batch)
```

The hot path accepts only values, coordinates, and graph membership. It does not
accept or create `edge_index`.

## 6. Complexity

Let

- \(N\) be the node count;
- \(G\) be the graph count;
- \(F=10S\) be the feature rank for `S` order-two scales;
- \(D\) be the transported value width;
- \(A\) be the number of applications;
- \(C\) be the bounded node chunk size.

The arithmetic order is

\[
T_{\rm implicit}=O(ANFD).
\]

For no-grad inference, the chunked reference schedule stores one graph statistic of shape
\(G\times F\times D\) and one temporary node chunk of shape
\(C\times F\times D\):

\[
M_{\rm implicit,infer}
=
O\left(
N(F+D)+GFD+CFD
\right).
\]

Eager autograd retains chunk contractions across the full node axis. Without
checkpointing, training therefore adds an \(O(ANFD)\) saved-activation term.
For fixed scales, value width, applications, and chunk size, arithmetic and
memory remain linear in node count; when the number of graphs grows with \(N\),
the \(GFD\) term is itself linear in \(N\).

The implementation accumulates graph sufficient statistics by chunked
`index_add`, then evaluates chunked contractions. It therefore avoids both an
\(N\times N\) pair matrix and a full \(N\times F\times D\) outer tensor.

A production backend may add direct single-graph, padded, bucketed, and ragged
GEMM schedules analogous to the global linear-attention backend. Those schedules
must evaluate the same finite-feature kernel.

## 7. Accuracy boundary

The operator approximates a smooth Gaussian mixture, not the indicator

\[
\mathbf 1[\lVert x_i-x_j\rVert<R_c].
\]

A sharper neighborhood requires more feature rank, a different basis, or
on-the-fly exact candidate discovery. Degree-two Taylor accuracy degrades when
\(|u_i^Tu_j|\) is large. Multiple length scales mitigate this but do not make the
kernel exact.

The implemented reference should therefore be evaluated against:

1. its dense feature-kernel reference — exact equality is required;
2. exact Gaussian pair sums — approximation error is reported;
3. explicit cutoff local transport — task-dependent quality is reported;
4. O(3), translation, batching, gradients, and double-backward contracts;
5. node-size scaling and peak memory.

It must not be described as an exact neighbor-list replacement until those
comparisons justify such a claim.

## 8. Relationship to prior work

The design is conceptually related to kernel factorizations that remove explicit
neighbor grouping, such as VecKM, and to reciprocal-space sufficient statistics
used by Ewald message passing. This implementation differs by using a compact
O(3)-compatible Gaussian--Taylor feature map matched to the `l<=2` Cartesian
carrier of equivariant linear attention.

For periodic long-range systems, an Ewald/Fourier feature backend is a more
natural future extension than graph centering. For exact short-range physics, an
on-the-fly cell-list kernel remains the preferred eventual backend.
