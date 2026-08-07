# Mathematical architecture

## 1. Persistent vector bundle

Each node carries

$$
H_i=H_i^{0e}\oplus H_i^{0o}\oplus H_i^{1o}\oplus H_i^{1e}
\oplus H_i^{2e}\oplus H_i^{2o}.
$$

Learned maps mix multiplicity axes only. Every gate and normalization
coefficient is invariant, so all updates commute with O(3), including
reflections.

## 2. Exact separable moments through order four

For a learned positive source lane `r`, define raw component sums

$$
S_r^{(a)}=\sum_j w_{jr}x_j^{\otimes a},
\qquad a=0,1,2,3,4.
$$

Receiver-centred moments follow from the symmetric binomial identity

$$
M_{ir}^{(k)}=
\frac{1}{S_r^{(0)}}
\sum_{a=0}^{k}{k\choose a}
(-x_i)^{\otimes(k-a)}\odot S_r^{(a)}.
$$

The self contribution cancels exactly. Therefore all selected higher-body
aggregate geometry is computed by node reductions, not pair or tuple
enumeration.

The angular decomposition is

$$
\operatorname{Sym}^1(\mathbb R^3)=1o,
$$

$$
\operatorname{Sym}^2(\mathbb R^3)=0e\oplus2e,
$$

$$
\operatorname{Sym}^3(\mathbb R^3)=1o\oplus3o,
$$

$$
\operatorname{Sym}^4(\mathbb R^3)=0e\oplus2e\oplus4e.
$$

The `3o` and `4e` tensors are transient. They affect the persistent state through

$$
3o\otimes2e\to1o,
\quad
3o\otimes2o\to1e,
$$

$$
3o\otimes1o\to2e,
\quad
3o\otimes1e\to2o,
$$

$$
4e\otimes2e\to2e,
\quad
4e\otimes2o\to2o.
$$

## 3. Self-adjoint relation geometry

### Content Gram operator

Projected scalar, pseudoscalar, polar, axial, and tensor blocks are concatenated
into an equivariant feature map `F`. Since each block transforms by an
orthogonal representation,

$$
K^{\mathrm{content}}_{ij}=\langle F_i,F_j\rangle
$$

is invariant and PSD. Graph/head trace normalization preserves self-adjointness
and bounds the spectral scale.

### Isotropic Mercer operator

For `gamma > 0`,

$$
e^{-\gamma\lVert x-y\rVert^2}
=e^{-\gamma\lVert x\rVert^2}e^{-\gamma\lVert y\rVert^2}
\sum_{k=0}^{\infty}
\frac{(2\gamma)^k}{k!}
\langle x^{\otimes k},y^{\otimes k}\rangle.
$$

ELA retains complete Cartesian powers through `k=4`. The finite feature map
remains exactly O(3)-invariant and PSD.

### SPD manifold atlas

Invariant logits define a partition of unity

$$
A_{ik}\ge0,
\qquad
\sum_k A_{ik}=1.
$$

Each chart has an equivariant centre, covariance, and regularized SPD metric.
Mahalanobis distances refine assignments. The induced relation

$$
K^{\mathrm{atlas}}=A D^{-1}W A^\top
$$

is symmetric, PSD, and rank-bounded by the chart count. Its action is evaluated
as node-to-chart-to-node transport.

A graph/head-level invariant convex combination of the content, Mercer, and
atlas operators remains self-adjoint and PSD.

## 4. Orthogonal Krylov spectral filter

One shared operator is reused:

$$
Z_1=RV,
\qquad
Z_2=R^2V,
\qquad
Z_3=R^3V.
$$

Modified Gram-Schmidt under the complete invariant irrep inner product produces
an orthogonal basis `B1,B2,B3`. Graph/head-level invariant coefficients then
form a learned low-order spectral filter. This is one relational algebra, not
three independently parameterized attention matrices.

## 5. Retained tensor closure

The persistent closure contains the parity-valid outputs through degree two from

$$
1\otimes1,
\qquad
1\otimes2,
\qquad
2\otimes2.
$$

Cartesian dot, cross, symmetric-traceless, commutator, Jordan, and vector-tensor
projections realize these couplings directly. The transient `3o` and `4e`
contractions supplement the retained bandwidth without increasing persistent
memory.

## 6. Natural-gradient quotient update

The atlas supplies a node-wise SPD metric `G_i`. A predicted polar field is
preconditioned by

$$
\widetilde v_i=(G_i+\epsilon I)^{-1}v_i.
$$

For a movable component,

$$
t=\frac1M\sum_i\widetilde v_i,
$$

$$
I=\sum_i(\lVert r_i\rVert^2I-r_ir_i^\top),
$$

$$
\omega=(I+\epsilon I)^{-1}
\sum_i r_i\times(\widetilde v_i-t),
$$

$$
u_i=\widetilde v_i-t-\omega\times r_i.
$$

The internal tangent has zero total translation and angular momentum. Fully
movable components are updated on shape space modulo `SE(3)`; partially movable
components may learn rigid pose motion relative to fixed context.

## 7. Exclusions

The architecture contains no explicit or predicted discrete edge list, radius
or k-nearest-neighbor search, sparse message path, pair state, triangle state,
topology cache, compatibility engine, or migration subsystem.
