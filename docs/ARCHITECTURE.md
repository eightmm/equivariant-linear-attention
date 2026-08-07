# Mathematical architecture

This document defines the mathematical operator implemented on `main`,
verified line-by-line against the source. The execution form (packing,
compact bases, fusion) is documented in `TENSOR_EXECUTION.md`; it changes
no formula below.

## 1. Public contract and persistent state

The public contract is

```text
ELAGraph -> ELA -> ELAGraph
```

`ELAGraph` carries node irreps `x`, Cartesian positions `pos`, an optional
graph id `batch`, an optional component id `group`, optional invariant
`condition` and `order` features, an optional boolean `update_mask`, and
targets/outputs. It has no edge, neighbor, cutoff, or topology field.

Each node carries one persistent state in six O(3) sectors,

$$
H_i=H_i^{0e}\oplus H_i^{0o}\oplus H_i^{1o}\oplus H_i^{1e}
\oplus H_i^{2e}\oplus H_i^{2o},
$$

with shapes

```text
0e: (C)    0o: (H)    1o: (H,3)    1e: (H,3)    2e: (H,5)    2o: (H,5)
```

where `C` is the scalar width and `H` the head count. Rank-two sectors are
symmetric-traceless (ST) tensors stored as the five components
`(xx, yy, xy, xz, yz)` with `zz = -xx - yy`. Every learned map mixes
multiplicity axes only; geometric axes pass exclusively through fixed
equivariant contractions. Every gate, normalization, and mixing
coefficient is an O(3) invariant, so all updates commute with rotations
and reflections.

## 2. Interaction segments

Attention scope is the interaction segment

$$
g=\operatorname{unique}(\mathrm{batch},\mathrm{group}),
$$

one segment per (graph, component) pair; without `group` the segment is
the graph. Centering, moments, the relation operator, Krylov
orthogonalization, and all mixture coefficients are computed inside one
segment. Attention therefore never crosses a component boundary; only the
final readout pools per graph. Nodes in one segment are provably isolated
from changes in another.

## 3. Normalized geometry

Per segment, coordinates are centered and scale-normalized:

$$
c_g=\frac{1}{|g|}\sum_{i\in g}x_i,
\qquad
\tilde x_i=x_i-c_g,
\qquad
r_g=\sqrt{\tfrac{1}{|g|}\sum_{i\in g}\lVert\tilde x_i\rVert^2+\epsilon},
\qquad
\hat x_i=\tilde x_i/r_g.
$$

All coordinate-dependent features are built from the complete symmetric
Cartesian monomials of the normalized coordinates through degree four,

$$
\psi(\hat x)\in\mathbb R^{35},
\qquad
35=1+3+6+10+15.
$$

Neither absolute coordinates nor the segment radius re-enters any
feature, so with frozen coordinates the represented function is exactly
invariant, per segment, to any similarity transform `x -> s R x + t`
with `R` orthogonal and `s > 0` (scale invariance up to the epsilon in
the radius).

## 4. Exact separable relative moments through order four

Each layer learns positive source lanes `r = 1..R` from invariants,

$$
w_{ir}=\operatorname{softplus}\!\big[(W_c H_i^{0e}+W_\rho\,\rho_i)/\tau_r\big]+\epsilon,
\qquad
\rho_i=\Big(\lVert\hat x_i\rVert^2,\ \log(1+\lVert\hat x_i\rVert^2),\
\tfrac{\lVert\hat x_i\rVert^2}{1+\lVert\hat x_i\rVert^2}\Big),
$$

and accumulates one raw sum per segment over the 35 monomials,

$$
S_{r,\beta}=\sum_{j\in g}w_{jr}\,\psi_\beta(\hat x_j).
$$

Receiver-centred moments follow from the multi-index binomial identity,
applied per monomial through a fixed signed translation table:

$$
M_{ir,\alpha}=
\frac{1}{S_{r,0}}
\sum_{\beta\le\alpha}\binom{\alpha}{\beta}(-\hat x_i)^{\alpha-\beta}S_{r,\beta}
\quad\Longleftrightarrow\quad
M^{(k)}_{ir}=\mathbb E_{w_r}\big[(\hat x_j-\hat x_i)^{\otimes k}\big],
\qquad k\le4.
$$

This is exact, uses node reductions only (no pair or tuple enumeration),
and the self term cancels identically for `k >= 1`. The angular content
decomposes as

$$
\operatorname{Sym}^1=1o,\quad
\operatorname{Sym}^2=0e\oplus2e,\quad
\operatorname{Sym}^3=1o\oplus3o,\quad
\operatorname{Sym}^4=0e\oplus2e\oplus4e.
$$

The extracted carriers are: `m1 -> 1o`; `m2 -> 0e` (trace) and `2e` (ST);
`m3 -> 3o` stored as its seven irreducible STF components;
`m4 -> 0e`, `2e` (trace matrix), and `4e` stored as nine irreducible STF
components. Odd sectors are generated from lane geometry alone,

$$
a_r=m_{1,r}\times m_{1,r+1}\ (1e),
\qquad
a_r\cdot m_{1,r+2}\ (0o),
\qquad
\operatorname{ST}\!\big(m_{1,r+2}\otimes a_r\big)\ (2o),
$$

with lane indices cyclic. All moment outputs pass through the bounded
maps of section 7.

## 5. One self-adjoint PSD relation operator

Three relations are expressed as explicit Gram factorizations and fused
into a single factor.

### Content Gram

A fused projection produces the packed value `V` and a content feature
`Fc` whose geometric blocks are normalized (vectors to the unit ball, ST
tensors to an orthonormal five-component basis). Because every block
transforms orthogonally,

$$
K^{\mathrm{content}}_{ij}=\langle F^c_i,F^c_j\rangle
$$

is invariant and PSD.

### Truncated Gaussian Mercer operator

For a per-head bandwidth `gamma_h = softplus(.) + 0.05`, the feature

$$
F^m_{i,h}
=e^{-\gamma_h\lVert\hat x_i\rVert^2}
\sqrt{\frac{(2\gamma_h)^{k}}{k!}}\,
\sqrt{\binom{k}{k_x\,k_y\,k_z}}\;
\psi(\hat x_i)\in\mathbb R^{35}
$$

(degree `k` and multinomial weight per monomial) satisfies exactly

$$
\langle F^m(x),F^m(y)\rangle
=e^{-\gamma(\lVert x\rVert^2+\lVert y\rVert^2)}
\sum_{k=0}^{4}\frac{(2\gamma)^k}{k!}(x\cdot y)^k,
$$

the order-four truncation of the Gaussian kernel
`exp(-gamma ||x - y||^2)`. The multinomial square roots make the
degree-`k` block inner product equal to `(x . y)^k` exactly.

### SPD manifold atlas

Invariant logits `l_i = W h_i + W_rho rho_i` define a two-pass partition
of unity over `K` charts. First pass:

$$
A^{(0)}=\operatorname{softmax}(\ell),
$$

then one packed reduction of the assignment against the degree-two basis
`[1, x, sym2(x)]` yields chart mass `n_k`, centre `mu_k`, and covariance
`Sigma_k`, and a regularized SPD metric

$$
G_k=\Sigma_k+\operatorname{softplus}(r_k)\,
\max\!\Big(\tfrac{\operatorname{tr}\Sigma_k}{3},1\Big)\,I.
$$

Mahalanobis distances refine the assignment with a mass-balance term:

$$
d^2_{ik}=(\hat x_i-\mu_k)^\top G_k^{-1}(\hat x_i-\mu_k)\ \ (\text{clamped to }[0,64]),
\qquad
A=\operatorname{softmax}\!\big(\ell-s_k\,d^2_{ik}-\tfrac14\log n_k\big).
$$

Statistics and metrics are recomputed from the refined assignment, the
metric is trace-normalized, and an effective dimension

$$
d^{\mathrm{eff}}_k=\frac{(\operatorname{tr}\Sigma_k)^2}{\lVert\Sigma_k\rVert_F^2}
\in[1,3],
\qquad
w_{hk}=\sigma\big(b_{hk}+s_{hk}(d^{\mathrm{eff}}_k-2)\big)
$$

sets per-head chart weights. The atlas factor and induced relation are

$$
\Phi^a_{i,h,k}=A_{ik}\sqrt{w_{hk}/n_k},
\qquad
K^{\mathrm{atlas}}=A\,\operatorname{diag}(w/n)\,A^\top,
$$

symmetric, PSD, and rank-bounded by the chart count. The node metric
`G_i = sum_k A_ik G_k` is exported to the coordinate update.

### Convex mixture and the unified factor

Per segment and head, the weights `(a_c, a_m, a_a)` are a softmax of the
segment-mean scalars. With trace normalizers `t = sum_i ||F_i||^2`, the
single fused factor is

$$
\Phi_i=\Big[\sqrt{\alpha_c/t_c}\,F^c_i,\
\sqrt{\alpha_m/t_m}\,F^m_i,\
\sqrt{\alpha_a}\,\Phi^a_i\Big],
\qquad
R=\Phi\Phi^\top,
\qquad
RV=\Phi(\Phi^\top V).
$$

Trace normalization is exact for the content and Mercer blocks; the atlas
block is mass-normalized. The relation is a scalar operator, so it acts
identically on every sector of the packed carrier `[N, H, Dh + 17]` and
one segmented Gram contraction transports all six sectors at once.

## 6. Orthogonal Krylov spectral filter

One operator is reused for three orders,

$$
Z_1=RV,\qquad Z_2=R^2V,\qquad Z_3=R^3V,
$$

followed by segment-wise modified Gram-Schmidt under the complete
invariant inner product (the mean of the six normalized sector inner
products):

$$
B_2=\operatorname{normalize}\big(Z_2-\operatorname{proj}_{Z_1}Z_2\big),
\qquad
B_3=\operatorname{normalize}\big(Z_3-\operatorname{proj}_{Z_1}Z_3-\operatorname{proj}_{B_2}Z_3\big),
$$

where projection coefficients are segment-level invariant ratios and
normalization divides by the root-mean node norm (capped). The message is
a learned low-order spectral filter,

$$
\mathrm{msg}=Z_1+c_2B_2+c_3B_3,
\qquad
(c_2,c_3)=\tanh\!\big(W\cdot\operatorname{mean}_g H^{0e}\big)
$$

per head, zero-initialized. This is one relational algebra, not three
independently parameterized attention matrices.

## 7. Parity-complete retained closure

Message sectors are written with a tilde; `E` and `O` are the even and
odd ST tensors in matrix form; `{A,B}` is the ST-projected Jordan
product; `[A,B]v` is the commutator axial vector; `⊙` is the ST
symmetric product of two vectors; `l2(v,T) = ST([v]x T - T [v]x)` is the
vector-tensor degree-two coupling. The retained update per sector is:

```text
0e <- W(msg 0e) + W(log(1+mass), tr m2, m4 scalar)
      + W(p·p~, a·a~, <E,E~>/5, <O,O~>/5)
0o <- mix(msg 0o) + mix(lane triple product)  + W(p·a~, <E,O~>/5)
1o <- mix(msg 1o) + mix(m1)                   + mix(p×a~ + [E,O~]v + E~p + O~a)
1e <- mix(msg 1e) + mix(lane axial)           + mix(p×p~ + a×a~ + [E,E~]v + [O,O~]v + E~a + O~p)
2e <- mix(msg 2e) + mix(m2 ST) + mix(m4 ST)   + mix(p⊙p~ + a⊙a~ + {E,E~} + {O,O~} + l2(a,E~) + l2(p,O~))
2o <- mix(msg 2o) + mix(lane 2o)              + mix(p⊙a~ + a⊙p~ + {E,O~} + {O,E~} + l2(a,O~) + l2(p,E~))
```

These are all parity-valid outputs through degree two of
`1 x 1`, `1 x 2`, and `2 x 2`. The transient `3o` and `4e` moment
carriers contract back into the persistent state through compact STF
formulas without materializing rank-three or rank-four tensors:

$$
3o\otimes2e\to1o,\quad
3o\otimes2o\to1e,\quad
3o\otimes1o\to2e,\quad
3o\otimes1e\to2o,
$$

$$
4e\otimes2e\to2e,\quad
4e\otimes2o\to2o,\quad
(4e\otimes2e)\otimes1o\to1o,\quad
(4e\otimes2o)\otimes1e\to1e\ \ (\text{and parity partners}).
$$

Every sector output passes through an equivariant bounded map,

$$
x\mapsto\frac{x}{\sqrt{1+x^2}},
\qquad
v\mapsto\frac{v}{\sqrt{1+\lVert v\rVert^2}},
\qquad
Q\mapsto\frac{Q}{\sqrt{1+\lVert Q\rVert^2/5}},
$$

and joins the residual stream through a learned scalar scale.

## 8. Equivariant feed-forward

The invariant vector

$$
u_i=\big[H^{0e},\ |H^{0o}|,\ \operatorname{rms}(H^{1o}),\
\operatorname{rms}(H^{1e}),\ \lVert H^{2e}\rVert/\sqrt5,\
\lVert H^{2o}\rVert/\sqrt5\big]\in\mathbb R^{C+5H}
$$

drives the scalar update `H0e <- MLP(u)` and sector gates
`g = 2 sigmoid(MLP(u))` in `(0, 2)` (zero-initialized, so `g = 1` at
start). Each geometric sector receives a parity-cross term through
pseudoscalar multiplication,

$$
H^{1o}\leftarrow g\big(\operatorname{mix}(H^{1o})+H^{0o}\operatorname{mix}(H^{1e})\big),
\qquad
H^{2e}\leftarrow g\big(\operatorname{mix}(H^{2e})+H^{0o}\operatorname{mix}(H^{2o})\big),
$$

and symmetrically for `1e`, `2o`, using only invariant coefficients.

## 9. Natural-gradient quotient coordinate update

Active only with `update_positions=True`. A polar field is predicted from
invariants and the `1o` carrier (zero-initialized head and node gate) and
preconditioned by the atlas node metric,

$$
\widetilde v_i=(G_i+\epsilon I)^{-1}v_i.
$$

Per segment, over the selected (movable) nodes:

$$
t=\operatorname{mean}\widetilde v,
\qquad
r_i=x_i-\bar x,
\qquad
\mathcal I=\sum_i\big(\lVert r_i\rVert^2I-r_ir_i^\top\big),
\qquad
\omega=(\mathcal I+\lambda I)^{-1}\sum_i r_i\times(\widetilde v_i-t),
$$

$$
u_i=\widetilde v_i-t-\omega\times r_i,
$$

so the internal shape tangent has zero total translation and zero
angular momentum. The applied step is

$$
\delta_i=g_t\,t+g_r\,(\omega\times r_i)+g_s\,u_i .
$$

If the selected nodes cover the entire component, the rigid gates are
forced to zero (`g_t = g_r = 0`): fully movable components evolve on
shape space modulo `SE(3)`. Partially movable components learn rigid pose
motion relative to their fixed context with `g_t, g_r = tanh(.)` and
`g_s = 1 + tanh(.)`. Steps are rescaled so the per-segment maximum norm
never exceeds `max_step / depth`, and the geometry of section 3 is
rebuilt after every layer.

## 10. Initialization is a near-identity operator

At initialization: `c2 = c3 = 0` so the message is exactly `RV`; all FFN
gates equal one; the coordinate step is exactly zero; residual scales are
`0.1 / sqrt(depth)`. The model starts as a stable single-relation
attention close to the identity.

## 11. Symmetry guarantees and verification

- **O(3)**: every learned coefficient is invariant and every geometric
  path is a fixed Clebsch-Gordan contraction, so each sector transforms
  exactly in its irrep, including parity.
- **Translation**: coordinates enter only through the centred normalized
  coordinates; features are translation invariant, updated positions are
  translation equivariant. Together with O(3) this gives full E(3)
  equivariance.
- **Scale**: per-segment RMS normalization makes the frozen-coordinate
  path invariant to segment-wise uniform scaling (section 3).
- **Permutation**: all reductions are segment sums; the operator is
  permutation equivariant, and interaction segments are exactly isolated.

These are enforced numerically in float64 at tolerance `5e-10` by
`tests/test_equivariance.py` (full model under rotation, reflection,
translation, and mixed irreps; node permutation; component isolation),
`tests/test_moments.py` (translation-invariant moment bank, compact vs
full-Cartesian oracle), `tests/test_relation.py` (invariant assignment,
equivariant atlas metric, packed vs dense PSD action), and
`tests/test_manifold.py` (rigid gauge removal and O(3) equivariance of
the quotient step).

## 12. Complexity and exclusions

All reductions are segment sums over nodes:

$$
O\big(N(C^2+KC)\big)
$$

time with node-linear memory, where the relation feature width combines
the content, Mercer (35), and atlas (`K`) blocks. The architecture
contains no explicit or predicted edge list, no radius or
k-nearest-neighbor search, no sparse message path, no pair or triangle
state, no topology cache, and no compatibility or migration subsystem.
