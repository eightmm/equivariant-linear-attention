# Soft-local geometry without explicit edges

The base ELA moment bank is receiver-centred but segment-global: every source
node in an interaction segment contributes to every receiver. The local Mercer
branch adds receiver-dependent spatial support while preserving the edge-free
public contract and node-linear memory.

## Degree-two Gaussian Mercer kernel

For normalized centred coordinates `x`, lane `r` learns a positive inverse
length scale

$$
\gamma_r = \mathrm{softplus}(\theta_r) + 0.05.
$$

The complete symmetric Cartesian basis through degree two gives the feature
map

$$
\phi_{r,\alpha}(x)
=
\exp(-\gamma_r\lVert x\rVert^2)
\sqrt{\frac{(2\gamma_r)^{|\alpha|}}{|\alpha|!}}
\sqrt{\binom{|\alpha|}{\alpha}}
 x^\alpha,
\qquad |\alpha|\leq2.
$$

Its inner product is

$$
K_r(x,y)
=
\exp[-\gamma_r(\lVert x\rVert^2+\lVert y\rVert^2)]
\left[1+2\gamma_r x\cdot y+2\gamma_r^2(x\cdot y)^2\right].
$$

This is the degree-two truncation of
`exp(-gamma_r ||x-y||^2)`. It is rotationally invariant, PSD by explicit Gram
factorization, and pointwise positive because `1 + t + t^2 / 2 > 0`.
Different lanes learn different bandwidths, from broad context to sharply
local density.

## Receiver-local moments by segmented reductions

Positive source weights `w_jr` are predicted from scalar content and radial
invariants. For the ten monomials `psi_beta` through degree two, one packed
segment reduction forms

$$
S_{g,r,p,\beta}
=
\sum_{j\in g}
w_{jr}\,\phi_{r,p}(x_j)\,\psi_\beta(x_j).
$$

Each receiver contracts its own feature map against that summary,

$$
T_{i,r,\beta}
=
\sum_p \phi_{r,p}(x_i)S_{g(i),r,p,\beta}
=
\sum_{j\in g(i)}w_{jr}K_r(x_i,x_j)\psi_\beta(x_j).
$$

The same fixed binomial translation table used by the global moment bank then
produces normalized receiver-centred `m1` and `m2`. No `N x N` kernel matrix,
edge list, radius graph, or persistent pair state is materialized.

## Local correlation order

Three learned lane projections `u`, `v`, and `w` are formed from local `m1`.
Products are taken after aggregation:

$$
\begin{aligned}
u\cdot v &\rightarrow 0e,\\
\mathrm{ST}(u\otimes v) &\rightarrow 2e,\\
u\times v &\rightarrow 1e,\\
(u\times v)\cdot w &\rightarrow 0o,\\
\mathrm{ST}[w\otimes(u\times v)] &\rightarrow 2o.
\end{aligned}
$$

Because each vector is already a local source sum, these products contain
implicit neighbour-pair and neighbour-triple correlations without enumerating
triplets. They provide local angle, anisotropy, coordination-shape, and
chirality signals.

## Fusion and initialization

The local carriers are fused into the existing global `MomentFeatures`
interface with learned per-lane gates. All gates are initialized to zero, so a
new model starts with exactly the previous global ELA operator. Training opens
only the local sectors that improve the objective; the downstream
parity-complete closure is unchanged.

## Symmetry and complexity

The kernel is scalar and O(3)-invariant. The translated first and second
moments transform as `1o` and `0e + 2e`; the correlation products have the
parities shown above. Segment reductions preserve permutation equivariance and
component isolation. Coordinates remain segment-centred and RMS-normalized, so
translation and uniform-scale guarantees are unchanged.

With local rank `R`, Mercer width `P=10`, and moment basis width `B=10`, the
additional work and transient storage are

$$
O(NRPB),
$$

which is linear in node count and independent of the number of node pairs.

## Current limitation

Locality is soft and measured in segment-normalized coordinates. It does not
impose a hard physical cutoff in angstroms. Tasks that require absolute bond or
contact length scales should additionally expose the segment radius or another
physical length invariant to the scalar stream.
