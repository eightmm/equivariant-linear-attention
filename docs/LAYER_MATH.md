# Factorized Moment Attention

For graph `g`, node `i`, and head `h`, map positive scalar-content features
`phi(q0_ih), phi(k0_jh) in R^D` to

```text
a_ih = phi(q0_ih) / sqrt(||phi(q0_ih)||^2 + eps)
b_jh = phi(k0_jh) / sqrt(||phi(k0_jh)||^2 + eps).
```

Both normalized scalar features and unit-ball vectors are multiplied by an
inward machine-precision margin `max(0.5, 1 - 4 D eps_machine)`, using their
last-axis dimension `D`. This offsets dot-product accumulation error so the
closed finite-precision kernel bound below is enforced for general directions,
not only axis-aligned endpoints.

Let `q1_ih, k1_jh in R^3` be polar-vector features mapped by the following
unit-ball transform:

```text
q1 = u / sqrt(1 + ||u||^2),   k1 = v / sqrt(1 + ||v||^2).
```

In exact arithmetic, scalar feature norms are at most one, vector norms are
strictly below one, and the learned scales satisfy `beta_h, gamma_h > 0` below
their configured maxima. Floating-point rounding can saturate endpoints, so
the implemented contract uses closed bounds. Let `delta_h=beta_h` when the
alignment-linear term is enabled and `delta_h=0` otherwise. The pair kernel is

```text
K_ijh = c + a_ih^T b_jh
        + beta_h + delta_h q1_ih^T k1_jh
        + gamma_h (q1_ih^T k1_jh)^2.
```

For `c > 0`, `0 <= beta_h <= beta_max`, and
`0 <= gamma_h <= gamma_max`, finite precision preserves the declared bound

```text
c <= K_ijh <= c + 1 + 2 beta_max + gamma_max.
```

Kernel initial values, maxima, floors, and init/max ratios must be normal
float32 values. Accepted initial coefficients therefore round-trip through the
inverse-logit parameterization instead of silently underflowing to zero.

## Optional architecture-v2 feature maps

The default scalar content above remains `scalar_content_mode="unit"`. The
opt-in `bounded` mode retains one bounded signal from the positive feature
magnitude. With `z=ELU(x)+1`, incumbent inward unit vector `u(z)`, and
`r=||z||_2/sqrt(D)`, it uses

```text
phi_bounded(x) = u(z) * 2 r / (1 + r).
```

The amplitude is monotone in `r` and below two, so the scalar-content dot
product remains nonnegative and is bounded above by four. This is a finite
feature kernel, not exact softmax attention. Simply removing normalization
would lose the declared finite kernel bound.

When persistent `2e` channels exist, the separate opt-in tensor-product kernel
lets that state influence attention weights. Channel-only maps form
symmetric-traceless query/key matrices, followed by the true Frobenius
unit-ball map

```text
U(T) = T / sqrt(1 + ||T||_F^2).
```

For a bounded positive per-head scale `eta_h`, the added pair term is

```text
K2_ijh = eta_h * (1 + <U(Q2_ih), U(K2_jh)>_F).
```

It lies in `[0, 2 eta_h]`. The implementation appends
`sqrt(eta_h) * [1, vec(U(Q2))]` and
`sqrt(eta_h) * [1, vec(U(K2))]` to the scalar feature maps, where `vec` is the
full `3 x 3` flattening. A raw dot product of the five stored tensor
coordinates would use the wrong Frobenius metric and is therefore not used.
The existing scalar graph summaries then factorize this term exactly, so fixed
width remains `O(N)` with no pair tensor. Since
`T -> R T R^T`, the Frobenius contraction is invariant under all `R in O(3)`.

With both options active, the finite pair-kernel bound becomes

```text
c <= K_ijh <= c + 4 + 2 beta_max + gamma_max + 2 eta_max.
```

The tensor-product option is invalid without persistent `2e` state and is not
registered with interacting multi-memory transport. Disabled options allocate
no tensor query/key/scale parameters and preserve the incumbent state schema.

The linear term distinguishes alignment from anti-alignment. Disabling it
removes only `beta_h q1_ih^T k1_jh`; the constant `beta_h` is retained in both
arms. This isolates alignment from a simultaneous change to the constant mass.

This is O(3)-invariant because inner products are preserved by every orthogonal
matrix, including reflections.

The individual coordinates of a flattened symmetric outer feature would be
signed. Positivity belongs to the complete quadratic kernel, not to every
feature coordinate. The production implementation therefore does not sum a
signed flattened feature and then take a dot product. It contracts structured
3x3 positive-semidefinite summaries.

For row scale `r_ih`, define

```text
Q0_gh = sum_i r_ih a_ih
Qr_gh = sum_i r_ih
Q1_gh = sum_i r_ih q1_ih
Q2_gh = sum_i r_ih q1_ih q1_ih^T.
```

Then each key mass is evaluated as

```text
m_jh = b_jh^T Q0_gh
       + (c + beta_h) Qr_gh
       + delta_h k1_jh^T Q1_gh
       + gamma_h k1_jh^T Q2_gh k1_jh.
```

The linear contraction can be signed, but when enabled the combined alignment
contribution `beta_h (Qr_gh + k1_jh^T Q1_gh)` is nonnegative because every
query/key vector has norm at most one. When disabled, `beta_h Qr_gh` remains.
The floor contribution `c Qr_gh` is strictly positive.
Value numerators use analogous scalar, constant, vector-valued, and
matrix-valued summaries. Their learned values may be signed, so numerator
summaries are not PSD and are not clamped. This is algebraically identical to
the explicit dense kernel and remains `O(N)` at fixed width and head count.

## One balancing cycle

For each graph and head, compute `m_jh = sum_i K_ijh` and use
`K'_ijh = K_ijh / m_jh`. The final row-normalized weights are

```text
A_ijh = K'_ijh / sum_l K'_ilh.
```

No pair matrix is needed. Tests compare both the balanced and row-normalized
structured factorizations with the explicit dense kernel in float64. One-cycle
balancing is the default; `--no-key-balancing` is retained only for a matched
normalization study. There is no private balance exponent or additional
Sinkhorn iteration. Positivity does not depend on balancing.

Balancing can remove a pure key-side aligned/anti-aligned preference when every
query is identical. That executable counterexample prevents interpreting one
cycle as an unconditional expressivity improvement.

## Kernel-baseline modes

Let `delta=1` when the alignment-linear term is enabled and `0` otherwise. The
default `fixed` global kernel is

```text
K_ij = a_i.b_j + c + beta + delta*beta*t_ij + gamma*t_ij^2.
```

Its strictly positive shifted baseline can grow linearly in `N_g`, forcing the
maximum normalized weight toward `O(1/N_g)`. The experimental
`inverse_graph_size` mode instead uses

```text
K_ij = a_i.b_j + (c + beta + delta*beta*t_ij)/N_g + gamma*t_ij^2.
```

Scaling only `c` would not remove the asymptotic positive baseline because
`beta*(1+t_ij)` remains nonnegative. Content and `gamma*t_ij^2` therefore stay
unscaled while the complete shifted-alignment baseline is scaled. This mode is
global and row-normalized only: it is rejected with key balancing, and local
attention never substitutes graph size for receiver degree.

## Exact relative moments

Let `x_i` be centered, graph-normalized coordinates and `a_jh` an invariant
gate. The relative vector is recovered exactly:

```text
sum_j A_ijh a_jh (x_j - x_i)
= sum_j A_ijh a_jh x_j - x_i sum_j A_ijh a_jh.
```

Define

```text
ST(x) = x x^T - ||x||^2 I / 3
ST(a,b) = (a b^T + b a^T)/2 - (a^T b) I / 3.
```

Then

```text
ST(x_j - x_i) = ST(x_j) + ST(x_i) - 2 ST(x_j, x_i),
```

which reconstructs the relative rank-2 moment from mass, first-moment, and
second-moment graph summaries. The stored five-component basis maps exactly to
a symmetric matrix with zero trace.

The optional radial trace uses the same mass and first moments plus the second
radial moment:

```text
sum_j A_ijh a_jh ||x_j - x_i||^2
= S2_ih + ||x_i||^2 m_ih - 2 x_i^T p_ih.
```

The corresponding scalar slot is reserved in both configurations and is
exactly zero when `use_radial_trace=False`.

## Optional persistent `2e` state

If `hidden_irreps` contains `C_2 x 2e`, node `i` carries a five-component
symmetric-traceless state `H_i` in addition to scalar and polar-vector state.
The five stored components represent a Cartesian matrix `T(H)` with
`T=T^T`, `tr(T)=0`, and transform under every `R in O(3)` as

```text
T(H_i) -> R T(H_i) R^T.
```

For one channel, the bounded state used inside a block is

```text
B(H) = H / sqrt(1 + ||T(H)||_F^2 / 5).
```

This denominator is evaluated directly. Computing `sqrt(||T||_F^2)` first is
forward-equivalent but has an undefined `sqrt(0)` derivative at the zero
initial tensor state; the direct squared-norm form keeps repeated float32
training updates finite.

Channel-only linear maps commute with this transformation. Let `M_i` be the
block's transient rank-2 head moment and `sbar_i` its invariant normalized
scalar state. The implemented residual update has the form

```text
Mcontext_i = M_i + W_to B(H_i)
H'_i = H_i + alpha_H B(W_from M_i) * tanh(G_H(sbar_i)).
```

The scalar updater reads invariant contractions including
`||T(Mcontext)||_F^2`, `q^T T(Mcontext) q`, and `||T(B(H))||_F^2`; the vector
updater reads `T(Mcontext) q`. The pointwise equivariant FFN then applies a
second invariant-gated channel mix,

```text
H''_i = H'_i + alpha_F B(W_F B(H'_i)) * tanh(G_F(sbar'_i)),
```

before its Frobenius norm is concatenated to the scalar FFN input. Therefore
the hidden tensor state can affect a scalar objective while remaining
reflection-even. No spherical harmonics or parity-odd tensor products are
introduced, and this path does not accept external `2e/2o` node inputs.

## Global transport controls

The learned mode uses the factorized kernel above. The uniform control replaces
its normalized pair weights by

```text
A_ij = 1/N_g  if b_i=b_j, else 0.
```

For every concatenated value/moment statistic `V`, the implementation computes

```text
U(V)_i = (sum_{j:b_j=b_i} V_j) / N_{b_i}
```

once per graph and broadcasts it, so the control is O(N) and never constructs
an `N x N` tensor. Because the weights are invariant scalars, this operator is
permutation consistent and O(3) equivariant. Substituting its graph means into
the same relative first/ST/radial identities above preserves translation
invariance exactly. It retains query-vector use in the shared equivariant
updater but removes learned query/key selectivity from transport weights.

The `none` control has zero global transport Jacobian. An all-global block skips
the query/key/value projections and the complete attention updater residual,
including its biases, then applies only the pointwise FFN. Feeding zero messages
through the updater would not be equivalent because its LayerNorm/MLP biases
could create a residual. Registered routes use all-local or all-global blocks;
mixed-head `none` is legal but its shared updater is not a head-separable
ablation.

## Local heads

Local heads use raw-coordinate directed edges `i <- j` within the same graph.
The optional public `edge_index` uses receiver row 0 and sender row 1. It is a
candidate list rather than a replacement kernel: direct callers validate
same-graph unique indices and complete self coverage, then apply the same strict
cutoff below. `GraphBatch` collation performs those content checks once and
sets a trusted fast-path assertion, so repeated model forwards check only tensor
metadata. Without supplied edges, the core constructs all same-graph candidates.
With cutoff `R_c`, define

```text
d_ij = (p_j - p_i) / R_c,
u_ij = ||d_ij||^2,
f_c(u) = 0.5 (1 + cos(pi u))  for u < 1, else 0.
```

The cutoff value and first coordinate derivative vanish at `u=1`, without a
square root at coincident points. Sixteen Gaussian RBFs of `u` feed a positive
radial gate with a fixed mixture floor; routing comparisons keep its learned
parameters frozen. The bounded degree-2 kernel times this radial gate is
normalized over each receiver's local senders. Local moments are evaluated
directly on retained edges, including self edges.

The opt-in edge-conditioned local transport replaces that normalized local
kernel while leaving global blocks unchanged. For a local block with invariant
normalized scalar state `sbar_i`, sender vector channels `v_j`, and RBF vector
`rho(u_ij)`, one block-local MLP gives

```text
[a_ij, g^v_ij, g^r_ij, g^T_ij]
    = MLP([sbar_i, sbar_j, rho(u_ij)]),  i != j.
```

At width 64 with four heads the frozen MLP is
`Linear(144,12) -> SiLU -> Linear(12,76)`: `a_ij` contains 64 scalar channels
and each gate contains four head channels. The equivariant sums are

```text
m^0_i = sum_j f_c(u_ij) reshape_H(a_ij),
m^v_i = sum_j f_c(u_ij) tanh(g^v_ij) v_j,
m^r_i = sum_j f_c(u_ij) tanh(g^r_ij) d_ij,
m^T_i = sum_j f_c(u_ij) tanh(g^T_ij) ST(d_ij).
```

When `normalize_edge_conditioned_local_by_sqrt_degree` is enabled, let
`D_i = |{j : (i,j) is a retained non-self candidate}|`. Every message family
above is replaced by

```text
mtilde_i = m_i / sqrt(max(D_i, 1)).
```

The degree is shared across heads and message types, counts candidates rather
than cutoff mass, and introduces no learned parameter. The default remains the
unnormalized sum.

Self edges are excluded from these four sums because self information remains
in the node residual. Invariant gates times polar vectors, relative polar
vectors, or even symmetric-traceless tensors preserve O(3); receiver sums
preserve node-permutation consistency and graph isolation. After edge contents
have been validated, the local transport is `O(E_local)` and the complete
fixed-width model hot path is `O(E_local + N)`. Duplicate/self validation and
neighbor construction are deliberately outside that claim. The original
registered operator uses the default unnormalized sum; the square-root-degree
variant remains opt-in pending its registered diagnostic screen.

### Gated same-feature local transport

The newer opt-in local transport changes only how the same node states and
geometry interact. It does not add a raw atom, bond, residue, segment, or label
feature. For head `h`, split the normalized scalar state into
`sbar_ih in R^d`, retain one polar vector `v_ih in R^3`, and assemble

```text
z_ijh = [
    sbar_ih,
    sbar_jh,
    rho(u_ij),
    <v_ih, v_jh>,
    ||v_ih||^2,
    ||v_jh||^2,
    <v_ih, d_ij>,
    <v_jh, d_ij>
].
```

A shared per-head MLP produces scalar content and five scalar gates:

```text
[a_ijh, g^s_ijh, g^i_ijh, g^j_ijh, g^r_ijh, g^T_ijh]
    = MLP(z_ijh).
```

Self edges are excluded. Let `D_i` be the retained nonself candidate count and
`C_i = sum_j f_c(u_ij)` its smooth cutoff mass. The equivariant aggregates are

```text
m^0_ih = LN_h(
    sum_j f_c(u_ij) sigmoid(g^s_ijh) a_ijh / sqrt(max(D_i, 1))
) + reshape_h(W_m [log(1 + D_i), log(1 + C_i)]),

m^v_ih = sum_j f_c(u_ij) [
    tanh(g^i_ijh) v_ih + tanh(g^j_ijh) v_jh
] / sqrt(max(D_i, 1)),

m^r_ih = sum_j f_c(u_ij) tanh(g^r_ijh) d_ij
    / sqrt(max(D_i, 1)),

m^T_ih = sum_j f_c(u_ij) tanh(g^T_ijh) ST(d_ij)
    / sqrt(max(D_i, 1)).
```

Every MLP input and gate is invariant under `O(3)`. Multiplying invariant
gates by polar vectors, relative polar vectors, or even-parity
symmetric-traceless tensors preserves the required transformation law.
Receiver sums are invariant to edge order and consistent under node
permutation. At fixed width, the local path is `O(E_local)` after candidates
are built, while an LGL stack retains the exact `O(N)` factorized global
block.

When `use_grouped_invariant_normalization` is enabled, define parameter-free
last-axis standardization `G(x)`. Before the incumbent learned update
LayerNorm, its invariant input becomes

```text
concat(
    G(m^0),
    G(concat(all angular vector/tensor contractions)),
    G(persistent-tensor Frobenius invariants)  if present
).
```

This prevents one invariant family from setting the scale of every other
family while leaving the learned update and all equivariant value paths
unchanged. Because the registered experiment tested gated-only and
gated-plus-grouped but not grouped-only, its performance supports the combined
package and does not identify either component as sufficient by itself.

The opt-in pairwise-content repair leaves those equivariant moments unchanged
and adds one invariant scalar message. Let `qbar_ih` and `kbar_jh` be the raw
scalar query/key projections before positive-feature normalization, and let
`rho(u_ij)` be the existing RBF vector. A head-shared two-layer MLP gives

```text
e_ijh = MLP([qbar_ih, kbar_jh, rho(u_ij)]),  i != j.
```

For retained nonself edges, define the unweighted neighbor degree and smooth
cutoff mass

```text
n_i = sum_j 1,
c_i = sum_j f_c(u_ij).
```

The added scalar message is

```text
m_pair_ih = alpha (
    sum_j f_c(u_ij) e_ijh / sqrt(max(1, n_i))
    + W_mass [log(1+n_i), log(1+c_i)]
).
```

`alpha` is one learned residual scalar, initialized to `0.1` by default. The
registered staged-initialization repair may set it to zero so the first forward
pass is the exact incumbent while `alpha` itself receives the first learning
signal. One shared module is reused by every local layer and every head, so the
registered width-64 model adds only 1,105 parameters. The unnormalized numerator retains pair content;
the explicit degree/cutoff-mass term restores neighborhood-size information
that receiver-row normalization removes. All inputs to this branch are
invariant scalars, so adding it to the scalar moment preserves O(3), reflection,
translation, permutation, and graph-batch isolation. Self edges remain in the
incumbent moment path but are excluded from this neighbor-content branch.
`use_pairwise_local_content=False` allocates no module and preserves the exact
default state schema and output.

## Optional latent-coordinate update

Let `s_i` be the invariant scalar state and `v_ic` the polar-vector state after
a nonfinal block. A learned channel mix and invariant scalar gate form

```text
r_i = tanh(g(LN(s_i))) sum_c w_c v_ic.
```

For graph `g`, center `r_i` and apply one shared graph scale:

```text
u_i = r_i - mean_{j in g} r_j,
a_g = min(1, 0.25 Angstrom / max_{j in g} ||u_j||),
x_i' = x_i + a_g u_i.
```

The shared scale retains `sum_i a_g u_i=0`, so the graph centroid is unchanged,
while every step is bounded by 0.25 Angstrom. An invariant gate times a polar
vector is O(3)-equivariant; centering and norm-based graph scaling preserve
O(3), translation, and permutation behavior. Updated global/local geometry is
recomputed before the next block. There is no final post-readout updater, so
every coordinate parameter can influence the scalar property loss.

The private dynamic EGNN control uses its invariant edge embedding `m_ij`:

```text
r_i = mean_{j != i} (x_i - x_j) tanh(phi_x(m_ij)),
```

then applies the same graph-centering and bound. This follows the EGNN
relative-vector update pattern but remains a same-harness internal control, not
an official implementation reproduction.

## Multi-memory global gate (HEMM)

For `M` slots, bounded invariant logits yield soft assignments
`pi_ihm >= 0`, `sum_m pi_ihm=1`. Weighted memory centers use their exact
positive occupancy, without adding `eps` to the denominator. An optional fixed
nonnegative cosine cutoff of squared center distance gives
`0 <= C_ghmn <= 1` and `C_ghmm=1`. The effective pair gate is

```text
G_ijh = sum_mn pi_ihm C_ghmn pi_jhn,
A_ijh proportional to K_ijh G_ijh.
```

Per-memory structured summaries evaluate this gate without an `N x N` matrix.
For `M=1`, `pi=1`, `C=1`, and the implementation dispatches to the incumbent
factorization exactly, including gradients. With interaction disabled,
`C=1` for all slot pairs, hence `G_ijh=1` algebraically for any `M`. Therefore
memory count alone is not a new mechanism; the registered experimental change
is interaction for `M=4/8` in the middle global block of `lgl`.

For one graph and head, let `mu=mean(G)`. The activation gate reports

```text
CV = sqrt(mean((G - mu)^2)) / mu,
D  = ||G - mu 11^T||_F / ||G||_F,
f_tau = mean(|G - mu| > tau mu),  tau=1e-3.
```

These satisfy `D=CV/sqrt(1+CV^2)`, so they are an internal consistency pair,
not independent evidence. If `G=k 11^T` for positive `k`, then `CV=D=f_tau=0`
and multiplication by `G` cancels exactly in row-normalized attention,
including after the registered one-cycle key balancing. Diagnostics therefore
compute these values separately inside every graph/head normalization domain;
pooling heads or graphs could manufacture false variation.

The present router replaces the earlier one-dimensional deterministic
projection with an `M`-independent invariant MLP. For each node/head it forms

```text
z_i = normalize(tanh(W_2 SiLU(W_1 b_i))) in R^8,
ell_im = 4 z_i^T c_m,
pi_i = softmax(ell_i),
```

where `c_m` are fixed unit DCT slot codes. Every layer allocates the same
router parameter schema regardless of route or memory count, and the exact
`M=1` dispatch occurs before the router is evaluated. Read and write
assignments remain shared, while the center coupling is symmetric and computed
from graph-normalized coordinates. The mechanism is therefore a
shape-relative low-rank gate rather than directed typed or persistent memory.

The radial cutoff is especially flat near coincident centers. For center
distance `d` and cutoff `R`,

```text
C_radial(d) = 0.5 (1 + cos(pi d^2/R^2))
            = 1 - (pi^2/4) d^4/R^4 + O(d^8),
```

so its center gradient is `O(d^3)`. The registered counterfactual repair was

```text
C_lambda = (1-lambda) C_radial + lambda I,
lambda in {0.10, 0.25, 0.50}.
```

It preserves symmetry, unit diagonal, bounds, slot-permutation covariance,
equivariance, and the exact `M=1` limit. It does not imply a PSD kernel. No
candidate passed the unchanged full Stage-0 matrix, so no residual-coupling
control was promoted to the public model configuration.

Assignment diagnostics distinguish marginal slot use from node dependence:

```text
pbar_m = mean_i pi_im,
H_marg = -sum_m pbar_m log(pbar_m) / log(M),
H_cond = -mean_i sum_m pi_im log(pi_im) / log(M),
I_slot = H_marg - H_cond.
```

`I_slot=0` exactly when every node has the same assignment distribution. The
registered widths 16/64, seeds 401--403, and four graph roles found the learned
router numerically nonconstant but functionally below the frozen transport and
gradient thresholds in 11 of 12 aligned M=4/8 lanes. Consequently all
interacting-memory accuracy/performance arms remain blocked.

Consistent permutation of assignment and coupling slot axes leaves the result
unchanged. Bounded soft assignments avoid an undefined empty-slot branch but
can still collapse on identical or symmetric inputs. No learned Cartesian
centers, learned coupling, hard top-k, persistent tensor memory, or higher
angular-order claim is included.

## Equivariance

For `R in O(3)` and translation `t`, centered coordinates satisfy
`x'_i = R x_i`. Scalar contractions are unchanged, vectors transform as `Rv`,
and rank-2 tensors transform as `R T R^T`. Graph sums and channel mixing commute
with these transformations. Therefore every block is O(3)-equivariant and
translation invariant; graph-wise sums also commute with node permutations.

## Numerical policy and complexity

Geometry preprocessing is scale-first: for each graph, coordinates are first
divided by their maximum absolute value, then centered and RMS-normalized in
the scaled frame. Physical log-radius/log-scale features are assembled without
forming an overflow-prone direct product. On ordinary float64 inputs this
agrees with the direct formula to the tested tolerance. It is initialized only
before the first learned or uniform global stage; local-only and `none` routes
do not execute it.

Geometry squares, angular and symmetric-traceless feature construction,
attention sums, masses, denominators, graph means, and the moment-invariant
LayerNorm use float32 for fp16/bf16 inputs and float64 for float64 inputs.
Scalar and vector residuals are cast back only after normalization/bounding;
rank-2 moment outputs remain float32 for low-precision inputs. This prevents
both an `eps=1e-12` clamp from becoming zero and valid normalized coordinates
near 256 from overflowing when squared in fp16.

Feature and coordinate precision are separate contracts. Low-precision model
features may be fp16/bf16, while coordinates remain float32 unless the complete
verification lane is float64. Finite FP32 coordinates are never cast through
fp16 before geometry preprocessing.

At fixed feature dimension, head count, depth, and memory count, global
structured attention is `O(N)` in nodes; the memory gate exposes
`O(NM + M^2)` work/storage terms. Local transport stores `O(E)` retained edge
state and reuses its edge/displacement/distance/RBF geometry across local
layers. The core fallback builds all same-graph Cartesian candidate pairs with
one stable batch sort and one vectorized expansion before applying the cutoff;
its candidate work is still `O(sum_g N_g^2)`. This changes repeated geometry
work from `L*N^2` to `N^2 + L*E` and removes the per-graph Python/GPU launch
loop. When precomputed `edge_index` is supplied, complete-pair discovery is
bypassed and candidate plus retained transport storage is O(E); fixed-width
local-layer geometry/transport after validation is `L*E`. Neighbor construction
remains the caller's cost, so this is not a production radius-backend claim.
The public wrapper derives graph metadata once and reuses it. Validation can
still create compile graph breaks; full-graph compilation and production
sparse-neighbor performance remain separate targets.

## Executable representation boundaries

The P2 counterexample suite records rather than hides the remaining limits:

- degree-2 moments collide for distributions with equal mass/mean/variance but
  different fourth moments;
- two-point global normalized geometry is independent of fragment separation,
  so the global branch has no structural cluster-decay guarantee;
- scalar output is invariant under global reflection;
- fixed positive global floor can dilute maximum attention as graph size grows;
- one balancing cycle can erase a pure key-side alignment preference;
- soft memory assignments can collapse and finite `M` does not remove
  degree-2/RBF representation collisions;
- coordinate gradients of invariant scalar sums transform equivariantly.

These tests constrain claims; they do not turn the global block into a local
force field or parity-complete SBDD model.
