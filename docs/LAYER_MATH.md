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

## Optional architecture-v3 angular features

The v3 path adds two independent, opt-in ways to increase angular selectivity.
They are finite feature maps, not spherical-harmonic tensor products and not an
approximation to softmax.

First, `angular_feature_rank=2` learns two gated polar-vector axes per head and
forms their direct sum:

```text
qbar_ih = U([q1_ih, q1extra_ih]) in R^6
kbar_jh = U([k1_jh, k1extra_jh]) in R^6,
```

where `U(x)=x/sqrt(1+||x||^2)`. The linear and quadratic angular terms use
`tbar_ijh=qbar_ih^T kbar_jh`. The configuration name counts independent
`1o` axes; it does **not** mean an irreducible `l=2` representation. The
separate persistent symmetric-traceless state remains the actual `2e` path.

Second, `use_quartic_kernel=True` adds a positive bounded coefficient
`0 <= kappa_h <= kappa_max` and the exact term

```text
kappa_h (q1_ih^T k1_jh)^4.
```

Here `q1,k1 in R^3` are the primary axes, even when the two-axis direct sum is
enabled. Let `alpha=(alpha_1,...,alpha_D)` range over multi-indices with
`|alpha|=4`. The implemented symmetric monomial map is

```text
Phi4(x)_alpha = sqrt(4! / prod_d alpha_d!) prod_d x_d^alpha_d,
```

so the multinomial theorem gives

```text
Phi4(q)^T Phi4(k) = (q^T k)^4.
```

For the implemented `D=3` primary axis this is 15 features per head. Appending
`sqrt(kappa_h) Phi4(q)` and `sqrt(kappa_h) Phi4(k)` to the scalar feature maps
therefore preserves the exact graph-summary factorization. With unit scalar
content and without the optional tensor-product kernel, the combined v3 kernel
is

```text
K_ijh = c + a_ih^T b_jh
        + beta_h + delta_h tbar_ijh
        + gamma_h tbar_ijh^2
        + kappa_h (q1_ih^T k1_jh)^4.
```

It obeys the closed bound

```text
c <= K_ijh <= c + 1 + 2 beta_max + gamma_max + kappa_max.
```

All graph summaries remain fixed width, hence the global node-count scaling is
still `O(N)` for fixed channels, heads, and angular rank. The larger feature
constant is real and must be measured at train-step level.

The degree-two term is stored without the redundant full `D x D` outer
product. Define the asymmetric compressed maps

```text
Phi2L(x) = [x_1^2,...,x_D^2,{2 x_a x_b}_{a<b}]
Phi2R(x) = [x_1^2,...,x_D^2,{  x_a x_b}_{a<b}].
```

Then

```text
Phi2L(q)^T Phi2R(k) = (q^T k)^2.
```

This reduces the quadratic summary width from `D^2` to `D(D+1)/2`: `9 -> 6`
for one axis and `36 -> 21` for two axes. The asymmetric scaling also avoids a
large float32 cancellation observed with a symmetric `sqrt(2)` basis.
Positivity belongs to the complete quadratic kernel. Mass/denominator
quadratic contractions are clamped at zero against roundoff, while signed
value numerators are never clamped.

For row scale `r_ih`, define

```text
Q0_gh = sum_i r_ih a_ih
Qr_gh = sum_i r_ih
Q1_gh = sum_i r_ih qbar_ih
Q2_gh = sum_i r_ih Phi2L(qbar_ih).
```

Then each key mass is evaluated as

```text
m_jh = b_jh^T Q0_gh
       + (c + beta_h) Qr_gh
       + delta_h kbar_jh^T Q1_gh
       + gamma_h Phi2R(kbar_jh)^T Q2_gh.
```

The linear contraction can be signed, but when enabled the combined alignment
contribution `beta_h (Qr_gh + kbar_jh^T Q1_gh)` is nonnegative because every
query/key vector has norm at most one. When disabled, `beta_h Qr_gh` remains.
The floor contribution `c Qr_gh` is strictly positive.
Value numerators use analogous scalar, constant, vector-valued, and compressed
quadratic summaries. Their learned values may be signed, so numerator summaries
are not clamped. This is algebraically identical to the explicit dense kernel
and remains `O(N)` at fixed width and head count.

## Adaptive multiscale spatial LGL

The opt-in exact three-layer LGL candidate adds a spatial feature block only to
the fully global middle stage. For normalized centered position `x_i` and
positive scale `s`, the existing ten-component degree-two Gaussian--Taylor map
`psi_s(x_i)` satisfies

```text
psi_s(x_i)^T psi_s(x_j)
  = exp(-s (||x_i||^2 + ||x_j||^2))
    * (1 + u_ij + u_ij^2 / 2),
u_ij = 2 s x_i^T x_j.
```

The polynomial factor is strictly positive in real arithmetic because
`1 + u + u^2/2 = ((u+1)^2 + 1)/2`. For four fixed scales, invariant scalar
projections produce separate query/key logits. With
`epsilon = finfo(logit_dtype).eps`, define

```text
pq_ihs = (softmax_s(lq_ihs) + epsilon) / (1 + S epsilon),
pk_jhs = (softmax_s(lk_jhs) + epsilon) / (1 + S epsilon),
gq_ihs = sqrt(pq_ihs),
gk_jhs = sqrt(pk_jhs),
Qsp_ih = concat_s(gq_ihs psi_s(x_i)),
Ksp_jh = concat_s(gk_jhs psi_s(x_j)).
```

Then the added pair term is

```text
Ksp_ijh = Qsp_ih^T Ksp_jh
        = sum_s gq_ihs gk_jhs psi_s(x_i)^T psi_s(x_j) >= 0.
```

The epsilon floor prevents a representable softmax zero from entering
`sqrt(0)` during backward. Renormalization keeps each squared scale profile
summed to one, preserves common-logit-shift invariance, and initializes to the
average of the four scale kernels when both projections are zero. The spatial
term is numerically nonnegative; extreme Gaussian underflow may make it zero.
The incumbent strictly positive kernel floor still keeps the full attention
kernel and denominator positive.

For key-balancing row scale `r_ih`, the spatial column mass is

```text
Qmass_gh = sum_i r_ih Qsp_ih,
msp_jh = Ksp_jh^T Qmass_gh.
```

After the combined structured-plus-spatial key scale `d_jh` is formed, value
transport is

```text
Ssp_gh = sum_j Ksp_jh tensor (d_jh v_jh),
osp_ih = Qsp_ih^T Ssp_gh.
```

These are exact evaluations of the defined kernel. With fixed scale count and
feature width, work and node storage are `O(N)` and no `N x N` tensor is
materialized. Gates depend only on invariant scalar state, while the spatial
basis uses centered/RMS-normalized coordinates, preserving O(3), translation,
batch, and permutation contracts subject to the documented coordinate-storage
precision boundary.

## Exact explicit-feature GEMM backend

The structured summaries above and an explicit feature factorization are two
execution strategies for the same kernel. Let `eta_g=1` for the fixed floor
and `eta_g=1/N_g` for the inverse-graph-size baseline. Using an isometric
symmetric-square map

```text
Sym2(x) = [x_1^2, x_2^2, x_3^2,
           sqrt(2)x_1x_2, sqrt(2)x_1x_3, sqrt(2)x_2x_3],
```

define

```text
PhiQ_ih = concat(
  a_ih,
  sqrt(eta_g (c + beta_h)),
  sqrt(eta_g delta_h) qbar_ih,
  sqrt(gamma_h) Sym2(qbar_ih),
  optional Qsp_ih
)

PhiK_jh = concat(
  b_jh,
  sqrt(eta_g (c + beta_h)),
  sqrt(eta_g delta_h) kbar_jh,
  sqrt(gamma_h) Sym2(kbar_jh),
  optional Ksp_jh
).
```

The shared `sqrt(2)` coordinates give
`Sym2(q)^T Sym2(k)=(q^T k)^2`; therefore
`PhiQ_i^T PhiK_j` is exactly the incumbent pair kernel, including distinct
adaptive spatial query/key features. For unbalanced transport,

```text
S_gh = sum_j PhiK_jh tensor [v_jh, 1],
y_ih = PhiQ_ih^T S_gh,
o_ih = y_ih[:-1] / y_ih[-1].
```

One-cycle key balancing first evaluates

```text
Q_gh = sum_i PhiQ_ih,
m_jh = PhiK_jh^T Q_gh,
S_gh = sum_j PhiK_jh tensor ([v_jh,1] / m_jh).
```

`feature_gemm` evaluates these equations as graph-major matrix products.
Unlike the compatibility numerator it never materializes a per-node
`N x H x F x V` outer. Storage is `O(NHF + GHFV + NHV)` at fixed feature
rank. A bounded-padding layout is used for ordinary batches; highly ragged
batches are grouped once, sliced by graph offsets, concatenated once, and
returned to node order with one inverse permutation. They execute the same two
GEMMs without `G` full-batch scans or a repeated full-size `index_copy` chain.

## Homogeneous sparse low-rank residual

The vNext block retains the full global message in every layer and adds a
separable local correction only on a static refresh set `R`:

```text
M_i^t = G_i^t + 1[t in R] S_i^t.
```

For rank `r=1..R`, node projections produce invariant scalars
`q_ir,k_jr`, polar vectors `u_ir,v_jr`, scalar values `z_jrd`, polar values
`p_jr`, and invariant value gates. With normalized polar edge direction
`d_ij=(x_j-x_i)/Rc`, the edge logit is

```text
ell_ijr = tanh(
  q_ir + k_jr + A_r RBF(||d_ij||^2)
  + alpha_r <u_ir,v_jr>
  + beta_r <u_ir,d_ij>
  + chi_r <v_jr,d_ij>
).
```

Every term in `ell` is O(3)-invariant: polar/polar and polar/direction
contractions remain even under reflection. Positive receiver-normalized
weights are

```text
abar_ijr = fc(||d_ij||^2) sigmoid(ell_ijr),
a_ijr = abar_ijr / sum_(k -> i) abar_ikr.
```

Rank-space sufficient statistics are

```text
S0_ird = sum_j a_ijr z_jrd
S1_ir  = sum_j a_ijr p_jr
Sd_ir  = sum_j a_ijr g^d_jr d_ij
S2_ir  = sum_j a_ijr (
           g^2_jr ST(d_ij tensor d_ij) + optional W2 H_j^(2e))
Sr_ir  = sum_j a_ijr g^r_jr ||d_ij||^2.
```

Channel-only maps `R -> H` return this tuple to the existing head update.
They are initialized at zero, so the enabled model starts as the exact
all-global control. At the first backward pass the output maps receive
gradient; after their first update the query/key/value projections receive a
nonzero learning signal. The sparse path uses only `E x R` invariant edge
latents and canonical `E x R x {D,3,5}` transported values. It has no
persistent edge state, hidden-width edge MLP, triplets, or `N x N` tensor.
Its scalar payload gives work `O(N + E R D_head)` in general and
`O(N + ER)` when rank and channel width are fixed.

## Receiver CSR and reverse key mass

With cutoff-retained edges sorted receiver-major, let `row_ptr[i]:row_ptr[i+1]`
index all `j -> i` edges. Receiver sums become

```text
out_i = segment_sum(edge_value[row_ptr[i]:row_ptr[i+1]]).
```

The optional reverse plan stably sorts the same edges by sender and supplies a
second offset array. Generic local key balancing uses it for
`m_j=sum_i K_ij`; the resulting edge weights return to receiver order before
the row denominator. Compact int32 offsets/senders are valid whenever node and
edge addresses fit signed 32-bit range, while tensor indexing is promoted only
at the point PyTorch requires it. For an already packed graph, cutoff masks
restrict the receiver/reverse pointers by prefix counts and map the reverse
permutation into retained forward order, so neither receiver nor sender rows
are resorted. CSR changes reduction order but not the defined sum; float64
value/gradient equivalence and bounded float32 error are tested separately.

## Static irrep planning boundary

For abstract irreps `(l_1,p_1)`, `(l_2,p_2)`, and `(L,p_L)`, the planner admits

```text
|l_1-l_2| <= L <= l_1+l_2,
p_L = p_1 p_2                         under O(3).
```

SE(3) planning may retain an explicitly requested parity-mixed path because
proper rotations do not distinguish the two parity labels. Layout parsing,
multiplicity merging, flattened slices, and path binding are construction-time
metadata. Only registered native Cartesian executors run numerically. The
current bindings are the existing `1o x 0e -> 1o` passthrough plus
`1o x 1o -> 2e`, `2e x 0e -> 2e`, and `2e x 1o -> 1o`; the planner does not
turn arbitrary `l` into an executable model.

## External equivariant inputs and irrep RMS normalization

The public forward can optionally accept polar-vector and reflection-even
symmetric-traceless tensor channels:

```text
node_vectors: (N, C1, 3),       v -> R v
node_tensors: (N, C2, 3, 3),    T -> R T R^T, T=T^T, tr(T)=0.
```

Channel-only maps inject them into the hidden `1o` and persistent `2e` states.
They must be enabled through `input_vector_dim=C1` and `input_tensor_dim=C2`;
tensor inputs additionally require hidden `2e` channels. This path is
equivariant under reflections as well as rotations and is permutation
equivariant. It does not accept `1e`, `2o`, or arbitrary `l`.

`use_irrep_rms_normalization=True` applies an invariant pre-normalization before
the attention update and again before the equivariant FFN. For vector channels
`V_c` and five-coordinate tensor channels representing matrices `T_c`, define

```text
r_i^-2 = (
    sum_c ||V_ic||_2^2 + sum_c ||T_ic||_F^2
) / (3 C1 + 5 C2) + eps.
```

The normalized states are `w_c^1 r_i V_ic` and `w_c^2 r_i T_ic`, with one
learned scalar gain per channel. The scale is invariant, channel mixing is
equivariant, and scalar state normalization remains separate. This follows the
separable-normalization motivation of higher-degree equivariant Transformers
without claiming to reproduce their spherical-harmonic implementation.

Finally, `checkpoint_gated_local_mlp=True` recomputes the latter part of the
gated edge MLP during backward. It changes the activation-storage schedule, not
the equations or parameters, and is valid only with gated local transport.

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
introduced. The v3 public API can inject external reflection-even `2e`
matrices into this state; external `2o` and arbitrary higher-degree inputs
remain unsupported.

### Optional global persistent-`2e` value lane

The base persistent path above is receiver-local: without an explicit value
lane, changing `H_j` at a remote sender cannot change receiver `i` through an
otherwise global-only block. The opt-in
`use_global_tensor_value_transport=True` closes that gap. Let

```text
V^T_jh = (W_to B(H_j))_h in R^5.
```

For the same invariant scalar kernel and denominator used by the incumbent
global read, the new lane computes

```text
Hbar_ih = sum_{j:b_j=b_i} K_ijh V^T_jh / Z_ih.
```

`V^T` is concatenated to the existing value statistics before factorization,
so each of its five coordinates reuses the same graph-segmented sufficient
statistics. At fixed head width, the work and storage increase by a constant
five value coordinates per global head and remain `O(N)`; no pair-weight
tensor is formed. The existing tensor reconstruction becomes

```text
M_i = M_i^position + Hbar_i,
```

after which the unchanged `W_from`, invariant gate, and bounded residual update
write to persistent `H_i` for the generic carrier. The static carrier has
`C_2=H`, uses the identity head layout, and applies its existing bounded
residual directly; it has no `W_from` or persistent-tensor gate. Since `K/Z` is
an `O(3)`-invariant scalar and every channel map that is present acts only on
multiplicity, `Hbar` transforms as `T(Hbar) -> R T(Hbar) R^T`, including
reflections. Graph-segmented sums give exact permutation equivariance and graph
isolation, while the value contains no coordinate origin and is translation
invariant. Learned, uniform, fixed spatial, adaptive spatial, memory, and
additive whitened-read configurations all share the same value-packing
argument; the whitened auxiliary lane itself still applies only to
scalar/vector values.

The flag is off by default and introduces no parameters or checkpoint keys.
It is rejected without hidden `2e`, without an active global transport mode,
or when every head in every layer is local.

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

## Optional whitened global read

`use_whitened_global_read` adds a second global lane that changes the metric of
the read rather than the kernel. Write the incumbent kernel through its explicit
feature map, using the isometric symmetric-quadratic basis

```text
Q(x) = [x1^2, x2^2, x3^2, sqrt2 x1x2, sqrt2 x1x3, sqrt2 x2x3],
phi_i = [q0_i, sqrt(c+alpha), sqrt(alpha_dot) q1_i, sqrt(gamma) Q(q1_i)],
psi_j = [k0_j, sqrt(c+alpha), sqrt(alpha_dot) k1_j, sqrt(gamma) Q(k1_j)],
```

so that `<Q(x),Q(y)> = (x.y)^2` and `phi_i . psi_j = K_ij` exactly, including the
tensor-product and quartic content extensions, which already live inside
`q0`/`k0`. With `V_j` the concatenated `0e` scalar and `1o` vector values,

```text
G_g = (1/N_g) sum_{j in g} psi_j psi_j^T,
S_g = (1/N_g) sum_{j in g} psi_j V_j^T,
lambda_g = ridge * tr(G_g) / F,
o_i = phi_i^T (G_{b_i} + lambda_{b_i} I)^-1 S_{b_i}.
```

The equivalent pair weights are `A_ij = phi_i^T (G + lambda I)^-1 psi_j / N_g`,
which is never materialized; cost is `O(N F^2)` plus `O(G H F^3)`. As `ridge`
grows, `(G + lambda I)^-1 -> I/lambda` and the lane becomes a scaled copy of the
*unnormalized* kernel moment `phi_i^T S`. That limit is the incumbent
numerator, not the incumbent read `phi_i^T S / phi_i^T m`: the omitted
denominator is query dependent and varies across nodes of one graph, so no
rescaling recovers the incumbent function from this lane. What does hold exactly
is the disabled and zero-initialized state, where the lane contributes nothing.
`G + lambda I` is positive definite because the constant block keeps `tr(G) > 0`.

Equivariance is exact but basis dependent. Rotations act on the feature vector by
`M(R) = diag(I, 1, R, D(R))`, where `D(R)` represents `S -> R S R^T` on the
symmetric rank-2 block. `M(R)` is orthogonal only in the isometric basis, and
only then does `(M G M^T + lambda I)^-1 = M (G + lambda I)^-1 M^T`, which is what
cancels the transform in `phi^T (G + lambda I)^-1 psi`. The compressed `1x`/`2x`
pairing used by the numerator gives the same kernel but is not norm preserving,
so it is not admissible here. `tr(G)` is invariant under the same action, and the
coefficients `(G + lambda I)^-1 phi_i` are invariant scalars, so `o_i` transforms
exactly like its values.

Because `(G + lambda I)^-1` is not a positive reweighting of the kernel, the
equivalent rows are signed. The lane is bounded, since the shifted inverse has
spectral norm at most `1/lambda`, and it is `O(3)` equivariant, but it is not a
convex attention distribution and is not described as one.

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

When `normalize_edge_conditioned_local_by_sqrt_degree` is enabled, let the
smooth cutoff mass be

```text
C_i = sum_j f_c(u_ij).
```

Every message family above is replaced by

```text
mtilde_i = m_i / sqrt(1 + C_i).
```

`C_i` is shared across heads and message types and introduces no learned
parameter. The additive one keeps the divisor finite when all edges vanish and,
critically, prevents a singleton factor `f_c` from cancelling against its own
normalizer. Thus both the message and its first coordinate derivative vanish
continuously when an edge reaches the cutoff. The historical public flag name
is retained for checkpoint/CLI compatibility; the default remains the
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

Self edges are excluded. Let

```text
C_i = sum_j f_c(u_ij),
S_i = sum_j f_c(u_ij)^2
```

be the smooth cutoff mass and effective degree. The equivariant aggregates are

```text
m^0_ih = LN_h(
    sum_j f_c(u_ij) sigmoid(g^s_ijh) a_ijh / sqrt(1 + C_i)
) + reshape_h(W_m [log(1 + C_i), log(1 + S_i)]),

m^v_ih = sum_j f_c(u_ij) [
    tanh(g^i_ijh) v_ih + tanh(g^j_ijh) v_jh
] / sqrt(1 + C_i),

m^r_ih = sum_j f_c(u_ij) tanh(g^r_ijh) d_ij
    / sqrt(1 + C_i),

m^T_ih = sum_j f_c(u_ij) tanh(g^T_ijh) ST(d_ij)
    / sqrt(1 + C_i).
```

Every MLP input and gate is invariant under `O(3)`. Multiplying invariant
gates by polar vectors, relative polar vectors, or even-parity
symmetric-traceless tensors preserves the required transformation law.
Receiver sums are invariant to edge order and consistent under node
permutation. At fixed width, the local path is `O(E_local)` after candidates
are built, while an LGL stack retains the exact `O(N)` factorized global
block. Scalar, vector, relative-vector, tensor, `C_i`, and `S_i` contributions
are partitioned into two width-balanced receiver reductions per gated local
stage. This bounds the temporary packed width while avoiding one scatter per
message family. The first edge-MLP affine map is also evaluated as
`W_i s_i + W_j s_j + W_r rbf_ij + W_I I_ij + b`, which is algebraically equal
to the concatenated map but does not materialize its full edge input. `S_i`
remains an explicit smooth concentration diagnostic for the learned mass
projection; it no longer controls message normalization. For `d` unit-weight
equal messages the aggregate scales as `d/sqrt(1+d)`, while a singleton is
attenuated as `f_c/sqrt(1+f_c)`.

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

### Static Cartesian tensor-product local transport

When persistent `2e` state is present, the opt-in CTP branch projects the
node tensor once into head layout. For normalized edge displacement
`d_ij=(x_j-x_i)/R_c`, define the symmetric-traceless edge quadrupole

```text
Q_ij = ST(d_ij outer d_ij).
```

The branch conditions its invariant gates on

```text
chi_ijh = [
    <T_ih,T_jh>_F,
    <T_ih,Q_ij>_F,
    <T_jh,Q_ij>_F,
    ||T_ih||_F^2,
    ||T_jh||_F^2
].
```

It reuses the incumbent edge latent and adds a channel-only projection of
`chi`; learned Cartesian parameters are never introduced. Three bounded
scalar gates multiply the native Cartesian bases

```text
b^v_ijh = T_jh d_ij,                  # 2e x 1o -> 1o
b^T0_ijh = T_jh,                      # 2e x 0e -> 2e
b^T1_ijh = ST(v_jh outer d_ij).       # 1o x 1o -> 2e
```

These values are added to the incumbent vector and tensor edge messages before
the same smooth-cutoff receiver reduction:

```text
m_ih = sum_j f_c(u_ij) message_ijh / sqrt(1 + sum_j f_c(u_ij)).
```

For any `R in O(3)`, `v,d -> Rv,Rd` and `T,Q -> R T R^T`.
Consequently every entry of `chi` is invariant, `T d` is polar `1o`, and both
tensor bases transform as reflection-even `2e`. Relative displacement gives
translation invariance; shared gates and receiver sums give edge-order and
node-permutation consistency. The final gate projection is initialized to
zero, which makes the first enabled forward exactly equal to the corresponding
persistent-`2e` control.

At fixed multiplicities, node projection is `O(N)` and the three contractions
are `O(E)` without an `N x N` pair tensor or neighbor triplets. The admitted
LBA specialization stores one compact `2e` tensor per head, carries it
unchanged through the global stage, and enables the full CTP path only in the
last local stage. This is not a general spherical-harmonic or arbitrary-`l`
tensor-product engine.

The opt-in pairwise-content repair leaves those equivariant moments unchanged
and adds one invariant scalar message. Let `qbar_ih` and `kbar_jh` be the raw
scalar query/key projections before positive-feature normalization, and let
`rho(u_ij)` be the existing RBF vector. A head-shared two-layer MLP gives

```text
e_ijh = MLP([qbar_ih, kbar_jh, rho(u_ij)]),  i != j.
```

For retained nonself edges, define the smooth cutoff mass and effective degree

```text
c_i = sum_j f_c(u_ij),
s_i = sum_j f_c(u_ij)^2.
```

The added scalar message is

```text
m_pair_ih = alpha (
    sum_j f_c(u_ij) e_ijh / sqrt(1 + c_i)
    + W_mass [log(1+c_i), log(1+s_i)]
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

An external sparse `edge_index` needs an explicit topology policy because
coordinates can move across the cutoff:

- `coordinate_neighbor_policy="error"` (default) rejects the ambiguous
  combination;
- `"fixed"` re-filters a fixed candidate set after every update and is an
  explicit approximation because omitted pairs cannot enter;
- `"rebuild"` discards the external candidates and rebuilds complete
  same-graph candidates from the current coordinates before every local stage.

The exact fallback is quadratic in each graph's node count. A production
cell-list or Verlet-skin backend is not implemented.

The private dynamic EGNN control uses its invariant edge embedding `m_ij`:

```text
r_i = mean_{j != i} (x_i - x_j) tanh(phi_x(m_ij)),
```

then applies the same graph-centering and bound. This follows the EGNN
relative-vector update pattern but remains a same-harness internal control, not
an official implementation reproduction.

## Optional ligand-pocket interaction readout

`readout_mode="interaction"` keeps the ligand mean scalar readout as a residual
baseline and adds an `O(E)` interface head after candidates are available. It
requires a Boolean ligand mask; its complement is the pocket. Without supplied
candidates, the current exact fallback still discovers them quadratically.
Normalized invariant node states produce ligand and pocket mean pools. For
directed ligand-receiver/pocket-sender edges inside the local cutoff, an
invariant edge MLP produces content `e_ij` and six gates `g_ija`. The
cross-interface content and polar moments are

```text
c_g = sum_(i<-j in cross_g) f_c(u_ij) e_ij / sqrt(1 + C_g),
P_ga = sum_(i<-j in cross_g) f_c(u_ij) tanh(g_ija) d_ij
       / sqrt(1 + C_g),
C_g = sum_(i<-j in cross_g) f_c(u_ij).
```

Two scalar triple products of the polar moments are pseudoscalars. The scalar
property head receives only the reflection-even combinations
`[chi_1^2, chi_2^2, chi_1 chi_2]`, together with ligand, pocket, and interface
pools. Thus the output remains invariant under full `O(3)`, including
reflections, while retaining parity-sensitive intermediate information. The
final interaction projection is zero initialized, so its initial graph scalar
is exactly the existing ligand mean. This is a parity-aware task head, not a
parity-complete `0o/1e/2o` hidden backbone.

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

## Sparse geometry-aware O(3)/SE(3) refinement

For a gated local layer, let `m^v_i`, `m^r_i`, and `m^T_i` be its incumbent
cutoff-weighted, receiver-normalized vector, relative-vector, and
symmetric-traceless aggregates. The opt-in refinement reuses them:

```text
v*_i = v_i + m^v_i + m^r_i,
T*_i = m^T_i (+ T_i when a persistent tensor carrier exists).
```

Invariant scalar gates produce bounded `q1,k1` from `v*` and bounded `q2,k2`
from `T*`. For retained nonself edges,

```text
ell_ijh =
    a0_h b_pair(h_ijh)
  + a1_h <q1_ih,k1_jh>
  + a2_h <q2_ih,k2_jh>_F,
alpha_ijh = receiver_softmax_j(5 tanh(ell_ijh/5), cutoff_ij).
```

All three score lanes are O(3)-invariant. The values form sparse scalar,
sender-vector, relative-vector, and `2e` sums, so fixed-width storage and work
are `O(E)`. The exact factorized global equations are unchanged.

When the declared symmetry is only SE(3), an additional value is legal:

```text
a_ijh = vee(T*_jh Q_ij - Q_ij T*_jh),
Q_ij = ST(d_ij d_ij^T).
```

This transforms as `Ra` for `det(R)=+1`, but picks up the axial parity under a
reflection. It is therefore excluded from the O(3) path and must not be
described as polar `1o` there.

## Equivariance

For `R in O(3)` and translation `t`, centered coordinates satisfy
`x'_i = R x_i`. Scalar contractions are unchanged, vectors transform as `Rv`,
and rank-2 tensors transform as `R T R^T`. Graph sums and channel mixing commute
with these transformations. Therefore the default path is O(3)-equivariant
and translation invariant; graph-wise sums also commute with node
permutations. The optional axial tensor-product path contracts only SE(3):
proper rotations and translations remain equivariant, while reflection
covariance is intentionally not required.

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
