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

## Local heads

Local heads use raw-coordinate directed edges `i <- j` within the same graph.
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
agrees with the direct formula to the tested tolerance.

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
layers, while the core fallback may still perform `O(N^2)` candidate search per
graph. This changes repeated geometry work from `L*N^2` to `N^2 + L*E`; it does
not turn discovery into a production sparse backend. The public wrapper derives
graph metadata once and reuses it. Validation can still create compile graph
breaks; full-graph compilation and production sparse-neighbor performance
remain separate targets.

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
