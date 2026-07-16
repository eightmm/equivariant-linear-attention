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
the implemented contract uses closed bounds. The pair kernel is

```text
K_ijh = c + a_ih^T b_jh
        + beta_h (1 + q1_ih^T k1_jh)
        + gamma_h (q1_ih^T k1_jh)^2.
```

For `c > 0`, `0 <= beta_h <= beta_max`, and
`0 <= gamma_h <= gamma_max`, finite precision preserves the declared bound

```text
c <= K_ijh <= c + 1 + 2 beta_max + gamma_max.
```

The linear term distinguishes alignment from anti-alignment. Disabling it
recovers the quadratic-only P1 control.

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
       + beta_h k1_jh^T Q1_gh
       + gamma_h k1_jh^T Q2_gh k1_jh.
```

The linear contraction can be signed, but the combined angular contribution
`beta_h (Qr_gh + k1_jh^T Q1_gh)` is nonnegative because every query/key vector
has norm at most one. The floor contribution `c Qr_gh` is strictly positive.
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
balancing is the default; `--no-key-balancing` is retained only for the matched
P1 normalization study. Positivity no longer depends on balancing.

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

## Equivariance

For `R in O(3)` and translation `t`, centered coordinates satisfy
`x'_i = R x_i`. Scalar contractions are unchanged, vectors transform as `Rv`,
and rank-2 tensors transform as `R T R^T`. Graph sums and channel mixing commute
with these transformations. Therefore every block is O(3)-equivariant and
translation invariant; graph-wise sums also commute with node permutations.

## Numerical policy and complexity

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

At fixed feature dimension, head count, depth, and normalization choice, time
and intermediate storage are `O(N)`. The public wrapper validates inputs and
derives graph count plus graph counts once, then reuses that metadata in every
layer and graph readout. Validation can still create compile graph breaks;
full-graph compilation remains a separate performance target.

## Executable representation boundaries

The P2 counterexample suite records rather than hides the remaining limits:

- degree-2 moments collide for distributions with equal mass/mean/variance but
  different fourth moments;
- two-point global normalized geometry is independent of fragment separation,
  so the global branch has no structural cluster-decay guarantee;
- scalar output is invariant under global reflection;
- coordinate gradients of invariant scalar sums transform equivariantly.

These tests constrain claims; they do not turn the global block into a local
force field or parity-complete SBDD model.
