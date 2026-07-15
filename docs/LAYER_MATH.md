# Factorized Moment Attention

For graph `g`, node `i`, and head `h`, let `q0_ih, k0_jh in R^D` be
positive scalar-content features. Let `q1_ih, k1_jh in R^3` be polar-vector
features mapped by the following unit-ball transform:

```text
q1 = u / sqrt(1 + ||u||^2),   k1 = v / sqrt(1 + ||v||^2).
```

In exact arithmetic, the vector norms are strictly below one and the learned
angular scale satisfies `0 < alpha_h < vector_kernel_max`. Floating-point
rounding can saturate either bound, so the implemented contract is
`||q1||, ||k1|| <= 1` and `0 <= alpha_h <= vector_kernel_max`. The pair kernel is

```text
K_ijh = q0_ih^T k0_jh + 1 + alpha_h (q1_ih^T k1_jh)^2 >= 1.
```

This is O(3)-invariant because inner products are preserved by every orthogonal
matrix, including reflections.

The individual coordinates of a flattened symmetric outer feature would be
signed. Positivity belongs to the complete quadratic kernel, not to every
feature coordinate. The production implementation therefore does not sum a
signed flattened feature and then take a dot product. It contracts structured
3x3 positive-semidefinite summaries.

For row scale `r_ih`, define

```text
Q0_gh = sum_i r_ih q0_ih
Qr_gh = sum_i r_ih
Q2_gh = sum_i r_ih q1_ih q1_ih^T.
```

Then each key mass is evaluated as

```text
m_jh = k0_jh^T Q0_gh + Qr_gh + alpha_h k1_jh^T Q2_gh k1_jh.
```

Every mass term is nonnegative and the constant term is strictly positive.
Value numerators use analogous scalar, constant, and matrix-valued summaries,
but their learned values may be signed, so these summaries are not PSD and the
quadratic numerator contraction is not clamped. This is algebraically identical
to the explicit dense kernel and remains `O(N)` at fixed width and head count.

## One balancing cycle

For each graph and head, compute `m_jh = sum_i K_ijh` and use
`K'_ijh = K_ijh / m_jh`. The final row-normalized weights are

```text
A_ijh = K'_ijh / sum_l K'_ilh.
```

No pair matrix is needed. Tests compare both the balanced and row-normalized
structured factorizations with the explicit dense kernel in float64. The single
balancing cycle remains a fixed architecture choice; whether it improves
selectivity is an empirical P1 question rather than a positivity requirement.

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

At fixed feature dimension, head count, depth, and one balancing cycle, time
and intermediate storage are `O(N)`. The public wrapper validates inputs and
derives graph count plus graph counts once, then reuses that metadata in every
layer and graph readout. Validation can still create compile graph breaks;
full-graph compilation remains a separate performance target.
