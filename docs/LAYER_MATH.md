# Factorized Moment Attention

For graph `g`, node `i`, and head `h`, let positive feature maps be
`q_ih, k_jh in R^D`. The unnormalized content kernel is

```text
K_ijh = q_ih^T k_jh >= 0.
```

The implementation builds each feature from positive scalar features and a
squared-vector feature map. If `u` and `v` are head vectors,

```text
<phi(u), phi(v)> = 1 + alpha (u^T v)^2 >= 1.
```

This is O(3)-invariant because inner products are preserved by every orthogonal
matrix, including reflections.

## One balancing cycle

For each graph and head, compute `m_jh = sum_i K_ijh` and use
`K'_ijh = K_ijh / m_jh`. The final row-normalized weights are

```text
A_ijh = K'_ijh / sum_l K'_ilh.
```

No pair matrix is needed. With

```text
s_h = sum_j k_jh / m_jh
S_h = sum_j (k_jh / m_jh) outer v_jh,
```

the denominator and numerator are `d_ih = q_ih^T s_h` and
`n_ih = q_ih^T S_h`, so the message is `n_ih / d_ih`. Tests compare this
factorization with the explicit dense kernel in float64.

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

At fixed feature dimension, head count, depth, and one balancing cycle, time
and intermediate storage are `O(N)`. The current wrapper still validates inputs
and derives graph count at runtime; full-graph compile behavior remains a
separate performance target.
