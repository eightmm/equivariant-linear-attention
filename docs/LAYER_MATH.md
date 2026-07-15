# Equivariant Moment Attention

## State

The layer keeps persistent scalar and vector channels:

```text
s_i in R^C0
V_i in R^(C1 x 3)
```

Rank-2 symmetric-traceless features are transient five-component messages, not
persistent hidden state. Graph coordinates are centered and RMS-scaled:

```text
mu_g = mean_i(x_i)
rho_g = sqrt(mean_i(||x_i - mu_g||^2) + eps)
y_i = (x_i - mu_g) / rho_g
```

`log1p(rho_g)` and absolute node radii remain scalar inputs so normalization
does not erase molecular length scale.

## Invariant Kernel

Each head has scalar query/key features and one equivariant routing vector. The
nonnegative kernel is

```text
kappa_ij = phi(q0_i)^T phi(k0_j) + 1 + lambda (q1_i . k1_j)^2
phi(z) = ELU(z) + 1
lambda = softplus(raw_lambda)
```

The constant stabilizer and angular term remain exactly factorized because

```text
(q1_i . k1_j)^2 = <q1_i q1_i^T, k1_j k1_j^T>_F.
```

The six-component symmetric outer-product feature therefore adds l=0 and l=2
angular routing without materializing pairwise attention.

An opt-in radial kernel makes the attention weights depend directly on pair
distance. For each graph, set `L = 2 max_i ||y_i||`, `u_i = y_i / L`, and use a
per-head shift `a_h > 1`. Then `||u_i-u_j||^2 <= 1` and

```text
rho_ijh = a_h - ||u_i-u_j||^2 > 0.
```

This kernel has the exact rank-five factorization

```text
p_i   = [1, ||u_i||^2, sqrt(2) u_i]
q_jh  = [a_h - ||u_j||^2, -1, sqrt(2) u_j]
rho_ijh = p_i . q_jh.
```

Multiplying the content kernel by `rho` remains factorized through Kronecker
features, `(chi_q tensor p_i)` and `(chi_k tensor q_jh)`. It therefore adds no
pairwise edge tensor and preserves exact O(3) invariance. The individual
factor features are signed even though the resulting pair kernel is positive,
so accumulated denominators require explicit numerical checks.

An experimental shifted-square kernel adds the missing first-order angular
term while remaining nonnegative and factorized. Routing vectors are mapped
equivariantly into the open unit ball, `u(v) = v / sqrt(1 + ||v||^2)`, and

```text
kappa_ij = phi(q0_i)^T phi(k0_j) + lambda (a + u(q1_i).u(k1_j))^2
a = 1 + softplus(raw_a).
```

Its ten angular features are

```text
[sqrt(lambda) a, sqrt(2 lambda a) u(v), sqrt(lambda) svec(u(v)u(v)^T)].
```

Their inner product is exactly the shifted square above. Since `a > 1` and
the normalized dot product is greater than `-1`, the angular kernel is
strictly positive.

## Linear Transport

For factorized query/key features `chi_q`, `chi_k`, the transport is

```text
A_i[f] = chi_q_i^T sum_j(c_j chi_k_j tensor f_j)
         / chi_q_i^T sum_j(c_j chi_k_j)
```

The default one-cycle key-mass preconditioner is

```text
c_j = 1 / (sum_i kappa_ij + eps).
```

It reduces key sinks but one cycle is not a converged doubly-stochastic
normalization. Additional exact factorized Sinkhorn cycles alternate

```text
u^(0) = 1
v^(t+1) = 1 / (K^T u^(t) + eps)
u^(t+1) = 1 / (K v^(t+1) + eps).
```

Neither matrix-vector product materializes `K`: `K v` is computed as
`chi_q^T sum_j(chi_k_j v_j)`, and `K^T u` swaps query and key. Setting the
iteration count to one is exactly the original key preconditioner followed by
row normalization.

An experimental per-head relaxation uses

```text
c_jh = exp(-eta_h log(m_jh + eps)),  eta_h = sigmoid(theta_h),
```

where `m_jh = sum_i kappa_ijh`. The default exact preconditioner is `eta=1`.

## Relative Moments

Exact relative moments are reconstructed from graph summaries:

```text
R_i = A_i[g y] - y_i A_i[g]
    = sum_j alpha_ij g_j (y_j - y_i)
```

For `ST(a,b) = (a b^T + b a^T)/2 - (a.b)I/3`, the transient l=2 moment is

```text
T_i = A_i[t ST(y,y)] + ST(y_i,y_i) A_i[t]
      - 2 ST(A_i[t y], y_i).
```

The complementary l=0 trace, available as an experimental scalar feature, is
reconstructed without pairwise edges:

```text
tau_i = A_i[t ||y||^2] + ||y_i||^2 A_i[t] - 2 y_i.A_i[t y]
      = sum_j alpha_ij t_j ||y_j - y_i||^2.
```

Vector updates combine transported vectors, `R_i`, and `T_i q1_i`. Scalar
updates use invariant contractions including `q1_i.R_i`, `T_i:T_i`, and
`q1_i^T T_i q1_i`.

## Invariant-Conditioned Dynamic Routing

The opt-in dynamic route enriches the fixed vector mixture without adding
pairwise work. For each transient symmetric-traceless tensor, define

```text
T2_i = ST(T_i^2) / (T_i:T_i + eps)
c3_i = tr(T_i^3) / (T_i:T_i + eps)^(3/2).
```

Invariant features built from bounded contractions of `q1`, transported base
vectors, `R`, `T q1`, `T2 q1`, and `c3` produce four node/head-specific
corrections `delta`:

```text
delta = delta_max tanh(MLP(invariants) + W_context LN(s))
M = (1 + delta_B) B
    + (c_R + delta_R) R
    + (c_T + delta_T) T q1
    + delta_T2 T2 q1.
```

The final invariant and context projections are zero initialized, making the
initial dynamic layer exactly equal to the static route. Each coefficient is a
scalar invariant and each routed value is `1o`, so the update is O(3)
equivariant. All tensor polynomials are pointwise after the existing linear
transport, preserving linear node scaling.

The optional full-Gram path additionally exposes the upper triangle of
`V_i V_i^T`, all state/message products `V_ic.M_ih`, `||B_ih||^2`, and
`B_ih.R_ih` to the scalar MLP. These are invariant because every spatial index
is fully contracted.

## Pointwise Equivariant FFN

The optional FFN provides a node-local nonlinear bypass around global
transport. After the attention residual, let `s_hat = LN(s)` and let bounded
vector channels be `V_hat`. The scalar branch is

```text
z_i = [s_hat_i, ||V_hat_i1||^2, ..., ||V_hat_iC1||^2]
(a_i, b_i) = W_in z_i
Delta s_i = W_out (SiLU(a_i) * b_i).
```

The vector branch mixes vector channels and gates them only with invariant
scalars:

```text
Delta V_ic = tanh((W_gate s_hat_i)_c) sum_d W_vector_cd V_hat_id.
```

The branches have independent learned residual scales initialized consistently
with the attention residual. Vector norms and scalar gates are invariant, while
channel mixing does not touch spatial indices, so the block is exactly
O(3)-equivariant. It adds `O(N(C0^2 + C0 C1 + C1^2))` pointwise work and no
pairwise storage or graph edges.

## Complexity

With kernel width `D = head_dim + 1 + 6`, all transported values are
concatenated into one segment reduction. For `S` Sinkhorn cycles, time remains
linear in node count:

```text
O(N H D (head_dim + 16 + S))
```

Persistent state storage is `O(N (C0 + 3 C1))`; transient tensors use five
components per head.

The shifted kernel changes the angular width from 7 to 10. Radial trace adds
one transported scalar per head. Full Gram adds
`C1(C1+1)/2 + C1 H + 2H` scalar inputs but does not change asymptotic scaling
in node count. None of these options constructs an `N x N` tensor.
The multiplicative radial-distance kernel multiplies the content feature width
by five while retaining linear node scaling.
The optional FFN changes channel-mixing cost but not the attention rank or its
linear scaling in node count.

## Symmetry Boundary

All routing weights and scalar updates are invariant contractions. Vector and
tensor messages transform under rotations, coordinate centering removes
translations, and segment reductions preserve permutations. The current
operations are also reflection-equivariant; chirality-sensitive scalar outputs
would require explicit parity-odd channels.

## Current Evidence

On the fixed-seed QM9 110k/10k/10k validation-only screen, radial trace
improved 500-step MAE from `0.78290` to `0.75905` at a 1.6% measured runtime
increase. Full Gram and a learnable balancing exponent worsened that screen.
Shifted-square reached `0.75683` but exceeded the baseline runtime by about
11.5%.

At 2,000 steps, neither trace-only (`0.62242`) nor trace plus shifted-square
(`0.61960`) improved the historical unchanged validation result (`0.61876`).
Therefore all enhancements remain explicit opt-in ablation flags and the
original kernel remains the default. These are single-seed adaptive validation
results, not final generalization claims.

From the transport-bypass perspective, a pointwise equivariant FFN produced the
first improvement that persisted at 2,000 steps. On a source-hashed matched
comparison, ratio-1 FFN reached `0.57584` validation MAE and ratio-2 reached
`0.51632`, versus `0.61710` without the FFN. Ratio 0.25 and ratio 4 both failed
their 500-step screens, and adding radial trace to the FFN did not help.

The ratio-2 FFN is therefore the default moment-linear block. On the final
data-verified tree, paired model seeds 41/42/43 reached mean validation MAE
`0.56704 +/- 0.03173`, versus `0.61664 +/- 0.01331` without the FFN. All three
paired differences favored the FFN, with mean improvement `0.04961 +/-
0.04156` (`8.0%`), but the n=3 95% paired t-interval included zero and the
pre-registered absolute threshold of `0.05` was narrowly missed.

The FFN comparison also changes capacity from `71.9k` to `150.9k` parameters
and increased mean measured runtime by `30.5%`. It is therefore a
performance-oriented default candidate on the adaptively reused random-row
warm validation split, not a statistically confirmed or mechanism-isolated
improvement. Frozen test and cold-split evaluation remain intentionally open.
