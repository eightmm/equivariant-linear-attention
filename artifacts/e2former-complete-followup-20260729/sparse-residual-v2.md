# Sparse residual v2 mechanics receipt

## Decision contract

- Question: can the homogeneous low-rank local residual stop cancelling its
  smooth cutoff while gaining useful invariant selectivity and explicit sparse
  neighbor semantics?
- Baseline: commit `7bf694d796c1b9a541b6f98df806ef3f33edce42`.
- Falsifier: any regression in default-off compatibility, O(3), translation,
  permutation, graph isolation, gradient wake-up, or the full fast repository
  gate.
- Scope: mechanics and software correctness only. No downstream-accuracy,
  CUDA-speed, or promotion claim.

## Reproduced defects

The prior edge weight was

```text
raw_ijr = cutoff_ij sigmoid(tanh(score_ijr))
w_ijr = raw_ijr / sum_j raw_ijr.
```

For a singleton receiver this gives `w=1`, exactly erasing the cutoff. The
bounded `sigmoid(tanh(.))` gate also had at most about `2.72x` contrast. RBF
features affected the score but not the five value families, the sparse route
ignored the legacy local-balancing flag, and missing sparse candidates silently
created a complete graph.

## Implemented equation

The invariant score is now

```text
s_ijr = q_ir k_jr + b_r + A_r RBF_ij
       + alpha_r <u_ir,v_jr>
       + beta_r <u_ir,d_ij><v_jr,d_ij>
       + chi_r (<u_ir,d_ij>^2 + <v_jr,d_ij>^2)
g_ijr = exp(L tanh(s_ijr/L)).
raw_ijr = cutoff_ij g_ijr.
```

The default positive lane computes all receiver statistics in one logical
`_local_receiver_sum` call:

```text
M_ir  = sum_j raw_ijr
M2_ir = sum_j raw_ijr^2
Y_irf = sum_j raw_ijr rho_ijrf V_ijrf
out_irf = Y_irf / (1 + M_ir)
rho_ijrf = 2 sigmoid(B_rf RBF_ij).
```

Consequences:

- singleton amplitude is `raw/(1+raw)`, so it still vanishes smoothly;
- identical unit-weight degree `0/1/64` amplitudes are `0`, `1/2`, `64/65`;
- scalar/vector/relative/tensor/radial values receive separate modulation from
  one compact RBF projection;
- `log1p(M)` and `log1p(M2)` feed a zero-initialized invariant scalar map;
- no normalized edge-weight tensor is formed in the default positive path.

The optional local-only softmax ablation is
`softmax_j(s + log(cutoff)) * cutoff`; exact global factorization is untouched.

## Public contracts

- `sparse_residual_normalization = "positive" | "softmax"`
- `sparse_residual_score_limit` lies in `[0.5, 4.0]`
- `sparse_residual_balancing = "receiver"`
- explicit legacy `use_local_key_balancing` is rejected for this route
- `sparse_residual_neighbor_policy = "require" | "complete_fallback"`
- complete fallback is rejected before construction when total nodes exceed
  `sparse_residual_complete_fallback_max_nodes`

All fields are appended to preserve positional config meaning and are propagated
by `build_regression_model`.

## Verification

- RED: 17 incumbent tests passed and 19 new contract tests failed for the
  expected missing controls/equations.
- Focused GREEN: `36 passed`.
- Related sparse/local/packed suite: `94 passed`.
- Repository gate: `bash scripts/check.sh fast`
  - `795 passed`
  - coverage `88.89%`
  - Ruff/compile checks passed
  - CPU ML smoke passed in float64

The tests cover cutoff value and first coordinate derivative, singleton and
two-edge crossing, degrees `0/1/64`, one fused receiver call, bounded positive
contrast, radial family isolation and gradient, mass features, local softmax,
config conflicts, bounded explicit fallback, O(3) including reflection,
translation, node/edge permutation, graph isolation, finite gradients,
zero-initialized incumbent equivalence, and second-step wake-up.

## Unverified

- CUDA train-step latency and allocator peak
- fully streamed custom CSR/ELL kernel and reverse-CSR backward
- downstream QM9/LBA/PDBBind/point-cloud utility
- promotion or default selection between positive and softmax
