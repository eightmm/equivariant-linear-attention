# HEMM coupling repair and performance scope

Date: 2026-07-17
Base commit: `591e7a241315f697a39c5354a48dd345639fed69`
Feedback source: https://chatgpt.com/share/6a59db49-8b50-83ee-9691-07bb73a472f4

## Research questions

1. On the current middle global layer, is the frozen Stage-0 failure caused by
   the assignment router, the radial center coupling, or cancellation after the
   effective pair gate?
2. Can the smallest registered identity-residual coupling

   ```text
   C_lambda = (1 - lambda) C_radial + lambda I,
   lambda in {0.10, 0.25, 0.50}
   ```

   make the mechanism materially active without changing the registered
   thresholds or the parameter/state schema?
3. If identity coupling is still functionally constant under the frozen
   Stage-0 threshold, can the feedback's shared multidimensional invariant MLP
   router activate the mechanism while preserving the symmetry and exact-bypass
   contracts?
4. Independently of HEMM admission, does the registered `lgl` backbone improve
   QM9 gap validation MAE over `ggg`, and what are the synchronized CUDA latency
   and peak-memory costs? Eligible memory arms are measured only after Stage-0.

The deliverable is a tested implementation and a reproducible evidence bundle,
not a default promotion. Test labels are never evaluated.

## Prior-information disclosure

Before this file was written, a read-only implementation auditor evaluated the
same-assignment identity counterfactual and reported that it was numerically
nonconstant but below the already-frozen `D >= 1e-2` material-activation
threshold for several registered-width seeds. That preview is retained as prior
information; the official counterfactual matrix below is therefore a
reproduction, not a blind confirmatory experiment. No threshold in this scope
was selected from that preview: the pair-gate, output, and promotion thresholds
come from `PROJECT.md` and the 2026-07-16 Stage-0 registration.

## Mathematical diagnostics

For assignments `pi_i` and `bar_pi = mean_i pi_i`, report per graph and head:

```text
H_marg = H(bar_pi) / log(M)
H_cond = mean_i H(pi_i) / log(M)
I_slot = H_marg - H_cond
```

`I_slot=0` exactly when all assignment rows are identical. Keep the existing
`H_cond` range but do not use it as the sole router diagnostic. The additional
assignment gates are:

```text
H_marg >= 0.05
I_slot >= 1e-3
minimum occupancy fraction >= 1e-4
```

For centers, report RMS spread about their mean, off-diagonal center-distance
minimum/median/maximum, and distance/cutoff quantiles. For coupling, report raw
radial and effective/mixed matrices separately, including off-diagonal nonunit
fraction and centered-Frobenius ratio. These explain the cause; the actual pair
gate and transport decide admission.

For the same assignment, evaluate:

```text
C_ones = 11^T
C_identity = I
C_radial = current radial coupling
C_lambda = (1 - lambda) C_radial + lambda I
```

The effective gate is `G = Pi C Pi^T`. Per graph/head metrics retain the frozen
`tau=1e-3` nonconstant tolerance and the CV/centered-Frobenius identity.

Mechanism differences use the symmetric relative RMS

```text
R_sym(a, b) = sqrt(2) ||a-b||_2 / sqrt(||a||_2^2 + ||b||_2^2).
```

Both-zero inputs give zero. Report this separately for the middle global
`scalar_message`, `vector_base`, `relative`, `tensor`, and radial-trace outputs;
for post-middle scalar, vector, and transient tensor states; and for gradients
of a fixed seeded random projection of the middle messages with respect to the
middle scalar input, middle vector input, and original positions. The aggregate
middle-message difference, aggregate post-middle-state difference, and each of
the three gradient differences must be at least `1e-5`. The full-model relative
output RMS retains its existing `1e-5` threshold as a secondary cancellation
check.

## Frozen counterfactual and repair decision

The official matrix uses CPU float64, heads=4, memory counts `{4,8}`, hidden
widths `{16,64}`, seeds `{401,402,403}`, and four deterministic 16-node graphs:

- A, feature-spatial aligned: qualification graph; every registered check must
  pass for every width/seed/head.
- B, feature-spatial crossed: robustness diagnostic, reported without changing
  qualification.
- C, spatial-only: geometry-limit diagnostic, reported without a promotion
  threshold.
- D, semantic-only compact geometry: semantic-routing diagnostic; any semantic
  activation claim additionally requires `I_slot >= 1e-3`, pair-gate
  `D >= 1e-2`, and nonconstant fraction `>=0.10` in the registered-width lane.

The existing A-graph checks remain unchanged and apply to every head:

```text
all values finite
0.05 <= H_cond <= 0.995
H_marg >= 0.05
I_slot >= 1e-3
minimum occupancy fraction >= 1e-4
effective coupling q00 <= 0.99
pair-gate centered-Frobenius ratio >= 1e-2
pair-gate nonconstant fraction >= 0.10
middle-message R_sym >= 1e-5
post-middle-state R_sym >= 1e-5
scalar/vector/position gradient R_sym >= 1e-5 each
full-output relative RMS versus exact M=1 >= 1e-5
```

Decision sequence:

1. Reproduce the clean `591e7a2` incumbent with `C_ones`, `C_identity`,
   `C_radial`, and the three fixed residual candidates.
2. If identity itself meets the material pair-gate threshold, choose the
   smallest residual candidate that passes every A-graph check for M=4 and M=8
   across both widths and all seeds. Do not select lambda from QM9 labels.
3. If identity is numerically variable but below the frozen material threshold,
   treat it as functionally constant for Stage-0. This is the quantitative form
   of the feedback's router-failure condition. Implement one preregistered
   shared invariant router:

   ```text
   z_i = normalize(tanh(W2 SiLU(W1 s_i))) in R^8
   logits_im = bounded(4 z_i^T e_m / temperature)
   pi_i = softmax(logits_i)
   ```

   where fixed unit DCT slot codes `e_m` are invariant, `W1/W2` are shared over
   heads and independent of M, all route/memory arms allocate the same router
   parameters, weights are not zero-initialized, M=1 remains an exact execution
   bypass, and the existing single spatial refinement is retained. The router
   dimension 8 and logit scale 4 are frozen before its outcome is observed.
4. Repeat the same residual selection. If no registered candidate passes, block
   interacting memory training and retain the falsification. Do not weaken a
   threshold, add an occupancy loss, or introduce read/write or typed memory.

For any residual implementation, `[0,1]`, symmetry, exact unit diagonal, slot
permutation covariance, O(3), translation, node permutation, M=1 exactness, and
structured-vs-dense forward/input-gradient equivalence are required. A passing
repair establishes assignment-similarity transport, not separated spatial
centers or a PSD kernel.

## Performance registration

The performance study is independent until an interacting memory arm is
admitted. It uses the existing QM9 `gap` target in eV, random-row split seed 42,
110k/10k train/validation rows from 130k loaded rows, model seeds 41/42/43,
FP32, batch 64, width 64, three layers, four heads, 2,000 steps, the same
optimizer/schedule, and no test evaluation.

First compare `ggg M=1` against `lgl M=1`. A candidate passes only if:

```text
mean paired validation-MAE improvement >= 0.010 eV
at least two of three seeds improve
worst paired regression <= 0.020 eV
paired parameter count, state schema, initial-state hash, source/data/split hash
latency increase <= 20% and peak CUDA allocation increase <= 20%
```

The CUDA decision uses eager FP32, synchronized warmup and measurement, five
fresh-process samples, and the median process mean. Measure forward and
forward/backward at 64 graphs with 18 nodes (representative) and 29 nodes
(bounded stress). Compile results, if any, are descriptive only. `lgl M=4` and
`M=8` are benchmarked only after Stage-0; a memory accuracy run additionally
requires `lgl` to pass the backbone rule and its own latency/memory ceiling.
M4 and M8 are decided separately; if both pass, prefer M4.

## Compute and stop envelope

- Environment: existing `uv.lock`; install only the locked `qm9` extra
  (`torch-geometric==2.8.0`, `rdkit==2026.3.3` and locked transitive packages).
- Inputs: existing local `data/qm9`; no dataset download or private data.
- Device: one local NVIDIA RTX PRO 6000 Blackwell GPU.
- Outputs: this run directory only, plus the project `.venv`; no lock rewrite.
- Budget: at most 30 cumulative GPU-minutes and approximately 0.2 GiB added
  environment storage. Stop launching new arms at 25 minutes, reserving five
  minutes for verification. Stop and record any individual 2,000-step run that
  exceeds 180 seconds before starting another.
- Smoke: focused CPU tests, fast/GPU project checks, then 1--2 QM9 steps on the
  maximum eligible path before any 2,000-step run.
- Cancellation: interrupt only processes launched for this run; preserve logs
  and partial outputs. No test labels, remote compute, credentials, destructive
  operations, or new network hosts beyond the locked package registry are in
  scope.

The user's request to implement the supplied feedback and measure performance
is treated as the one-time approval for this bounded package/GPU envelope.
