# Frozen scope: bounded positive content and shifted `2e` kernel

Date: 2026-07-23
State: confirmed by the user's instruction to implement substantial
architecture improvements and validate them on real data

## Decision

Determine whether two exact-factorization-compatible changes raise useful
selectivity without sacrificing the project's O(3), positivity, finite-bound,
permutation, or linear-global-memory contracts:

1. retain bounded scalar-content magnitude instead of projecting every
   positive feature vector to essentially unit norm;
2. let persistent symmetric-traceless `2e` state affect transport weights
   through a shifted Frobenius inner-product kernel.

The deliverable is opt-in code, minimal mathematical and public-boundary
verification, matched strict-CUDA QM9 evidence, a cached train-only
ATOM3D-LBA/PDBBind capacity comparison, provenance, independent review,
documentation, commit, and push.

This is not a test-set result, official EGNN reproduction, PDBBind
generalization result, docking result, force model, softmax equivalence claim,
chirality model, or proof that finite-degree moments are universal.

## Architecture

### A1: bounded-magnitude positive scalar content

For `z = ELU(x) + 1 > 0`, let `u(z)` be the incumbent inward unit-norm map and
`r = ||z||_2 / sqrt(D)`. The candidate uses

```text
phi_bounded(z) = u(z) * 2r / (1 + r).
```

Its norm is below two, so `phi(q)^T phi(k)` is positive and bounded above by
four while retaining a monotone signal from the projected feature magnitude.
The default `unit` mode is byte-for-byte unchanged. The candidate is not
described as exact softmax or as an unbounded exponential feature map.

### A2: shifted persistent-`2e` product kernel

Separate channel mixes form symmetric-traceless query/key tensors, each mapped
inside the unit Frobenius ball. With `0 < eta_h < eta_max`,

```text
K2_ijh = eta_h * (1 + <Q2_ih, K2_jh>_F) >= 0.
```

Appending `sqrt(eta_h) * [1, vec(Q2)]` and
`sqrt(eta_h) * [1, vec(K2)]` to the scalar query/key features makes this term
an ordinary feature dot product. Existing graph summaries therefore evaluate
it exactly in `O(N)` at fixed width without pair storage. Under
`T -> R T R^T`, the Frobenius contraction is invariant for every `R in O(3)`.
The option is invalid without persistent `2e` hidden channels and allocates no
query/key tensor parameters when disabled.

## Claims and falsifiers

- **C1 — default compatibility:** both options disabled preserve the current
  state schema, initialization, outputs, and run configuration.
- **C2 — mathematical contract:** bounded scalar content remains positive and
  finite; the shifted tensor term is nonnegative and agrees with the explicit
  Frobenius formula; the factorized result agrees with a dense reference.
- **C3 — symmetry contract:** full scalar outputs remain O(3)/reflection and
  translation invariant, permutation consistent, and batch isolated.
- **C4 — QM9 utility:** at least one candidate improves the strict 500-update
  screen by `0.010 eV` while finite and within the `+0.020 eV` guard. Only the
  lowest admitted candidate advances.
- **C5 — QM9 confirmation:** across seeds 41--45 and 2,000 updates, the selected
  candidate improves mean validation MAE over the rerun incumbent by at least
  `0.020 eV`, improves at least four pairs, and has worst regression no larger
  than `0.020 eV`. EGNN competitiveness is a separate lower-mean plus
  three-paired-win condition.
- **C6 — large-complex capacity:** on frozen ATOM3D-LBA train rows 0--15, the
  combined candidate reaches train MAE at most `0.10 pK` within 3,000 updates.
  Incumbent and private EGNN are matched descriptive controls.

Any failed threshold remains a null or negative result; thresholds and arm
identities are not moved after outcome inspection.

## Data and execution boundary

- QM9 `gap`, cached PyG data, random-row split seed 42, train/validation
  `110000/10000`, test disabled, train-only target normalization, FP32,
  strict deterministic CUDA, batch 64.
- Screen arms: incumbent unit/no-`2e` kernel; bounded content; unit plus
  persistent `4x2e` shifted kernel; combined bounded plus persistent `4x2e`.
- Confirmation arms, only after C4: rerun incumbent, selected candidate, and
  width-91 private static EGNN for model seeds 41--45, 2,000 updates.
- ATOM3D-LBA revision
  `f93dd2d150a47c270f624620f84e07451a158705`, cached train rows 0--15,
  pocket and ligand copies only, opaque atom tokens, co-crystal coordinates in
  Angstrom, supplied pK label, ligand readout, no validation/test access.
- PDBBind arms: existing spatial GGG plus persistent `4x2e`; the same model
  with A1+A2; near-parameter private static EGNN. Deterministic batches,
  constant AdamW, batch two, gradient clipping 1.0, at most 3,000 updates.
- One local RTX PRO 6000, cumulative GPU wall at most 1,800 seconds. Expected
  disk growth below 250 MB for JSON/log artifacts; no network, dependency,
  raw-data redistribution, checkpoint publication, or remote compute.
- Start with the smallest strict-CUDA smoke. On nonfinite state, deterministic
  operator failure, OOM, or budget exhaustion, save the terminal result.
  Cancellation targets only this run and first requests a graceful interrupt.

## Verification emphasis

Testing is intentionally narrow: exact feature/kernel algebra, default
compatibility, one dense/factorized reference, and the public
O(3)/translation/permutation/batch boundary. The primary decision evidence is
the matched real-data execution, not test count.
