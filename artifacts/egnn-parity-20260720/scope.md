# Frozen scope: private-EGNN parity

Date: 2026-07-20
State: confirmed by the user's `확정`

## Question and target

- Question: can the single public EquivariantAttention architecture beat the
  near-parameter-matched private static EGNN on the existing QM9 `gap`
  validation harness by repairing its local pairwise representation?
- Baseline: `internal_static_egnn_baseline`, width 91, three layers, static
  coordinates.
- Attention baseline: LGL, width 64, learned middle global transport, static
  coordinates, memory interaction and radial trace off.
- Parity: five-seed mean validation MAE at most `0.408932 eV`.
- Promotion: mean validation MAE at most `0.398932 eV`, at least three of five
  seed-wise wins against a rerun private static EGNN, and worst paired
  improvement at least `-0.020 eV`.

## Ordered architecture iterations

1. Enable the already allocated learned local radial gate and change nothing
   else.
2. Add a parameter-bounded invariant local edge-content path conditioned on
   receiver scalar state, sender scalar state, and the existing RBF distance
   basis. Expose neighborhood degree or pre-normalization attention mass to the
   scalar updater. Keep the existing equivariant vector/tensor moment path and
   factorized global transport.
3. Only after inspecting iterations 1--2, choose one isolated topology or
   optimization repair supported by their diagnostics. Do not combine unrelated
   repairs or revisit coordinate updates/multi-memory as substitutes.

Public defaults remain unchanged until a promotion pass. New dependencies and
a second public model family are excluded.

## Evaluation and compute

- Existing pinned local PyG QM9 rows and hashes; target index 4 `gap` in eV;
  supplied equilibrium coordinates in Angstrom.
- Random-row warm split seed 42: 110k train, 10k validation, 10k test. Fit target
  normalization on train only. Test evaluation is disabled throughout.
- FP32, batch size 64, AdamW, existing cyclic data order, three layers. Model
  seeds 41--45 for confirmation.
- Every iteration starts with a seed-42/500-update numerical screen. It advances
  only when finite, provenance-valid, and not clearly regressing. A screen does
  not promote a model.
- Confirmation reruns candidate and private static EGNN at seeds 41--45 for
  2,000 updates under the same source/data/split boundary.
- Record parameter and nonzero-gradient counts, fixed train-probe and validation
  metrics, pre-clip gradient norm and clip fraction, residual scales,
  synchronized elapsed time, peak CUDA memory, source/data/split/state hashes,
  and `test_evaluated=false`.
- One local GPU, at most three architecture iterations, at most 3,600 cumulative
  GPU-wall seconds. Stop on the first promotion pass or the first exhausted
  bound. No 10,000-step run or checkpoint publication.

## Required verification

- Write RED tests before each new capability.
- Preserve O(3), reflection, translation, permutation, batch isolation, mixed
  precision, and static-default state/output contracts.
- Run `scripts/check.sh fast` and `scripts/check.sh gpu` before registered QM9
  compute.
- Execute claims through `oms research-runner` and append every positive, null,
  or failed outcome to `docs/EXPERIMENTS.jsonl`.

## Claim boundary

This is adaptive validation evidence on a random-row warm split against a
private EGNN-style control. Even a pass is not an official EGNN reproduction,
scaffold/cold-molecule generalization result, test result, force model, relaxed
geometry result, or molecular-dynamics claim.
