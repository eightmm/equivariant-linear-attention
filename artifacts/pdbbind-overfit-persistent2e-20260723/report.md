# PDBBind train-only persistent-2e packet

## Decision

The implementation and data path are admitted. The CPU fallback overfit gate
failed; the exact registered CUDA overfit and systems claims are not verified.

- C1, loader/readout correctness: supported within the frozen train-only
  packet. The exact sample IDs, node counts, and ligand counts are now enforced
  by the runner.
- C2, persistent Cartesian `2e`: supported for the implemented O(3),
  translation, node-permutation, batch-isolation, gradient, and disabled-path
  contracts in the float64 CPU reference lane. CUDA mixed-precision execution
  remains not verified. This does not support `2o`, external tensor inputs,
  SH/CG, or arbitrary `l>2`.
- C3, attention train MAE at most `0.10 pK`: the CPU fallback instantiation is
  rejected because neither arm reached the threshold within 3,000 updates.
  The frozen CUDA instantiation remains not verified.
- C4, CUDA resource comparison: unfulfilled/rejected as registered because
  unavailable or non-comparable measurement is its frozen falsifier. The
  underlying CUDA evidence status is not verified, so no performance
  conclusion is allowed.

## Frozen data

The packet used rows 0 through 15 of the pinned
`vector-institute/atom3d-lba` train Parquet at revision
`f93dd2d150a47c270f624620f84e07451a158705`. Pocket and ligand copies were
retained, the full-protein duplicate was removed, and affinity was pooled over
ligand nodes. Observed `input_ids` were treated as opaque categorical tokens.

A failed generic-builder preflight physically materialized local caches for all
splits. No validation/test row was indexed, printed, selected, or evaluated.
The production loader names only the two pinned train Parquet files.

## Result

| arm | parameters | best train MAE | final train MAE | CPU median step | elapsed |
|---|---:|---:|---:|---:|---:|
| edge-free GGG + persistent `4x2e` | 167,115 | 0.151798 pK @ 2,950 | 0.199863 pK | 0.117964 s | 373.890 s |
| private static EGNN, width 92 | 167,260 | 0.163536 pK @ 2,850 | 0.163732 pK | 0.252309 s | 823.408 s |

The latency medians are from one sequential CPU run after excluding the first
ten train steps. They time `train_regression_step` only: radius construction,
batch-index selection/collation, and periodic full-train evaluation are
excluded. These are not repeated end-to-end benchmark timings.

Both arms had finite gradients and changed state hashes. Attention ended with
157,022 nonzero gradient elements and EGNN with 155,482. Attention's lower
best observed MAE and 2.139x CPU step-rate ratio are descriptive only; they
were not promotion gates and do not rescue C3.

The curves oscillated late in training. That pattern is consistent with the
fixed `1e-3` learning rate being too aggressive near a memorizing solution,
but this packet does not establish causality.

## Verification and limitations

The final CPU gate passes all repository tests and float64 CPU smoke. The
registered CUDA gate could not run: CUDA was unavailable inside the sandbox,
and approved external execution was rejected by the current Codex usage limit.
The raw CPU result's `peak_cuda_memory_bytes=0` is a historical sentinel, not
a measurement; `result-correction.json` supersedes its interpretation with
unavailable/null.

This is not a validation result, test result, affinity generalization result,
official EGNN reproduction, force model, docking evaluation, or claim of
protein/point-cloud superiority.

## Smallest next experiment

Register a non-adaptive optimizer diagnostic that changes only the learning
rate schedule for both arms: constant `1e-3` versus a frozen decay ending near
`1e-4`, with the same rows, ordering, architecture, parameter matching, and
3,000-update cap. The preregistered prediction should be that attention crosses
`0.10 pK` without increasing model capacity. Full ID30 validation remains
blocked until the basic CUDA overfit gate passes and CUDA execution is
available.
