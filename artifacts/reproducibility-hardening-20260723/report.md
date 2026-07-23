# Reproducibility hardening result

## Verdict

The implementation packet passes its CPU software and reference-lane gates.
The historical seeded behavior remains the default, while `strict` is opt-in
and fails loudly when deterministic execution cannot be honored.

Five independent strict CPU synthetic runs produced:

- validation MAE: `0.07374836206436157` in all five runs;
- validation-MAE range: `0.0`;
- sample standard deviation: `0.0`;
- unique final-state hashes: `1`;
- strict repeat verdict: passed.

This supports CPU reference-lane repeatability for the recorded tiny synthetic
configuration only. It is not evidence for CUDA, QM9, mixed precision,
cross-machine reproducibility, model accuracy, or performance.

## Implemented

- `configure_reproducibility(seed, mode)` centralizes and reports the effective
  PyTorch, cuDNN, and cuBLAS controls.
- `scripts/train_compare.py --determinism {seeded,strict}` configures the
  process before model construction or CUDA work and records the result.
- `scripts/summarize_reproducibility_runs.py` validates run identity, computes
  metric mean/sample standard deviation/range, counts final-state hashes, and
  emits a nonzero exit status when the selected gate fails.
- Identity validation requires meaningful 64-hex source/state/split/data
  hashes, complete run configuration, and a complete effective reproducibility
  state; identical null or empty placeholders are rejected.
- The registered same-seed near-threshold rule uses five fresh processes and a
  maximum validation-MAE range of `0.005 eV`. The verdict is derived from the
  recorded mode: strict always requires identical metric values and final-state
  hashes, while an optional expected-mode assertion cannot weaken that rule.

## Verification

- Focused: `23 passed`.
- Repository: `446 passed`, total coverage `88.59%`.
- CPU ML smoke: passed in float64.
- Strict CPU repeat summary: passed.

The initial independent record/method review found two major and one minor
software-contract gaps. All three were repaired with adversarial tests before
the authoritative raw CPU runs and manifest were regenerated; the original
finding response is retained for re-review.

## Not verified

- CUDA deterministic-algorithm compatibility.
- The historical seeded QM9 CUDA noise floor after this instrumentation.
- CUDA mixed precision, multi-GPU, or another hardware/software stack.

The next compute action is a separately approved five-process CUDA
repeatability packet. No architecture ablation should use a near-`0.01 eV`
promotion claim before that result is available.
