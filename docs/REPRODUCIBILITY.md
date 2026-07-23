# REPRODUCIBILITY

Single-command repro for any reported number.

## Repro Command

```bash
git checkout <commit>
uv sync
uv run python scripts/train.py --config configs/<exp>.yaml --seed <seed>
uv run python scripts/eval.py  --config configs/<exp>.yaml --ckpt <path>
```

## Determinism modes

`scripts/train_compare.py` exposes two explicit process-level modes:

- `--determinism seeded` is the backward-compatible default. It seeds Python
  and PyTorch, disables deterministic algorithms, leaves cuDNN deterministic
  mode off, and disables cuDNN benchmarking. It is reproducibly configured but
  is not promised to be bitwise deterministic on CUDA.
- `--determinism strict` additionally calls
  `torch.use_deterministic_algorithms(True, warn_only=False)`, enables
  `cudnn.deterministic`, and requires `CUBLAS_WORKSPACE_CONFIG` to be
  `:4096:8` or `:16:8`. If unset, the runner sets `:4096:8` before CUDA work.
  An unsupported nondeterministic operation fails instead of silently warning.

Every result stores the effective values under `reproducibility`. The previous
document listed strict controls as pins even though the registered QM9 runner
did not apply them; that documentation/implementation mismatch is corrected by
the 2026-07-23 reproducibility-hardening packet.

Dataset shuffling must continue to use an explicit local
`torch.Generator().manual_seed(seed)` where applicable. A process seed alone is
not a substitute for a recorded data/split identity.

## Same-seed repeat gate

For effects near the existing `0.01 eV` promotion threshold, run the exact same
source, command semantics, data/split, initial state, and model seed in five
fresh processes. Summarize the outputs with:

```bash
uv run python scripts/summarize_reproducibility_runs.py \
  --metric-path val_mae \
  --max-metric-span 0.005 \
  --min-runs 5 \
  --output artifacts/<run-id>/repeat-summary.json \
  artifacts/<run-id>/repeat-{1,2,3,4,5}.json
```

The seeded noise-floor gate passes only when the five-run validation-MAE range
is at most `0.005 eV`. The gate derives its decision rule from the identical
recorded determinism mode: strict runs always require one unique final-state
hash and exactly equal metric values. `--expected-mode strict` may additionally
assert the caller's intended lane, but it cannot weaken the recorded rule. A
strict CUDA error is evidence that the current operator path is unsupported,
not a reason to switch modes after inspecting outcomes.

### Recorded strict QM9 CUDA result

The 2026-07-23 packet under
`artifacts/qm9-strict-cuda-repeatability-20260723/` executed five fresh FP32
CUDA processes for the current static LGL control (500 updates, model/split
seed 42, random-row train/validation sizes 110000/10000). All five produced
validation MAE `0.6988662062644958 eV` and final-state SHA-256
`22a337b66efe53d63909b4abf376dd65ef9c47c2d6aecce81f14cffc5d902976`.
The metric range and sample standard deviation were both zero, and the strict
gate passed without fallback. Test evaluation was disabled.

This conclusion is intentionally narrow: the current source, one model seed,
one random-row split, FP32, and the recorded RTX PRO 6000. It does not establish
cross-hardware determinism, multi-seed accuracy, generalization, or superiority
over EGNN. The validation MAE must not be compared as an architecture
improvement against historical runs from different source hashes. All five
runs clipped gradients on 456/500 updates, so the packet also confirms that
the high-clipping diagnostic remains unresolved.

## What Pins a Run

| Element | Pinned by |
|---------|-----------|
| Code | git commit |
| Deps | `uv.lock` |
| Data | hash in `DATA.md` |
| Hardware | `~/.oh-my-setting/local/machine.md` |
| Hyperparams | config yaml in `configs/` |
| Seed | `--seed` arg |

## Known Non-Determinism

- The historical seeded CUDA lane produced `0.778593` and `0.767847 eV` from
  matching source, command semantics, data/split, and initial state. Atomic
  reductions are plausible but not yet isolated as the cause.
- Some CUDA kernels (atomic add, scatter) can remain nondeterministic in the
  seeded lane.
- Mixed precision: results may differ across GPU generations
- Multi-GPU reduction order

## Update Triggers

Any new source of randomness -> add pin + note here.
