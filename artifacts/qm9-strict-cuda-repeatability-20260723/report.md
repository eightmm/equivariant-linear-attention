# Strict QM9 CUDA Repeatability

## Verdict

The registered gate passed. Five fresh strict CUDA processes produced exactly
the same validation MAE and canonical final model state.

| Measure | Result |
|---|---:|
| Fresh processes | 5 |
| Updates per process | 500 |
| Validation MAE | 0.6988662062644958 eV |
| Validation-MAE span | 0.0 eV |
| Sample standard deviation | 0.0 eV |
| Unique final-state hashes | 1 |
| Clipped updates per process | 456/500 (91.2%) |
| Test evaluated | No |
| Cumulative training wall time | 80.646 s |

The common final-state SHA-256 is
`22a337b66efe53d63909b4abf376dd65ef9c47c2d6aecce81f14cffc5d902976`.
Every result records strict deterministic algorithms, deterministic cuDNN,
disabled cuDNN benchmarking, `warn_only=false`, and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. The two-step strict CUDA smoke and all five
full runs completed without deterministic-operator fallback.

## Interpretation

For this exact source, static LGL configuration, seed, random-row split, FP32
mode, and RTX PRO 6000 environment, same-seed runtime noise did not perturb the
selected validation metric or canonical final state. The registered
`0.005 eV` range criterion passed with a measured range of zero; the stricter
bitwise criterion also passed.

The result does not show that the model is accurate enough or that it beats
EGNN. The value is not a controlled architecture comparison with historical
results because the source identity differs. It also does not establish
cross-hardware, mixed-precision, multi-seed, or distributed determinism.

## Diagnostic retained

Gradient clipping remained active on 91.2% of updates in every process. This
supports prioritizing a controlled local-message scaling intervention before
further small-effect accuracy decisions. It does not by itself prove that
clipping causes the accuracy gap or that degree normalization will fix it.

## Data and evaluation boundaries

- Cached QM9 data only; no download or dependency change.
- Random-row split with split seed 42.
- Target normalization fitted on the training subset.
- Validation was used for the registered repeatability metric.
- Test evaluation remained disabled.
- One model seed and one hardware/software environment were tested.
