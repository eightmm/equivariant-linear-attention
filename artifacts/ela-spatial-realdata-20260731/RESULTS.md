# ELA spatial-operator real-data results

Date: 2026-07-31

The implementation and focused/full mechanics gates passed before the run.
All compared spatial arms had exactly the same parameter count, state schema,
initial state, optimizer, split, and update budget. Test evaluation remained
disabled.

## QM9 gap screen

Seed 42, 500 updates, 110k/10k seeded random-row train/validation split:

| Arm | Val MAE (eV) | Median step (s) | Peak CUDA (MB) | Clip fraction |
|---|---:|---:|---:|---:|
| explicit | 0.671843 | 0.19988 | 262.6 | 1.000 |
| implicit | **0.627661** | **0.18172** | 322.8 | 0.976 |
| hybrid | 0.638748 | 0.20132 | 390.6 | 1.000 |

Relative to explicit:

- implicit improved MAE by `0.044182 eV`, used `0.909x` step latency, and
  `1.229x` peak memory;
- hybrid improved MAE by `0.033096 eV`, used `1.007x` step latency, and
  `1.487x` peak memory.

The preregistered hybrid accuracy and latency conditions passed, but the
`<=1.35x` memory condition failed. Hybrid is therefore not promoted. Implicit
is the strongest one-seed screen result, but it is not confirmed across seeds
and its memory is worse on these small graphs.

All arms had 249,816 parameters, state schema
`47d127f82ae71f5500706cbc3e0dc467f67f10f7ce7903d08904ddb6c4d186c1`,
and initial state
`f99c99060494ede487cad14623d372ff474bfba21ad6997c6c0dfe110f309062`.
Every arm reported zero nonfinite gradient parameters.

## ATOM3D-LBA train-only capacity

Frozen 16-complex train subset, ligand-mask readout, strict seed 20260723,
maximum 1,000 updates:

| Arm | Train MAE (pK) | Steps | Median step (s) | Peak CUDA (MB) | Capacity gate |
|---|---:|---:|---:|---:|---:|
| explicit | **0.051845** | 900 | **0.15151** | 915.6 | pass |
| implicit | 0.286117 | 1,000 | 0.16045 | **360.5** | fail |
| hybrid | 0.198044 | 1,000 | 0.16963 | 1,064.2 | fail |

The explicit arm reached the registered `0.10 pK` threshold at step 900.
Implicit used only `0.394x` the explicit peak memory but did not fit the
training subset. Hybrid recovered some local capacity but still missed the
threshold while using `1.162x` memory.

All arms had 258,072 parameters and initial state
`de131c6b5b27ea1392e6d819c8176d6f5fcdc2b4852ee6b319fd8431a7c652c3`.
No arm evaluated validation or test data, and every arm reported zero
nonfinite gradients.

## Interpretation

The edge-free Gaussian Taylor operator is a useful smooth global spatial
channel, not a complete replacement for sparse local geometry:

- on small QM9 graphs it improved the one-seed validation screen and was
  slightly faster, but its moment workspace raised memory;
- on larger LBA complexes it delivered the expected memory advantage, yet
  failed the sharp local-interaction capacity probe;
- naively adding both paths did not dominate: hybrid missed the QM9 memory
  gate and the LBA capacity gate.

The architecture should therefore retain explicit local geometry as the
default for interaction-heavy 3D graphs. The implicit path remains
experimental as a selectively scheduled long-range residual. The next
architecture experiment should reduce its workspace and gate it by graph
scale/layer rather than running it at every layer.

These are one-seed QM9 validation and train-only LBA capacity results, not test
performance or a generalization claim.
