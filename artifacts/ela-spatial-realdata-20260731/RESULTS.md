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

## Current-source contextual controls

The controls were run after freezing the spatial result. LGL has since been
retired by project decision and is excluded from architecture selection.

### QM9

| Model | Val MAE (eV) | Median step (s) | Peak CUDA (MB) | Parameters |
|---|---:|---:|---:|---:|
| ELA implicit | **0.627661** | 0.18172 | 322.8 | 249,816 |
| frozen unified | 0.664455 | 0.16560 | 250.7 | 209,229 |
| private static EGNN | 0.718706 | **0.00543** | **133.8** | 162,154 |

ELA implicit improved MAE by `0.036794 eV` over frozen unified at
`1.097x` latency and `1.288x` memory. It improved MAE by `0.091045 eV`
over the private EGNN but used `33.49x` latency and `2.412x` memory. This is an
accuracy win at the fixed 500-update screen, not an efficiency win or an
official-EGNN result.

### LBA train-only

| Model | Train MAE (pK) | Steps | Median step (s) | Peak CUDA (MB) |
|---|---:|---:|---:|---:|
| ELA explicit | **0.051845** | **900** | 0.15151 | 915.6 |
| frozen unified | 0.078332 | 1,000 | 0.13875 | 902.7 |
| private static EGNN | 0.473982 | 1,000 | **0.00783** | **779.1** |

ELA explicit improved train MAE by `0.026487 pK` over frozen unified while
using `1.092x` latency and `1.014x` memory. It improved MAE by `0.422137 pK`
over private EGNN but used `19.36x` latency and `1.175x` memory. This is
train-set memorization capacity only.

The combined evidence favors an architecture policy, not one universal arm:

- explicit sparse local geometry for interaction-heavy or sharp local tasks;
- implicit smooth spatial transport only as a selectively scheduled global
  residual where its accuracy gain justifies the workspace;
- no always-on hybrid until its memory and optimization behavior improve.

## Periodic implicit follow-up

`implicit_every=3` in this three-layer implementation executes the implicit
residual at zero-based layer `[0]` only. This is a frequency-and-placement
ablation, not another matched operator-replacement comparison.

### QM9

| Arm | Val MAE (eV) | Median step (s) | Peak CUDA (MB) |
|---|---:|---:|---:|
| explicit | 0.671843 | 0.19988 | 262.6 |
| implicit, every layer | **0.627661** | 0.18172 | 322.8 |
| implicit, layer `[0]` | 0.652378 | **0.17440** | **239.5** |
| hybrid, every layer | 0.638748 | 0.20132 | 390.6 |
| hybrid, layer `[0]` | 0.632676 | 0.20079 | 306.1 |

Scheduled implicit improved explicit by `0.019466 eV`, with `0.873x` latency
and `0.912x` memory. It nevertheless regressed always-on implicit by
`0.024716 eV`, exceeding the registered `0.020 eV` guard. It therefore failed
promotion and remains experimental.

Scheduled hybrid is the best exploratory resource/accuracy compromise in this
one seed: versus explicit it improved MAE by `0.039167 eV` at `1.005x`
latency and `1.166x` memory; versus always-on hybrid it improved MAE by
`0.006072 eV` and reduced memory to `0.784x`. The preregistration did not give
this companion arm a numeric promotion gate, so this is a Pareto observation,
not a promotion or default change.

### LBA train-only

| Arm | Train MAE (pK) | Median step (s) | Peak CUDA (MB) | Capacity gate |
|---|---:|---:|---:|---:|
| explicit | **0.051845** | **0.15151** | **915.6** | pass |
| hybrid, every layer | 0.198044 | 0.16963 | 1,064.2 | fail |
| hybrid, layer `[0]` | 0.241412 | 0.16680 | 965.0 | fail |

Scheduling did not recover sharp-interaction capacity. It was `1.101x` the
explicit latency and `1.054x` its memory, while missing the `0.10 pK` capacity
threshold. The exact latency ratio was `1.100904x`, so it also exceeded the
registered `<=1.10x` ceiling by `0.000904x`; the memory ceiling passed.
Relative to always-on hybrid it used `0.907x` memory but fit worse. Thus
explicit remains the only admitted LBA-capacity policy.

The raw run receipts predate the corrected schedule metadata and retain their
original fields. Their hashes and the non-destructive semantic correction are
recorded in `SCHEDULED_METADATA_CORRECTION.json`. Future receipts now record
the actual active layer indices and reject periods longer than the model
stack. No LGL result participates in this decision.
