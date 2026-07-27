# ATOM3D-LBA matched-budget multi-seed study (2026-07-27)

## Outcome

The current squared-RBF gated-plus-grouped LGL passed the frozen exploratory
gate against the preceding LGL on the official ATOM3D-LBA ID30 validation
split. Across model/data-order seeds 41--43, its mean best-checkpoint RMSE was
`1.598765 pK`, versus `1.619865 pK` for the incumbent. The paired improvement
was `0.021099 pK`, and the candidate won all three seeds.

This is useful real-data evidence for retaining the gated local transport and
grouped invariant normalization. It is not test evidence or a published-model
superiority result. The matched-35-epoch protocol was chosen after partial
seed-41 curves had been observed while repairing an execution-budget defect,
so its evidence grade is exploratory rather than preregistered confirmation.

## Protocol

- cached `vector-institute/atom3d-lba` revision
  `f93dd2d150a47c270f624620f84e07451a158705`;
- all 3,507 official ID30 train and 466 validation complexes;
- no test-loading or test-evaluation path;
- identical atom features, coordinates, ligand readout, and
  segment-balanced sparse topology for both arms;
- strict deterministic FP32 CUDA on an NVIDIA RTX PRO 6000 Blackwell;
- AdamW at `3e-4`, weight decay `0.01`, batch size 16, clip norm `1.0`;
- five warmup epochs followed by a 35-epoch cosine schedule;
- exactly 35 epochs / 7,700 optimizer updates per arm and seed.

The candidate had 168,815 parameters and the incumbent 161,541
(`1.045x`). Both used the current topology hash
`1eea0af8bca4bd3106457f704677265acb09b322abcd3a1c9b7ed92bed42399c`
with 32,303,244 directed edges over train plus validation.

## Validation accuracy

| seed | candidate RMSE | incumbent RMSE | paired improvement | candidate best epoch | incumbent best epoch |
|---:|---:|---:|---:|---:|---:|
| 41 | 1.578512 | 1.629213 | +0.050701 | 21 | 33 |
| 42 | 1.593418 | 1.598576 | +0.005158 | 20 | 19 |
| 43 | 1.624366 | 1.631805 | +0.007439 | 27 | 27 |
| mean | 1.598765 | 1.619865 | +0.021099 | — | — |

Candidate and incumbent sample standard deviations were `0.023390` and
`0.018482 pK`. The paired-improvement sample standard deviation was
`0.025661 pK`; most of the mean gain came from seed 41, although no seed
regressed.

All frozen decision criteria passed:

- mean paired improvement at least `0.020 pK`: `0.021099`;
- at least two improving seeds: `3/3`;
- worst regression no larger than `0.050 pK`: worst result was a
  `+0.005158 pK` improvement;
- median train-step latency ratio at most `1.25`: `0.93472`;
- median peak-allocation ratio at most `1.50`: `1.37233`.

The candidate mean is numerically close to the separately published
ATOM3D-GNN reference of `1.601 pK`, but that reference is not same-harness and
was not used in the decision.

## Efficiency and optimization diagnosis

| arm | median step | median peak CUDA allocation | mean clip fraction | mean pre-clip norm |
|---|---:|---:|---:|---:|
| candidate | 27.644 ms | 1,728,283,648 bytes | 0.99139 | 12.4345 |
| incumbent | 29.578 ms | 1,258,568,192 bytes | 0.99216 | 14.8994 |

The accepted candidate was `6.53%` faster per step but used `37.23%` more peak
allocation. Its gated edge MLP therefore remains the main memory tradeoff.

The candidate reduced mean pre-clip norms in every shared path: input
`7.87 -> 6.10`, global/shared update `7.47 -> 5.57`, FFN `8.89 -> 7.27`,
and readout `2.34 -> 2.15`. Clipping nevertheless affected about 99% of steps
in both arms. Since one global scalar rescales all gradients before AdamW,
clip frequency alone does not identify a single architectural defect; any
clip-policy change still needs a matched accuracy experiment.

## Provenance and limitations

The completed packet used source hash
`68afeea170b66b9e2f545cf6097df85883732ec4a0872b468a1d89644a33681a`.
Seed 41 ran at commit `8f52c00`; seeds 42--43 ran at `13eca29`, whose only
model-independent change added coordinator resume/provenance handling. All
three model-source hashes match. The combined packet wall time was
`1873.08 s`, including the reused completed seed-41 run.

Earlier equal-budget and `2:1` launches are retained locally but excluded.
They exposed two coordinator defects: unequal arm training opportunity and an
outer timeout with no result-finalization grace. The repaired packet gives
both arms identical updates and reserves finalization time. A historical
topology constant was also corrected after two current-code executions
reproduced the new hash and edge count.

The result does not establish cold-target generalization, pose robustness,
sequence-only folding, docking performance, test performance, or superiority
to an official EGNN/Equiformer/SE(3)-Transformer reproduction. The completed
real-batch operator profile traced the candidate's memory excess to its larger
gated edge-MLP activation path and motivated an opt-in, equation-preserving
activation-checkpoint mode. An independent clean preregistration is still
required before elevating the accuracy result beyond exploratory evidence.

The local summary is
`artifacts/lba-multiseed-confirmation-20260727/id30-3seed-fixed35/summary.json`
with SHA-256
`954d63145899be1b8c4259d8d9391dcfe7f51815079c36deddac6dcf471c90bf`.
