# ATOM3D-LBA gradient-clipping diagnosis (2026-07-27)

## Conclusion

The candidate's approximately 99% clip frequency is not just a harmless
counter: removing global clipping improved fixed-budget validation RMSE in the
frozen seed-44 screen. The last-epoch RMSE changed from `1.628645` with clip 1
to `1.600802 pK` without clipping, a `0.027843 pK` improvement. This passed
the preregistered `0.020 pK` one-seed threshold.

The result does not yet change the public default. It is one-seed,
validation-only evidence and the run exposed a separate cross-run topology
identity issue that must be repaired before confirmation.

## Frozen protocol

- cached `vector-institute/atom3d-lba` revision
  `f93dd2d150a47c270f624620f84e07451a158705`;
- all 3,507 official ID30 train and 466 validation complexes;
- current squared-RBF gated-plus-grouped LGL candidate;
- policies: global L2 clip 1, global L2 clip 10, and no clipping;
- model/data-order seed 44, unused in the preceding LBA three-seed packet;
- strict deterministic FP32 CUDA, AdamW `3e-4`, weight decay `0.01`;
- batch 16, five warmup epochs, cosine decay;
- exactly 20 epochs / 4,400 updates per policy;
- primary metric: last-epoch validation RMSE;
- no test-loading or test-evaluation path.

The conservative pre-outcome prediction was that relaxation would change the
gradient scale but fail to improve RMSE by `0.020 pK`.

## Results

| policy | last RMSE | best RMSE (epoch) | last MAE | clip fraction | effective scale |
|---|---:|---:|---:|---:|---:|
| clip 1 | 1.628645 | 1.617674 (15) | 1.316065 | 0.9855 | 0.1719 |
| clip 10 | 1.611120 | **1.593766 (11)** | 1.296232 | 0.3818 | 0.8644 |
| none | **1.600802** | 1.598508 (15) | **1.286896** | 0 | 1.0000 |

No clipping beat clip 1 in 15 of 20 validation epochs and in every one of the
final six epochs. Best RMSE improved by `0.019166 pK`. Clip 10 showed the same
direction and achieved the best single checkpoint, but its `0.017524 pK`
last-epoch gain missed the frozen primary threshold.

Resource criteria passed. Unclipped/clip-1 ratios were `1.01014x` median
synchronized step latency and `0.99876x` peak CUDA allocation. Every loss and
recorded scalar was finite, all initial-state hashes were identical, and all
arms completed 4,400 updates.

## What was actually clipping?

For clip 1, the pre-clip norm averaged `11.2714` with standard deviation
`8.7129` and maximum `71.8154`. Norms exceeded 10 in `44.39%` and 20 in
`15.41%` of updates. The applied scale averaged only `0.1719`.

Mean squared-gradient-norm shares were:

| path | share |
|---|---:|
| FFN | 36.25% |
| global/shared update | 24.64% |
| input | 19.26% |
| readout | 15.81% |
| gated local | 4.04% |

This falsifies the earlier idea that the gated local edge branch is primarily
responsible. The threshold jointly rescales several shared paths. Relaxation
also changed the learned trajectory: mean raw norm fell to `9.8463` at clip 10
and `8.7908` without clipping.

## Topology identity finding

All policies used one in-memory edge list, so the clipping comparison is
internally controlled. That list contained 32,303,245 directed edges with hash
`344158d83490...`. The immediately preceding LBA three-seed packet recorded
32,303,244 edges and hash `1eea0af8bca4...`, despite matching sample identity
and topology code. A new seed-41 build returned the current `344158...` hash,
excluding model seed as the cause; the historical 2026-07-24 run also had that
hash.

The likely weak point is the CPU `torch.cdist` plus tied kth-boundary
construction. An explicit squared-distance probe returned 32,302,953 edges and
hash `3ee52f...`, showing that distance/tie semantics affect more than the
single drifting edge. It is therefore a candidate data-contract repair, not a
substitution that can be made after seeing this outcome.

## Decision and next experiment

The one-seed screen admits `grad_clip=None` for confirmation but does not
authorize a default change. `train_lba_id30.py` now exposes
`--grad-clip none`, and the default remains `1.0`.

Before the confirmation:

1. freeze a deterministic squared-distance and tie policy that preserves node
   permutation equivariance;
2. verify its edge hash across fresh processes;
3. preregister new paired seeds for clip 1 versus no clipping;
4. require mean improvement, seed consistency, a bounded worst seed, finite
   training, and unchanged resource ceilings.

Primary output:
`artifacts/lba-gradient-clipping-20260727/id30-seed44-20epoch/summary.json`,
SHA-256
`544eec6733b3fd9f3908c21e284875ae81b67f605559c685e0fb7d68e287f4de`.
