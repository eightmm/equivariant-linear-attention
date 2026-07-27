# Distance-RBF five-seed confirmation

## Outcome

The distance-spaced local radial basis is not promoted. Its one-seed,
500-update improvement did not survive the preregistered five-seed,
2,000-update confirmation.

All runs used cached QM9 `gap`, the same random-row split seed 42, model seeds
41--45, strict deterministic FP32 CUDA, batch size 64, three layers, identical
raw features and coordinates, and no test evaluation.

| arm | validation MAE, mean +/- sample SD | median step | median peak CUDA |
|---|---:|---:|---:|
| current LGL, squared RBF | **0.371793 +/- 0.020792 eV** | 29.787 ms | 190.6 MB |
| current LGL, distance RBF | 0.383284 +/- 0.019450 eV | 29.322 ms | 190.6 MB |
| private EGNN, complete graph | 0.418467 +/- 0.065334 eV | 5.384 ms | 269.3 MB |
| private EGNN, matched 2.5-A candidates | 0.417983 +/- 0.011102 eV | 5.360 ms | 133.5 MB |

Distance spacing changed validation MAE relative to squared spacing by
`-0.036583`, `-0.018727`, `-0.002922`, `-0.023929`, and `+0.024707 eV`
for seeds 41--45, where positive means better. The mean improvement was
`-0.011491 eV`; only one of five seeds improved and the worst regression was
`0.036583 eV`. The frozen gate required mean improvement at least `0.010 eV`,
at least three improving seeds, worst regression no larger than `0.020 eV`,
and latency and memory ratios no larger than `1.20`. Only the two resource
criteria passed.

The result identifies the earlier `+0.029220 eV` seed-42/500-update screen as
an early-training false positive. Distance spacing remains an opt-in,
state-schema-compatible capability because it is mathematically valid and may
be useful at other physical cutoffs, but it is not a QM9 default and does not
authorize the conditional LBA run.

## Revised EGNN comparison

The current squared-RBF LGL itself is stronger than the historical attention
numbers that motivated the EGNN chase. Against the internal controls it had:

- `0.046674 eV` lower mean MAE than complete-graph EGNN, winning three of five
  paired seeds;
- `0.046191 eV` lower mean MAE than topology-matched EGNN, winning all five
  paired seeds.

This is a same-feature, same-training-harness validation comparison. The EGNN
is private and static-coordinate, not an official reproduction, so these
numbers do not establish published-model superiority. They do establish that
the current local-global-local architecture no longer has the previously
reported same-harness accuracy deficit.

The systems result goes the other way on these small molecular batches. LGL
was `5.53x` slower than complete EGNN and `5.56x` slower than matched EGNN. It
used less peak allocation than complete EGNN (`0.708x`) but more than the
topology-matched control (`1.427x`). This agrees with the existing scaling
evidence: exact factorized attention wins only once the graph is sufficiently
large or dense for EGNN's edge-linear work to dominate its larger constant
factor.

## Interpretation and next stage

The useful architectural gains came from the selective local path plus the
near-uniform global graph-mean path. Increasing global selectivity, polynomial
order, angular state, cutoff, or changing the radial coordinate has not
survived the registered gates. Those rejected paths should not be compounded.

The next scientific priority is therefore:

1. confirm the already positive full ATOM3D-LBA ID30 result across model seeds,
   using the squared-RBF gated-plus-grouped model and keeping test structurally
   unavailable;
2. profile and reduce small-graph execution constants without changing the
   accepted equations;
3. compare with an official external equivariant baseline before making a
   family-level superiority claim.

The full local evidence is under
`artifacts/distance-rbf-confirmation-20260727/qm9-2k/summary.json`
(SHA-256
`82dcafb2b623b0d22d193385115d52cbc0efdb474fbee9576027c45eb15b3d42`).
The packet completed 20/20 registered executions in `1111.41 s`, evaluated
validation only, and never evaluated the test split.
