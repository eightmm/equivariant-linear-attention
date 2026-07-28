# Whitened global read: schema-matched follow-up

## Decision

The finite-sample-gated whitened global read is not promoted. The bounded QM9
safety condition passed, but the official ATOM3D-LBA ID30 three-seed accuracy
condition failed. Public defaults remain disabled.

## Why the comparison changed

The original QM9 candidate/whitened comparison produced `0.640651/0.696559 eV`
validation MAE. A later run kept the whitened mixes exactly zero but still
produced `0.720716 eV`. Thus zero-initialized forward equivalence did not imply
an identical floating-point training trajectory: the auxiliary autograd branch
altered gradient accumulation under a regime where roughly 91% of updates were
clipped.

The repaired control uses the identical whitening modules, full initial state,
ridge, rank gate, forward/autograd graph, optimizer schema, data and updates.
Gradient hooks hold only the eight mix parameters at zero. The active arm
allows those gradients to update.

## Architecture repair

The optional reliability is

`rho_g = max(0, n_g - 2F) / n_g`,

where `F` is the actual kernel-feature dimension. If no graph in a batch has
more than `2F` nodes, the auxiliary branch is skipped entirely. This preserves
permutation and O(3) contracts, retains linear node complexity, and makes the
all-small-graph path the incumbent computation rather than an algebraically
equal extra branch.

## Results

The strict-CUDA QM9 safety run used seed 42 and 500 updates. Frozen and active
arms both reached MAE `0.6406513190 eV` and RMSE `0.8343833076 eV`; step latency
ratio was `1.013299`, peak allocation ratio was `1.0`, and test evaluation was
disabled.

The LBA run used the official train/validation split (`3,507/466`), seeds
41--43, 20 epochs and 4,400 updates per arm, batch 16, and one frozen
`32,302,952`-edge topology (`57f40fb1...`).

| seed | frozen last RMSE | active last RMSE | improvement |
| ---: | ---: | ---: | ---: |
| 41 | 1.599205 | 1.586143 | +0.013061 |
| 42 | 1.602016 | 1.614437 | -0.012421 |
| 43 | 1.611601 | 1.615970 | -0.004369 |

Mean paired last-epoch improvement was `-0.001243 pK` (sample SD
`0.013026 pK`), only one seed improved, and mean best-checkpoint effect was
`-0.000892 pK`. The preregistered requirements were at least `+0.020 pK` mean
improvement and at least two wins. Resource, identity, finiteness, completion,
worst-regression and no-test boundaries all passed.

The vector mix was largest on seed 42, which regressed most. Mix magnitude
therefore proves activity but does not identify a beneficial route.

## Boundary and next action

This is same-validation-harness architecture evidence, not a cold-target,
pose-robustness, test-set, published-model, or affinity-SOTA claim. Since the
primary LBA gate failed, the conditional clipping interaction and
scalar/vector-only ablations were not executed. Retain the rank gate as an
opt-in safety mechanism, keep all whitening defaults off, and move the next
architecture effort away from this global-read metric.
