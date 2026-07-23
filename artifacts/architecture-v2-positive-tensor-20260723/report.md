# Architecture v2: bounded content and shifted persistent-`2e` attention

## Outcome

The implementation is mathematically valid and remains available as an
experimental opt-in, but it is **not promoted for performance**. The repaired
real-data study admitted no QM9 candidate, and no ATOM3D-LBA arm reached the
registered train-overfit threshold. Defaults stay unchanged.

## Implemented architecture

For positive scalar content `z = ELU(x) + 1`, the new bounded mode keeps the
incumbent direction `u(z)` but restores bounded magnitude:

```text
r = ||z||_2 / sqrt(D)
phi_bounded(z) = u(z) * 2r / (1 + r),     ||phi_bounded||_2 < 2.
```

For persistent symmetric-traceless `2e` state, separate query/key channel
mixes produce tensors in the open Frobenius unit ball. Per head:

```text
K2(i,j,h) = eta_h * (1 + <Q2(i,h), K2(j,h)>_F) >= 0.
```

Appending `sqrt(eta_h) * [1, vec(Q2)]` and the analogous key feature makes this
an ordinary feature dot product. The existing graph summaries therefore
evaluate it exactly with fixed-width `O(N)` work and storage; no `N x N`
attention tensor is introduced. Frobenius contraction under
`T -> R T R^T` preserves the `O(3)` contract, including reflections.

Both features are opt-in. The default scalar map remains `unit`, and the
tensor kernel allocates no query/key modules when disabled.

The final repository gate passed 481 tests at 88.69% coverage, together with
static checks, compilation, and the CPU float64 ML smoke. A strict deterministic
CUDA smoke of the combined path completed two updates with zero nonfinite
gradient parameters.

## Numerical integrity repair

Two implementation issues were found before final interpretation:

1. The initial unit-ball map evaluated `sqrt(||T||_F^2)` at zero. Although the
   forward value was finite, its derivative was undefined and produced a NaN
   on the second float32 update. The implementation now uses the algebraically
   identical direct denominator `sqrt(1 + ||T||_F^2 / 5)`.
2. Constructing optional tensor query/key modules consumed the global CPU RNG
   before later incumbent modules. Original v2 tensor arms therefore did not
   share common initial weights with the control. The modules are now created
   inside `torch.random.fork_rng(devices=[])`, which restores the CPU RNG
   stream. Exact common-state pairing is tested across all three layers.

The original v2 outputs remain preserved as negative results for their exact
source hashes. They are not pooled with v2.1 and do not support a causal
tensor-kernel claim.

## QM9 validation screen

The repaired v2.1 screen used strict deterministic FP32 CUDA, seed 42,
500 updates, QM9 `gap`, train/validation sizes 110,000/10,000, train-only
target normalization, and no test evaluation.

| arm | validation MAE (eV) | delta vs incumbent | parameters | train time | peak CUDA |
|---|---:|---:|---:|---:|---:|
| incumbent | 0.709287 | 0 | 153,285 | 16.24 s | 171.6 MiB |
| bounded | 0.722743 | +0.013456 | 153,285 | 16.69 s | 177.8 MiB |
| tensor package | 0.811461 | +0.102174 | 158,967 | 22.18 s | 200.4 MiB |
| combined | 0.716811 | +0.007524 | 158,967 | 22.97 s | 206.5 MiB |

The frozen admission rule required at least `0.010 eV` improvement and no more
than `0.020 eV` regression. No arm passed. The combined arm is the closest,
but it is still worse than the incumbent. Consequently the five-seed
confirmation and conditional private-EGNN comparison were not run.

The incumbent and bounded outputs reproduce the original v2 values exactly.
Original-to-v2.1 tensor results change materially, confirming that the old
component interpretation was unsafe. This four-arm screen evaluates complete
packages; it does not contain a fresh v2.1 persistent-`2e`-without-kernel
control, so it does not isolate the tensor term.

## ATOM3D-LBA train-only capacity

The repaired comparison used cached ATOM3D-LBA revision
`f93dd2d150a47c270f624620f84e07451a158705`, train rows 0--15, ligand-only
readout, deterministic batches of two, 3,000 updates, and no validation or
test access. This is a capacity check, not a generalization benchmark.

| arm | parameters | best observed train MAE | final MAE / RMSE | median train step | peak CUDA |
|---|---:|---:|---:|---:|---:|
| incumbent | 167,115 | 0.143626 pK @ 2,650 | 0.201581 / 0.302229 | 36.99 ms | 282.9 MiB |
| combined candidate | 167,223 | 0.184002 pK @ 2,850 | 0.420520 / 0.494234 | 42.24 ms | 336.3 MiB |
| private static EGNN | 167,260 | 0.191400 pK @ 2,900 | 0.249922 / 0.396028 | 5.31 ms | 743.0 MiB |

All arms failed the frozen `train MAE <= 0.10 pK` capacity gate. The candidate
used about 45% of EGNN's peak CUDA allocation in this exact run, but its
measured train step was about 8 times slower. It was also slower, larger in
memory, and less accurate than the incumbent. These are single-run,
train-only, harness-bound observations.

## Decision and limitations

- C1--C3 (compatibility, kernel algebra, and tested symmetry) are supported.
- C4 and C6 are rejected/unsupported by the saved real-data measurements.
- C5 is unavailable because no candidate was admitted to confirmation.
- `bounded` and the shifted tensor kernel remain experimental and disabled by
  default.
- There is no claim of QM9 improvement, EGNN competitiveness, PDBBind
  generalization, docking quality, force consistency, chirality sensitivity,
  softmax equivalence, or universal finite-degree expressivity.
- The private EGNN is a same-harness internal control, not an official-paper
  reproduction.

The next architecture effort should not add more polynomial degree blindly.
The evidence instead points toward a separately registered redesign that
improves local pairwise chemical selectivity while retaining the factorized
global channel for large/dense systems.
