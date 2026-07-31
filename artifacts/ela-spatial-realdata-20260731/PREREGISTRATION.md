# ELA spatial-operator real-data screen

Date: 2026-07-31

## Question

When the exact global equivariant linear-attention backbone, parameter schema,
initialization, optimizer, input features, split, and readout are held fixed,
what is contributed by:

1. an explicit sparse short-range operator;
2. an edge-free smooth implicit spatial residual; and
3. their hybrid?

The architecture is a generic point-cloud / 3D-graph model. QM9 and ATOM3D LBA
are validation tasks, not architecture-specific definitions.

## Frozen arms

| Arm | Exact global ELA | Sparse local | Implicit residual |
|---|---:|---:|---:|
| `explicit` | yes | yes | no |
| `implicit` | yes | no | yes |
| `hybrid` | yes | yes | yes |

All three instantiate the same state dict. Input multipoles and tensor closure
use the same edge-free context; sparse geometry is supplied only to the local
message in the explicit and hybrid arms. The implicit kernel uses Gaussian
Taylor moments at `(R, 2R, 4R)`, order 2, with a zero-initialized learnable
residual. Coordinates are frozen.

## QM9 screen

- target: gap, index 4, eV
- data: 130,000 processed rows
- split: seeded random-row warm split, 110,000 train / 10,000 validation /
  10,000 unopened test
- split seed/model seed: 42/42
- updates: 500
- batch size: 64
- hidden/layers/heads: 64/3/4
- optimizer: AdamW, lr `3e-4`, weight decay `1e-2`, clip `1.0`
- precision/determinism: float32, strict CUDA determinism

Primary metric is validation MAE. Secondary metrics are train-step latency,
peak CUDA allocation, clipping fraction, and finite-gradient coverage.

Prediction: hybrid improves validation MAE by at least `0.010 eV` over explicit
without exceeding `1.35x` latency or memory. Implicit is exploratory: because
it is a smooth global kernel rather than a compact-cutoff replacement, it must
not be promoted merely for matching the explicit lane on this small-molecule
task.

## LBA train-only capacity probe

- data: frozen first 16 ATOM3D-LBA train complexes at revision
  `f93dd2d150a47c270f624620f84e07451a158705`
- labels: train only; validation and test closed
- readout: ligand-mask mean
- steps: 1,000
- batch size: 2
- seed/order seed: 20260723/20260723
- optimizer: AdamW, lr `1e-3`, weight decay `0`, clip `1.0`
- precision/determinism: float32, strict CUDA determinism

Primary metric is final train MAE in pK. The probe asks only whether the
architecture is wired and can fit the small subset. A train MAE at or below
`0.10 pK` passes the registered capacity threshold. It is not a generalization
or affinity-ranking claim.

## Stop and interpretation rules

- Any NaN/Inf, data/split drift, state-schema mismatch, test evaluation, or
  nonzero nonfinite-gradient count invalidates the affected arm.
- A one-seed QM9 result is a screen, never a default-changing confirmation.
- LBA train-only overfit cannot promote downstream superiority.
- A failed prediction remains recorded; no post-hoc threshold changes.
