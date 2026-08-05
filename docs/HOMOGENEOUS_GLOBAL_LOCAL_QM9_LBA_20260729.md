# Homogeneous Global + Local Validation (2026-07-29)

> **Historical:** the reproduction commands below invoke retired experiment
> runners that are not shipped by the current architecture-only package. See
> `docs/EXPERIMENTS.md` for why and how to reproduce from the recorded Git
> revision.

## Architecture under test

The primary candidate in this packet is **not** LGL.  It is the homogeneous
update

$$
H_{t+1} =
\text{FFN}_t\!\left(
H_t + G_t(H_t, X)
+ \mathbf{1}[t \in \{0,2\}]S_t(H_t, X, E)
\right),
$$

where every one of the three blocks retains all four exact factorized global
heads.  `S_t` is an additive rank-4 sparse local residual at blocks 0 and 2;
it does not replace or route away any global head.

The frozen attribution pair was:

- `global_only`: `local_head_counts=(0,0,0)`, exact `feature_gemm` global
  reduction in every block, and no sparse residual.
- `global_local`: the identical global stack plus the rank-4 materialized
  sparse residual at blocks 0 and 2.

`lgl` (`local_head_counts=(4,0,4)`) and the private static EGNN are historical
controls only.  Neither defines the candidate architecture.

## Preregistered decisions

- QM9 admission required `global_local` to improve validation MAE by at least
  `0.010 eV` over `global_only` and to regress by no more than `0.020 eV`
  against historical LGL at 500 updates.
- LBA admission required `global_local` to improve best validation RMSE by at
  least `0.020 pK` over `global_only` and to beat historical LGL after exactly
  20 epochs / 4,400 updates per arm.
- Test labels remained closed.  A failed admission gate stops multi-seed
  confirmation and leaves the capability opt-in.

## QM9 result

All rows use the cached 130k QM9 data, target index 4 (`gap`), the fixed
110k/10k warm random-row train/validation split, strict CUDA FP32, seed 42,
batch size 64, 2.5-Angstrom precomputed candidates, and 500 updates.

| Arm | Parameters | Validation MAE (eV) | Median step (s) | Peak allocated (bytes) |
|---|---:|---:|---:|---:|
| `global_only` | 153,285 | 0.795491 | 0.033941 | 126,229,504 |
| `global_local` | 167,327 | 0.794915 | 0.047115 | 150,985,216 |
| historical `lgl` | 160,559 | 0.718769 | 0.029438 | 186,313,216 |
| private static EGNN | 162,154 | 0.718706 | 0.005347 | 133,837,312 |

The additive local residual improved its direct all-global control by only
`0.000576 eV`, below the registered `0.010 eV` gate, while increasing median
step time by `1.388x`.  It trailed historical LGL by `0.076146 eV`.
Therefore this candidate was rejected without a 2,000-update confirmation.

An earlier LGL-derived persistent-`2e` / transient-`l=3` screen was also not a
promotion of the project candidate.  Its first 2,000-update confirmation seed
regressed LGL by `0.024087 eV`, crossing its registered `0.020 eV` maximum, so
that branch was rejected too.

## ATOM3D-LBA result and provenance boundary

The bounded run used the cached official ID30 train/validation rows
(`3,507/466`), bound-complex 140D atom features, ligand pooling, strict CUDA
FP32, seed/order 44, batch size 16, and exactly 20 epochs / 4,400 updates per
arm.

| Arm | Best validation RMSE (pK) | Median step (s) | Peak allocated (bytes) |
|---|---:|---:|---:|
| `global_only` | 1.676701 | 0.044682 | 474,735,616 |
| `global_local` | 1.675044 | 0.057120 | 1,044,828,672 |
| historical `lgl` | 1.604758 | 0.028691 | 1,750,139,904 |

The local residual improved its direct control by `0.001657 pK`, below the
`0.020 pK` gate, and regressed historical LGL by `0.070286 pK`.  This rejects
the architecture hypothesis within the shared run.

The run is not admissible as a frozen cross-packet LBA result.  It produced
exactly `32,302,952` edges but topology digest
`d8d4f04a1241ccd6edbc515000395bc386c36e887fcb55da028d4bafa0e0d829`
instead of the frozen
`57f40fb157e6416558db5507d95c3a5e4f828881e0bc92e142e1b85de802dc6c`.
Git history isolates this to commit `b3307b5`, which made sample IDs
label-blind; the edge builder and edge count did not change.  The runner now
fails closed before full training until that digest-schema migration is
explicitly resolved.

## Exact commands

The corrected QM9 attribution pair was run with:

```bash
uv run --locked python scripts/train_compare.py --benchmark-model factorized_moment --architecture-arm global_only --dataset qm9 --data-root data/qm9 --qm9-target-index 4 --num-samples 130000 --train-size 110000 --val-size 10000 --batch-size 64 --steps 500 --hidden-dim 64 --num-layers 3 --num-heads 4 --local-cutoff 2.5 --no-key-balancing --precompute-local-edges --lr 0.0003 --weight-decay 0.01 --grad-clip 1.0 --seed 42 --split-seed 42 --model-seed 42 --determinism strict --device cuda --amp-dtype none --metrics-out artifacts/matched-vnext-qm9-lba-20260729/qm9-screen/global-only-seed42.json
uv run --locked python scripts/train_compare.py --benchmark-model factorized_moment --architecture-arm global_local --dataset qm9 --data-root data/qm9 --qm9-target-index 4 --num-samples 130000 --train-size 110000 --val-size 10000 --batch-size 64 --steps 500 --hidden-dim 64 --num-layers 3 --num-heads 4 --local-cutoff 2.5 --no-key-balancing --precompute-local-edges --lr 0.0003 --weight-decay 0.01 --grad-clip 1.0 --seed 42 --split-seed 42 --model-seed 42 --determinism strict --device cuda --amp-dtype none --metrics-out artifacts/matched-vnext-qm9-lba-20260729/qm9-screen/global-local-seed42.json
```

The LBA screen was run with:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 uv run --locked python -u scripts/train_lba_id30.py artifacts/matched-vnext-qm9-lba-20260729/lba/global-local-screen-seed44 --data-root data/atom3d_lba --device cuda --arms vnext_global_local vnext_global candidate --batch-size 16 --max-epochs 20 --min-epochs 20 --patience 20 --warmup-epochs 5 --learning-rate 0.0003 --weight-decay 0.01 --grad-clip 1.0 --min-lr-ratio 0.05 --amp-dtype none --model-seed 44 --order-seed 44 --budget-seconds 1800
```

## Raw receipt integrity

The raw outputs remain local ignored artifacts.  These hashes make the tracked
summary auditable without pretending the large artifact directory is versioned.

| Local receipt | SHA-256 |
|---|---|
| `qm9-screen/global-only-seed42.json` | `4e5a78ee39f71e316b8c23f6a635c3762532912d76b6be6e06fec677de088441` |
| `qm9-screen/global-local-seed42.json` | `5c4da982447ef8a0c9f730a78039dd19cd5bd0c9bd2f4664945bbbc4592be20c` |
| `qm9-screen/lgl-seed42.json` | `3f22af9d05abb287d748ebc18fb682e419ceb6d417186c1da0c49c2f91b89488` |
| `qm9-screen/egnn-seed42.json` | `4416ec2cb863c0c2f447076c2750efc6f5369f655543a159499d74b2a9ee5e30` |
| `qm9-screen/lgl-2e-l3-seed41-2000.json` | `a060c25c4a01911fa56bead2152f59ba15f9f1cbda537e106e20832848d2f0d5` |
| `lba/global-local-screen-seed44/result.json` | `82f1c4e98fe57e98b810f8c44d6db890000a6a328d0d564dcac4bdf10849bbe0` |
| `lba/topology-fail-closed-probe/result.json` | `598455cc6bc799ad696b11603d312b1655484107e0c3f6cad8787fbc572282a0` |

## Claim boundary

These are same-harness rejection screens, not SOTA claims.  QM9 uses a warm
random-row validation split and a private EGNN control.  LBA validation has
been accessed before and uses bound structures; it does not establish
cold-target, apo, sequence-only, or pose-robust performance.  No test labels
were evaluated.
