# QM9 strict CUDA repeatability contract

Frozen before CUDA execution on 2026-07-23.

## Decision

Question: can the current same-source static LGL QM9 screen be reproduced
bitwise across five fresh strict-CUDA processes, so architecture effects near
`0.01 eV` can be interpreted against a trustworthy execution contract?

Prediction: all five runs complete, record strict deterministic controls,
produce exactly equal validation MAE values, and produce one canonical final
state hash. Any metric mismatch, final-state mismatch, identity drift,
nonfinite output, unsupported deterministic CUDA operator, or missing run
falsifies the strict gate.

## Frozen system and data

- Prediction unit: one QM9 molecule with its supplied equilibrium 3D geometry.
- Target: QM9 `gap`, target index 4, reported in eV.
- Source snapshot: local pinned `data/qm9` files whose three expected SHA-256
  values are checked and recorded by the harness.
- Split: seeded random-row warm-start split, split seed 42, with 110,000 train,
  10,000 validation, and 10,000 held-out test rows.
- Test evaluation: disabled. Test membership is hashed but labels/metrics are
  not evaluated.
- Target normalization: fit on train rows only.
- Model: factorized moment attention, static `lgl`, learned global transport,
  width 64, three layers, four heads, no coordinate update, one noninteracting
  memory, default fixed positive kernel.
- Optimization: AdamW, learning rate `3e-4`, weight decay `0.01`, gradient
  clipping at `1.0`, FP32, batch size 64, 500 updates.
- Dataset, split, and model seeds: 42.

## Execution

Run a two-step strict CUDA smoke first using the same dataset/model contract.
If strict deterministic execution is unsupported, retain the error and stop;
do not run a seeded substitute. If it passes, run five independent sequential
processes with:

```text
uv run --locked python scripts/train_compare.py
  --dataset qm9 --data-root data/qm9
  --num-samples 130000 --train-size 110000 --val-size 10000
  --batch-size 64 --steps 500 --hidden-dim 64
  --num-layers 3 --num-heads 4 --routing lgl
  --memory-count 1 --split-seed 42 --model-seed 42
  --determinism strict --device cuda --amp-dtype none
  --skip-test-eval --metrics-out <fresh-run-path>
```

One local GPU is authorized for at most 900 cumulative seconds. Runs are
sequential. No package, dependency, data download, remote compute, or test
evaluation is authorized.

## Gate

The summarizer must validate identical source, QM9 data hashes, split hashes,
 run configuration, initial state, model seed, and effective strict runtime
state. It then evaluates `val_mae` with:

- minimum fresh process count: 5;
- recorded mode: strict;
- exact equality of all selected metric values;
- exactly one final-state hash;
- maximum reported metric range: `0.005 eV`.

Strict equality controls the verdict. The `0.005 eV` value is retained as the
future near-threshold decision scale and cannot turn a strict mismatch into a
pass.

## Boundary

This gate measures repeatability, not model accuracy or generalization.
The random-row warm split is retained only for same-harness continuity and is
not a scaffold/OOD claim. Passing does not establish cross-hardware,
multi-GPU, mixed-precision, or official EGNN reproducibility.
