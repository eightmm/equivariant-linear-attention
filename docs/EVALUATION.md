# Evaluation

The repository evaluates one architecture. There is no model-selector flag.

```bash
uv run python scripts/train_compare.py \
  --dataset synthetic --steps 10 --split-seed 42 --model-seed 42
```

The script reports train loss, validation MAE/RMSE, split hashes, model seed,
parameter count, source hash, optimizer settings, target normalization, and
whether test evaluation occurred. During architecture work use
`--skip-test-eval` and select only on validation.

QM9 numbers use target `gap` in eV and a random-row warm split. They do not
measure scaffold, protein-target, temporal, or cold-complex generalization.
Frozen final evaluation requires a separate preregistered split and multiple
data/model seeds.
