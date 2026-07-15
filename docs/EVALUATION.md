# EVALUATION

Evaluation protocol. Locked — changes invalidate prior numbers.

## Metrics

| Metric | Definition | Direction | Primary? |
|--------|------------|-----------|----------|
| MAE | mean absolute error of graph scalar target | down | yes |
| RMSE | root mean squared error of graph scalar target | down | no |

- Implementation: `src/equivariant_attention/training.py`
- Target normalization: fit mean/std on train split only; metrics are reported after inverse transform.
- Locked commit SHA:

## Test Split

- Path:
- Size:
- Frozen since (date / commit):
- DO NOT touch during model selection.

## Baseline

| Model | Metric | Value | Commit | Run ID |
|-------|--------|-------|--------|--------|
| EGNNBaseline | MAE/RMSE | TBD | TBD | TBD |
| RichEquivariantAttention(local) | MAE/RMSE | TBD | TBD | TBD |

## Eval Command

```bash
uv run python scripts/train_compare.py --dataset synthetic --model egnn --steps 10
uv run python scripts/train_compare.py --dataset synthetic --model rich_local --steps 10
```

## Reporting

- Mean ± std over N seeds:
- CI (bootstrap): yes/no
- Per-class / per-subgroup breakdown: yes/no

## Regression Policy

- Drop > X% on primary metric vs baseline -> block merge.
- New baseline requires PR review + `EXPERIMENTS.md` entry.

## Update Triggers

Metric definition, test split, or eval code change -> bump eval version + re-baseline.
