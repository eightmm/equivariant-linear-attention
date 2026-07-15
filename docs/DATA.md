# DATA

No dataset is part of this prototype.

Benchmark smoke data is available through `SyntheticMoleculeDataset`; it is
deterministic and uses an invariant pair-distance target for pipeline tests.
Real QM9 loading is optional through `load_qm9_samples(...)` and requires
`torch-geometric` plus RDKit in the environment.

Expected future input contract:

| Field | Shape | Notes |
|-------|-------|-------|
| `node_feats` | `(N, node_dim)` | invariant atom/residue features |
| `pos` | `(N, 3)` | Cartesian coordinates in one unit convention |
| `edge_feats` | `(N, N, edge_dim)` | optional dense pair context; dense attention only |
| `batch` | `(N,)` | graph id for batched global attention |
| `neighbor_index` | `(N, K)` | optional local-mode absolute neighbor ids for O(NK) attention |
| `neighbor_mask` | `(N, K)` | optional bool mask for padded local neighbors |

Before training on molecular or protein data, define standardization, dedup,
split policy, label semantics, leakage boundary, metric, and baseline here.

## Benchmark Commands

```bash
uv run python scripts/download_dataset.py --dataset qm9 --data-root data/qm9
uv run python scripts/train_compare.py --dataset synthetic --model egnn --steps 10
uv run python scripts/train_compare.py --dataset synthetic --model rich_local --steps 10
```

QM9:

```bash
uv run python scripts/train_compare.py --dataset qm9 --data-root data/qm9 --model egnn --num-samples 1000
uv run python scripts/train_compare.py --dataset qm9 --data-root data/qm9 --model rich_local --num-samples 1000
```
