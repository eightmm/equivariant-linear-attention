# Data Contract

`GraphSample` stores node features, coordinates, a graph target, and a stable
sample ID. `collate_graphs` concatenates nodes and creates contiguous integer
graph IDs; no neighbor or dense pair tensor is generated.

| Tensor | Shape | Meaning |
|---|---:|---|
| `node_feats` | `(N, F)` | floating node features |
| `pos` | `(N, 3)` | Cartesian coordinates |
| `batch` | `(N,)` | graph ID in `0..G-1` |
| `target` | `(G, T)` | graph regression target |

Synthetic smoke data is deterministic by seed. QM9 loading requires the `qm9`
optional dependency group and target index 4 is documented as HOMO-LUMO gap in
eV.

```bash
uv run python scripts/train_compare.py --dataset synthetic --steps 10
uv run --extra qm9 python scripts/train_compare.py \
  --dataset qm9 --data-root data/qm9 --qm9-target-index 4
```

The current QM9 split is a seeded random-row warm-start split. See
`QM9_CONTRACT.md` before interpreting results.
