# Data Contract

`GraphSample` stores node features, coordinates, a graph target, and a stable
sample ID. `collate_graphs` concatenates nodes and creates contiguous integer
graph IDs; no neighbor or dense pair tensor is generated.

| Tensor | Shape | Meaning |
|---|---:|---|
| `node_feats` | `(N, F)` | O(3)-invariant scalar (`0e`) features |
| `pos` | `(N, 3)` | Cartesian coordinates, stored in float32+ |
| `batch` | `(N,)` | graph ID in `0..G-1` |
| `target` | `(G, T)` | graph regression target |

Synthetic smoke data is deterministic by seed. QM9 loading requires the `qm9`
optional dependency group and target index 4 is documented as HOMO-LUMO gap in
eV. Loaded QM9 sample IDs keep the processed row, PyG's zero-based raw-record
`data.idx`, and the dataset molecule name as distinct fields, for example
`qm9-row-3055-raw-index-3111-name-gdb_3112`. The raw index is deliberately not
called a GDB9 molecule identifier because the external name is one-based.

`GraphBatch.to(dtype=...)` applies `dtype` to model features and targets, not
coordinates. Coordinates remain float32, or float64 when either the requested
feature lane or source coordinates are float64. An explicit `geometry_dtype=`
may be supplied when needed.

```bash
uv run python scripts/train_compare.py --dataset synthetic --steps 10
uv run --extra qm9 python scripts/train_compare.py \
  --dataset qm9 --data-root data/qm9 --qm9-target-index 4
```

The current QM9 split is a seeded random-row warm-start split. See
`QM9_CONTRACT.md` before interpreting results.
