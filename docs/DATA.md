# Data Contract

`GraphSample` stores node features, coordinates, a graph target, a stable
sample ID, and optional local candidates/readout mask. `collate_graphs`
concatenates nodes and creates contiguous integer graph IDs; no neighbor or
dense pair tensor is generated unless samples already carry candidates.

| Tensor | Shape | Meaning |
|---|---:|---|
| `node_feats` | `(N, F)` | O(3)-invariant scalar (`0e`) features |
| `pos` | `(N, 3)` | Cartesian coordinates, stored in float32+ |
| `batch` | `(N,)` | graph ID in `0..G-1` |
| `target` | `(G, T)` | graph regression target |
| `readout_mask` | `(N,)` | optional boolean graph-pooling selection |

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

## ATOM3D-LBA train-only overfit packet

The optional `pdbbind` dependency group loads the public
`vector-institute/atom3d-lba` Parquet conversion at immutable revision
`f93dd2d150a47c270f624620f84e07451a158705`. The registered loader accepts only
the `train` split and explicit unique row indices. The first packet freezes
rows `0..15`; validation and test rows are never selected, indexed, or
evaluated.

One failed preflight call through the generic `datasets` builder materialized
local Arrow caches for train, validation, and test even though `split=train`
was requested. No validation/test row or label was indexed, printed, selected,
or used. The production loader was then narrowed to the two immutable train
Parquet URLs. Thus the holdout rows were not consumed, but the local cache is
explicitly not claimed to be physically pristine or train-only.

Each row contains a full-protein copy (`token_type_id=0`), pocket copy
(`1`), and ligand (`2`). The loader discards the duplicated full-protein copy,
keeps pocket plus ligand coordinates, and uses the observed `input_ids` values
as opaque categorical atom tokens rather than claiming they are atomic
numbers. Node input is a 138-way token one-hot plus two-way pocket/ligand
identity (`F=140`). All retained nodes participate in transport, while
`readout_mask` selects ligand nodes for the graph prediction. Labels are the
supplied affinity `pK` values and normalization is fitted only on the selected
16 train rows.

The downloaded data stays under ignored `data/` and is not redistributed.
The upstream license and provenance must be reviewed before reuse outside the
confirmed non-commercial research packet.
