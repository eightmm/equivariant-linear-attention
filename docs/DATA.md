# Data Contract

> **Historical:** this is the retired benchmark data contract. The current
> dependency-free boundary uses one `ELAGraph` input/output type and
> `ELAGraph.collate` for mini-batches. See [the current data API](DATA_API.md).

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

## Precomputed candidate topology

`segment_balanced_knn_edge_index` retains `i <- j` exactly when the float64
squared displacement is below the squared cutoff, always keeps self edges, keeps
the `k` smallest squared distances per relation, and retains every exact tie at
the boundary, so a receiver degree may exceed its budget. The float64 promotion
is mandatory whatever the storage dtype: a matrix-multiplication Euclidean
distance is neither translation invariant nor thread-order reproducible in
float32 and must not decide retention. The promotion makes retention exact and
reproducible *given the stored coordinates*; translation invariance is limited by
the storage dtype itself, exactly to at least a `1e3 Angstrom` float32 offset
against a `61.58 Angstrom` operating range. Candidate order is receiver major, then
self, intra-segment, cross-segment, then ascending sender index.

Historical packets used the joint `topology_sha256`, which hashes sample IDs
and edge bytes together. New packets must record `sample_identity_sha256` and
`edge_topology_sha256` separately (and may retain the joint digest for backward
compatibility). This prevents a label-blind sample-ID migration from looking
like a changed candidate graph. `scripts/verify_lba_topology.py` must report one
edge count and one edge-only hash across fresh processes before a multi-seed
claim. See `TOPOLOGY_CONTRACT_20260727.md` for the historical frozen joint
identity and the effect on older numbers.

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

## ATOM3D-LBA ID30 validation packet

`load_atom3d_lba_split_samples` is the separate held-out evaluation loader. It
accepts only the pinned official `train` or `val` split and structurally rejects
`test`. The 2026-07-24 study used all 3,507 train and 466 validation complexes.
It retained the same pocket/ligand filtering, 140-dimensional node features,
ligand readout mask, and `pK` target as the train-only packet. Target
normalization was fitted on the complete train split only.

The split identity hashes were:

- train: `94d0468cd2c6eb579f5625f9fc74e12c1473c82f44d52186e90bbda17faf3998`
- validation:
  `ed4565afc9e87adb926798dd1909a3987fc849a7f0e1f5e3ba92d52c10e7d99c`

The loader relies on the already-materialized immutable Hugging Face cache in
offline mode. The test Arrow file exists in that cache from an earlier generic
builder preflight, but this evaluation runner neither constructs a test dataset
object nor reads a test row or label.
