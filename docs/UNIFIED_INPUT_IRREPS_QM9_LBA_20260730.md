# Flattened input irreps: QM9 and LBA packet (2026-07-30)

## Outcome

The canonical generic-3D model now accepts one flattened `l<=2` input carrier
containing arbitrary multiplicities of `0e`, `0o`, `1e`, `1o`, `2e`, and
`2o`. Positions remain a separate geometry input. The mechanics packet passes,
and the same scalar raw features used by the controls execute on both QM9 and
ATOM3D-LBA.

The bounded real-data results are mixed:

- QM9 gap improved materially at 500 updates, from LGL validation MAE
  `0.718769` to unified `0.664455 eV`.
- On 16 LBA train complexes, unified reduced train MAE from `0.773502` to
  `0.203864 pK` at 1,000 updates.
- The new executor was substantially more expensive: `5.55x` QM9 and `4.25x`
  LBA median step latency; `1.35x` and `3.27x` peak CUDA allocation.

This admits the flattened-irrep interface and its training wiring. It does not
admit an efficiency claim or establish multi-seed accuracy superiority.

## Implemented contract

`Unified3DConfig` now declares:

```python
Unified3DConfig(
    input_irreps="32x0e + 2x1o + 1x2e",
    output_irreps="1x0e",
)
```

The model accepts `model(node_irreps, positions, prepared_graph)`.
`pack_irreps` and `split_irreps` convert between flattened and per-sector
views. `matrix_to_st5` and `st5_to_matrix` define the compact symmetric
traceless basis `[xx, yy, xy, xz, yz]`, with `zz=-xx-yy`.

Every external sector is projected only into the matching hidden sector.
Only `0e` has a bias. External features are summed with geometry-derived
multipoles in the same parity sector. `input_irreps="0"` is a supported
geometry-only path; persistent `l>2` is rejected.

`UnifiedRegressionModel` connects scalar-only datasets as `C x 0e`, preserves
their sparse topology, and supports graph-mean or supplied node-mask pooling.

## Mechanics evidence

- all six input sectors receive finite, nonzero gradients;
- proper/improper O(3), translation, node/edge ordering, and graph isolation
  contracts remain covered;
- coordinate double backward passes;
- CUDA BF16 forward/backward passes;
- focused new tests: `7 passed`;
- project GPU gate: BF16 and FP32 smoke pass;
- first full fast gate covered `1218 passed, 1 skipped` and found four
  compatibility failures caused solely by renaming the
  historical `_with_egnn_radius_edges` helper and were corrected with a
  compatibility alias. The focused affected suite then passed `11/11`.
- final full fast gate: `1222 passed, 1 skipped`, coverage `85.98%`, CPU
  float64 smoke passed.

## QM9 screen

Both arms used target index 4 (gap, eV), raw 11D atom features, the same
seed-42 random-row split (`110,000/10,000/10,000`), precomputed 2.5 Å radius
candidates, AdamW, strict determinism, batch size 64, and 500 updates. Test
labels were not evaluated.

| metric | LGL control | unified |
|---|---:|---:|
| validation MAE (eV) | 0.718769 | 0.664455 |
| validation RMSE (eV) | 0.883672 | 0.821382 |
| parameters | 160,559 | 209,229 |
| median step (ms) | 29.674 | 164.695 |
| peak CUDA allocation (MB) | 186.3 | 250.7 |
| clipped-step fraction | 0.914 | 0.998 |

The MAE difference is `-0.054314 eV` in favor of unified. This is a planned
one-seed screen, not confirmation. The arms use identical input information
and split, but are not parameter matched: unified has `1.303x` parameters.

## LBA train-only capacity probe

Both arms used the same 16 cached ATOM3D-LBA ID30 train rows, the same 140D
scalar node features, targets, cyclic batch order, AdamW, strict model seed,
batch size 2, and ligand-mask mean readout. No validation or test row was
evaluated. The incumbent attention arm is the historical edge-free GGG overfit
control; unified uses 6 Å sparse candidates because its canonical local path
requires a prepared graph.

The dataset revision, row order, node counts, and ligand counts match the
frozen packet. Current sample IDs use the documented label-blind digest, so
the receipt records
`label_blind_ids_with_frozen_node_and_ligand_counts`. This does not repair the
separate official LBA validation topology-hash contract.

| metric | existing attention | unified |
|---|---:|---:|
| updates | 1,000 | 1,000 |
| train MAE (pK) | 0.773502 | 0.203864 |
| train RMSE (pK) | 1.174286 | 0.231072 |
| parameters | 167,115 | 217,485 |
| median step (ms) | 25.734 | 109.378 |
| peak CUDA allocation (MB) | 276.4 | 902.7 |

Neither arm reached the preregistered `0.10 pK` threshold by 1,000 updates.
Unified nevertheless shows substantially higher bounded train capacity. The
comparison is not parameter matched and does not measure generalization.

## Why the current executor costs more

The accuracy change and the efficiency regression have the same architectural
source. Unified persists six parity/degree sectors, transports vector and
rank-2 tensor values in every block, and runs both exact global and sparse
local work at every depth. The LGL control persists a much smaller
`0e + 1o` carrier and alternates local/global routes. The regression adapter
also repacks receiver CSR for each dynamic training batch in Python. On LBA,
saved activations scale with the five-coordinate `2e/2o` carriers and the
large sparse candidate list.

The next optimization target is therefore executor specialization, not removal
of the new input API:

1. prepack/carry `Prepared3DGraph` in the collated batch instead of rebuilding
   CSR inside every forward;
2. avoid materializing inactive parity sectors when the declared input/output
   and task cannot reach them;
3. checkpoint or fuse tensor local/global intermediates;
4. parameter-match and repeat the QM9 comparison across seeds before any
   architecture promotion.

## Raw receipts

Local ignored artifact directory:
`artifacts/unified-input-irreps-qm9-lba-20260730/`

- `qm9-lgl-seed42-500.json`:
  `e5bdd0e743b6994e2f301fbdb98f72695b971a87b754c4c2fa9dac50486d24ae`
- `qm9-unified-seed42-500.json`:
  `847adf9fb6542b7abe472cb7d8c9838af750874a42a7d399cbb842af2d328a39`
- `lba-train-only-overfit.json`:
  `024fa7484bb304c67a3f6e2e564cb9f64acf21b36b4304f9a3917487887e2201`

Reproducibility level: `rerunnable`, not independently reproduced.
