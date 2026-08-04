# Project contract

Equivariant Linear Attention exposes exactly one public execution path:

```text
ELAGraph -> ELA -> ELAGraph
```

```python
from equivariant_linear_attention import ELA, ELAGraph
```

`ELA` is the only public model. `ELAGraph` is the only public input, batch, and
output container. Coordinate updates are declared on `ELA` with
`update_positions`; there is no runtime refinement object or alternate model.

The numerical core uses a private packed receiver-major CSR representation,
but that representation is not part of ordinary user code. Public edges are
always sender-to-receiver. Representations are declared only with irreps through
`l <= 2`.

The canonical layer is fixed global-plus-local exact transport with parity-valid
updates, tensor closure, residuals, and an equivariant FFN. Learned branch
routing, persistent edge state, dense attention, and automatically promoted
Triton execution are not part of the canonical architecture.

## Scope and applicability

ELA is a general-purpose architecture for finite 3D point clouds and sparse 3D
graphs. QM9 and ATOM3D-LBA are validation harnesses, not definitions of the
model's domain. Current evidence applies to scalar property prediction on small
molecules and bound protein-pocket/ligand complexes. It does not establish
protein folding, force quality, cold-target generalization, apo-to-bound
transfer, pose robustness, periodic-cell behavior, or arbitrary ``l > 2``
support.

## Registered real-data validation contracts

All claim-bearing runs use `scripts/validate_realdata.py`. The inference-time
boundary is one `ELAGraph` containing invariant node features, 3D coordinates,
optional sender-to-receiver sparse edges and invariant edge types. Targets,
split membership, normalization statistics, and dataset-wide identifiers never
enter `ELA.forward`. Dataset adapters and task readouts remain outside the
public model.

### QM9 gap

- Prediction and label: HOMO-LUMO `gap`, target index 4, in eV. Lower validation
  MAE is better; RMSE is secondary.
- Immutable entity key: the cached PyG row index under the recorded raw and
  processed-file SHA-256 identities.
- Split key: seed-42 permutation of the first 130,000 rows; 110,000 train,
  10,000 validation, and 10,000 unused test rows by default.
- Train-only transform: scalar target mean and standard deviation.
- Baseline: the paired `full` ELA arm from the same source, split, initialized
  state, optimizer, topology, and update budget. Architecture arms change only
  the named private lane.
- Leakage boundary: PyG loads one monolithic processed object containing all
  labels. Receipts therefore record test-label storage as materialized; test
  indices are not indexed, used for normalization, selected on, or evaluated.
  This is a warm random-row architecture screen, not scaffold/cold-molecule
  evidence. Historical test access means the local test partition is not a
  pristine final holdout.

### ATOM3D-LBA affinity

- Prediction and label: supplied bound-complex affinity `pK`. Lower validation
  RMSE is the primary ID30 metric; MAE is secondary. The train-only overfit task
  reports train MAE/RMSE as a capacity diagnostic, not generalization.
- Immutable entity key: pinned dataset revision
  `f93dd2d150a47c270f624620f84e07451a158705`, split name, row index, and the
  label-blind pocket/ligand sample identity recorded in each receipt.
- Split key: only the allowlisted ID30 train and validation Arrow shards.
  Resolving or opening `test` is rejected before dataset import.
- Train-only transform: scalar affinity mean and standard deviation.
- Baseline: the paired `full` ELA arm with identical 140D invariant atom and
  pocket/ligand features, initialized state, relation schema, sparse topology,
  optimizer, and update budget. The `stagewise` coordinate arm is a separately
  labelled functionality arm because its parameter schema differs.
- Leakage boundary: validation may report metrics and has historical
  model-selection use. A cached test row was accidentally materialized during a
  2026-08-04 schema audit, so the local test split is contaminated and must not
  support a final-holdout claim. Strongest admitted evidence is bound-complex
  ID30 train/validation; it is not cold-target, apo, or pose-robust evidence.

Claim-bearing experiments must preregister the exact arm set, model/data seeds,
optimizer-update budget, primary metric and decision threshold. Short one-seed
screens are exploratory process and mechanism evidence regardless of outcome.
Commands, hashes, metrics, and null or negative results go to
`docs/EXPERIMENTS.jsonl`. Raw data and unselected/generated run outputs stay
outside git. A sanitized, manifest-validated evidence bundle may be committed
when it is intentionally selected for a release record; it must omit secrets,
host identifiers, raw examples, and unapproved claim outputs.
