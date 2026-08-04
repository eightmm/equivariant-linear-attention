# Canonical real-data validation

`scripts/validate_realdata.py` is the bounded QM9 and ATOM3D-LBA runner for the
current package. It executes only the root public contract:

```python
from equivariant_linear_attention import ELA, ELAGraph
```

It does not restore a legacy model, LGL route, alternate graph container, or
task-specific model API. Dataset adaptation and task readout remain in the
script; every model call is `ELAGraph -> ELA -> ELAGraph`.

## Commands

The commands below document the adapters. For the frozen completion study they
must not be launched directly: use the separately authorized `data` phase of
`scripts/run_completion_packet.py`, which admits only the byte-for-byte G3-G5
argv in `gpu-job-packet.json`. The current phase is deferred because the local
GPU is occupied; no current QM9/LBA receipt is claimed.

Install only the optional data adapter needed by the selected task:

```bash
# Bounded QM9 gap screen (random-row validation only)
uv run --locked --extra qm9 python scripts/validate_realdata.py \
  qm9 artifacts/qm9-ela.json --device cuda

# Train-only 16-complex LBA capacity check
uv run --locked --extra pdbbind python scripts/validate_realdata.py \
  lba-overfit artifacts/lba-overfit-ela.json --device cuda

# Bounded official ID30 train/validation screen
uv run --locked --extra pdbbind python scripts/validate_realdata.py \
  lba-id30 artifacts/lba-id30-ela.json --device cuda
```

The defaults are intentionally screens: 100, 250, and 220 optimizer updates
per arm, respectively. A longer study must state its budget explicitly, for
example `--steps 4400`; a completed process is not a convergence claim. Add
`--include-stagewise` only for a separately labelled coordinate-update
functionality arm. Use `--help` after a subcommand for all controls.

The CPU contract tests need no dataset extras and read no real-data files:

```bash
uv run pytest -q tests/test_realdata_validation.py
```

## Fixed data boundary

QM9 uses the cached PyG representation, target index 4 (`gap`, eV), and verifies
the recorded SHA-256 identities of both raw files and `processed/data_v3.pt`
before loading. PyG materializes this processed dataset monolithically, so the
receipt explicitly says that storage containing all labels was loaded; it does
not claim that test labels remained unread at the storage level. A seed-42
permutation of the first 130,000 rows defines 110,000 train, 10,000 validation,
and 10,000 unused test rows by default. Test indices are not indexed or used for
normalization or metrics. Target normalization is fit on train targets only.
The split is a warm random-row architecture screen, not scaffold or cold-entity
generalization.

LBA accepts only the pinned cache revision
`f93dd2d150a47c270f624620f84e07451a158705`. The resolver contains an explicit
allowlist for the two train Arrow shards and one validation Arrow shard; it does
not glob the cache. Any split other than `train` or `val`, including `test`, is
rejected before a path is resolved or optional dataset code is imported.

For each LBA complex the adapter:

- discards the duplicated full-protein nodes (`token_type_id=0`);
- retains pocket (`1`) and ligand (`2`) nodes;
- validates opaque atom tokens in the frozen range 1 through 137 and forms 140
  invariant scalar channels from a 138-way token one-hot plus
  the two-way pocket/ligand indicator;
- computes the prediction as a ligand-only mean using that retained input
  indicator, not a model-generated mask; and
- uses the supplied affinity `pK` label, normalized with train labels only.

The deterministic float64 topology builder keeps self edges plus up to 16
within-segment and 16 cross-segment neighbors inside 6 Angstrom, retaining
distance ties. Public `edge_index` is sender-to-receiver. Edge types are
`pocket-pocket`, `ligand-ligand`, and `cross`. Sample identity, edge indices,
edge relations, topology, their joint hash, and edge count are recorded
separately for train and validation. A combined receipt hashes those two split
receipts. A full ID30 run fails before optimization unless it observes exactly
3,507 train complexes, 466 validation complexes, and 32,302,952 directed edges.
Initial topology materialization and Arrow decoding happen before
training timing; each measured training step does include cached sample access,
`ELAGraph.collate`, device transfer, public-graph ingestion/CSR preparation,
forward, backward, clipping, and optimizer update.

## Architecture arms

Static arms share one initialized `ELA` state and parameter schema:

| arm | isolated change |
|---|---|
| `full` | all canonical lanes enabled |
| `no-relation` | freezes relation-conditioned transport at its zero initial state; LBA only |
| `no-cg12` | disables and freezes the `1 tensor-product 2` closure lane |
| `no-multiscale` | disables and freezes the learned multi-scale local lane |

Initial state/schema hashes, disabled parameter names, and initial prediction
hashes make pairing auditable. The optional `stagewise` arm constructs
`ELA(update_positions=True)` and preserves hidden state across layerwise
coordinate updates, but its extra coordinate parameters make it a separate
functionality arm rather than a paired one-variable ablation.

## Result receipt and limits

The JSON file is written before training starts and after each arm. It records
source and git identity, data/shard/split/topology hashes, label-access flags,
train-only normalization, exact update counts, diagnostic minibatch losses,
same-set initial/final metrics for the 16-complex capacity screen,
validation or train MAE/RMSE, clipping statistics, measured step latency, CUDA
peak allocated memory, model state hashes, and limitations. CPU peak memory is
reported as `null`; it is not estimated. An interrupted or failed run is not a
completed result.

Arms execute sequentially in the recorded `arm_execution_order`. Their step
latencies therefore include order, warm-cache, and process-history effects and
are observational diagnostics, not an unbiased speed ranking. Use the dedicated
alternating-order CUDA profiler for performance decisions.

These runs are same-harness architecture checks. They do not establish SOTA,
protein-family generalization, pose robustness, apo-to-bound transfer, force
quality, or folding accuracy. Validation has historical model-selection use,
and QM9 has historical test-partition access.

During a 2026-08-04 local schema audit, one cached ATOM3D-LBA test row and its
label were inadvertently materialized. The value is intentionally not
reproduced or used. That event contaminates any claim that the local LBA test
holdout is pristine. This runner therefore cannot resolve, open, index, or
evaluate the test shard; its strongest admitted LBA evidence is ID30
train/validation evidence.
Receipts distinguish that historical contamination from the current run with
`test_label_storage_materialized_by_this_run=false` and an explicit historical
materialization flag.
