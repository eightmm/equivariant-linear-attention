# Frozen scope: PDBBind overfit with persistent 2e

Date: 2026-07-23
State: confirmed by the user's instruction to download and proceed

## Decision

Determine whether the edge-free spatial-linear attention implementation has
enough correctly wired protein-ligand capacity to memorize a deterministic
16-complex ATOM3D-LBA train subset, while preserving its O(3), translation,
permutation, batch-isolation, and linear-global-memory contracts.

The deliverable is source code, tests, a locally cached and provenance-pinned
dataset, a bounded attention-versus-private-EGNN overfit result, a resource
comparison after the fit threshold, documentation, and a review receipt.

This is not an affinity generalization benchmark, docking experiment,
prospective prediction, test-set result, force model, commercial-data
workflow, or claim that persistent `2e` is generally superior.

## Data and inference boundary

- Source: `vector-institute/atom3d-lba`, an auto-converted Parquet view of the
  ATOM3D LBA task derived from the PDBBind 2019 refined set.
- License boundary: local non-commercial research under the upstream
  CC-BY-NC-ND-4.0 terms. Raw or transformed records and checkpoints are not
  redistributed or committed.
- Prediction unit: one co-crystallized protein-ligand complex.
- Initial subset: 16 deterministic rows from the published train split,
  selected before label inspection and identified by row index plus content
  digest. Validation and test rows are not selected, indexed, or used; see the
  disclosed cache-builder side effect below.
- Input available at inference: upstream categorical atom tokens, deposited co-crystal
  coordinates in Angstrom, and the supplied protein/pocket/ligand token type.
- Retained atoms: supplied pocket copy (`token_type_id=1`) and ligand
  (`token_type_id=2`). The full-protein copy (`token_type_id=0`) is excluded to
  avoid duplication and unbounded full-protein graph sizes.
- Node input: one-hot upstream atom token plus pocket/ligand type. The dataset
  card describes `input_ids` as atomic numbers, but schema preflight on the
  pinned train parquet found IDs `122..137`; these are therefore treated as
  opaque categorical tokens rather than assigned unsupported element
  semantics. Ligand nodes form the graph readout mask after pocket-to-ligand
  transport.
- Label: supplied scalar pK, defined upstream as `-log(Ki)` or `-log(Kd)`.
  Target normalization is fitted only on the 16 training rows.

## Hypotheses and claims

### C1: data and readout correctness

The loader preserves atom-coordinate-type alignment, removes type 0, retains
at least one pocket and ligand atom, assigns a nonempty ligand readout mask,
and remains invariant to joint node permutation.

Falsifier: any schema mismatch, nonfinite coordinate/label, missing segment,
empty readout, or permutation-dependent scalar result.

### C2: persistent rank-2 capability

An opt-in symmetric-traceless `2e` hidden state can persist between blocks,
interact through invariant gated residual and FFN paths, receive finite nonzero
training gradients, and transform as `T -> R T R^T` for every tested
`R in O(3)`. With zero hidden tensor multiplicity, forward values, output keys,
and state schema remain unchanged.

Falsifier: failed disabled compatibility, nonfinite gradients, tensor-trace or
symmetry drift, or O(3)/translation/permutation/batch error above `1e-6` in
float64.

### C3: bounded overfit

The edge-free attention arm reaches training MAE at most `0.10 pK` within
3,000 optimizer updates on the frozen 16-complex subset.

Baseline: the private static EGNN control receives identical nodes,
coordinates, targets, readout mask, ordering, and optimizer/update budget. It
uses a precomputed directed 6-Angstrom radius graph with self candidates
(self edges are removed by the private EGNN implementation); the attention arm
receives no edge tensor.

Falsifier: the attention arm does not reach the threshold within the bound.
An EGNN pass does not rescue an attention failure.

### C4: descriptive resource comparison

After an arm first reaches `0.10 pK`, record synchronized steady-state step
latency, peak CUDA memory, parameters, nonzero-gradient parameters, and
time-to-threshold. These are descriptive on this hardware and subset; there is
no preregistered requirement that attention beat EGNN.

## Architecture intervention

- Add an optional permutation-consistent node readout mask to graph samples,
  batches, attention pooling, and the private EGNN control. Omitting it retains
  current whole-graph mean pooling.
- Permit persistent `2e` channels in `hidden_irreps`; initialize them to zero,
  mix the current transported rank-2 moment into the hidden tensor state, use
  tensor norm/contractions as scalar invariants, and apply invariant-gated
  tensor residual/FFN updates.
- Use the existing edge-free multiscale spatial kernel for the attention arm.
- Do not add e3nn, spherical harmonics, `2o`, general `l>2`, chirality-sensitive
  claims, learned coordinate updates, or another public architecture.

## Execution and acceptance

- Dependency: one optional `pdbbind` extra containing `datasets`; update and
  commit the project lock.
- Download: about 473 MB from Hugging Face into ignored local caches. The
  production loader names only the two pinned train Parquet shards. The first
  failed `datasets` builder attempt also materialized val/test files despite a
  train split request; no val/test row was indexed, printed, selected, or used,
  but the physical cache is no longer described as a pristine unopened
  holdout. Record
  repository revision, file/cache identifiers, schema, row indices, and
  content digests without copying dataset records into artifacts.
- Start with RED unit and integration tests, then a CPU real-row loader smoke,
  `scripts/check.sh fast`, and `scripts/check.sh gpu`.
- Train sequentially on one local GPU. FP32 is the reference precision.
  Cumulative GPU wall time is capped at 1,800 seconds across both arms, with
  at most 900 seconds per arm. Both use deterministic cyclic batches of two,
  AdamW with learning rate `1e-3`, zero weight decay, and gradient clipping at
  `1.0`.
- Stop an arm on threshold, 3,000 updates, nonfinite loss, or remaining-budget
  exhaustion. Do not access validation/test labels.
- Cancel only this run's process with a graceful interrupt; preserve its
  partial log and result.
- Every positive, null, or failed arm is appended to
  `docs/EXPERIMENTS.jsonl`.

## Risks and controls

- A 16-example overfit can measure wiring/capacity but cannot measure
  generalization or prevent memorization through shortcuts.
- PDBBind labels mix Ki and Kd assay semantics and the co-crystal pose leaks
  bound-state geometry by design; claims stay within the supplied LBA task.
- Ligand-only pooling reduces pocket-size dilution but makes the readout
  entity-specific; the segment mapping and mask are explicit and tested.
- The public Parquet conversion may drift. Execution pins the immutable
  revision resolved at download time.
- A later ID30 validation comparison, protein-only/ligand-only shortcuts, and
  leakage-resistant affinity claims require a separate confirmed packet. That
  packet must recover or independently verify the upstream `122..137` atom
  token mapping before making element-semantic claims.
