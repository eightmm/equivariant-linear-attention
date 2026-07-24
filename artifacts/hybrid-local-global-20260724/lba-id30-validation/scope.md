# ATOM3D-LBA ID30 validation scope

## Decision

Determine whether the current opt-in gated-local plus grouped-normalization LGL
architecture generalizes better than the previous LGL and a
parameter-matched private static EGNN when all three receive identical raw
features, coordinates, official split membership, sparse candidates, target
normalization, batches, optimizer, and model-selection policy.

The prediction unit is one bound protein-pocket/ligand crystal complex. The
label is the supplied affinity `pK`. Inference assumes that the bound pocket,
bound ligand pose, atom-token categories, pocket/ligand identity, and Cartesian
coordinates are available. This run does not claim sequence-only, apo-structure,
novel-pocket, pose-generation, virtual-screening, or prospective affinity
performance.

## Frozen evidence lanes

- Dataset: cached public `vector-institute/atom3d-lba` at immutable revision
  `f93dd2d150a47c270f624620f84e07451a158705`.
- Split: complete official ID30 `train` (3,507 complexes) and `val` (466
  complexes). The runner rejects `test`; test rows and labels are not loaded,
  selected, or evaluated.
- Representation: the existing deterministic 138-way opaque atom-token one-hot
  plus two-way pocket/ligand identity; full-protein duplicates are excluded.
- Topology: identical self-containing segment-balanced candidates for every
  arm, with `intra_k=16`, `cross_k=16`, and a 6 Angstrom cutoff.
- Readout: ligand-mask mean readout for all arms.
- Target transform: mean and population standard deviation fitted on train
  labels only, then inverted before reporting metrics.
- Model arms:
  1. `candidate`: current LGL with gated local transport and grouped invariant
     normalization.
  2. `incumbent`: otherwise matched previous LGL without those two options.
  3. `egnn`: private static EGNN with the closest parameter-matched width.
- Cheap baseline: constant prediction equal to the train target mean.
- External context only: published ATOM3D ID30 GNN RMSE `1.601 pK`. Because its
  model and featurization are not the same harness, it is not used for the
  registered architecture decision.

## Falsifiable claim

Primary hypothesis: the candidate lowers best official ID30 validation RMSE by
at least `0.02 pK` relative to the same-harness incumbent. Candidate-versus-EGNN
RMSE is reported as a separate direct comparison. MAE, Pearson, Spearman,
training time, synchronized step latency, peak CUDA allocation, gradient
clipping, and parameter counts are secondary diagnostics.

This is one deterministic seed. It can falsify a large claimed gain, but cannot
establish a stable population mean or publication-grade superiority.

## Training and selection

- AdamW, learning rate `3e-4`, weight decay `0.01`, gradient clipping `1.0`.
- Batch size `16`.
- Five-epoch linear warmup followed by cosine decay to `0.05x`.
- At most 100 epochs, validation once per epoch, minimum 30 epochs, patience 15.
- Best checkpoint selected by validation RMSE. No test access.
- FP32 strict deterministic CUDA by default.

## One-time local-compute envelope

- Existing locked project environment via `uv run --locked`; no package or
  lockfile changes.
- Existing local GPU only; no remote compute and no private-data transfer.
- Existing 1.7 GB cache; network access and new downloads are disabled.
- Smoke command:

  ```bash
  uv run --locked python scripts/train_lba_id30.py \
    artifacts/hybrid-local-global-20260724/lba-id30-validation/smoke \
    --device cuda --arms candidate incumbent egnn \
    --train-limit 32 --val-limit 16 --batch-size 4 \
    --max-epochs 2 --min-epochs 1 --patience 1 --warmup-epochs 1 \
    --budget-seconds 300
  ```

- Full command:

  ```bash
  uv run --locked python scripts/train_lba_id30.py \
    artifacts/hybrid-local-global-20260724/lba-id30-validation/full \
    --device cuda --arms candidate incumbent egnn \
    --batch-size 16 --max-epochs 100 --min-epochs 30 \
    --patience 15 --warmup-epochs 5 --budget-seconds 7200
  ```

- Maximum envelope: two GPU-hours total, expected peak allocation below 16 GB,
  less than 0.1 GB new checkpoints/results, and no new dataset storage.
- Cancellation: interrupt only this command. Each epoch writes history and
  best/last portable checkpoints; completed arm results remain in `result.json`.

## Stop conditions

Stop and preserve partial evidence on nonfinite loss, deterministic-operator
failure, unavailable cached split, schema/split/topology drift, GPU OOM, or the
two-hour bound. Do not reinterpret a partial arm as a completed comparison.
