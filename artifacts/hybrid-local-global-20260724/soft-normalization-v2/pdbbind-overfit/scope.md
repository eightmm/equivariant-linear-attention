# Soft-normalization v2 ATOM3D-LBA train-only overfit

## Decision

Determine whether the corrected gated-plus-grouped LGL model still has enough
protein-ligand capacity to memorize the same frozen ATOM3D-LBA/PDBBind-derived
train subset, and whether its convergence changed relative to the immediately
preceding cutoff-squared implementation.

This is a wiring/capacity diagnostic. It cannot establish affinity
generalization, ranking quality, cold-target performance, or superiority on
PDBBind validation/test data.

## Claims and falsifiers

- **C1 — finite capacity:** corrected combined LGL reaches train MAE
  `<= 0.10 pK` within 3,000 updates. Failure to cross is a rejection.
- **C2 — preserved convergence:** corrected combined crosses no later than the
  prior combined result at update 1,050. Crossing later rejects this stronger
  prediction even if C1 passes.
- **C3 — matched comparison:** incumbent, corrected combined, and private
  static EGNN receive identical raw node features, coordinates, ligand masks,
  normalized targets, batches, and sparse candidate edges. Any identity drift
  invalidates the comparison.

## Frozen inputs and execution

- Dataset: cached `vector-institute/atom3d-lba` conversion of PDBBind 2019.
- Split and rows: train only, frozen indices 0--15.
- Prediction unit: one protein-pocket/ligand complex.
- Label: supplied affinity `pK`; normalizer fitted only to these 16 train rows.
- Topology: segment-balanced directed candidates with self edges,
  `intra_k=16`, `cross_k=16`, cutoff 6 Angstrom; identical for every arm.
- Arms: incumbent LGL, corrected gated-plus-grouped LGL, and near-parameter
  private static EGNN.
- Optimizer/order/model seeds, batch size, evaluation interval, and threshold
  are inherited unchanged from the registered runner.
- Strict deterministic CUDA, existing `uv.lock`, cached data, no download,
  dependency change, validation access, or test access.

## Metrics and interpretation

Primary metric is updates and wall time to first train MAE `<= 0.10 pK`.
Secondary diagnostics are final/best train MAE, synchronized median step
latency, peak CUDA allocation, parameter count, finite gradients, input hashes,
and exact edge counts.

The previous combined result (`1,050` updates, `27.60 s`) is a versioned
historical comparator. The current incumbent and EGNN reruns are the
same-source controls. A faster threshold crossing is evidence of optimization
on these 16 rows only.

## Resource envelope

- One local NVIDIA RTX PRO 6000 GPU.
- At most 3,000 updates per arm and 600 seconds cumulative GPU wall time.
- Output:
  `artifacts/hybrid-local-global-20260724/soft-normalization-v2/pdbbind-overfit/result.json`.
- Cancel only the process launched for this packet, first with a graceful
  interrupt. Preserve partial output on timeout or failure.
