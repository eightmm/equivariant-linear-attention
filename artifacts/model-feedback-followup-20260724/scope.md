# Model-feedback follow-up contract

## Decision

Implement the model-relevant findings from the shared review while disabling
automatic GitHub Actions. Preserve default `mean` readout and static-coordinate
behavior exactly. The follow-up may change only opt-in local transports,
dynamic-coordinate sparse-edge handling, and an opt-in protein-ligand readout.

## Claims and falsifiers

- **C1 — cutoff regularity:** gated, edge-conditioned normalized, and pairwise
  local paths use only smooth cutoff mass for normalization and mass features.
  A candidate fails if the registered float64 cutoff probes exceed `2e-4`
  output error or `2e-3` coordinate-gradient error across `cutoff ± 1e-5 Å`.
- **C2 — reduction efficiency:** gated local scalar/vector/relative/tensor and
  mass statistics use one packed receiver reduction per local stage. Relative
  to commit `626275a`, the CUDA diagnostic must reduce `index_add` calls and
  must not regress profiled full-step time or peak allocation by more than 10%.
- **C3 — dynamic-neighbor correctness:** external sparse candidates with
  coordinate updates are rejected unless the caller explicitly selects fixed
  approximate candidates or exact per-layer rebuilding. In rebuild mode, a
  pair that crosses the cutoff must enter the next local layer.
- **C4 — interaction readout:** an opt-in readout combines ligand, pocket, and
  cross-interface pools with parity-even products of two learned
  pseudoscalars. It must preserve global O(3), translation, node permutation,
  graph isolation, and the existing output schema. Its final projection is
  zero initialized so common initialization matches `mean` readout.

## Evidence boundary

- Architecture-only QM9 and ATOM3D-LBA raw node features remain unchanged.
- The PDBBind check uses cached `vector-institute/atom3d-lba` revision
  `f93dd2d150a47c270f624620f84e07451a158705`, train rows 0–15 only.
- No validation or test label is read. A short train-only run is a wiring and
  optimization smoke, not affinity generalization evidence.
- One local RTX PRO 6000 may be used for at most 120 cumulative GPU seconds.
- No new dependency, official EGNN claim, default model promotion, or
  long-running GitHub CI is included.
