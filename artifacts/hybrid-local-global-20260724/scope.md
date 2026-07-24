# Frozen scope: same-feature gated local/global architecture

Date: 2026-07-24
State: confirmed by the user's instruction to keep raw features matched and
continue with the architecture changes judged most useful

## Decision

Determine whether a stronger short-range equivariant interaction can improve
the existing LGL model while preserving the exact edge-free `O(N)` global
channel. Every matched arm receives the same raw node feature tensor,
coordinates, split, and (where used) sparse edge candidates. No atom, bond,
residue, protein-language-model, or label-derived feature is added.

The intervention has two independently inspectable switches:

1. a gated local edge transport with a width-matched per-head MLP, invariant
   vector/geometry contractions, explicit neighborhood mass, and equivariant
   scalar/vector/rank-2 messages;
2. grouped pre-normalization of scalar-message, angular, and persistent-tensor
   invariant families before the incumbent update normalization.

The existing factorized global attention, readout, coordinate policy, raw
features, targets, and public defaults remain unchanged. Candidate-only module
construction must preserve the incumbent RNG stream.

## Hypotheses and gates

- **H1 — software/mathematical contract:** both switches are opt-in; disabled
  output/state behavior is unchanged; scalar outputs retain O(3), translation,
  permutation, batch-isolation, finite-gradient, and sparse-edge contracts.
- **H2 — QM9 screen:** on strict deterministic CUDA, seed 42, 500 updates,
  cached QM9 `gap`, train/validation `110000/10000`, and test disabled, gated
  local transport improves the matched LGL validation MAE by at least
  `0.010 eV` with regression no larger than `0.020 eV`. Grouped normalization
  is a second arm; it is attributed only relative to gated-local alone.
- **H3 — optimization diagnostic:** the selected candidate reduces the
  incumbent clipping fraction by at least `0.05` absolute or reduces mean
  pre-clip norm by at least `20%`, without failing H2's regression guard.
- **H4 — ATOM3D-LBA capacity:** using the cached immutable PDBBind-derived
  ATOM3D-LBA train rows 0--15, all compared models receive identical
  segment-balanced sparse candidates derived from the existing segment IDs and
  coordinates. The selected hybrid reaches best observed train MAE
  `<=0.10 pK` within 3,000 updates. This is train-only capacity evidence, not
  affinity generalization.

No screen result can establish official EGNN superiority, test performance,
PDBBind generalization, docking quality, or the value of additional chemical
features.

## Execution boundary

- Existing `uv.lock`, cached QM9 and ATOM3D-LBA data, one local CUDA GPU.
- No dependency, network, download, test-label, checkpoint-publication, or
  remote-compute change.
- Small CPU mathematical/behavioral checks first.
- GPU execution requires a one-time exact approval packet and is capped at
  900 cumulative GPU-wall seconds.
- QM9 arms: incumbent LGL, gated-local LGL, gated-local plus grouped invariant
  normalization. All use identical precomputed 2.5-Angstrom candidates.
- ATOM3D-LBA arms: incumbent LGL, selected hybrid, and private static EGNN on
  the same segment-balanced candidates, 6-Angstrom cutoff, ligand-only readout,
  deterministic batches, and no validation/test access.
- Failed and null results remain in the ledger; defaults change only after the
  frozen gate passes.

## Planned evidence

1. focused RED/GREEN tests and a CPU optimizer-step smoke;
2. strict CUDA smoke;
3. matched QM9 screen and clipping diagnostics;
4. conditional ATOM3D-LBA 16-complex overfit comparison;
5. commands, hashes, environment, metrics, limitations, and review record.
