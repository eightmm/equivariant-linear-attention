# Soft-normalization v2 and grouped-only attribution

## Outcome

The review correctly identified a mathematical defect. Normalizing a
cutoff-weighted singleton by `sqrt(sum f_c^2 + eps)` cancels almost all radial
attenuation. All applicable normalized paths now use

```text
C_i = sum_j f_c(u_ij)
m_i = sum_j f_c(u_ij) phi_ij / sqrt(1 + C_i)
```

The squared-cutoff statistic remains an explicit learned diagnostic in the
gated and pairwise paths, but it no longer controls the divisor.

The exact-cutoff case also exposed an empty-edge packing failure, which is now
fixed. The interaction readout casts stable layer-normalized states back to the
model dtype before its linear projections; direct CUDA bfloat16
forward/backward is finite.

## QM9 2x2 attribution

The strict deterministic screen held raw node features, coordinates, targets,
split, batches, and sparse candidates fixed. It used seed 42, 500 updates,
110,000 train rows, 10,000 validation rows, and did not evaluate test labels.

| arm | validation MAE | delta vs incumbent | clipped | mean pre-clip norm |
|---|---:|---:|---:|---:|
| incumbent | 0.709287 eV | 0 | 455/500 | 5.435 |
| grouped only | 0.737526 eV | +0.028239 eV | 455/500 | 4.825 |
| gated only | 0.683842 eV | -0.025445 eV | 440/500 | 4.507 |
| gated + grouped | 0.647637 eV | -0.061650 eV | 454/500 | 5.152 |

Grouped-only is not the explanation for the earlier gain. It regressed without
gated transport. Corrected gated transport improved independently, and grouped
normalization added another `0.036205 eV` only in that gated representation.
The favorable difference-in-differences is `0.064444 eV`; combined beat
grouped-only by `0.089889 eV`, so the registered interaction threshold passed.

The combined model remains the selected opt-in candidate. This packet does not
promote it to a default: one 500-update validation seed is insufficient for a
general accuracy claim.

## Evidence

- Raw frozen records: `qm9-screen/*.json`
- Frozen scope and decision thresholds: `scope.md`
- Direct CUDA bfloat16 smoke: `cuda-bfloat16-smoke.json`
- Machine-readable synthesis: `results-summary.json`
- Stale descriptive-label disclosure: `provenance-correction.md`

## Next bounded experiment

Confirm only incumbent, corrected gated-only, and corrected combined across
multiple seeds before changing defaults. If the interaction survives, the next
task-level architecture experiment should be a ligand-centered per-ligand
interface readout that passes mean, size-normalized sum, and cutoff mass to the
head. Chirality features and a full parity-complete backbone remain later,
separate hypotheses.
