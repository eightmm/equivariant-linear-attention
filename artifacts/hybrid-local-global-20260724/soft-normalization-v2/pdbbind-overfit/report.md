# Soft-normalization v2 ATOM3D-LBA train-only overfit

## Outcome

The corrected gated-plus-grouped LGL model passed the frozen capacity gate. On
the same 16 cached ATOM3D-LBA/PDBBind-derived train complexes, it first reached
the evaluated `train MAE <= 0.10 pK` threshold at step 950. The preceding
cutoff-squared version reached it at step 1,050.

| arm | threshold result | train MAE at stop | median step | peak CUDA |
|---|---:|---:|---:|---:|
| incumbent LGL | pass at 1,800 / 51.15 s | 0.085251 pK | 25.23 ms | 268.9 MB |
| prior v1 combined | pass at 1,050 / 27.60 s | 0.098200 pK | 23.63 ms | 402.2 MB |
| current v2 combined | pass at 950 / 25.03 s | 0.099318 pK | 23.56 ms | 423.7 MB |
| private static EGNN | miss at 3,000 | 0.116225 pK | 4.25 ms | 326.1 MB |

Threshold metrics were evaluated every 50 updates, so these are first observed
crossings on that grid rather than per-update crossing times.

## Interpretation

Relative to v1, v2 needed 100 fewer updates (`9.52%`) and 2.56 fewer measured
seconds (`9.29%`) to cross the threshold. Median step latency was effectively
unchanged (`0.997x`), while peak CUDA allocation increased `5.35%`.

Relative to the same-run incumbent, v2 needed `47.22%` fewer updates and
`51.06%` less wall time. Its median step was `6.62%` faster, but peak CUDA
allocation was `1.576x` higher.

The private EGNN remained the systems winner on these small complexes: the v2
median step was `5.54x` slower and used `1.300x` peak memory. EGNN nevertheless
failed this particular memorization threshold at 3,000 updates. That is
finite-sample optimization/capacity evidence, not an accuracy ranking.

## Matched-comparison audit

The current and historical runs have identical dataset revision, sample IDs,
sample identity hash, per-sample node counts, and per-sample edge counts. The
candidate initial-state hash is also identical. The rerun incumbent and EGNN
reproduced their historical final-state hashes exactly. All three current arms
received 153,029 identical directed candidates with self edges, identical raw
features, coordinates, targets, ligand masks, and cyclic batches.

Execution used strict deterministic CUDA on an NVIDIA RTX PRO 6000, the locked
environment, cached public data, and base commit
`6dd4dcc0e9fcb03197c69424ec26391e443200ed`. No validation or test labels were
read.

## Claim boundary

This run establishes that the current path is wired, trainable, and capable of
memorizing the frozen 16-complex subset. It does not establish PDBBind affinity
generalization, ranking quality, target/cluster transfer, or superiority to an
official EGNN implementation. One deterministic seed and a train-only subset
also cannot establish a robust convergence advantage. The combined path
therefore remains opt-in.

## Evidence

- Frozen scope: `scope.md`
- Raw immutable result: `result.json`
- Machine-readable comparison: `results-summary.json`
- Compute environment: `compute-environment.json`
- Reference provenance: `reference-use-ledger.json`
- Verification summary: `verification-summary.json`

Independent review was not performed for this follow-up. The raw identities,
thresholds, and arithmetic were self-audited and are retained for later review.
