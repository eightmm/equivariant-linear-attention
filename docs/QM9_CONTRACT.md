# QM9 Probe Contract

This project uses the PyTorch Geometric QM9 representation processed locally
with PyG `2.8.0`. Coordinates are the supplied equilibrium 3D geometries. The
current probe predicts target index 4, `gap`, in `eV`.

## Data Identity

```text
raw/gdb9.sdf
98c4e97d50ac549b8c9f0b2114b348a9a944718e17e50d9a724b729f1deaa28e

raw/gdb9.sdf.csv
73a67793e3cfa9660f001278bd019c143f57e4785db537a01811cf2ce72aa7eb

processed/data_v3.pt
9254af077d7bc651631bb56a3a689fb41004731b413bdd0ec8c6efa318229f83
```

## Split Semantics

The 130,000-row boundary is permuted with `torch.randperm` seed 42 and sliced
into 110,000 train, 10,000 validation, and 10,000 test rows. This is a seeded
random-row warm-start split, not a scaffold, series, temporal, or prospective
holdout. Chemically similar molecules may occur across partitions. It is used
only for matched architecture probes and must not be described as a cold-entity
generalization result.

The earlier packaged run records binary index-list hashes in
`artifacts/20260713-moment-linear-qm9-2k/manifest.json`. New runs also emit
canonical text index hashes directly in their metrics JSON.

Target normalization is fit on training targets only. Architecture selection
uses validation only; the test split is skipped by `--skip-test-eval`.

## Ledger Semantics

`docs/EXPERIMENTS.jsonl` is written by `oms research-runner`. Its `gate` field
means the preflight check and experiment command executed successfully. It does
not mean the scientific hypothesis met its registered threshold. Hypothesis
outcomes for the enhanced-moment study are recorded separately in
`outputs/ablations/summary.json`.
