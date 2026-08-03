# QM9 Probe Contract

> **Historical:** the QM9 loader and training runner described here are not
> shipped by the current architecture-only package. Use the recorded Git
> revision when reproducing these results.

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
Each loaded entity carries
`qm9-row-{row}-raw-index-{data.idx}-name-{data.name}`. This distinguishes the
processed PyG row, zero-based raw-record index, and one-based molecule name so
external joins cannot silently treat `data.idx` as the GDB9 name.

Target normalization is fit on training targets only. Architecture selection
uses validation only. Test evaluation is disabled by default and occurs only
when the caller explicitly supplies `--evaluate-test`.

Registered adaptive comparisons use three-layer/four-head `ggg`, `lgl`, or
mechanistic-control `lll` routing; memory counts 1, 4, or 8; and isolated
alignment, balancing, pair-floor, memory-interaction, and radial-trace changes.
These are capabilities under study, not promoted performance defaults.

The 2026-07-18 mechanism study additionally registers `lgg`, `ggl`, and
learned/uniform/none global transport. Its private static-coordinate EGNN arm
uses the exact same PyG features, row split, training-only target transform,
optimizer/update budget, and graph-mean readout. It is a local layer baseline,
not the official-paper data/training recipe.

The separately approved dynamic-coordinate study preserves these data and
label boundaries. It compares static/dynamic `ggg`, `lgl`, and private EGNN
controls while varying only the coordinate-update switch within a family.
Updated positions are latent property-model state; the study makes no relaxed
geometry, force, dynamics, or energy-conservation claim. Its screen and
conditional five-seed confirmation remain validation-only under a fresh
1,500 GPU-second ceiling.

That packet completed all 26 registered arms in 944.3 GPU-wall seconds with
test evaluation disabled. The five-seed mean validation MAEs were
`0.582946/0.585535 eV` for attention static/dynamic and
`0.408932/0.410428 eV` for private EGNN static/dynamic. Neither family passed
the complete accuracy/resource rule, so coordinate updates remain optional and
off by default. Every dynamic confirmation arm had nonzero coordinate
gradients, maximum per-layer step no greater than `0.25000003 Angstrom`, and
maximum centroid drift `4.92e-7 Angstrom`.

The approved 2026-07-19 validation-only execution completed the six-arm screen
and five-seed learned/uniform/none confirmation without test-label access.
Both global modes improved validation MAE over `none`, but at least one frozen
20% latency/memory ceiling failed for every candidate. Transport was therefore
not locked, defaults did not change, and the conditional EGNN arm remained
unexecuted. The immutable ledger row records 819 seconds and
`transport_locked=false`.

Interacting M=4/M=8 runs additionally require the frozen Stage-0 mechanism
gate to pass at widths 16/64 and seeds 401--403 before any QM9 labels are used.
The 2026-07-17 full matrix rejected every registered residual-coupling
candidate, so this contract admits only the independent M=1 `ggg` versus `lgl`
comparison. Its validation-only promotion rule is paired across model seeds
41--43: mean MAE improvement at least 0.010 eV, improvement in at least two
seeds, and worst-seed regression no larger than 0.020 eV.

Before the three-seed comparison, synchronized eager CUDA measurements use 64
graphs with 18 and 29 nodes, 20 warm-up plus 50 measured iterations, five
fresh processes, and process-mean medians. Both forward and forward/backward
latency and peak allocated memory must remain within 20% of `ggg`. Test labels
remain disabled throughout this adaptive gate.

## Ledger Semantics

`docs/EXPERIMENTS.jsonl` is the immutable historical run ledger for this task.
Its `gate` field means the preflight check and experiment command executed
successfully; it does not mean the scientific hypothesis met its registered
threshold. Historical test access prevents describing this random-row split as
a pristine confirmatory holdout.
