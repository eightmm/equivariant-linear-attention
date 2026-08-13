# ATOM3D PSR results

Status date: 2026-08-11. This is a retrospective record of completed runs, not
a preregistered confirmation study. The tracked ELA core was
`a9357d2967e6c00f6a980ec82f9beeb007d4ae88`. The PSR scripts were untracked at
run time. `EXPERIMENTS.jsonl` preserves only a later working-tree snapshot;
the exact script bytes used by each earlier run cannot be reconstructed.

## Evaluation contract

- Dataset: official ATOM3D PSR split-by-year LMDB, converted to local PyTorch
  shards.
- Split sizes: 25,400 train decoys, 2,800 validation decoys, and 16,014 test
  decoys. Validation contains 56 scored targets and test contains 85.
- Prediction unit: one CASP decoy structure.
- Grouping key: CASP target ID.
- Label: `GDT_TS`; higher is better.
- Primary selection metric: validation mean per-target Spearman correlation.
- Reported test metrics: mean per-target Spearman and global Spearman.
- Common optimizer: AdamW, learning rate `3e-4`, weight decay `0.01`, 500-step
  warmup, cosine decay, gradient clip `10`, node budget 12,000, seed 0.
- `resatom` is a 167-class residue/heavy-atom identity. `element` uses element
  identity only. `element-local` adds heavy-atom neighbor counts inside 3.5 and
  5.0 Angstrom, scaled by 10 and 30.

Hydrogens and unsupported atoms are removed by the conversion script. Local
counts are derived only from inference coordinates and contain no labels, but
their current generator materializes a dense distance matrix. Their observed
gain shows that this combined absolute-Angstrom local-density feature is useful;
it does not separate the contributions of absolute scale and locality, nor does
it establish an end-to-end node-linear preprocessing path.

Dataset file hashes, split ID fingerprints, and the target-level leakage check
are recorded in `PSR_DATA_MANIFEST.json`.

## Completed runs

Every test value comes from the checkpoint with the best validation mean
per-target Spearman. Epochs are zero-indexed.

| Run | Features / loss | Model | Best val mean RS (epoch) | Test mean RS | Test global RS | Peak allocated VRAM |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `psr-full-64x3` | residue+atom / MSE | 64x3 eager | 0.435630 (16) | 0.254333 | 0.652668 | 9.15 GiB |
| `psr-full-64x3-element` | element / MSE | 64x3 eager | 0.351600 (18) | 0.282845 | 0.622863 | 9.14 GiB |
| `psr-96x6-element-compile` | element / MSE | 96x6 compile | 0.363313 (30) | 0.256907 | 0.583081 | 2.33 GiB |
| `psr-fact-rank` | element / MSE + rank | 64x3 compile | 0.446697 (14) | 0.335993 | 0.322232 | 0.89 GiB |
| `psr-fact-local` | element+local / MSE | 64x3 compile | 0.509176 (14) | 0.412641 | 0.788466 | 0.85 GiB |
| `psr-fact-both` | element+local / MSE + rank | 64x3 compile | **0.583917 (18)** | **0.458579** | 0.530895 | 0.85 GiB |

`rank` is a combined configuration: within-target pairwise logistic ranking
loss with weight 1.0 plus target-grouped batching. Grouping also changes the
number and ordering of batches while the scheduler length is derived from the
ungrouped loader. Increasing element-only ELA from 64x3 to 96x6 did not improve
the test result. Adding local contact counts produced the strongest balanced
result. The combined rank configuration had higher within-target ranking and
substantially lower global calibration, but this experiment does not isolate
the ranking loss from batching and schedule effects.

## Published reference points

The original ATOM3D proceedings paper reports:

| Model | Mean RS | Global RS |
| --- | ---: | ---: |
| ATOM3D 3DCNN | 0.431 | 0.789 |
| ATOM3D GNN | 0.411 | 0.750 |
| contemporary external baseline | 0.432 | 0.796 |
| ELA element+local | 0.413 | 0.788 |
| ELA element+local+rank | 0.459 | 0.531 |

Source: [ATOM3D proceedings paper, Table 6](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/c45147dee729311ef5b5c3003946c48f-Paper-round1.pdf).

A later GVP-GNN benchmark reports three-run means of `0.515 +/- 0.010` mean RS
and `0.755 +/- 0.004` global RS for its reference GNN, and `0.511 +/- 0.010`
mean RS and `0.845 +/- 0.008` global RS for GVP-GNN. These are the more
conservative comparison points because the published ATOM3D GNN value differs
between versions. Source: [GVP-GNN, Table 2](https://arxiv.org/abs/2106.03843).

ELA therefore does not yet dominate the published benchmark. The local-only
variant nearly matches the original 3DCNN global score but remains below the
later mean-RS baselines. The local+rank variant exceeds the original paper's
mean RS while losing cross-target calibration.

## Execution probe

On one same-batch diagnostic, eager FP32 measured 0.2004 seconds/step,
55,168 atoms/second, and 9.10 GiB peak allocation. Compiled FP32 measured
0.0637 seconds/step, 173,436 atoms/second, and 0.90 GiB, with maximum forward
drift `1.71e-7`. This is about 3.15x throughput and 0.10x peak allocation after
compilation. The first compiled step took 157 seconds, and execution still
reported graph breaks and recompilation-limit warnings. These figures are a
local optimization probe, not a hardware-independent performance claim.

## Evidence limits

1. All ELA accuracy results are one seed. Published references use three-run
   means and standard deviations.
2. The test split was evaluated repeatedly during development. It is no longer
   an untouched final test and all current test comparisons are descriptive.
3. The local feature generator is an offline `torch.cdist` O(N^2) diagnostic;
   it is not evidence that the complete input pipeline is node-linear.
4. The runner records model weights and histories but not optimizer state,
   exact environment, strict deterministic mode, or a self-contained run
   manifest.
5. Raw checkpoints and output JSON files are ignored by Git. The ledger keeps
   their observed values and hashes, but a clean clone cannot independently
   re-hash the raw artifacts until they are published separately.
6. The exact untracked script bytes used by the earlier runs are unavailable;
   the recorded script hashes are a later retrospective snapshot.
7. `torch.compile` is fragmented by graph breaks. Its speed result applies to
   the measured harness only.
8. Current coordinate normalization removes absolute segment scale from the
   ELA core. The local counts reintroduce absolute Angstrom-scale information,
   but this ablation does not isolate scale from local density/contact effects.

## Current conclusion and next discriminating test

The strongest supported architectural diagnosis is that the combined
absolute-Angstrom local-density features helped more than increasing
width/depth in these single-seed runs. The next architecture experiments should
separate absolute scale from locality, implement the useful channel without
dense pair construction, and compare against the element-only 64x3 baseline on
validation across at least three seeds. A grouped-MSE control is also required
before attributing the rank-configuration result to the ranking loss. A new
held-out protocol is required before making a final test claim.

## Rerun commands

```bash
uv run python scripts/psr_localfeat.py
uv run python scripts/train_psr.py --device cuda --compile \
  --features element-local --labels gdt_ts --epochs 20 \
  --width 64 --depth 3 --node-budget 12000 \
  --out outputs/psr-fact-local

uv run python scripts/train_psr.py --device cuda --compile \
  --features element-local --labels gdt_ts --rank-weight 1.0 \
  --epochs 20 --width 64 --depth 3 --node-budget 12000 \
  --out outputs/psr-fact-both
```

Test evaluation must be added only for a declared final run; the current test
split must be treated as development-exposed.
