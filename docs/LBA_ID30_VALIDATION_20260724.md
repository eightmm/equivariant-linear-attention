# ATOM3D-LBA ID30 validation study (2026-07-24)

## Outcome

The first full held-out ATOM3D-LBA experiment is complete. This replaces the
earlier 16-complex train-only capacity check as the relevant affinity
generalization evidence.

On the complete official ID30 train/validation split, the current
gated-local plus grouped-normalization LGL candidate reached validation RMSE
`1.550035 pK`. The matched previous LGL reached `1.592008 pK`, and the private
parameter-matched static EGNN reached `1.692812 pK`.

The registered single-seed gate passed: candidate minus incumbent RMSE was
`-0.041973 pK`, exceeding the frozen `0.02 pK` improvement threshold. The
candidate also beat the private EGNN by `0.142776 pK`.

This is positive architecture evidence, but not yet a default-promotion or
publication-grade superiority claim:

- a paired bootstrap over the 466 validation complexes gave a candidate versus
  incumbent 95% interval of `[-0.130138, +0.043411] pK`; it crosses zero;
- only one deterministic model seed was trained;
- every arm clipped more than 99% of optimizer updates;
- the EGNN is an internal same-feature control, not an official reproduction;
- the test split remained unopened.

## Frozen protocol

| item | value |
|---|---|
| dataset | `vector-institute/atom3d-lba` |
| revision | `f93dd2d150a47c270f624620f84e07451a158705` |
| split | official ID30 train 3,507 / validation 466 |
| test | structurally rejected by the loader and runner |
| prediction unit | bound pocket-ligand crystal complex |
| target | supplied affinity `pK` |
| input | opaque atom-token one-hot, pocket/ligand identity, bound coordinates |
| readout | ligand-node mean |
| candidates | identical self + intra-16 + cross-16, 6 Å cutoff |
| optimizer | AdamW, `3e-4`, weight decay `0.01`, clip `1.0` |
| schedule | 5-epoch warmup, cosine decay to `0.05x` |
| selection | best validation RMSE, max 100 epochs, min 30, patience 15 |
| numeric lane | strict deterministic FP32 CUDA |
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q, 101,971,591,168 bytes |

The train-only target normalizer was mean `6.523873 pK`, standard deviation
`2.000890 pK`. The precomputed topology contained 32,303,245 directed
candidates across train and validation and took 56.25 seconds to build. Every
model consumed the same sample order and topology.

## Accuracy

| arm | params | best epoch | train RMSE | validation MAE | validation RMSE | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|
| gated + grouped LGL | 168,815 | 37 | 1.295006 | 1.254561 | **1.550035** | **0.637805** | 0.610718 |
| previous LGL | 161,541 | 15 | 1.560473 | 1.297191 | 1.592008 | 0.622750 | **0.615140** |
| private static EGNN | 167,260 | 16 | 1.696543 | 1.349694 | 1.692812 | 0.537693 | 0.532804 |
| train-mean constant | — | — | — | 1.614885 | 2.039959 | — | — |

The candidate improves the primary RMSE and MAE over both learned controls.
The incumbent has slightly higher Spearman correlation, so the candidate does
not dominate every metric.

### Paired validation bootstrap

Ten thousand deterministic nonparametric resamples of the 466 validation
complexes were used. These intervals measure sensitivity to the finite
validation set; they do not measure model-seed or protein-cluster uncertainty.

| comparison | point RMSE delta | bootstrap 95% interval | P(candidate lower) |
|---|---:|---:|---:|
| candidate - incumbent | -0.041973 | [-0.130138, +0.043411] | 0.8304 |
| candidate - private EGNN | -0.142776 | [-0.230663, -0.055042] | 0.9995 |

Thus the private-EGNN advantage is consistent across this held-out set, while
the smaller incumbent improvement remains suggestive rather than conclusive.
A multi-seed confirmation is required before changing defaults.

## Runtime and memory

The synchronized step measurement includes batch transfer plus
forward/backward/update, but excludes one-time topology construction and
epoch-level validation.

| arm | median step | p90 step | approximate complexes/s | peak CUDA allocation |
|---|---:|---:|---:|---:|
| gated + grouped LGL | 24.726 ms | 26.041 ms | 647.1 | 1.732 GB |
| previous LGL | 26.517 ms | 27.291 ms | 603.4 | 1.253 GB |
| private static EGNN | 9.674 ms | 10.438 ms | 1,653.9 | 1.545 GB |

The candidate was 6.75% faster per step than the incumbent despite the extra
local path, consistent with the preceding hot-path refactor. It used 38.3% more
peak allocation than the incumbent. The private EGNN remained 2.56x faster per
step and used 10.8% less peak allocation than the candidate on these
small-to-medium sparse complexes. The project’s large/dense-graph efficiency
claim therefore must remain separate from this LBA accuracy result.

## Optimization diagnosis

| arm | clipped updates | clip fraction | mean pre-clip norm | maximum |
|---|---:|---:|---:|---:|
| candidate | 11,375 / 11,440 | 0.9943 | 10.245 | 79.866 |
| incumbent | 6,544 / 6,600 | 0.9915 | 12.381 | 84.325 |
| private EGNN | 6,791 / 6,820 | 0.9957 | 9.928 | 112.884 |

The earlier concern about chronic clipping is confirmed on real held-out LBA
training, and it is not specific to one architecture. The candidate reduced
the incumbent mean pre-clip norm, but not enough to escape the clipping regime.
Validation RMSE also oscillated materially between epochs. The next
optimization study should change only learning-rate/clip policy or introduce
pathwise gradient scaling, while keeping this split and architecture fixed.

## Published context

The original ATOM3D work defines LBA as a standardized 3D structural benchmark
and emphasizes that architecture choice materially affects results
([ATOM3D paper](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/c45147dee729311ef5b5c3003946c48f-Abstract-round1.html)).
The ICLR 2024 BindNet comparison reports LBA-30 RMSE values of `1.601` for
Atom3D-GNN, `1.568` for Atom3D-ENN, `1.416` for Atom3D-CNN, `1.451` for
GeoSSL, `1.434` for Uni-Mol, and `1.340` for BindNet
([BindNet Table 1](https://proceedings.iclr.cc/paper_files/paper/2024/file/2c28efa5a86dca4b603a36c08f49f240-Paper-Conference.pdf)).

The current `1.550` is therefore competitive with the older GNN/ENN references,
but is not state of the art and trails strong CNN/pretrained methods. These are
descriptive cross-paper comparisons only: training features, preprocessing,
pretraining, parameter counts, seeds, and code are not controlled as they are
inside the same-harness comparison.

## Reproduction and evidence

```bash
uv run --locked python scripts/train_lba_id30.py \
  artifacts/hybrid-local-global-20260724/lba-id30-validation/full \
  --device cuda --arms candidate incumbent egnn \
  --batch-size 16 --max-epochs 100 --min-epochs 30 \
  --patience 15 --warmup-epochs 5 --budget-seconds 7200

uv run --locked python scripts/analyze_lba_id30.py \
  artifacts/hybrid-local-global-20260724/lba-id30-validation/full/result.json \
  artifacts/hybrid-local-global-20260724/lba-id30-validation/full/paired-analysis.json \
  --device cuda --bootstrap-replicates 10000
```

Key hashes:

- result JSON: `05ad57de67ef1f5daa123b11237cbf7065e503e40ec93a30f17744c519e0cd93`
- candidate best checkpoint:
  `8ee0b987f84b773be38a3a9fbd54fabf67ff7947a04e8f57388fb22854179650`
- incumbent best checkpoint:
  `15ac292ea08d18df6f5523720c06def958743ae2f79b257467d3a3fda97ee735`
- EGNN best checkpoint:
  `3814d838fffc316e0696b8b0db48eac7f46a99610b72cea8ec0410fb54e51d57`

The complete local artifact directory is
`artifacts/hybrid-local-global-20260724/lba-id30-validation/`. Aggregate
results and provenance can be versioned; per-complex labels/predictions and
checkpoints should remain local because of dataset licensing and artifact size.
Automated focused checks and checkpoint re-evaluation passed. An independent
review was not performed in this task, so the scientific packet remains
review-pending.
