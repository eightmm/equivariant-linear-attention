# Real-batch LBA train-step profile (2026-07-27)

## Result

The accepted gated-plus-grouped LGL is faster than the preceding LGL on the
profiled real ATOM3D-LBA batch while launching fewer indexing/scatter
operations and more matrix multiplications. This is consistent with lower
dispatch/indexing cost despite more arithmetic. Its extra peak memory is
consistent with the dense edge-MLP activation path. Activation checkpointing
exposes a useful memory/speed tradeoff without changing equations or
parameters.

## Profile lane

- cached ATOM3D-LBA revision
  `f93dd2d150a47c270f624620f84e07451a158705`;
- first 16 official ID30 train complexes only;
- 7,378 nodes and 153,029 directed sparse edges;
- strict FP32 CUDA, model seed 41;
- two warmup steps, 20 synchronized timing steps, three profiled steps;
- identical batch, topology, loss, optimizer, and clipping for all arms;
- no validation or test evaluation.

| arm | parameters | median synchronized step | peak CUDA allocation |
|---|---:|---:|---:|
| candidate | 168,815 | **23.379 ms** | 1,456,019,456 B |
| checkpointed candidate | 168,815 | 28.529 ms | 1,160,661,504 B |
| incumbent | 161,541 | 31.747 ms | **1,062,962,688 B** |

Relative to the incumbent, the uncheckpointed candidate used `0.7364x` step
time and `1.3698x` peak allocation on this batch. Relative to that candidate,
activation checkpointing used `1.2203x` step time and `0.7971x` peak
allocation: a `20.29%` peak-memory reduction for a `22.03%` latency cost. The
checkpointed candidate remained faster than the incumbent on this one batch
and reduced its memory excess to about `1.092x`.

These timings are post-outcome diagnostics, not replacements for the full-run
medians. The three-seed LBA study measured a smaller candidate latency
advantage (`0.9347x`).

## Operator diagnosis

Over the three profiled steps, candidate/incumbent counts were:

| operator | candidate | incumbent |
|---|---:|---:|
| `aten::_index_put_impl_` | 84 | 138 |
| `aten::index` | 93 | 123 |
| `aten::bmm` | 144 | 180 |
| `aten::mul` | 1,734 | 2,076 |
| `aten::mm` | 264 | 231 |

The gated path's factorized edge MLP replaces part of the repeated
gather/scatter work with larger matrix multiplications. The observed operator
mix is consistent with lower dispatcher/indexing wall time and more stored edge
activations. The checkpointed mode recomputes the latter edge-MLP segment
during backward, matching the observed lower memory and higher latency.

## Implementation decision

`train_lba_id30.py` now exposes
`--checkpoint-gated-local-mlp`. It is opt-in because the resource tradeoff is
workload-dependent. The core implementation already had an exact
forward/backward equivalence test; the new LBA path records the option in the
run configuration and a strict-CUDA real-data smoke completed with finite
gradients and `test_evaluated=false`.

The reusable profiler is `scripts/profile_lba_train_step.py`. Its profiler
operator times include profiling overhead and are diagnostic only. The
artifact is `artifacts/lba-operator-profile-20260727/profile-final.json` with
SHA-256
`bc08a1b90653ce2b6e815e84cf7b09f2273a806f0e8f9f59fe2bc22b2bcfd009`.
