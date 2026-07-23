# Edge-free multi-scale spatial linear attention

## Outcome

The repository now has an opt-in edge-free Euclidean kernel for the existing
all-global factorized attention. It accepts no `edge_index`, creates no
node-pair tensor, keeps the public default off, and adds no learned parameter.
The existing coordinate updater remains supported and recomputes spatial
features after each coordinate step.

On the recorded RTX PRO 6000 Blackwell FP32 forward workload, the static
spatial path crossed the private static EGNN at `N=8192,k=64`. A 100-repeat
confirmation measured 11.359 ms for static spatial attention versus 11.672 ms
for EGNN, a 1.03x speed advantage. At `k=128`, the result was
11.359/25.407 ms, or 2.24x faster. Measured working allocation plus the
prebuilt EGNN edge tensor was 121.58/753.00 MiB at `k=64` and
121.58/1506.56 MiB at `k=128`, corresponding to 6.19x and 12.39x lower memory.

The coordinate-updating spatial path did not beat EGNN at `k=64`: it measured
12.093/11.672 ms, 3.6% slower using EGNN latency as the denominator. It crossed
by the exploratory `k=80` point (12.093/15.587 ms, 1.29x faster) and reached
2.10x at `k=128`.

## Kernel

For graph-centered and RMS-normalized position `x` and head scale `a`, define
`t = sqrt(2a) x` and

```text
phi_a(x) = exp(-a ||x||^2)
           [1, tx, ty, tz,
            tx^2/sqrt(2), ty^2/sqrt(2), tz^2/sqrt(2),
            tx*ty, tx*tz, ty*tz].
```

The head kernel is

```text
phi_a(x_i) dot phi_a(x_j)
  = exp(-a(||x_i||^2 + ||x_j||^2)) (1 + z + z^2/2),
z = 2a x_i dot x_j.
```

It is strictly positive because `1 + z + z^2/2` has no real root, and it is
O(3)-invariant because it depends only on norms and a dot product. Graph
centering supplies translation invariance. The implementation adds this
kernel to the incumbent content/vector kernel through graph-segmented
sufficient statistics. With fixed feature width 10, head count, and value
width, its transport cost and storage are linear in node count.

The four heads use fixed log-spaced scales `[0.125, 0.25, 0.5, 1.0]`. The
static GGG and static spatial models have identical trainable parameter counts
(153,081), total parameter counts (153,285), parameter bytes (613,140), and
state SHA-256
`dc4987e9e345acc34f2bc649bb5c7aeaa26a3eaba48d8baa6b6aacd6b5458511`.
The scale buffer is nonpersistent. In the retained first-run JSON,
`model_parameters` likewise means trainable parameters; the final benchmark
records that semantic explicitly and also records total parameters.

## Correctness

- A materialized dense reference matches the factorized spatial transport
  below `1e-10` in float64 with and without key balancing, for single- and
  multi-graph batches.
- Direct full-model checks cover node and graph scalar, vector, and tensor
  outputs under proper rotations, reflections, translations, within-graph
  permutations, and batch isolation. Optional updated positions and finite
  train-path gradients also pass.
- Matched default and explicit-off models have byte-identical state tensors
  and outputs. Matched opt-in spatial models retain the same state tensors but
  execute a measurably different transport.
- `scripts/check.sh fast` passed 396 tests at 89.35% coverage. CUDA bf16 and
  fp32 smoke gates passed.

## Registered performance result

The primary grid used one graph per forward, widths 64/91 for
attention/EGNN, three layers, four attention heads, model and graph seed
20260723, five warmups, and fifteen synchronized repeats.

| N | static spatial | dynamic spatial | EGNN k=128 | static/EGNN |
|---:|---:|---:|---:|---:|
| 128 | 6.333 ms | 7.470 ms | 0.719 ms | 8.81 |
| 512 | 6.362 ms | 7.524 ms | 1.360 ms | 4.68 |
| 2,048 | 6.366 ms | 7.472 ms | 5.462 ms | 1.17 |
| 8,192 | 11.354 ms | 12.070 ms | 24.833 ms | 0.457 |

The candidate latency is independent of the EGNN edge multiplier because the
candidate receives no edge tensor and is measured once per node count. EGNN
latency and memory grow with `E=kN`.

The first implementation was retained in `gpu-benchmark.json`. Inspection
found two spatial-only overheads: fixed scale validation caused GPU-to-CPU
synchronization inside every layer, and denominator/value spatial statistics
used separate reductions. Removing the synchronization and sharing one
transport reduction improved the `N=8192` static path from 12.374 to
11.354 ms (8.2%) and the dynamic path from 13.167 to 12.070 ms (8.3%). The
optimized result was rerun after adding a nonfinite-output preflight.

At `N=8192`, the confirmation results were:

| path | k / edges | median | peak delta plus edge |
|---|---:|---:|---:|
| current GGG | no edges | 10.029 ms | 120.33 MiB |
| spatial static | no edges | 11.359 ms | 121.58 MiB |
| spatial dynamic | no edges | 12.093 ms | 121.67 MiB |
| private static EGNN | 64 / 524,288 | 11.672 ms | 753.00 MiB |
| private static EGNN | 80 / 655,360 | 15.587 ms | 939.97 MiB |
| private static EGNN | 96 / 786,432 | 18.829 ms | 1128.72 MiB |
| private static EGNN | 128 / 1,048,576 | 25.407 ms | 1506.56 MiB |

Relative to current GGG, static spatial overhead was 1.133x latency and 1.010x
memory; dynamic spatial overhead was 1.206x and 1.011x. Both pass the
registered 2.5x diagnostic bound.

## Interpretation boundary

This is a systems result for one GPU, one dtype, fixed model shapes, synthetic
coordinates, one-graph forward execution, and a private near-parameter-matched
EGNN. The candidate and EGNN do not execute the same equations or topology.
Omitting edges discards arbitrary graph adjacency: two examples with identical
features and coordinates but different omitted adjacency are indistinguishable.

The run excludes neighbor construction, transfer, backward timing, optimizer
state, multi-graph throughput, QM9 training or test evaluation, accuracy,
forces, chirality, and molecule/protein/point-cloud generalization. In
particular, the result does not repair the previously observed QM9 accuracy
gap. Low-edge and small-node EGNN remains substantially faster, and at
`N=8192,k=4` also uses less working memory.

## Reproduction

```bash
scripts/check.sh fast
scripts/check.sh gpu
.venv/bin/python scripts/benchmark_sparse_scaling.py \
  --edge-free-spatial-grid \
  --sizes 128 512 2048 8192 \
  --edge-multipliers 4 16 64 128 \
  --device cuda --seed 20260723 --model-seed 20260723 \
  --warmup 5 --repeats 15 --max-wall-seconds 300 \
  --metrics-out artifacts/edge-free-spatial-linear-20260723/gpu-benchmark-optimized.json
```
