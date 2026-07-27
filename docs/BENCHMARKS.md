# Benchmarks

The standalone microbenchmark covers the single `factorized_moment`
implementation under real batched semantics. Its defaults remain the public
`ggg`, `M=1`, interaction-off, radial-trace-off configuration.

```bash
uv run python scripts/bench_attention.py \
  --device cuda \
  --graphs 1 8 32 128 \
  --nodes-per-graph 16 32 \
  --iters 20 \
  --warmup 8
```

Registered local/global routes and memory/radial lanes use the same command and
the same `EquivariantAttention` class. For example:

```bash
uv run python scripts/bench_attention.py \
  --device cuda \
  --routing lgl \
  --memory-count 4 \
  --memory-interaction \
  --radial-trace \
  --graphs 1 8 32 \
  --nodes-per-graph 16 32
```

`--routing` accepts `ggg`, `lgg`, `ggl`, `lgl`, and `lll`;
`--global-transport-mode` accepts `learned`, `uniform`, and `none`; and
`--memory-count` accepts the registered `1`, `4`, and `8` values. Interacting
multi-memory transport is
registered only for the middle global stage of `lgl`, so combinations such as
`lll --memory-interaction` are rejected by the model configuration validator.
The local and memory geometry controls are exposed as `--local-cutoff`,
`--num-rbf`, `--memory-assignment-temperature`, `--memory-assignment-scale`,
and `--memory-interaction-cutoff`.

Recorded columns are graph count, nodes per graph, total nodes, forward versus
forward/backward pass, mean latency, peak allocated CUDA memory, implementation
name, route/local-head layout, local cutoff/RBF configuration, memory count and
interaction controls, radial-trace state, dtype, and compile configuration.
Use `--dtype bf16` and `--compile` for additional lanes. The `compiled` column
records whether compilation was actually applied: the existing behavior
compiles forward inference only, so forward/backward rows remain eager even
when `--compile` is requested.

On the verified PyTorch 2.12.1 x86_64 environment, CPU Inductor code generation
fails on the model's `index_add` atomic scatter in both `default` and
`reduce-overhead` modes. Eager CPU is the supported lane here. CUDA Inductor
forward smoke passes for both `ggg` and interacting `lgl`/M4; this is a backend
compatibility statement, not a compile speedup claim.

Do not compare numbers from removed implementations with this benchmark: their
semantics and batch shapes differ. Route, local cutoff/RBF, interacting
`M=4/8`, radial trace, and fixed-versus-graph-size-scaled positive-baseline
accuracy comparisons belong in matched training/instrumented runs.

## Bounded transport/baseline CPU screen (2026-07-18)

A deterministic 64-row synthetic, 20-step CPU screen exercised learned
`ggg/lgg/ggl/lgl`, exact-uniform LGL, disabled-global LGL, and the private
static EGNN runner on identical split/model seeds. All seven arms completed
with finite gradients and validation metrics, emitted matching split hashes,
and kept `test_evaluated=false`. The factorized controls retained the same
12,480 trainable parameters and state-schema hash.

On the inspected seven-node graph, the learned middle LGL head was already
nearly uniform (entropy/log-N 0.999998; effective support 6.999973), while the
analytic control reported exactly 1.0 and 7.0 to floating-point tolerance.
Local diagnostics normalized each receiver/head row to within 1e-7 and reported
degree 4--7 and mean entropy/log-degree 0.92095. These numbers validate the
diagnostic and control semantics; the tiny synthetic screen is not an accuracy
or throughput comparison.

The complete nine-arm reproduction, including the sequential wider pair, is:

```bash
bash scripts/run_bounded_control_screen.sh \
  artifacts/egnn-matched-baseline-development-20260718
```

The recorded run completed in 15.80 seconds of outer wall time on CPU, measured
by GNU `time` around the exact tracked command. Its one-shot arm times are not
promoted throughput measurements.

A sequential same-split width screen also completed for LGL H=64 and static
EGNN H=91. For the synthetic eight-channel input it reported 152,889 and
151,792 trainable parameters. On the registered 11-channel QM9 input the counts
are LGL 153,285 total / 153,081 trainable and EGNN 152,065 total/trainable, a
0.664% trainable-count gap. At that 2026-07-18 stage no QM9 or GPU run was made
because the fresh 25-GPU-minute approval gate was still closed; the later
approved result is recorded below.

## Registered transport mechanism result (2026-07-19)

The approved validation-only study used local PyG QM9 `gap`, the frozen
110k/10k/10k random-row split, FP32, batch size 64, 2,000 updates, LGL width 64,
and model seeds 41--45. Source, data, split, state-schema, and matched-seed
initialization hashes agreed; all values were finite and every arm recorded
`test_evaluated=false`.

| seed | learned MAE | uniform MAE | none MAE |
|---:|---:|---:|---:|
| 41 | 0.535650 | 0.548641 | 0.681638 |
| 42 | 0.551381 | 0.550999 | 0.706867 |
| 43 | 0.519668 | 0.542929 | 0.680377 |
| 44 | 0.514849 | 0.556477 | 0.706019 |
| 45 | 0.456891 | 0.474836 | 0.684202 |
| mean | 0.515688 | 0.534776 | 0.691821 |

The frozen rule required every accuracy and efficiency condition. Ratios below
are candidate divided by baseline; positive MAE improvement favors the
candidate.

| candidate vs baseline | mean MAE improvement | improving seeds | worst improvement | elapsed ratio | memory ratio | result |
|---|---:|---:|---:|---:|---:|---|
| learned vs uniform | 0.019088 | 4/5 | -0.000383 | 1.175 | 1.359 | fail memory |
| learned vs none | 0.176133 | 5/5 | 0.145989 | 1.512 | 1.414 | fail elapsed/memory |
| uniform vs none | 0.157044 | 5/5 | 0.132998 | 1.288 | 1.040 | fail elapsed |

Thus the runs provide matched validation-accuracy evidence for global
transport and learned selectivity at this update budget, but fail the registered
accuracy/efficiency promotion rule. No transport mode was locked, `ggg learned`
remains the public default, and the conditional width-91 private-EGNN arms were
not run. This is not an EGNN, test-set, cold-molecule, or final-training claim.

The six preceding 500-step screen arms were numerical checks only. Their
expanded LGL diagnostic sampled 32 validation graphs spanning 4--29 nodes and
covered both local layers and all four heads. Maximum receiver row-mass error
was below `2.1e-7`; aggregate mean entropy/log-degree was 0.8034 in layer 0 and
0.7946 in layer 2. Diagnostics were excluded from elapsed and peak-memory
metrics.

The budget-enforced runner completed 21 GPU arms in 819.2 wall seconds under a
1,500-second ceiling. The exact command and scalar outcome are recorded in
`docs/EXPERIMENTS.jsonl`; reproduce or inspect the immutable plan with:

```bash
uv run --locked python scripts/run_registered_transport_study.py \
  artifacts/transport-study-reproduction --dry-run
```

## Registered dynamic-coordinate/EGNN result (2026-07-19)

The independent coordinate packet used the same local QM9 `gap` data identity,
110k/10k random-row warm validation protocol, FP32, batch size 64, and three
layers. Its seed-42/500-step screen admitted `ggg` attention and dynamic EGNN;
`lgl` dynamic regressed by 0.056577 eV and did not advance. Confirmation paired
static and dynamic initializations at seeds 41--45 for 2,000 updates.

| seed | attention static | attention dynamic | improvement | EGNN static | EGNN dynamic | improvement |
|---:|---:|---:|---:|---:|---:|---:|
| 41 | 0.512575 | 0.510856 | 0.001718 | 0.429862 | 0.395129 | 0.034733 |
| 42 | 0.586021 | 0.605571 | -0.019550 | 0.431229 | 0.417954 | 0.013275 |
| 43 | 0.623774 | 0.629016 | -0.005241 | 0.421278 | 0.372225 | 0.049053 |
| 44 | 0.598871 | 0.589326 | 0.009545 | 0.396939 | 0.449499 | -0.052560 |
| 45 | 0.593489 | 0.592905 | 0.000584 | 0.365354 | 0.417335 | -0.051981 |
| mean | 0.582946 | 0.585535 | -0.002589 | 0.408932 | 0.410428 | -0.001496 |

Positive improvement favors dynamic coordinates. The frozen rule required
every listed gate:

| family | mean improvement | improving seeds | worst improvement | elapsed ratio | memory ratio | result |
|---|---:|---:|---:|---:|---:|---|
| attention `ggg` | -0.002589 | 3/5 | -0.019550 | 1.179 | 1.010 | fail mean gain |
| private EGNN | -0.001496 | 3/5 | -0.052560 | 1.456 | 1.008 | fail mean/worst/elapsed |

Neither family is promoted. Attention keeps `coordinate_updates=False` as the
public default; the dynamic private EGNN remains an equation-level diagnostic
control. All ten dynamic confirmation arms had active nonzero coordinate
gradients. Maximum observed per-layer step was `0.25000003 Angstrom`, maximum
graph-centroid drift was `4.92e-7 Angstrom`, and every run kept
`test_evaluated=false`. These are latent-coordinate and validation findings,
not physical geometry or official EGNN reproduction claims.

The budget-enforced runner completed 26 GPU arms in 944.3 wall seconds under a
1,500-second ceiling. Inspect the frozen packet with:

```bash
uv run --locked python scripts/run_registered_coordinate_study.py \
  artifacts/coordinate-study-reproduction --dry-run
```

### Competitiveness assessment against the private EGNN

The coordinate ablation answered whether latent coordinate motion helped each
family; it did not establish attention/EGNN parity. Under the same 2,000-update
QM9 validation harness, the current static GGG attention remains substantially
behind the near-parameter-matched private static EGNN:

| path | trainable parameters | mean final-batch train loss | mean validation MAE | median recorded elapsed | peak CUDA memory |
|---|---:|---:|---:|---:|---:|
| static GGG attention | 153,081 | 0.377288 | 0.582946 eV | 47.934 s | 245,148,672 B |
| static LGL learned | 153,081 | 0.328162 | 0.515688 eV | 41.545 s | 181,456,896 B |
| private static EGNN | 152,065 | 0.168751 | 0.408932 eV | 7.148 s | 269,261,824 B |

GGG trails EGNN by 0.174014 eV and takes 6.706x its recorded elapsed time,
while using 9.0% less peak memory. The stronger LGL result narrows the
descriptive MAE gap to 0.106756 eV and uses 32.6% less peak memory, but still
takes 5.812x the elapsed time. LGL and EGNN values come from separate registered
studies and source hashes, so that row-to-row comparison is diagnostic rather
than a paired promotion result. `train_loss` is the last normalized-target
minibatch loss, not a train-set mean; its large gap is an optimization or
capacity warning, not proof of a particular cause.

Code inspection suggests five falsifiable bottlenecks:

1. The registered local attention arms froze `learn_local_radial_gate=false`.
   Their 16 RBF values therefore do not learn a radial logit; distance affects
   the weight mainly through the fixed cosine cutoff. EGNN instead puts raw
   squared distance directly inside a learned receiver/sender edge MLP.
2. Attention normalizes each receiver row before aggregation. EGNN sums edge
   messages, directly retaining neighborhood mass and coordination. The moment
   scalar updater currently receives neither the local denominator nor an
   explicit degree/mass invariant.
3. The attention content has the separable form of a scalar pair weight times a
   sender value. EGNN can change edge-message content jointly with receiver,
   sender, and distance features.
4. The private EGNN uses every directed same-graph nonself pair, whereas local
   attention uses a 2.5-Angstrom cutoff plus self edges. The middle factorized
   global block compresses geometry into bounded degree-2 moments and may not
   recover all missing pairwise distance distinctions.
5. Attention residual branches start at `0.1/sqrt(num_layers)`, and recorded
   final gradient norms sit at the clipping boundary. Periodic fixed-train-probe
   loss and pre-clip norms are needed before attributing the train-loss gap only
   to representation.

These are inferences from the implementation and registered metrics. They are
not causal findings. The highest-priority candidate is a sparse, invariant
receiver/sender/RBF edge-content branch inside the existing local moment block,
with explicit degree or attention-mass invariants, while retaining factorized
global attention. Coordinate updates, multi-memory interaction, higher moment
degree, and indiscriminate width/depth increases remain downstream choices.

## Scaling-aware EC-LGL result (2026-07-22)

This packet implemented end-to-end optional sparse-edge plumbing and a
per-local-layer edge-conditioned equivariant sum inside the existing public
`EquivariantAttention`. It separately measured identical global-kernel
factorization, same-edge model constants, and the different-edge-density regime
that motivated the work. Neighbor-list construction was excluded from all
timings; the convenience QM9 radius builder is a dense per-molecule scan.

For the exact same normalized float64 kernel on the RTX PRO 6000 Blackwell:

| nodes | dense time | factorized time | dense peak delta | factorized peak delta |
|---:|---:|---:|---:|---:|
| 1024 | 0.3777 ms | 0.8638 ms | 41.94 MB | 0.85 MB |
| 2048 | 0.5036 ms | 1.6544 ms | 167.77 MB | 1.69 MB |
| 4096 | 3.3744 ms | 0.7611 ms | 671.09 MB | 3.38 MB |

Maximum dense/factorized output error was `2.406e-15`; the factorized path
materialized no `N x N` tensor. The first measured runtime crossover was 4096
nodes. Its negative last-window timing slope reflects finite GPU occupancy
variation and is not a subconstant-complexity claim.

The measured model path used content-validated synthetic edge tensors; one-time
validation and neighbor construction were excluded. At fixed degree 16, full
EC-LGL latency changed from 5.135 ms at 32 nodes to 5.595 ms at 4096 nodes.
Static EGNN on the same sparse edges was faster at every size (0.599 to
1.402 ms). Against complete-edge EGNN, however, the first
descriptive crossover appeared at 512 nodes:

| 512-node system | candidate edges | median time | peak CUDA delta |
|---|---:|---:|---:|
| EC-LGL, degree 16 | 8,192 | 5.243 ms | 11.71 MB |
| static EGNN, complete | 262,144 | 5.577 ms | 391.59 MB |

This crossover does not compare identical computations: EC-LGL caps expensive
local geometry and retains one factorized global stage, whereas EGNN processes
every local pair. At the same complete 512-node edge set, EC-LGL remained
slower (6.675 versus 5.561 ms). The valid conclusion is therefore a memory and
edge-regime crossover, not a universally faster layer.

The repeated seed-42/500-step QM9 screen then rejected the frozen EC operator:

| arm | run 1 | run 2 | mean validation MAE |
|---|---:|---:|---:|
| EC-LGL | 0.743185 | 0.861202 | 0.802194 eV |
| static LGL | 0.708419 | 0.715938 | 0.712178 eV |

The `+0.090015 eV` mean regression exceeded the `+0.020 eV` confirmation
ceiling, so no 2,000-step or EGNN accuracy confirmation ran. The parameter
ratio gate passed (158,537 candidate versus 152,065 EGNN trainable parameters),
and the symmetry/sparse-path gates passed. The large EC repeat range and 92%
clip fraction suggest optimization sensitivity, but do not prove its cause.
The feature stays opt-in and no default changed.

Reproduce the scaling harness with:

```bash
uv run python scripts/benchmark_sparse_scaling.py --device cuda \
  --metrics-out artifacts/sparse-scaling.json
```

Frozen scope, exact JSON results, screen arms, verification, and limitations are
under `artifacts/ec-lgl-sparse-scaling-20260722/`.

## Exact `E=kN` same-edge scaling (2026-07-23)

The follow-up benchmark removes the earlier fixed-degree versus complete-edge
ambiguity. For every `(N,k)` cell it constructs one deterministic directed
graph with exactly `E=kN` candidate edges, including one self edge and exactly
`k` incoming candidates per node, and passes that identical edge tensor to
width-64 EC-LGL and width-91 private static EGNN. Graph construction is `O(E)`
and excluded from timing. Model parameters are 158,537 and 152,065.

All 24 registered cells for `N={128,512,2048,8192}` and
`k={4,8,16,32,64,128}` completed on the RTX PRO 6000 Blackwell. The full grid
used three warmups and seven synchronized repeats. A separate high-density
confirmation fixed model seed 20260723 and varied three topology seeds, using
ten warmups and 31 repeats per model:

| N | k | candidate edges | EC-LGL mean | static EGNN mean | EC/EGNN range | EC wins/seeds |
|---:|---:|---:|---:|---:|---:|---:|
| 8,192 | 32 | 262,144 | 8.615 ms | 5.400 ms | 1.593--1.597 | 0/3 |
| 8,192 | 64 | 524,288 | 11.408 ms | 11.559 ms | 0.985--0.988 | 3/3 |
| 8,192 | 128 | 1,048,576 | 17.127 ms | 25.365 ms | 0.673--0.678 | 3/3 |

The preregistered no-crossover prediction was falsified; the first confirmed
same-edge crossover is `N=8192,k=64`. The margin at `k=64` is small, so the
stronger claim is the `k=128` high-density win. At seed 20260723 and `k=128`,
peak CUDA allocation delta was about 1.410 GB for EC-LGL and 1.563 GB for EGNN.
A local `k={32,64,128}` fit gave 10.832 versus 25.413 ms per million candidate
edges, but its negative EGNN intercept forbids extrapolation to zero edges.

This crossover is expected from the computation mix: EC-LGL uses two local
edge stages plus one exact factorized `O(N)` global stage; EGNN uses three edge
message stages. At low density the EC-LGL fixed/global/pointwise overhead still
dominates by a large factor. At a fixed 262,144 total edges, increasing nodes
from 2,048 to 8,192 raised EC-LGL latency 1.281x while EGNN changed 0.986x,
which isolates the extra per-node work. Profiling shows EC-LGL cost distributed
over gather, scatter, concatenation, GEMM and elementwise kernels, so optimizing
only `index_add` is insufficient.

The first generator revision used one affine traversal over globally flattened
pairs. It met count/uniqueness checks but produced receiver-degree skew, so a
new RED test required exact degree `k`; the original outputs are retained with
an `affine-exploratory` label and every authoritative cell was rerun. This
amendment occurred after inspecting initial timings and is explicitly part of
the record.

Reproduce the primary grid with:

```bash
uv run python scripts/benchmark_sparse_scaling.py --edge-multiplier-grid \
  --sizes 128 512 2048 8192 --edge-multipliers 4 8 16 32 64 128 \
  --device cuda --seed 20260723 --model-seed 20260723 \
  --warmup 3 --repeats 7 \
  --max-wall-seconds 120 --metrics-out artifacts/exact-edge-grid.json
```

The result is forward-only and excludes graph construction, backward/training,
multi-graph batching and task accuracy. Synthetic topology does not establish
molecule, protein or point-cloud quality. Raw grids, seed confirmation,
profile, analysis and provenance are under
`artifacts/edge-multiplier-scaling-20260723/`.

## Edge-free spatial-linear scaling (2026-07-23)

This follow-up removes the candidate edge path entirely. Each all-global head
adds a fixed ten-dimensional degree-two Gaussian-Taylor feature map at one
log-spaced spatial scale. Its dot product is positive and O(3)-invariant, while
graph centering/RMS normalization supplies translation invariance. The spatial
kernel is added through segmented sufficient statistics, so neither an
`edge_index` nor an `N x N` pair tensor is materialized. The option is off by
default, adds no learned parameters, and preserves the incumbent state hash
under matched initialization.

The registered forward-only CUDA grid used width-64/four-head/three-layer GGG,
static and coordinate-updating spatial GGG, and the width-91/three-layer private
static EGNN. It covered `N={128,512,2048,8192}` and EGNN
`k={4,16,64,128}` with five warmups and fifteen synchronized repeats. All
cells completed. Graph construction was excluded; EGNN received deterministic
prebuilt receiver-regular `E=kN` tensors, while the attention candidates
received no topology. Trainable parameter counts were 153,081 for static
attention, 153,475 for coordinate-updating attention, and 152,065 for EGNN;
the corresponding total counts were 153,285, 153,679, and 152,065.

After the first grid, two spatial-only overheads were removed: fixed scale
validation no longer synchronizes GPU values back to the CPU inside each
layer, and spatial denominator/value statistics share one transport reduction.
The original grid is retained. At `N=8192`, the optimized static/dynamic paths
improved from 12.374/13.167 ms to 11.354/12.070 ms in the final 15-repeat grid.

A 20-warmup/100-repeat confirmation then measured the high-density crossover:

| N | path | k / edges | median latency | peak delta plus edge index |
|---:|---|---:|---:|---:|
| 8,192 | current GGG | no edges | 10.029 ms | 120.33 MiB |
| 8,192 | spatial static | no edges | 11.359 ms | 121.58 MiB |
| 8,192 | spatial dynamic | no edges | 12.093 ms | 121.67 MiB |
| 8,192 | private static EGNN | 64 / 524,288 | 11.672 ms | 753.00 MiB |
| 8,192 | private static EGNN | 80 / 655,360 | 15.587 ms | 939.97 MiB |
| 8,192 | private static EGNN | 96 / 786,432 | 18.829 ms | 1128.72 MiB |
| 8,192 | private static EGNN | 128 / 1,048,576 | 25.407 ms | 1506.56 MiB |

Thus static spatial attention was 1.03x faster than EGNN at `k=64` and 2.24x
faster at `k=128`, while its measured working-plus-edge memory was 6.19x and
12.39x lower. The coordinate-updating path was 3.6% slower at `k=64`, crossed
by `k=80` (1.29x faster), and was 2.10x faster at `k=128`. Relative to current
GGG, static spatial overhead at `N=8192` was 1.13x latency and 1.01x memory;
dynamic spatial overhead was 1.20x and 1.01x. The registered 2.5x overhead
bound passed.

The benefit is confined to sufficiently large, edge-dense workloads. At
`N<=2048`, EGNN remained faster even at `k=128`; at `N=8192,k=4`, EGNN was
about 11.9x faster than static spatial attention and also used less working
memory. The comparison does not preserve arbitrary omitted adjacency, does not
time neighbor construction, and says nothing about backward/training,
accuracy, forces, or molecule/protein/point-cloud generalization.

Reproduce the registered grid with:

```bash
uv run python scripts/benchmark_sparse_scaling.py --edge-free-spatial-grid \
  --sizes 128 512 2048 8192 --edge-multipliers 4 16 64 128 \
  --device cuda --seed 20260723 --model-seed 20260723 \
  --warmup 5 --repeats 15 --max-wall-seconds 300 \
  --metrics-out artifacts/edge-free-spatial-grid.json
```

Raw initial/optimized grids, the 100-repeat crossover confirmation,
correctness gates, interpretation, and provenance are under
`artifacts/edge-free-spatial-linear-20260723/`.

## Full train-step edge-free scaling (2026-07-23)

The forward-only result above did not establish training efficiency. A
separate eager-FP32 grid therefore timed the complete
`zero_grad -> forward -> MSE -> backward -> AdamW.step` operation with
optimizer state initialized during five warmups. It used twenty synchronized
repeats, isolated each model for absolute peak CUDA allocation, and compared
edge-free static/dynamic spatial attention with the private static EGNN at
`N={512,2048,8192}` and exact EGNN receiver degree `k={16,64,128}`.
Graph construction and host-to-device transfer were measured separately.

The preregistered first grid completed all nine cells but falsified the static
latency hypothesis. At `N=8192,k=128`, static attention took `109.884 ms`
versus EGNN `62.601 ms` (`1.755x`) while using `1.252 GB` versus `8.884 GB`
peak allocation (`0.141x`). A post-outcome three-step profile attributed
`254.573 ms` aggregate device time to duplicate-index `IndexBackward` in the
single-graph attention path.

The only optimization changed one-graph graph-summary expansion from
`summary[batch]`, where every index is zero, to the mathematically identical
stride-zero broadcast of `summary[0]`. Multi-graph execution remains indexed.
Focused mathematical/equivariance/gradient tests and the repository fast gate
passed before the full grid was rerun. The optimized results were:

| N | k | static ms | dynamic ms | EGNN ms | static / EGNN | dynamic / EGNN | static peak / EGNN |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 16 | 21.806 | 26.139 | 2.181 | 9.997 | 11.984 | 1.039 |
| 512 | 64 | 21.740 | 26.052 | 2.606 | 8.342 | 9.996 | 0.394 |
| 512 | 128 | 21.830 | 26.129 | 3.954 | 5.521 | 6.608 | 0.227 |
| 2,048 | 16 | 22.002 | 26.303 | 2.556 | 8.608 | 10.291 | 1.079 |
| 2,048 | 64 | 21.991 | 26.254 | 6.682 | 3.291 | 3.929 | 0.312 |
| 2,048 | 128 | 22.040 | 26.403 | 14.008 | 1.573 | 1.885 | 0.160 |
| 8,192 | 16 | 25.455 | 36.284 | 6.505 | 3.913 | 5.577 | 1.079 |
| 8,192 | 64 | 25.482 | 36.330 | 30.000 | 0.849 | 1.211 | 0.274 |
| 8,192 | 128 | 25.471 | 36.319 | 62.626 | 0.407 | 0.580 | 0.138 |

Thus static edge-free attention is 1.18x faster at `N=8192,k=64` and 2.46x
faster at `k=128`, while using 3.65x and 7.26x less peak allocation. The
coordinate-updating path crosses only at `k=128`, where it is 1.72x faster and
uses 7.10x less peak allocation. The optimized profile no longer contains
`IndexBackward`; `bmm` becomes the largest recorded operator.

This optimization and rerun followed observation of the failed first grid, so
they are explicitly post-outcome evidence rather than a preregistered
confirmation. The comparison uses one synthetic graph, one GPU, a private
same-harness EGNN, and a synthetic loss. It excludes task accuracy,
convergence, data loading, arbitrary graph topology, and domain
generalization. Raw grids, profiles, contracts, and review records are under
`artifacts/train-step-scaling-20260723/`.

Independent review identified one minor protocol deviation: the scope said
target construction was excluded, while the timed loss stage creates the
one-element constant target with `torch.full_like`. The same scalar operation
is included in every arm, so raw timings and ratios are retained rather than
rewritten. The mismatch is recorded as `PD-001`; a future benchmark revision
must pass a preconstructed target.

## Registered EGNN-parity result (2026-07-20)

The confirmed packet kept static coordinates, width-64 LGL learned transport,
the QM9 `gap` random-row split, FP32, batch size 64, and test evaluation off.
It allowed three sequential architecture iterations and 3,600 cumulative
GPU-wall seconds. Each iteration first reran a seed-42/500-step static LGL
control; only finite, active, parameter-bounded candidates no more than 0.020
eV worse could advance.

| iteration | LGL screen | candidate screen | candidate minus LGL | outcome |
|---|---:|---:|---:|---|
| learned radial gate | 0.778593 | 0.759655 | -0.018938 | confirmed |
| pairwise content, `alpha=0.1` | 0.767847 | 0.840664 | +0.072816 | screen reject |
| pairwise content, `alpha=0` | 0.738009 | 0.712453 | -0.025556 | confirmed |

The pairwise branch adds 1,105 parameters (154,390 total versus 153,285 for
LGL and 152,065 for EGNN), and all 1,105 received finite nonzero gradients.
Because `alpha=0.1` worsened both the fixed train probe and validation while its
clip fraction was close to LGL, the final evidence-selected repair changed only
`alpha` initialization to zero. It preserved the exact incumbent first forward,
then learned `alpha=0.038083` by the 500-step screen.

The two admitted five-seed confirmations were:

| seed | radial | EGNN rerun A | EGNN minus radial | staged pairwise | EGNN rerun B | EGNN minus pairwise |
|---:|---:|---:|---:|---:|---:|---:|
| 41 | 0.493019 | 0.398689 | -0.094330 | 0.488625 | 0.481749 | -0.006875 |
| 42 | 0.497891 | 0.443284 | -0.054607 | 0.571414 | 0.436437 | -0.134977 |
| 43 | 0.540666 | 0.421719 | -0.118947 | 0.513891 | 0.420657 | -0.093234 |
| 44 | 0.504323 | 0.390677 | -0.113645 | 0.523597 | 0.410540 | -0.113058 |
| 45 | 0.461642 | 0.451627 | -0.010015 | 0.447514 | 0.441955 | -0.005559 |
| mean | 0.499508 | 0.421199 | -0.078309 | 0.509008 | 0.438268 | -0.070741 |

Neither candidate reached the absolute 0.398932 eV threshold, won a paired
seed, or respected the -0.020 eV worst-regression floor. Radial and staged
pairwise took 5.831x and 6.238x the median EGNN elapsed time; their median peak
memory ratios were 0.676x and 0.762x. Candidate update clip fractions averaged
0.9737 and 0.9729. Staged pairwise's final `alpha` values were
`[-0.105471, 0.088824, -0.139261, -0.095576, -0.094521]`, so the additive
message direction itself was seed-unstable.

The packet stopped after its third architecture iteration at 850.7 cumulative
GPU-wall seconds. No default changed; pairwise content and its initialization
remain explicit experimental controls. Exact scalar outcomes and commands are
in `docs/EXPERIMENTS.jsonl`, while the frozen scope and compact summaries live
under `artifacts/egnn-parity-20260720/`.

There is also a material reproducibility limitation. Iterations 1 and 2 reran
the same LGL screen command with identical source, data/split, and initial-state
hashes, but obtained 0.778593 and 0.767847 eV and different final-state hashes.
This proves the current seeded CUDA lane is not bitwise deterministic; CUDA
atomic reductions are a plausible implementation source, not yet a confirmed
cause. The large confirmation failures remain larger than this observed drift,
but future claims near 0.01 eV require a deterministic or repeated-run gate.

## Registered M=1 routing result (2026-07-17)

On clean commit `a8bda61`, the validation-only QM9 `gap` comparison used the
registered random-row warm split, 2,000 FP32 steps, and paired model seeds
41--43. Only routing/local-head layout changed; source, data, split, parameter
count (153,285), state schema, and each seed's initial state hash matched.

| seed | GGG MAE (eV) | LGL MAE (eV) | improvement (eV) |
|---:|---:|---:|---:|
| 41 | 0.530218 | 0.507433 | 0.022785 |
| 42 | 0.589037 | 0.572078 | 0.016958 |
| 43 | 0.631137 | 0.512559 | 0.118578 |

Mean improvement was 0.052774 eV, all three seeds improved, and the worst
improvement was 0.016958 eV. This passes the registered 0.010 eV / 2-of-3 /
-0.020 eV rule. No test metric was evaluated.

The synchronized eager-FP32 resource gate used 64 graphs, 18 and 29 nodes,
20 warm-up plus 50 measured iterations, and five fresh processes. Relative to
GGG, LGL forward was 7.1--8.7% faster and forward/backward was 13.7--14.8%
faster. Its largest peak-memory increase was 6.3%; other lanes used less
memory. The local candidate builder remains `O(sum_g N_g^2)` but its exact
same-graph Cartesian expansion is vectorized across the batch.

This is adaptive three-seed evidence for this one QM9 validation protocol, not
a test-set, cold-molecule, EGNN, or public-default claim. Interacting M=4/M=8
remains blocked by the independent Stage-0 mechanism gate.

The helpers in `equivariant_attention.diagnostics` are pure and bounded: they
detach inputs, return JSON-safe Python scalars, do not mutate model state, and
reject nonfinite or invalid inputs. Attention summaries report entropy/log-N,
mean maximum weight, effective support, and column CV; kernel helpers report
component quantiles and bounded `beta`/`gamma` summaries. Effective rank is
explicitly opt-in and rejects matrices larger than its configured limit because
it performs a full SVD.

For the registered synthetic kernel-size lane, run
`uv run python scripts/probe_kernel_scaling.py`. Its default sizes are
`16,32,64,128,512,2048`; `--block-rows` bounds the largest exact statistics
block and `--probe-rows` bounds the differentiable value/output probe. Runtime
is explicitly synchronized on CUDA. This probe compares formulas and scaling
diagnostics, not model throughput or accuracy, and never computes effective
rank.

Local transport retains `O(E)` edge state but the core fallback may use
`O(N^2)` candidate search. Global structured attention avoids an `N x N`
matrix; fixed-memory HEMM adds `O(NM + M^2)` terms. These are implementation
structure statements, not measured end-to-end performance claims.

`uv run python scripts/ml_smoke.py cuda auto compile` uses the same nontrivial
batch for eager and compiled inference and checks output equality on the CUDA
lane. Tensor-dependent input validation may still create graph breaks; no
fullgraph claim is made.

## ATOM3D-LBA official ID30 validation (2026-07-24)

The first full held-out protein-ligand study used all 3,507 official ID30 train
complexes and 466 validation complexes at pinned dataset revision
`f93dd2d150a47c270f624620f84e07451a158705`. Test access was structurally
disabled. Raw features, target transformation, ligand readout, batches, and
32,303,245 directed sparse candidates were identical across arms.

| arm | params | validation MAE | validation RMSE | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|
| gated + grouped LGL | 168,815 | 1.254561 | **1.550035** | **0.637805** | 0.610718 |
| previous LGL | 161,541 | 1.297191 | 1.592008 | 0.622750 | **0.615140** |
| private static EGNN | 167,260 | 1.349694 | 1.692812 | 0.537693 | 0.532804 |
| train-mean constant | — | 1.614885 | 2.039959 | — | — |

The candidate-minus-incumbent RMSE delta was `-0.041973 pK`, passing the
registered one-seed `-0.02 pK` gate. A 10,000-replicate paired bootstrap over
validation complexes yielded `[-0.130138, +0.043411] pK`, so it does not yet
establish a stable incumbent advantage. Candidate minus private EGNN was
`-0.142776 pK`, with interval `[-0.230663, -0.055042]`.

Candidate/incumbent/private-EGNN median synchronized steps were
24.726/26.517/9.674 ms at batch size 16, and peak CUDA allocations were
1.732/1.253/1.545 GB. Every arm clipped more than 99% of updates. The complete
protocol, external published context, hashes, and interpretation are in
`docs/LBA_ID30_VALIDATION_20260724.md`.

## ATOM3D-LBA matched 35-epoch multi-seed result (2026-07-27)

The current squared-RBF gated-plus-grouped LGL was compared with the preceding
LGL on the same official ID30 train/validation data, strict FP32 CUDA, and
model/data-order seeds 41--43. Each arm received exactly 35 epochs / 7,700
updates. Test remained structurally inaccessible.

| arm | mean validation RMSE (pK) | sample SD | median step | median peak CUDA |
|---|---:|---:|---:|---:|
| gated + grouped LGL | **1.598765** | 0.023390 | **27.644 ms** | 1,728,283,648 B |
| previous LGL | 1.619865 | 0.018482 | 29.578 ms | **1,258,568,192 B** |

Paired candidate improvements were `0.050701`, `0.005158`, and `0.007439 pK`;
the mean was `0.021099 pK` with `3/3` wins. The latency and memory ratios were
`0.93472x` and `1.37233x`, so every frozen accuracy/resource criterion passed.
The candidate also reduced mean pre-clip norm from `14.8994` to `12.4345`, but
both arms still clipped about 99% of updates.

This is exploratory validation evidence: the matched-epoch protocol was
selected after partial seed-41 curves existed while repairing the coordinator.
It is not a test, cold-target, or published-model comparison. Full provenance
and limitations are in `docs/LBA_MULTISEED_CONFIRMATION_20260727.md`.

## ATOM3D-LBA clipping-policy screen (2026-07-27)

The accepted candidate was trained on the full official ID30 train/validation
split for 20 fixed epochs at model/data-order seed 44. All policies shared raw
features, bound coordinates, target normalization, one precomputed topology,
initial parameters, batches, AdamW schedule, and 4,400 updates.

| global clip | last validation RMSE | best validation RMSE | clip fraction | mean effective scale |
|---:|---:|---:|---:|---:|
| 1 | 1.628645 | 1.617674 | 98.55% | 0.1719 |
| 10 | 1.611120 | **1.593766** | 38.18% | 0.8644 |
| none | **1.600802** | 1.598508 | 0% | 1.0000 |

No clipping improved the registered primary metric by `0.027843 pK` and passed
the frozen one-seed screen. Clip 10 improved `0.017524 pK`, just below the
`0.020 pK` threshold. The unclipped trajectory beat clip 1 in 15/20 validation
epochs and all final six epochs. Its latency and peak-allocation ratios were
`1.0101x` and `0.9988x`.

This is exploratory optimization evidence, not authorization to change the
default. The packet also found a one-edge topology-hash drift relative to the
preceding LBA study. It does not confound the within-packet comparison, but the
distance/tied-neighbour contract must be made cross-run deterministic before a
multi-seed confirmation. See `docs/LBA_GRADIENT_CLIPPING_20260727.md`.
