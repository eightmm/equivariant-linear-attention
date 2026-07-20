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
