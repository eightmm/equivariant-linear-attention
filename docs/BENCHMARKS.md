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

`--routing` accepts `ggg`, `lgl`, and `lll`; `--memory-count` accepts the
registered `1`, `4`, and `8` values. Interacting multi-memory transport is
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
