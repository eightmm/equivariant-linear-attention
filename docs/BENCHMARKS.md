# Benchmarks

The benchmark covers the single `factorized_moment` implementation under real
batched semantics rather than treating a molecular mini-batch as one graph.

```bash
uv run python scripts/bench_attention.py \
  --device cuda \
  --graphs 1 8 32 128 \
  --nodes-per-graph 16 32 \
  --iters 20 \
  --warmup 8
```

Recorded columns are graph count, nodes per graph, total nodes, forward versus
forward/backward pass, mean latency, peak allocated CUDA memory, and
implementation name. Use `--dtype bf16` and `--compile` for additional lanes.

Do not compare numbers from removed implementations with this benchmark: their
semantics and batch shapes differ. No performance claim is made until eager and
compiled batched outputs/backward are checked on identical inputs and the
environment is recorded.
