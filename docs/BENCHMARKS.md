# BENCHMARKS

## Attention Microbenchmark

Command:

```bash
uv run python scripts/bench_attention.py --device cuda --nodes 128 512 2048 --iters 20 --warmup 8
```

Hardware: NVIDIA RTX PRO 6000 Blackwell Max-Q, CUDA 13 path, float32.
Model: `node_dim=32`, `hidden_dim=64`, `num_layers=3`, `num_heads=4`.

| Mode | Nodes | Forward ms | Fwd+Bwd ms | Peak MiB |
|------|------:|-----------:|-----------:|---------:|
| linear | 128 | 14.556 | 15.598 | 65.9 |
| linear | 512 | 14.834 | 15.937 | 70.5 |
| linear | 2048 | 15.116 | 16.208 | 88.9 |
| linear_sh | 128 | 15.021 | 21.110 | 66.1 |
| linear_sh | 512 | 14.995 | 21.301 | 71.3 |
| linear_sh | 2048 | 15.572 | 21.026 | 91.9 |
| local_indexed | 2048 | 16.136 | 22.250 | 331.2 |
| local fallback | 128 | 77.572 | 126.106 | 81.3 |
| local fallback | 512 | 77.947 | 126.426 | 132.7 |
| local fallback | 2048 | 79.184 | 132.859 | 349.7 |
| dense | 128 | 17.123 | 20.409 | 120.0 |
| dense | 512 | 21.063 | 26.878 | 943.8 |
| dense | 2048 | 89.969 | 181.283 | 14039.8 |

## Rich Irreps Microbenchmark

Command:

```bash
uv run python scripts/bench_attention.py --device cuda --modes rich_linear rich_local linear_sh --nodes 512 2048 --iters 10 --warmup 4
```

Model: rich modes use `hidden_irreps="64x0e + 8x1o + 4x2e"`,
`output_irreps="1x0e + 1x1o + 1x2e"`.

| Mode | Nodes | Forward ms | Fwd+Bwd ms | Peak MiB |
|------|------:|-----------:|-----------:|---------:|
| rich_linear | 512 | 8.404 | 16.773 | 76.1 |
| rich_linear | 2048 | 6.503 | 19.027 | 111.2 |
| rich_local | 512 | 10.363 | 17.800 | 138.9 |
| rich_local | 2048 | 11.538 | 20.330 | 352.3 |
| linear_sh | 512 | 8.263 | 14.607 | 71.3 |
| linear_sh | 2048 | 8.822 | 15.524 | 91.9 |

## BF16 Compile Smoke

Command:

```bash
uv run python scripts/bench_attention.py --device cuda --modes linear_sh --nodes 1024 --dtype bf16 --compile --iters 10 --warmup 4
```

Result:

| Mode | Nodes | Forward ms | Fwd+Bwd ms | Peak MiB |
|------|------:|-----------:|-----------:|---------:|
| linear_sh bf16 compile | 1024 | 7.506 | 15.015 | 71.1 |

## Notes

- `linear_sh` is the default richer linear-scale path.
- `rich_linear` and `rich_local` keep persistent scalar/vector/tensor hidden states with explicit irreps-like specs.
- `dense` is still required for arbitrary `(N, N, edge_dim)` attention bias.
- `local_indexed` uses supplied `(N, K)` neighbors and benchmarks the O(NK) layer path.
- `local` fallback uses quadratic `cdist` for neighbor selection; use it only for smoke/prototyping.
- Regression threshold: investigate >10% slowdown or >10% memory increase.
