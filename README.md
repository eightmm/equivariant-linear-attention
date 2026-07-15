# Equivariant Linear Attention

Research-oriented PyTorch layers for 3D structural graphs. The package explores
SE(3)-equivariant attention with global linear kernels, local neighborhoods,
dense pair attention, explicit Cartesian irreps-like states, and factorized
relative moments.

The repository is a prototype for mathematical and performance experiments. It
includes focused symmetry tests, synthetic training smoke tests, QM9 probe
utilities, and a locked Python environment.

## Highlights

- `EquivariantAttention`: `linear`, `linear_sh`, `local`, and `dense` modes.
- `RichEquivariantAttention`: persistent scalar, vector, and rank-2 Cartesian
  states with linear or local transport.
- `EquivariantMomentAttention`: persistent scalar/vector state with transient
  symmetric-traceless moments and factorized global transport.
- Scalar, vector, and rank-2 node and graph outputs.
- cuEquivariance-first geometry backend with e3nn and Cartesian fallback paths.
- Rotation, translation, reflection, permutation, backward, and
  dense-vs-factorized tests.

## Environment

- Python `>=3.12`
- PyTorch `>=2.12.1`
- CUDA 13 cuEquivariance packages in the locked environment
- `uv` for dependency management

```bash
uv sync --locked
scripts/check.sh fast
```

The fast check compiles the source, runs Ruff, executes the CPU test suite with
coverage, and performs a model/backward smoke test. A short CUDA smoke is
available with:

```bash
scripts/check.sh gpu
```

## Minimal example

```python
import torch

from equivariant_attention import (
    EquivariantMomentAttention,
    EquivariantMomentAttentionConfig,
)

model = EquivariantMomentAttention(
    EquivariantMomentAttentionConfig(
        node_dim=16,
        hidden_irreps="64x0e + 4x1o",
        output_irreps="1x0e + 1x1o + 1x2e",
        num_layers=3,
        num_heads=4,
    )
)

node_feats = torch.randn(12, 16)
pos = torch.randn(12, 3)
batch = torch.tensor([0] * 5 + [1] * 7)

out = model(node_feats, pos, batch=batch)
print(out["graph_scalars"].shape)  # (2, 1)
print(out["graph_vectors"].shape)  # (2, 1, 3)
print(out["graph_tensors"].shape)  # (2, 1, 3, 3)
```

See [`docs/MODEL.md`](docs/MODEL.md) for the complete input/output contract and
[`docs/LAYER_MATH.md`](docs/LAYER_MATH.md) for the factorized moment equations.

## Attention paths

| Path | Connectivity | Node scaling | Geometry |
|---|---|---:|---|
| `linear` | global | `O(N)` | positive scalar kernel and vector moments |
| `linear_sh` | global | `O(N)` | linear path plus rank-2 moments |
| `local` | indexed or top-k | `O(NK)` with supplied neighbors | relative vectors and tensors |
| `dense` | all pairs | `O(N^2)` | distance, tensor moments, optional pair bias |
| `moment_linear` | global | `O(N)` | squared-vector kernel and exact relative moments |

The asymptotic statements assume a fixed hidden width, number of heads, and
number of balancing iterations. Local fallback neighbor construction uses
`torch.cdist` and is quadratic; supply `neighbor_index` for an `O(NK)` layer
path.

## Current mathematical boundary

The default-parity `moment_linear` path has focused float64 tests for O(3),
translation, permutation, exact kernel factorization, Sinkhorn equivalence, and
degenerate single-node graphs.

The broader prototype still has known limitations that should be fixed before
treating every advertised path as a strict irreducible-representation
implementation:

- `rich_linear` vector-to-tensor transport can leave the symmetric-traceless
  subspace.
- The Cartesian `l=2` fallback basis is not normalized for an invariant ordinary
  Euclidean norm.
- Indexed local attention does not yet reject cross-graph neighbor indices.
- Top-k fallback selection is not permutation-stable at exact distance ties.
- General `e/o` parity labels are descriptive; the implemented O(3) contract is
  the default `0e`, polar `1o`, and even `2e` combination.

These limitations do not invalidate the tested default moment path, but they
do narrow repository-wide symmetry claims. Contributions should add a focused
counterexample test before changing the corresponding implementation.

## Experiments and benchmarks

- [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md): CUDA attention microbenchmarks.
- [`docs/QM9_CONTRACT.md`](docs/QM9_CONTRACT.md): target, split, and provenance
  boundary for QM9 probes.
- [`docs/EXPERIMENTS.jsonl`](docs/EXPERIMENTS.jsonl): append-only run ledger.
- `scripts/bench_attention.py`: forward/backward latency and memory benchmark.
- `scripts/train_compare.py`: synthetic or QM9 regression probes.

QM9 data and generated metrics are intentionally not tracked. The current QM9
split is a seeded random-row warm-start architecture probe, not a cold-molecule
generalization benchmark.

## Development

```bash
uv run ruff check .
uv run pytest -q
scripts/check.sh fast
```

The repository currently has no declared open-source license. Contact the
repository owner before redistributing substantial portions of the code.
