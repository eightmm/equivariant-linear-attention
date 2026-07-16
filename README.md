# Equivariant Linear Attention

A focused PyTorch prototype of one O(3)-equivariant global attention
architecture for 3D graphs. The model keeps scalar and polar-vector states,
transports exact relative first and symmetric-traceless second moments through a
factorized kernel, and never materializes an `N x N` attention matrix.

The repository intentionally exposes one implementation:
`EquivariantAttention`. Earlier dense, local, rich, and backend-dependent
variants were removed so mathematical changes can be tested against one stable
contract.

## Install and verify

Core installation requires only PyTorch:

```bash
uv sync --locked
scripts/check.sh fast
```

QM9 tooling is optional:

```bash
uv sync --locked --extra qm9
```

## Example

```python
import torch

from equivariant_attention import EquivariantAttention, EquivariantAttentionConfig

model = EquivariantAttention(
    EquivariantAttentionConfig(
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

## Mathematical contract

- O(3) equivariant, including reflections; translation invariant and
  permutation consistent.
- Persistent `0e` scalars and polar `1o` vectors; transient `2e`
  symmetric-traceless moments.
- `node_feats` contains invariant scalar (`0e`) channels only; coordinates are
  stored in float32+ independently of feature/model precision.
- Exact graph-wise factorization with `O(N)` node scaling at fixed width,
  depth, head count, and a fixed normalization choice.
- Unit-normalized positive scalar content, unit-ball vector queries/keys,
  bounded linear/quadratic angular scales, and structured vector/3x3
  mass/denominator contractions avoid signed
  flattened-feature cancellation; signed value numerators remain unclamped.
- fp16/bf16 geometry squares, angular/ST features, moment reductions, and
  invariant normalization use float32; float64 stays float64. Rank-2 moment
  outputs therefore remain float32 for low-precision model inputs.
- Integer, nonnegative, contiguous graph IDs starting at zero.

Scalar outputs are O(3)-invariant and therefore cannot distinguish
isolated mirror pairs. Chirality-sensitive prediction requires a future,
explicitly tested parity-complete extension.

See [the model contract](docs/MODEL.md) and [the derivation](docs/LAYER_MATH.md).

## Research boundary

Existing QM9 results are adaptive random-row warm-start probes, not evidence of
scaffold or cold-target generalization. Historical experiments remain in
`docs/EXPERIMENTS.jsonl`; removed architecture variants are not public APIs.
The global centroid/RMS-normalized model is not a size-consistent interatomic
potential; its intended role is a bounded-size property probe or a global
context block above a local equivariant encoder.

```bash
uv run python scripts/train_compare.py --dataset synthetic --steps 10
uv run python scripts/bench_attention.py --graphs 1 8 32 --nodes-per-graph 16 32
```

Use `--no-linear-kernel` and `--no-key-balancing` only for controlled P1
ablations; they do not create separate model implementations.

The repository currently has no declared open-source license. Contact the owner
before redistributing substantial portions of the code.
