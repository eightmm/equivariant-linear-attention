# MODEL

## Overview

- Family: SE(3)-equivariant attention layers for 3D structural graphs.
- Scope: prototype layer family, no training recipe.
- Backend: cuEquivariance by default; e3nn and Cartesian fallback are runtime fallbacks.
- Default mode: `linear_sh`.
- Rich API: `RichEquivariantAttention` accepts explicit `0/1/2` irreps-like specs.
- Moment API: `EquivariantMomentAttention` keeps persistent `0/1` states and transient `2` moments.

## Attention Modes

- `linear`: global positive-kernel attention, O(N) memory, vector first moments.
- `linear_sh`: default global positive-kernel attention plus l<=2 SH geometry and rank-2 tensor moments.
- `local`: local attention with vector and rank-2 tensor messages. Use precomputed neighbors for O(NK).
- `dense`: all-pairs softmax attention with distance, l=0 SH, and optional dense edge bias.

`edge_feats` require `attention_mode="dense"`. Arbitrary dense pair bias is
quadratic, so it is not accepted in linear-scale modes.

If `neighbor_index` is omitted, `local` falls back to `torch.cdist` + top-k,
which is convenient but quadratic. Supplying `neighbor_index` and optional
`neighbor_mask` makes the layer itself O(NK).

Rich `linear` mode uses segment-sum batched kernel attention when `batch` is
provided. It does not consume `neighbor_index`, so training utilities use empty
neighbor slots for `rich_linear` unless `--max-neighbors` is explicitly set.
`rich_linear_light` removes rank-2 tensor hidden channels and halves vector
channels for faster scalar regression probes.

`moment_linear` adds a factorized squared-vector routing kernel, exact relative
first/second moments, and one-pass key-mass balancing. See `LAYER_MATH.md` for
the equations and symmetry boundary.

## Input API

```python
out = model(
    node_feats,
    pos,
    edge_feats=None,
    batch=None,
    neighbor_index=None,
    neighbor_mask=None,
)
```

- `node_feats`: `(N, node_dim)` invariant scalar node features.
- `pos`: `(N, 3)` Cartesian coordinates.
- `edge_feats`: optional `(N, N, edge_dim)` dense pair context for `dense` mode.
- `batch`: optional `(N,)` graph id. Attention is restricted to equal graph ids.
- `neighbor_index`: optional `(N, K)` absolute node indices for `local` mode.
- `neighbor_mask`: optional `(N, K)` bool mask for padded neighbors.

## Rich Irreps API

```python
from equivariant_attention import RichEquivariantAttention, RichEquivariantAttentionConfig

model = RichEquivariantAttention(
    RichEquivariantAttentionConfig(
        node_dim=32,
        hidden_irreps="64x0e + 8x1o + 4x2e",
        output_irreps="1x0e + 1x1o + 1x2e",
        attention_mode="local",
        vector_edge_bias=True,
        residual_scale_init=0.1,
    )
)
```

Supported terms:

- `0e` / `0o`: scalar channels, stored as `(N, C0)`.
- `1e` / `1o`: vector channels, stored as `(N, C1, 3)`.
- `2e` / `2o`: rank-2 symmetric-traceless channels, stored as `(N, C2, 3, 3)`.

`CartesianIrreps.dim` reports irreducible dimension `C0 + 3*C1 + 5*C2`.
`storage_dim` reports actual tensor storage `C0 + 3*C1 + 9*C2`.

`RichEquivariantAttention` keeps `(scalar, vector, tensor)` hidden state across
layers. Updates include scalar attention, vector transport, `T @ r`, rank-2
relative moments, and vector-to-tensor symmetric-traceless coupling.

The rich stack uses bounded coordinate features, bounded vector/tensor message
norms, scalar output normalization, and a learnable residual layer scale
initialized from `residual_scale_init / sqrt(num_layers)`. These stabilizers are
scalar gates or invariant normalizations, so they preserve SE(3) equivariance.

`vector_edge_bias=True` is available for rich `local` attention. It builds a
bounded per-head scalar logit bias from invariant contractions of hidden vector
states with local edge directions: `v_i . rhat_ij`, `v_j . rhat_ij`,
`v_i . v_j`, and `log1p(||r_ij||)`. The final bias weights are zero-initialized,
so enabling the module starts from the previous local-attention behavior.

## Output

- `node_scalar`: `(N, 1)`, rotation/translation invariant.
- `node_vector`: `(N, 3)`, SE(3)-equivariant vector.
- `node_tensor`: `(N, 3, 3)`, rank-2 symmetric-traceless equivariant tensor.
- `graph_scalar`: `(G, 1)`, graph-level invariant scalar.
- `graph_vector`: `(G, 3)`, graph-level equivariant vector.
- `graph_tensor`: `(G, 3, 3)`, graph-level rank-2 equivariant tensor.
- `backend`: active geometry backend string.
- `attention_mode`: active attention mode.

## Architecture

```text
node_feats + invariant geometry features
  -> repeated equivariant attention layers
       linear: kernel sums over scalar values and l=1 vector moments
       linear_sh: linear + l<=2 SH norm + symmetric-traceless tensor moments
       local: softmax over precomputed or fallback local relative geometry
       dense: all-pairs softmax with optional dense pair bias
  -> node and graph scalar/vector/tensor heads
```

Scalar paths only consume invariant contractions. Vector outputs combine
invariant weights with relative Cartesian vectors. Tensor outputs combine
invariant weights with symmetric-traceless relative outer products.

## Inference

```python
from equivariant_attention import prepare_for_inference

model = prepare_for_inference(model, device="cuda", dtype="bf16", compile_model=True)
```

## Benchmark

```bash
uv run python scripts/bench_attention.py --device cuda --nodes 128 512 2048
uv run python scripts/bench_attention.py --device cuda --modes local_indexed local --nodes 2048
uv run python scripts/bench_attention.py --device cuda --modes rich_linear rich_local --nodes 512 2048
uv run python scripts/bench_attention.py --device cuda --modes linear_sh --dtype bf16 --compile
uv run python scripts/train_compare.py --dataset qm9 --model rich_linear --steps 2000 --device cuda
uv run python scripts/train_compare.py --dataset qm9 --model rich_linear_light --num-layers 3 --steps 2000 --device cuda
uv run python scripts/train_compare.py --dataset qm9 --model moment_linear --num-layers 3 --steps 2000 --device cuda
```

## Verification

```bash
uv run pytest -q
scripts/check.sh fast
scripts/check.sh gpu
```

- Tests assert rotation, translation, and permutation error below `1e-6`.
