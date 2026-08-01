# Equivariant Linear Attention

A general-purpose, parity-aware 3D layer implemented in PyTorch.

The repository exposes one model and one layer:

```text
ELA
ELALayer
```

Every layer combines:

\[
\boxed{
\text{exact global equivariant linear attention}
+
\text{exact sparse short-range residual}
}
\]

The core package depends only on PyTorch. PyG and DGL are not required for graph
construction, batching, or model execution.

## Install

```bash
uv sync --locked
```

## Quick start

### Scalar node features

```python
from equivariant_attention import ELA

model = ELA.scalar(
    node_dim=32,
    output_dim=1,
    width=128,
    depth=8,
    cutoff=6.0,
)

# x: [N, 32], pos: [N, 3]
out = model(x, pos)

node_prediction = out["node_irreps"]
graph_prediction = out["graph_irreps"]
```

If no graph is supplied, ELA builds exact radius candidates from `pos`. This is
convenient for examples and small graphs. Repeated or large workloads should
prepare and reuse the graph.

### Generic irreps

```python
model = ELA(
    input_irreps="32x0e + 4x1o + 1x1e",
    output_irreps="1x0e + 2x1o",
    width=128,
    depth=8,
    cutoff=6.0,
)

out = model(node_irreps, pos)
```

Supported sectors are `0e`, `0o`, `1o`, `1e`, `2e`, and `2o`, with arbitrary
multiplicity and `l <= 2`.

Positions are a separate affine geometry input:

\[
x_i\mapsto Rx_i+t.
\]

They are not an ordinary `1o` feature, which transforms only as `v -> Rv`.

## Mini-batches without PyG

### Flat packed batch

```python
# x: [N_total, D]
# pos: [N_total, 3]
# batch: [N_total], graph IDs 0 ... B-1
out = model(x, pos, batch=batch)
```

With supplied candidates:

```python
out = model(
    x,
    pos,
    batch=batch,
    edge_index=edge_index,  # [2, E], receiver row first
)
```

### Padded batch plus mask

```python
# x: [B, M, D], pos: [B, M, 3], mask: bool [B, M]
out = model(x, pos, mask=mask)

out["node_irreps"]      # [B, M, D_out]
out["graph_irreps"]     # [B, D_out]
out["node_mask"]        # [B, M]
```

Masked node outputs and coordinate deltas are zero; masked positions are returned
unchanged.

Padded edges may be given as:

```python
model(x, pos, mask=mask, edge_index=edges, edge_mask=edge_mask)
# edges: [B, 2, E_max] or [B, E_max, 2]

model(x, pos, mask=mask, edge_index=[edges_0, edges_1, ...])
# each edge tensor: [2, E_b]

model(x, pos, mask=mask, adjacency=adjacency)
# bool adjacency: [B, M, M]
```

### Plain dictionary dataset

A sample is an ordinary mapping:

```python
sample = {
    "x": node_features,
    "pos": positions,
    "edge_index": edge_index,  # optional
    "edge_type": relation_id,  # optional alias
    "y": target,
    "id": sample_id,
}
```

Use the built-in collator:

```python
from torch.utils.data import DataLoader
from equivariant_attention import ELA

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    pin_memory=True,
    collate_fn=ELA.collate,
)

for batch in loader:
    batch = batch.to("cuda", non_blocking=True)
    out = model(batch)
    loss = loss_fn(out["graph_irreps"], batch["target"])
```

`ELA.collate` returns an `ELABatch`, offsets variable-size graph edges, creates
the packed batch vector, and carries targets, sample IDs, conditions, and semantic
order. Common aliases include `x`, `pos`, `y`, and `edge_type`.

See [`examples/train_without_pyg.py`](examples/train_without_pyg.py) for a mixed
precision training loop.

## Prepared graph hot path

Automatic radius discovery and COO-to-CSR packing are not free. For a fixed
candidate graph, prepare it once:

```python
graph = model.prepare_graph(
    pos,
    batch=batch,
    edge_index=edge_index,  # omit to build radius candidates once
)

for step in range(num_steps):
    out = model.forward_prepared(x, pos, graph)
```

`forward_prepared` assumes the packed node order and graph membership have
already been validated. It bypasses public input packing and is the preferred
path for repeated training, inference, profiling, and compilation.

```python
compiled_forward = torch.compile(
    model.forward_prepared,
    mode="reduce-overhead",
)
out = compiled_forward(x, pos, graph)
```

The normal `model(x, pos, graph)` path remains available when runtime validation
is preferred.

## Automatic radius candidates

The built-in radius builder is an exact, chunked PyTorch reference:

```python
from equivariant_attention import radius_graph

edge_index = radius_graph(
    pos,
    batch=batch,
    cutoff=6.0,
    max_neighbors=64,
)
```

It avoids cross-graph pairs and bounds temporary memory, but still performs
quadratic pair tests within each graph:

\[
O\left(\sum_g N_g^2\right).
\]

For large or moving systems, provide precomputed candidates, a cell-list/Verlet
provider, or a future fused on-the-fly backend. With a supplied graph, the ELA
stack has

\[
O\left(L(N+E)\right)
\]

arithmetic at fixed widths and ranks.

Geometric radius candidates cannot infer bonds, mesh connectivity, temporal
transitions, or typed relations. Supply explicit edges and relation IDs when
those semantics matter.

## Condition, semantic order, and refinement

The same `ELA` class supports optional runtime context:

```python
model = ELA(
    input_irreps="32x0e",
    output_irreps="1x0e + 1x1o",
    width=128,
    depth=8,
    cutoff=6.0,
    condition_dim=256,
    order_dim=1,
    coordinate_refinement=True,
)

out = model(
    x,
    pos,
    batch=batch,
    edge_index=edge_index,
    condition=time_embedding,
    order=residue_rank,
    order_group=chain_id,
    order_mask=is_ordered_node,
    refine_steps=4,
    max_coordinate_step=0.2,
    update_mask=movable_nodes,
)
```

- `condition` is invariant `0e` data and may be shared, graph-level, or
  node-level.
- `order` is a semantic sequence/grid coordinate, never the current tensor row
  index.
- `order_mask` permits ordered and unordered node types in one graph.
- refinement is a bounded outer loop and is not a conservative integrator.

For conservative forces, use a scalar energy:

\[
F_i=-\nabla_{x_i}E.
\]

Reusable advanced context is available through `ELAContext`.

## Advanced configuration

```python
from equivariant_attention import (
    ELA,
    ELAConfig,
    ELAFeatures,
    SparseGeometry,
)

config = ELAConfig(
    input_irreps="32x0e + 4x1o",
    output_irreps="1x0e",
    width=128,
    depth=8,
    geometry=SparseGeometry(
        cutoff=6.0,
        num_rbf=16,
        relation_cutoffs=(4.0, 6.0),
    ),
    features=ELAFeatures(
        condition_dim=256,
        order_dim=1,
        coordinate_refinement=True,
    ),
)
model = ELA(config)
```

Head count, local rank, hidden irreps, normalization, residual scale, tensor
closure, and chirality construction are derived internally rather than exposed
as architecture choices.

## Layer equation

For hidden state \(h^\ell\):

\[
\bar h^\ell
=
\operatorname{EqRMSNorm}_{\rm attn}(h^\ell),
\]

\[
G^\ell
=
\operatorname{ExactGlobalELA}_{l\le2}(\bar h^\ell),
\qquad
L^\ell
=
\operatorname{ExactSparseLocal}_{l\le2}
(\bar h^\ell,x,\mathcal E).
\]

An invariant zero-initialized router combines global and local messages. At
initialization:

\[
w_G^\tau=w_L^\tau=1,
\qquad
M_i^\tau=G_i^\tau+L_i^\tau.
\]

The fused message is followed by parity-valid update, low-order tensor closure,
residual scaling, and an equivariant FFN. There is no dense `N x N` attention
tensor and no persistent edge hidden state.

## Kernel optimization

The PyTorch prepared path is the numerical contract. `torch.compile` should be
measured first. The highest-value custom-kernel target is the receiver-major
local branch: fuse geometry, radial basis, content score, positive weight, and
all receiver reductions so `[E,R,...]` intermediates are not materialized.

Global ELA is already dominated by GEMM/BMM and is a lower-priority custom Triton
target. A production edge-free user API should eventually use an internal exact
cell-list/Verlet traversal rather than the quadratic reference radius builder.

Triton or CUDA kernels remain optional execution backends and must not change the
model class, config, checkpoint, or mathematical output.

## Documentation

- [`docs/DATA_API.md`](docs/DATA_API.md)
- [`docs/ELABATCH.md`](docs/ELABATCH.md)
- [`docs/KERNEL_OPTIMIZATION.md`](docs/KERNEL_OPTIMIZATION.md)
- [`docs/CANONICAL_ELA.md`](docs/CANONICAL_ELA.md)
- [`docs/API_POLICY.md`](docs/API_POLICY.md)
- [`docs/SCALING.md`](docs/SCALING.md)

## Validation

```bash
scripts/check.sh fast
scripts/check.sh gpu
```

Focused canonical suite:

```bash
ELA_SUITE_MODE=full \
ELA_SUITE_DEVICE=cuda \
ELA_SUITE_DTYPE=bfloat16 \
  bash scripts/run_canonical_ela_suite.sh \
  artifacts/canonical-ela/final
```

Automated push/PR CI is disabled; validation is explicit.
