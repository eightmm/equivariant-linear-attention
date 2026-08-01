# Equivariant Linear Attention

A general-purpose, parity-aware 3D layer implemented in PyTorch.

The repository exposes one model and one layer:

```text
ELA
ELALayer
```

Its spatial equation is fixed:

\[
\boxed{
\text{exact global equivariant linear attention}
+
\text{exact sparse short-range residual}
}
\]

The core package depends only on PyTorch. PyG is not required for graph data,
batching, edge packing, or radius-neighbor construction.

## Install

```bash
uv sync --locked
```

## 20-second example

For ordinary scalar node features:

```python
from equivariant_attention import ELA

model = ELA.scalar(
    node_dim=32,
    output_dim=1,
    width=128,
    depth=8,
    cutoff=6.0,
)

# x:   [N, 32]
# pos: [N, 3]
out = model(x, pos)

node_prediction = out["node_irreps"]   # [N, 1]
graph_prediction = out["graph_irreps"] # [1, 1]
```

When no graph is supplied, ELA builds radius candidates from `pos`. This is a
convenient exact PyTorch reference path. For repeated training or large graphs,
prepare and reuse the graph as shown below.

## Generic irreps

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

The optimized path supports arbitrary multiplicities of:

```text
0e  ordinary scalar
0o  pseudoscalar
1o  polar vector
1e  axial vector
2e  even symmetric-traceless tensor
2o  odd symmetric-traceless tensor
```

with `l <= 2`.

Positions remain a separate affine geometry input:

\[
x_i\mapsto Rx_i+t.
\]

They must not be packed as an ordinary `1o` feature, which transforms only as
`v -> Rv`.

## Batching without PyG

ELA accepts three public batch layouts. They are converted internally to one
packed ragged node axis.

### 1. Flat packed graphs

```python
# x:     [N_total, D]
# pos:   [N_total, 3]
# batch: [N_total], values 0 ... B-1
out = model(x, pos, batch=batch)
```

Radius candidates are built independently inside each graph.

You can provide an existing candidate graph:

```python
out = model(
    x,
    pos,
    batch=batch,
    edge_index=edge_index,  # [2, E], receiver row first
)
```

### 2. Padded dense batch plus mask

```python
# x:    [B, M, D]
# pos:  [B, M, 3]
# mask: [B, M], bool
out = model(x, pos, mask=mask)

node_prediction = out["node_irreps"] # [B, M, D_out]
graph_prediction = out["graph_irreps"] # [B, D_out]
```

Masked node outputs and coordinate deltas are zero. Masked positions are
returned unchanged.

Padded edges may be supplied in any of these forms:

```python
model(x, pos, mask=mask, edge_index=edges, edge_mask=edge_mask)
# edges: [B, 2, E_max] or [B, E_max, 2]

model(x, pos, mask=mask, edge_index=[edges_0, edges_1, ...])
# each item: [2, E_b]

model(x, pos, mask=mask, adjacency=adjacency)
# adjacency: bool [B, M, M]
```

Negative endpoints in a padded edge tensor are treated as edge padding.

### 3. Plain dictionaries and `DataLoader`

A dataset item can be an ordinary Python dictionary:

```python
sample = {
    "x": node_features,       # aliases: node_irreps, node_features
    "positions": positions,  # alias: pos
    "edge_index": edge_index, # optional
    "target": target,         # ignored by the model
    "sample_id": sample_id,
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
    collate_fn=ELA.collate,
)

for batch in loader:
    out = model(batch)
    loss = loss_fn(out["graph_irreps"], batch["target"])
```

`ELA.collate` concatenates node tensors, creates the batch vector, offsets
per-graph edges, and carries optional targets, sample IDs, conditions, and order
coordinates. No framework-specific graph object is required.

## Reuse a prepared graph

Automatic graph construction and CSR packing are intentionally convenient, not
free. For a fixed topology, cache it once:

```python
graph = model.prepare_graph(
    pos,
    batch=batch,
    edge_index=edge_index, # omit to build radius candidates once
)

for step in range(num_steps):
    out = model(x, pos, graph)
```

This is the preferred hot path for training, inference, `torch.compile`, and
profiling.

The built-in radius builder is chunked to bound temporary memory but performs
quadratic pair tests within each graph:

\[
O\left(\sum_g N_g^2\right).
\]

For large or dynamically moving systems, provide a cell-list/Verlet neighbor
provider or a future fused on-the-fly backend. The ELA layer itself remains

\[
O\left(L(N+E)\right)
\]

for a supplied candidate graph at fixed widths and ranks.

## Optional condition, order, and refinement

Optional features use the same model class. They can be passed directly as
keywords; the lower-level `ELAContext` API remains available for reusable
contexts.

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

- `condition` is an invariant `0e` tensor and may be shared, graph-level, or
  node-level.
- `order` is a semantic sequence/grid coordinate, never the current tensor row
  index.
- `order_mask` permits ordered and unordered node types in one graph.
- coordinate refinement is a bounded outer loop and does not imply conservative
  dynamics.

For conservative forces, use a scalar energy:

\[
F_i=-\nabla_{x_i}E.
\]

## Advanced configuration

The explicit config objects remain useful for checkpoint provenance and complex
relation cutoffs:

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

For each irrep sector, an invariant zero-initialized router combines global and
local messages. Initially it is exactly the ordinary sum:

\[
w_G^\tau=w_L^\tau=1,
\qquad
M_i^\tau=G_i^\tau+L_i^\tau.
\]

Then:

\[
\begin{aligned}
M^\ell &= \operatorname{Fuse}(G^\ell,L^\ell),\\
\widetilde h^\ell
&=h^\ell+
\operatorname{AttnResidual}
\left[
\operatorname{ParityUpdate}(M^\ell)
+
\operatorname{TPClosure}_{l\le2}
\right],\\
h^{\ell+1}
&=\widetilde h^\ell+
\operatorname{EqFFN}
\left(
\operatorname{EqRMSNorm}_{\rm ffn}
(\widetilde h^\ell)
\right).
\end{aligned}
\]

There is no dense `N x N` attention tensor and no persistent edge hidden state.

## Performance path

Start with the standard PyTorch implementation and a prepared graph:

```python
graph = model.prepare_graph(pos, batch=batch, edge_index=edge_index)
compiled_model = torch.compile(model, mode="reduce-overhead")
out = compiled_model(x, pos, graph)
```

The highest-value custom-kernel target is the receiver-major local branch:
geometry, radial basis, content score, positive weight, and all receiver
reductions should eventually be fused so `[E, R, ...]` intermediates are not
materialized. The global ELA branch is already dominated by GEMM/BMM operations
and is a lower-priority Triton target.

Triton remains optional. The PyTorch reference is the numerical contract and the
fallback for unsupported devices or higher-order gradient checks.

See:

- `docs/DATA_API.md`
- `docs/KERNEL_OPTIMIZATION.md`
- `docs/CANONICAL_ELA.md`
- `docs/API_POLICY.md`
- `docs/SCALING.md`

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

Automated GitHub Actions are disabled; validation is explicit.
