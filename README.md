# Equivariant Linear Attention

A general-purpose, parity-aware 3D neural layer implemented in PyTorch.

The public architecture is deliberately small:

```text
ELA       one model
ELALayer  one reusable layer
ELABatch  one graph container
```

Every layer combines

\[
\boxed{
\text{exact global equivariant linear attention}
+
\text{exact sparse short-range geometry}
}
\]

without constructing a dense `N x N` attention matrix. The core package does not
require PyG or DGL.

## Install

```bash
uv sync --locked
```

Triton is optional. If a compatible Triton runtime is present, supported CUDA
reductions are selected automatically; otherwise ELA uses the PyTorch reference.

## Quick start

Representations are always declared with irreps. A scalar-only input with 32
channels is written as `"32x0e"`.

```python
import torch

from equivariant_attention import ELA, ELABatch

model = ELA(
    input_irreps="32x0e",
    output_irreps="1x0e",
    width=128,
    depth=8,
    cutoff=6.0,
)

batch = ELABatch(
    node_irreps=torch.randn(24, 32),
    positions=torch.randn(24, 3),
)

output = model(batch)

node_prediction = output["node"]
graph_prediction = output["graph"]
```

When `edge_index` is omitted, ELA constructs exact geometric radius candidates
from `positions` and the model cutoff.

## General equivariant inputs and outputs

```python
model = ELA(
    input_irreps="32x0e + 4x1o + 1x1e",
    output_irreps="1x0e + 2x1o",
    width=128,
    depth=8,
    cutoff=6.0,
)
```

Supported sectors are:

```text
0e  ordinary scalar
0o  pseudoscalar
1o  polar vector
1e  axial vector
2e  even symmetric-traceless tensor
2o  odd symmetric-traceless tensor
```

Arbitrary multiplicities are supported for `l <= 2`. Input tensors are flattened
according to `input_irreps`; helpers such as `pack_irreps` and `split_irreps` are
available for structured construction.

Positions remain a separate affine input because

\[
x_i \mapsto R x_i + t,
\]

whereas an ordinary `1o` feature transforms only as `v -> Rv`.

## Explicit graph topology

Use receiver/sender COO when graph semantics matter:

```python
batch = ELABatch(
    node_irreps=x,
    positions=pos,
    ptr=ptr,
    edge_index=edge_index,          # [2, E]
    edge_relation_id=edge_type,     # [E], optional
)

output = model(batch)
```

The convention is:

```text
edge_index[0] = receiver
edge_index[1] = sender
```

Automatic radius candidates express geometric proximity only. Supply explicit
edges for bonds, mesh connectivity, temporal transitions, metal coordination,
or other typed relations.

## Mini-batches without PyG

ELA uses one packed ragged representation:

```text
node_irreps:      [N_total, D]
positions:        [N_total, 3]
ptr:              [B + 1]
edge_index:       [2, E] or None
edge_relation_id: [E] or None
```

### From flat tensors

```python
batch = ELA.batch(
    x,
    pos,
    batch=batch_index,       # graph-major IDs 0 ... B-1
    edge_index=edge_index,
)

output = model(batch)
```

`ptr` may be supplied instead of `batch`.

### From padded tensors

```python
batch = ELA.padded(
    x_padded,       # [B, M, D]
    pos_padded,     # [B, M, 3]
    mask=node_mask, # bool [B, M]
)

output = model(batch)
packed_node_output = output["node"]
padded_node_output = batch.restore_nodes(packed_node_output)
```

Padded tensors are a convenience input only. Valid nodes are packed before the
numerical core runs, so masked dummy nodes do not consume layer compute.

### Plain dictionary datasets

Dataset samples may be ordinary mappings:

```python
sample = {
    "x": node_features,
    "pos": positions,
    "edge_index": edge_index,  # optional
    "edge_type": relation_id,  # optional
    "y": target,
    "id": sample_id,
}
```

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    pin_memory=True,
    collate_fn=ELA.collate,
)

for batch in loader:
    batch = batch.to("cuda", dtype=torch.bfloat16, non_blocking=True)
    output = model(batch)
    loss = loss_fn(output["graph"].float(), batch.target.float())
```

The collator packs variable-size graphs, offsets local edge indices, builds
`ptr`, and carries targets, sample IDs, conditions, and semantic order.

See [`examples/train_without_pyg.py`](examples/train_without_pyg.py).

## Output contract

The precise names and concise aliases are both returned:

```text
node_irreps       alias: node
graph_irreps      aliases: graph, graph_mean
graph_sum
positions         alias: pos
coordinate_delta  alias: delta
```

`graph_irreps` is the graph-wise mean. `graph_sum` is provided for extensive
quantities such as additive total energies.

## Prepared hot path

Graph discovery and COO-to-CSR packing should not be repeated for fixed topology.
Prepare an `ELABatch` once:

```python
batch = model.prepare(batch)

for step in range(num_steps):
    output = model.forward_prepared(batch)
```

The normal `model(batch)` call prepares an unprepared batch automatically.
`forward_prepared` is the intended path for repeated training, inference,
profiling, and backend comparison.

For static shapes it can be compiled independently:

```python
compiled_forward = torch.compile(
    model.forward_prepared,
    mode="reduce-overhead",
)

output = compiled_forward(batch)
```

## Automatic radius candidates

Small graphs use an exact chunked dense reference. Larger graphs use an exact
3D cell list followed by distance filtering. Both return directed candidates and
never connect different graphs.

Under fixed cutoff and bounded spatial density, the cell-list path has expected

\[
O(N + E)
\]

work. Worst-case behavior remains quadratic when many points occupy one cell.
Periodic cells and minimum-image geometry are not inferred; provide an explicit
topology for those workloads.

## Conditions, semantic order, and coordinate refinement

Optional capabilities remain fields of the same model and batch.

```python
from equivariant_attention import OrderContext, RefinementRequest

model = ELA(
    input_irreps="32x0e",
    output_irreps="1x0e + 1x1o",
    condition_dim=256,
    order_dim=1,
    coordinate_refinement=True,
)

batch = ELABatch(
    node_irreps=x,
    positions=pos,
    edge_index=edge_index,
    condition=time_embedding,
    order=OrderContext.sequence(
        residue_rank,
        segment_id=chain_id,
        enabled=is_ordered_node,
    ),
    refinement=RefinementRequest(
        steps=4,
        max_step=0.2,
        update_mask=movable_nodes,
    ),
)

output = model(batch)
```

- `condition` is invariant `0e` information.
- Semantic order is node-attached information such as residue rank or time, never
  the current tensor row index.
- Coordinate refinement is a bounded learned outer loop, not a conservative
  integrator.

For conservative forces use a scalar energy:

\[
F_i = -\nabla_{x_i} E.
\]

## Advanced configuration

```python
from equivariant_attention import ELAConfig, ELAFeatures, SparseGeometry

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

Head count, local rank, hidden irreps, normalization, tensor closure, chirality,
and execution backend are internal deterministic choices rather than public
architecture variants.

## Layer equation

For hidden state `h^l`:

\[
\bar h^l = \operatorname{EqRMSNorm}(h^l),
\]

\[
G^l = \operatorname{ExactGlobalELA}_{l\le2}(\bar h^l),
\qquad
L^l = \operatorname{ExactSparseLocal}_{l\le2}(\bar h^l, x, \mathcal E).
\]

An invariant, identity-initialized router combines the two branches. At
initialization:

\[
w_G^\tau = w_L^\tau = 1,
\qquad
M_i^\tau = G_i^\tau + L_i^\tau.
\]

The fused message enters parity-valid updates, low-order tensor closure, residual
scaling, and an equivariant FFN. There is no persistent edge hidden state.

## Complexity

For a prepared graph with `N` nodes, `E` directed candidates, `L` layers, and
fixed widths/ranks:

\[
T = O\left(L(N+E)\right).
\]

Node-linear scaling additionally requires `E = O(N)`. Neighbor discovery must be
reported separately when it is included.

## Kernel backends

The PyTorch implementation is the numerical reference. Backend selection does
not alter the model, config, checkpoint, or equations.

```bash
ELA_KERNEL_BACKEND=auto    # default
ELA_KERNEL_BACKEND=torch   # force reference
ELA_KERNEL_BACKEND=triton  # fail if unsupported
```

The current Triton path accelerates receiver-major CSR reductions and uses
memory-bounded payload groups in the local operator. Unsupported devices, dtypes,
or graph regimes fall back to PyTorch in `auto` mode.

Benchmark the same prepared batch and model state:

```bash
uv run python scripts/benchmark_ela.py \
  --input-irreps "32x0e" \
  --output-irreps "1x0e" \
  --nodes 4096 \
  --degree 32 \
  --width 128 \
  --depth 8 \
  --device cuda \
  --dtype bfloat16 \
  --output artifacts/ela-kernels.json
```

## Documentation

- [`docs/DATA_API.md`](docs/DATA_API.md)
- [`docs/CANONICAL_ELA.md`](docs/CANONICAL_ELA.md)
- [`docs/API_POLICY.md`](docs/API_POLICY.md)
- [`docs/KERNEL_OPTIMIZATION.md`](docs/KERNEL_OPTIMIZATION.md)
- [`docs/SCALING.md`](docs/SCALING.md)
- [`docs/CANONICAL_ELA_VALIDATION.md`](docs/CANONICAL_ELA_VALIDATION.md)

## Validation

```bash
scripts/check.sh fast
scripts/check.sh gpu
```

Focused suite:

```bash
ELA_SUITE_MODE=full \
ELA_SUITE_DEVICE=cuda \
ELA_SUITE_DTYPE=bfloat16 \
  bash scripts/run_canonical_ela_suite.sh \
  artifacts/canonical-ela/final
```

Automated push/PR CI is disabled; validation is explicit.
