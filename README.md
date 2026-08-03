# Equivariant Linear Attention

A general-purpose, parity-aware 3D neural layer implemented in PyTorch.

The public architecture is deliberately small:

```text
ELA       one model
ELALayer  one reusable layer
ELABatch  one graph container
```

Every layer combines

$$
\boxed{
\text{exact global equivariant linear attention}
+
\text{exact sparse short-range geometry}
+
\text{invariant global/local fusion}
}
$$

without constructing a dense `N x N` attention matrix. The core package does not
require PyG or DGL.

Start with [installation](#install), [the minimal example](#quick-start),
[batching](#mini-batches-without-pyg), [task outputs](#output-contract),
[flow matching](#flow-matching-velocity-and-learned-displacement), or
[performance](#prepared-hot-path) and [validation](#validation).

## Install

Source installation requires Python 3.12+ and
[`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/eightmm/equivariant-linear-attention.git
cd equivariant-linear-attention
uv sync --locked
uv run python examples/train_without_pyg.py
```

The repository and distribution use the hyphenated name
`equivariant-linear-attention`; Python imports use the matching underscore form
`equivariant_linear_attention`. The former pre-release import root
`equivariant_attention` is not shipped.

Triton is optional. PyTorch remains the default numerical backend. A compatible
Triton runtime can be forced for contract-tested experiments and benchmarks.

## Quick start

Representations are always declared with irreps. A scalar-only input with 32
channels is written as `"32x0e"`.

```python
import torch

from equivariant_linear_attention import ELA, ELABatch

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
from `positions` and the model cutoff. These candidates are directed and include
self edges; provide explicit topology when that is not the desired graph
semantics.

## General equivariant inputs and outputs

```python
from equivariant_linear_attention import ELA

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

$$
x_i \mapsto R x_i + t,
$$

whereas an ordinary `1o` feature transforms only as `v -> Rv`.

Build mixed inputs with the declared layout instead of concatenating blocks by
hand:

```python
from equivariant_linear_attention import pack_irreps

input_irreps = "8x0e + 2x1o"
node_irreps = pack_irreps(
    input_irreps,
    {
        "0e": scalar_features.unsqueeze(-1),  # [N, 8, 1]
        "1o": polar_vectors,                  # [N, 2, 3]
    },
)
```

## Explicit graph topology

Use receiver/sender COO when graph semantics matter:

```python
from equivariant_linear_attention import ELA, ELABatch

model = ELA(
    input_irreps="32x0e",
    output_irreps="1x0e",
    num_edge_types=4,
)

batch = ELABatch(
    node_irreps=x,
    positions=pos,
    ptr=ptr,
    edge_index=edge_index,          # [2, E]
    edge_relation_id=edge_type,     # [E], values in [0, 4)
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

If a sample supplies `edge_type`, construct the model with matching
`num_edge_types` or `relation_cutoffs`. Untyped models should omit `edge_type`.

```python
import torch
from torch.utils.data import DataLoader

from equivariant_linear_attention import ELA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    compute_dtype = (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
else:
    compute_dtype = torch.float32
model = ELA(
    input_irreps="32x0e",
    output_irreps="1x0e",
).to(device)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    pin_memory=device.type == "cuda",
    collate_fn=ELA.collate,
)

for batch in loader:
    # Keep model parameters, input features, and geometry in FP32. Autocast
    # selects lower precision only for supported CUDA operators.
    batch = batch.to(device, non_blocking=True)
    with torch.autocast(
        device_type=device.type,
        dtype=compute_dtype,
        enabled=device.type == "cuda",
    ):
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

For `D_out = sum(multiplicity * (2*l + 1))`, common choices are:

| task | example `output_irreps` | readout |
|---|---|---|
| intensive graph property | `"1x0e"` | `output["graph"]` |
| sum readout for three extensive targets | `"3x0e"` | `output["graph_sum"]` |
| five node scalar labels | `"5x0e"` | node `0e` block |
| velocity or displacement | `"1x1o"` | node `1o` block |
| axial-vector target | `"1x1e"` | node `1e` block |
| symmetric-traceless tensor | `"1x2e"` or `"1x2o"` | node tensor block |

`graph_sum` selects a sum readout; global interaction means this alone does not
prove physical size-extensivity or disconnected-component additivity.

Use `model.split_output` instead of manually slicing a mixed output:

```python
from equivariant_linear_attention import ELA, st5_to_matrix

model = ELA(
    input_irreps="32x0e",
    output_irreps="2x0e + 1x1o + 1x2e",
)
output = model(batch)
blocks = model.split_output(output["node"])

node_scalars = blocks["0e"][..., 0]  # [N, 2]
polar_vector = blocks["1o"][:, 0]    # [N, 3]
st_tensor = blocks["2e"][:, 0]       # [N, 5]
st_matrix = st5_to_matrix(st_tensor)  # [N, 3, 3]
```

### Energy and forces

A direct `1o` prediction is an equivariant vector but is not necessarily a
conservative force. When conservation is required, predict one invariant node
energy, sum it per graph, and differentiate:

```python
from equivariant_linear_attention import ELA, ELABatch, conservative_forces

positions = positions.detach().requires_grad_(True)
batch = ELABatch(
    node_irreps=node_features,
    positions=positions,
    ptr=ptr,
    edge_index=edge_index,
)
energy_model = ELA(
    input_irreps="32x0e",
    output_irreps="1x0e",
)

output = energy_model(batch)
energy = output["graph_sum"][:, 0]
forces = conservative_forces(
    energy,
    batch.positions,
    create_graph=True,  # required when a force loss is differentiated
)
```

Force/HVP training needs higher-order autograd. Keep it on eager PyTorch unless
the exact compiled workload has passed a separate double-backward contract.

## Flow matching velocity and learned displacement

A coordinate-space flow velocity is a node-wise polar vector. Declare it as
`1x1o` and read it from the node irreps. It is **not** the refinement-only
`output["delta"]` value.

```python
import torch
import torch.nn.functional as F

from equivariant_linear_attention import ELA

flow_model = ELA(
    input_irreps="32x0e",
    output_irreps="1x1o",
    condition_dim=1,  # invariant flow time t
    width=128,
    depth=8,
    cutoff=6.0,
)

# x0 and x1 are corresponding, consistently centered [N, 3] coordinates.
# batch_index contains graph-major IDs and t has shape [B, 1].
num_graphs = int(batch_index[-1].item()) + 1
t = torch.rand(num_graphs, 1, device=x0.device)
t_node = t[batch_index]
x_t = (1.0 - t_node) * x0 + t_node * x1
target_velocity = x1 - x0  # only for this straight conditional path

batch_t = ELA.batch(
    node_features,
    x_t,
    batch=batch_index,
    edge_index=edge_index,  # omit to build radius candidates at x_t
    condition=t,
)
output = flow_model(batch_t)
velocity = flow_model.split_output(output["node"])["1o"].squeeze(-2)
loss = F.mse_loss(velocity, target_velocity)

dt = 0.01
delta_x = dt * velocity
x_next = x_t + delta_x
```

The compact loss above weights nodes equally. For ragged mini-batches with
different graph sizes, prefer the graph-balanced loss in the complete example
linked below so that large point clouds do not dominate the objective.

For a non-linear interpolation schedule the target is `d x_t / d t`, not
automatically `x1 - x0`. Source and target nodes must also have the same
correspondence; symmetric or unmatched point sets need a matching/OT policy
before the displacement target is defined.

`1o` specifies polar O(3) covariance, while ELA's relative/centered geometry
makes the field translation-invariant. Rotation augmentation is therefore not
required to impose that symmetry. Use the same per-graph centering convention
for `x0` and `x1`, then derive the target from those centered endpoints. If
exact zero center-of-mass velocity is part of the task, subtract the predicted
graph mean explicitly; the generic `1o` output is intentionally neither centered
nor bounded.

With scalar-only conditions and isotropic noise, exact O(3) equivariance also
makes a generated distribution reflection-symmetric. A chiral application must
provide a physically meaningful odd-parity input, such as a `0o` handedness
signal or a parity-declared polar/axial frame, rather than relabeling an ordinary
scalar as odd. The `0o` value must change sign when a reflected input is used.

When coordinates move and local radius candidates are used, rebuild candidates
whenever their membership may change. Reuse a prepared topology only for fixed
bonds or an intentionally fixed sparse candidate set. A complete runnable
training step is in
[`examples/flow_matching_velocity.py`](examples/flow_matching_velocity.py).

The built-in collator treats `target` as graph-level metadata. For ragged node
targets such as `[N_i,3]` velocities, a task collator should call `ELA.collate`
for the graph batch, concatenate only the velocity targets in exactly the same
graph-major node order, and return the two together.

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
never connect different graphs. Both paths use float64 geometry for float64
coordinates and float32 geometry otherwise, so float16/bfloat16 topology does
not change at the dense/cell dispatch threshold.

`model.prepare(batch, max_neighbors=k)` keeps the nearest distance shells. If
several candidates tie at the k-th distance, the complete shell is retained;
the realized degree may exceed `k` only in that degenerate case. This preserves
node-permutation consistency instead of selecting tied nodes by storage index.

Under fixed cutoff and bounded spatial density, the cell-list path has expected

$$
O(N + E)
$$

work. Worst-case behavior remains quadratic when many points occupy one cell.
Periodic cells and minimum-image geometry are not inferred; provide an explicit
topology for those workloads.

## Conditions, semantic order, and coordinate refinement

Optional capabilities remain fields of the same model and batch.

```python
from equivariant_linear_attention import ELA, ELABatch, OrderContext, RefinementRequest

model = ELA(
    input_irreps="32x0e",
    output_irreps="1x0e",
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

`output["delta"]` is the accumulated refinement displacement. Each step is
bounded by `max_step` and is centered according to the requested policy, which
may explicitly be `"none"`. Without a `RefinementRequest` the value is zero,
even if `output_irreps` contains a `1o` block. Use node `1o` for supervised
velocity/displacement fields and use refinement only when the model should
iteratively mutate its input geometry.

For conservative forces use a scalar energy:

$$
F_i = -\nabla_{x_i} E.
$$

## Advanced configuration

```python
from equivariant_linear_attention import ELA, ELAConfig, ELAFeatures, SparseGeometry

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

`relation_cutoffs=(4.0, 6.0)` declares two semantic relation types. Every batch
for this configuration must therefore provide `edge_relation_id` values in
`[0, 2)`. Remove `relation_cutoffs` when using an untyped automatic radius graph.

Head count, local rank, hidden irreps, normalization, tensor closure, chirality,
and execution backend are internal deterministic choices rather than public
architecture variants.

## Layer equation

For hidden state `h^l`:

$$
\bar h^l = \operatorname{EqRMSNorm}(h^l),
$$

$$
G^l = \operatorname{ExactGlobalELA}_{l\le2}(\bar h^l),
\qquad
L^l = \operatorname{ExactSparseLocal}_{l\le2}(\bar h^l, x, \mathcal E).
$$

An invariant, identity-initialized router combines the two branches. At
initialization:

$$
w_G^\tau = w_L^\tau = 1,
\qquad
M_i^\tau = G_i^\tau + L_i^\tau.
$$

The fused message enters parity-valid updates, low-order tensor closure, residual
scaling, and an equivariant FFN. There is no persistent edge hidden state.

## Complexity

For a prepared graph with `N` nodes, `E` directed candidates, `L` layers, and
fixed widths/ranks:

$$
T = O\left(L(N+E)\right).
$$

Node-linear scaling additionally requires `E = O(N)`. Neighbor discovery must be
reported separately when it is included.

## Kernel backends

The PyTorch implementation is the numerical reference. Backend selection does
not alter the model, config, checkpoint, or equations. The
`ELA_KERNEL_BACKEND` environment variable accepts `auto` (the default), `torch`
(force the reference), or `triton` (force Triton and fail if unsupported). To
set it for subsequent commands, export exactly one value:

```bash
export ELA_KERNEL_BACKEND=triton
```

The current Triton path uses degree-dynamic receiver-major CSR reductions,
cat-free pair/triple payload groups, fused inference gather-weight-reduce for
four local value families, and direct first-order backward gathers/broadcasts.
Unsupported devices or dtypes fail closed when forced. `auto` deliberately stays
on PyTorch because the current complete-stack latency measurements do not satisfy
the promotion threshold; the forced path is useful for contract and memory
experiments.

For static prepared shapes, `torch.compile(model.forward_prepared,
mode="reduce-overhead")` is currently the strongest measured speed path. The
measurements, hardware-sensitive scope, and limitations are recorded in
[`docs/KERNEL_OPTIMIZATION.md`](docs/KERNEL_OPTIMIZATION.md). This is not a
ragged-shape or double-backward claim; force/HVP workloads must retain the eager
path unless a separate compiled higher-order-autograd contract passes.

Tests and profilers can avoid process-global environment mutation:

```python
from equivariant_linear_attention.kernels import kernel_backend

with kernel_backend("triton"):
    output = model(batch)
```

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

Date-stamped studies, `docs/EXPERIMENTS.jsonl`, and documents marked
**Historical** preserve prior experiments. Their commands may require the Git
revision recorded in the document and are not part of the current package API.

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
