# Equivariant Linear Attention

A parity-aware O(3)-equivariant neural network for general 3D graphs and point
clouds, implemented in PyTorch without a PyG or DGL dependency.

The public API has one model and one data type:

```text
ELAGraph  ->  ELA  ->  ELAGraph
```

There are no separate tensor, dictionary, PyG, padded, batch, refinement, or
output APIs. `ELAGraph` is both the model input and the model result.

## Install

```bash
git clone https://github.com/eightmm/equivariant-linear-attention.git
cd equivariant-linear-attention
uv sync --locked
```

Python 3.12 or newer is required. Triton is optional; PyTorch is the default
numerical backend.

## One canonical example

```python
import torch

from equivariant_linear_attention import ELA, ELAGraph

model = ELA(
    "32x0e",          # input irreps
    "1x0e",           # output irreps
    width=128,
    depth=8,
    cutoff=6.0,
)

graph = ELAGraph(
    x=torch.randn(24, 32),
    pos=torch.randn(24, 3),
)

out = model(graph)

node_prediction = out.x          # [N, D_out]
graph_prediction = out.graph_x   # [B, D_out], mean readout
graph_total = out.graph_sum      # [B, D_out], sum readout
final_positions = out.pos        # [N, 3]
coordinate_delta = out.delta     # [N, 3]
```

When `edge_index` is omitted, ELA builds an exact radius graph from `pos` and the
model cutoff. Automatic topology is directed, excludes self edges, and never
connects different graphs.

## The graph contract

```python
graph = ELAGraph(
    x=node_features,       # [N, D_in]
    pos=positions,         # [N, 3]
    edge_index=edges,      # optional [2, E]
    batch=batch_index,     # optional [N]
    edge_type=edge_type,   # optional [E]
    condition=condition,   # optional invariant condition
    update_mask=mask,      # optional coordinate-update mask
    y=target,              # optional training target
)
```

Public edges always use one convention:

```text
edge_index[0] = sender/source
edge_index[1] = receiver/target
```

`batch` must be graph-major and contiguous: `0, ..., 0, 1, ..., 1, ...`. A
missing `batch` means one graph. The numerical core converts this contract once
to a private receiver-major packed representation; layers never re-parse user
inputs.

## Mini-batches

Dataset samples are `ELAGraph` objects and the DataLoader collator is always
`ELAGraph.collate`:

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,                         # Dataset[ELAGraph]
    batch_size=16,
    shuffle=True,
    collate_fn=ELAGraph.collate,
)

for graph in loader:
    graph = graph.to("cuda", non_blocking=True)
    out = model(graph)
    loss = loss_fn(out.graph_x, graph.y)
```

The collator concatenates nodes, offsets edges, creates `batch`, and carries
`edge_type`, conditions, update masks, targets, and sample IDs without dummy
padded nodes.

For repeated execution with static topology, an explicit immutable-storage
contract enables the E-independent prepared-cache path:

```python
static_graph = graph.to("cuda").assume_immutable()
with torch.inference_mode():
    model(static_graph)       # prepares and attaches private CSR metadata
    out = model(static_graph) # reuses it without scanning every edge
```

`assume_immutable()` clones `pos`, `edge_index`, `batch`, `edge_type`, and
`group` before opting in. Do not mutate those returned tensors or export mutable
aliases of them. Ordinary `ELAGraph` execution remains the safe default and
revalidates cached topology content exactly, including storage that may be
changed through NumPy or DLPack aliases. Seal the graph before entering
`torch.inference_mode()`; inference tensors do not expose mutation counters, so
sealing inside that context safely falls back to exact validation.

## Coordinate updates

Coordinate mutation is selected once when the model is declared:

```python
model = ELA(
    "32x0e",
    "1x0e",
    update_positions=True,
    max_coordinate_step=0.2,
)

out = model(graph)
updated_positions = out.pos
step = out.delta
```

With `update_positions=False`, `out.pos` equals the input positions and
`out.delta` is zero. With `update_positions=True`, ELA predicts a polar-vector
step after every layer, carries the hidden state forward, and lets the next
layer consume the updated geometry. Selected updates are centered per graph;
the sum of all stage steps is bounded by `max_coordinate_step`. Automatic
radius topology is rebuilt between stages when its preparation provenance no
longer permits reuse. `graph.update_mask` selects movable nodes; unselected
nodes receive exactly zero displacement.

This built-in coordinate update is for learned geometry refinement. To predict a
flow-matching velocity without mutating coordinates, keep
`update_positions=False`, declare `output_irreps="1x1o"`, and read the `1o`
block from `out.x`.

## Equivariant inputs and outputs

Representations are declared only with irreps:

```python
model = ELA(
    "32x0e + 4x1o + 1x1e",
    "2x0e + 1x1o + 1x2e",
)
```

Supported sectors are:

```text
0e  scalar
0o  pseudoscalar
1o  polar vector
1e  axial vector
2e  even symmetric-traceless tensor
2o  odd symmetric-traceless tensor
```

Arbitrary multiplicities are supported for degree `l <= 2`. Positions remain a
separate affine input; they are not an ordinary `1o` feature.

Structured mixed irreps are available from the advanced module:

```python
from equivariant_linear_attention.advanced import pack_irreps, split_irreps

x = pack_irreps(
    "8x0e + 2x1o",
    {
        "0e": scalar_features.unsqueeze(-1),
        "1o": vector_features,
    },
)

blocks = split_irreps("2x0e + 1x1o", out.x)
```

## Graph and node tasks

```text
task                         output_irreps       result
intensive graph property     "1x0e"              out.graph_x
extensive graph property     "1x0e"              out.graph_sum
node scalar labels           "Kx0e"              out.x
velocity / vector field      "Kx1o"              1o block of out.x
axial-vector target          "Kx1e"              1e block of out.x
rank-two tensor target       "Kx2e" or "Kx2o"   tensor block of out.x
```

For conservative forces, predict invariant node energies, use `out.graph_sum`,
and differentiate with `conservative_forces` from the advanced module. A direct
`1o` output is equivariant but is not necessarily conservative.

## Typed and disconnected topology

Use explicit edges for bonds, meshes, temporal transitions, metal coordination,
or other semantic relations:

```python
model = ELA("32x0e", edge_types=4)

graph = ELAGraph(
    x=x,
    pos=pos,
    edge_index=edge_index,
    edge_type=edge_type,       # values in [0, 4)
)
```

`group` can split one sample into disconnected interaction components. Global
attention and automatic local edges remain inside each component, while
`graph_x` and `graph_sum` are still pooled per sample:

```python
graph = ELAGraph(x=x, pos=pos, batch=batch, group=component_id)
```

## Architecture

Each layer computes an exact global linear-attention branch and an exact sparse
short-range branch, then applies their fixed sum:

$$
M_i^\tau = G_i^\tau + L_i^\tau.
$$

The hidden carrier is parity-complete through `l=2`. Geometric multiplicity and
local rank scale deterministically with width, so users do not choose heads,
ranks, hidden irreps, normalization variants, or branch routers. The global
kernel includes pseudoscalar-aware and radial routing; the local operator uses
unit-direction angular terms, relation-conditioned radial/value transport, and
first/second-moment chiral channels.

No branch constructs a dense `N x N` attention matrix. For `E = O(N)` sparse
candidates and fixed widths, stack work is `O(L(N + E))`.

## Performance

Topology preparation, COO-to-CSR conversion, and backend dispatch are internal.
A fixed-position output may carry a private prepared topology for compatible
subsequent models. Radius topology records cutoff, neighbor policy, relation
schema, and reference positions, and is rebuilt rather than silently reused
when stale.

PyTorch is the canonical backend. Compile ELA through the inference helper so
public validation, cache lookup, topology discovery, pooling, and output
wrapping remain eager while only the private numerical core is compiled:

```python
from equivariant_linear_attention.inference import prepare_for_inference

inference_model = prepare_for_inference(model, device="cuda", compile_model=True)
with torch.inference_mode():
    out = inference_model(graph)
```

Compilation is an optimization attempt, not a numerical requirement: recognized
Dynamo/Inductor lowering failures warn once and permanently fall back to the
exact eager core. Other runtime errors remain visible. Validate latency on the
intended shapes and hardware.
Triton remains an explicit memory-oriented experimental backend and is never
selected automatically.

See [performance](docs/PERFORMANCE.md), [advanced configuration](docs/ADVANCED.md),
[task recipes](docs/TASKS.md), [real-data validation](docs/REALDATA_VALIDATION.md),
[the architecture contract](docs/CANONICAL_ELA.md), and the
[evidence-scoped completion checklist](docs/COMPLETION_STATUS.md).

## Validation

```bash
uv run pytest -q
uv run python scripts/ml_smoke.py cpu
uv run python examples/train_without_pyg.py
uv run python examples/flow_matching_velocity.py
```

The suite covers proper and improper O(3) actions, translation, node and edge
permutations, graph isolation, coordinate double backward, topology provenance,
ST5 tensor metrics, relation conditioning, and optional CUDA/Triton parity.
