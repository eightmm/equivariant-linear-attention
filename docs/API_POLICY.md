# Public API policy

Equivariant Linear Attention has one public model, one public graph type, and one
execution form:

```python
from equivariant_linear_attention import ELA, ELAGraph

graph = ELAGraph(x=node_features, pos=positions)
model = ELA("32x0e", "1x0e")
out = model(graph)
```

The package root exports exactly:

```text
ELA
ELAGraph
```

## One graph contract

`ELAGraph` is both the model input and the model output. It carries graph data,
optional supervision, and predictions without introducing a second output type.

Input fields:

```text
x            [N, D_in] node irreps flattened in declared order
pos          [N, 3] Cartesian coordinates
edge_index   [2, E] optional source-to-target edges
batch        [N] optional graph IDs; omitted for one graph
edge_type    [E] optional semantic edge relation IDs
group        [N] optional disconnected interaction-component IDs
condition    [B, C] or [N, C] optional invariant conditioning
order        optional semantic-order context
update_mask  [N] optional mask for coordinate updates
y            optional target
ids          optional sample identifiers
```

Output fields use the same object:

```text
out.x          [N, D_out] node prediction
out.graph_x    [B, D_out] graph-mean prediction
out.graph_sum  [B, D_out] graph-sum prediction
out.pos        [N, 3] final coordinates
out.delta      [N, 3] total coordinate displacement
```

Input topology and metadata remain attached to the returned graph. There are no
public dictionaries, string-key aliases, padded-output wrappers, or separate
output classes.

## One edge convention

Public `edge_index` follows the common source-to-target convention:

```text
edge_index[0] = sender/source
edge_index[1] = receiver/target
```

The numerical core uses a receiver-major packed representation. Conversion is
private and occurs exactly once at the public boundary.

When `edge_index` is omitted, ELA builds an exact directed radius graph. Automatic
radius graphs do not include self edges. Explicit edges are required for bonds,
mesh connectivity, temporal transitions, metal coordination, or other semantic
topologies.

## One coordinate-update policy

Coordinate behavior is declared when the model is constructed:

```python
fixed = ELA("32x0e", "1x0e", update_positions=False)
moving = ELA(
    "32x0e",
    "1x0e",
    update_positions=True,
    max_coordinate_step=0.2,
)
```

Calling either model remains identical:

```python
out = model(graph)
```

`update_positions=False` returns `out.pos == graph.pos` and a zero `out.delta`.
`update_positions=True` performs one hidden-state-preserving coordinate update
at every layer boundary and returns the final coordinates in the same
`ELAGraph`. The cumulative displacement remains bounded by
`max_coordinate_step`.

There is no public refinement request, rebuilder callback, or separate refiner
object.

## Batching

Dataset samples are ordinary `ELAGraph` objects. Variable-size graphs are packed
with one collator:

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=16,
    collate_fn=ELAGraph.collate,
)

for graph in loader:
    out = model(graph)
```

`ELAGraph.collate` offsets edges, builds graph IDs, and packs optional fields.
The numerical core never depends on PyG or DGL.

## Advanced configuration

Advanced reproducibility settings live outside the package root:

```python
from equivariant_linear_attention.advanced import ELAConfig, SparseGeometry

config = ELAConfig(
    input_irreps="32x0e",
    output_irreps="1x0e",
    geometry=SparseGeometry(cutoff=6.0, max_neighbors=64),
)
model = ELA.from_config(config)
```

Advanced configuration does not create another execution API. The call remains
`model(ELAGraph(...))` and the result remains `ELAGraph`.

## Internal contracts

Packed batches, prepared CSR topology, kernel outputs, and layer state are private
implementation details. They may be used by repository tests and benchmarks but
are not compatibility promises for downstream users.

A public API change is accepted only when it preserves or further simplifies the
following contract:

```text
ELAGraph -> ELA -> ELAGraph
```
