# Task recipes

All tasks use the same boundary:

```python
from equivariant_linear_attention import ELA, ELAGraph

out = model(ELAGraph(x=x, pos=pos, batch=batch))
```

## Graph properties

Use `output_irreps="Kx0e"`. Read `out.graph_x` for an intensive mean readout or
`out.graph_sum` for an extensive sum readout.

## Node, vector, and tensor targets

Read `out.x`. Declare polar vector targets as `Kx1o`, axial targets as `Kx1e`,
and symmetric-traceless tensor targets as `Kx2e` or `Kx2o`. Use
`split_irreps` from the advanced module to recover structured blocks. For ST5
targets, use `st5_mse`; ordinary five-coordinate MSE is not rotation invariant.

## Flow matching and score fields

Keep `update_positions=False`, set `output_irreps="Kx1o"`, and interpret the
polar-vector block of `out.x` as velocity or score. Time or noise level is an
invariant `condition` field on `ELAGraph`.

## Learned coordinate updates

Set `update_positions=True` and read `out.pos` and `out.delta`. An optional
boolean `update_mask` fixes selected nodes. This is a learned refinement path;
it is not required to produce an equivariant vector field.

## Conservative energy and forces

Predict invariant scalar node energies, sum them with `out.graph_sum`, and use
`conservative_forces` from the advanced module to differentiate with respect to
input coordinates. A direct `1o` output is equivariant but is not necessarily a
conservative force.

## Typed or disconnected graphs

Supply explicit source-to-target `edge_index` and `edge_type` for semantic
relations. Use `group` to isolate disconnected interaction components inside a
sample; global and local transport stay inside each group while readout remains
sample-level.
