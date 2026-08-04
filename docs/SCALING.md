# Scaling contract

Let:

- `N` be packed node count;
- `E` be directed prepared candidate count;
- `L` be layer count;
- `C` summarize fixed scalar/geometric widths;
- `S` be model-declared coordinate-update steps.

## Prepared stack

Every layer contains exact global sufficient-statistic attention, exact sparse
local reduction, fixed additive branch fusion, tensor closure, and pointwise
updates. At fixed widths and ranks:

$$
T_{\text{stack}}=O\left(L(N+E)\right).
$$

Node-linear arithmetic additionally requires `E=O(N)`. The global branch does
not build a dense `N x N` matrix. Fixed `G+L` fusion is pointwise and adds no
learned routing MLP or message-RMS pass.

Activation memory is implementation and autograd dependent, but the structural
prepared tensors are `O(N+E)`.

## Radius discovery

Small interaction groups use a chunked exact dense reference. Larger groups use
an exact batched 3D cell list followed by distance filtering. Under fixed cutoff
and bounded spatial density the expected work is

$$
O(N+E).
$$

Worst-case work remains quadratic when many nodes occupy one cutoff cell.
`max_neighbors` keeps nearest complete distance shells and therefore may exceed
`k` only at an exact tie. Automatic candidates exclude self edges.

Radius discovery, COO validation/sorting, CSR construction, and graph-layout
planning are excluded from prepared-stack timings unless explicitly stated.

## Moving coordinates

With skin `s`, candidates are built to cutoff `r_c+s` and reused while the
maximum node displacement from the reference is at most `s/2`. Geometry and the
physical cutoff are recomputed every forward. A rebuild restores the exact
candidate superset.

`max_neighbors` and skin caching are mutually exclusive because a capped nearest
set is not guaranteed to remain exact while coordinates move.

## Coordinate updates

A model configured with `S` bounded coordinate updates performs approximately
`S+1` stack evaluations:

$$
T_{\text{update}}=
O\left((S+1)L(N+E)\right),
$$

plus any required radius reconstruction. The update count is a model property,
but the public execution remains one `model(graph)` call.

## Width scaling

Scalar width is `width`. Geometric multiplicity is derived monotonically from
width and can grow beyond eight copies for wider models; local rank also grows
for the wider regime. Users do not select raw head or rank values. The exact
component cost is therefore configuration dependent even though the asymptotic
`N+E` dependence is unchanged.

## Interaction groups

`ELAGraph.group` partitions global summaries and radius construction without
changing sample-level output pooling. Complexity depends on total node and edge
counts; splitting one sample into components can reduce the worst-case global
layout and radius-discovery constants while enforcing component additivity.

## Backends

PyTorch is the numerical reference. `prepare_for_inference(...,
compile_model=True)` isolates compilation to the private numerical core;
repository benchmarks may separately time a private prepared stack. Compiler
failure warns and falls back to exact eager execution. Triton is an explicit
optional backend whose current value is
primarily memory reduction; backend selection does not
change the equations or complexity class. Higher-order autograd workloads remain
on eager execution unless separately validated.
