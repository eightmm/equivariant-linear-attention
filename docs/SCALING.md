# ELA scaling contract

This document states exactly when the single canonical ELA architecture may be
described as linear. Algorithmic arithmetic, wall-clock time, neighbor
construction, refinement, and training activation memory are separate claims.

## 1. Base layer stack

Let

- \(N\) be the node count;
- \(E\) be the directed candidate-edge count;
- \(L\) be the number of `ELALayer` applications;
- channel widths, head count, local rank, radial rank, and multipole rank be
  fixed.

One layer contains:

- an exact finite-feature global operator with \(O(N)\) arithmetic;
- an exact sparse local operator with \(O(E)\) arithmetic;
- an invariant branch router and pointwise update with \(O(N)\) arithmetic.

Therefore

\[
\boxed{
T_{\rm ELA}=O\left(L(N+E)\right)
}
\]

and no `N x N` attention tensor is materialized.

## 2. Node-linear condition

The expression above does not by itself imply node-linear scaling. Node-linear
scaling requires a graph family satisfying

\[
E=O(N).
\]

Examples:

- fixed-degree kNN;
- bounded-density radius graphs;
- fixed-valence meshes;
- fixed-cutoff Verlet candidates under bounded density.

A complete candidate graph has

\[
E=\Theta(N^2),
\]

so the local part becomes

\[
O(LN^2).
\]

Candidate edges outside the smooth cutoff still incur indexing and geometry
cost if they are present in the prepared graph.

## 3. Optional context

Invariant condition and semantic-order PE are node-level context. With fixed
context width they add

\[
O(LN)
\]

and do not change the base asymptotic order.

If both are absent, their layer modulation is bypassed entirely. Allocating the
capability does not force its runtime cost on a context-free call, aside from the
small parameter storage of dormant modules.

## 4. Coordinate refinement

A `RefinementRequest` with \(S\) outer update steps evaluates the ELA stack once
per update and once more at the final geometry:

\[
T_{\rm refine}
=
O\left((S+1)L(N+E)\right)
\]

while candidate topology is fixed.

Each update recomputes graph centering, edge displacement, distance, cutoff,
radial features, and node multipoles.

Graph reconstruction is external. Typical provider costs are:

- cell/Verlet list under bounded density: expected \(O(N+E)\);
- spatial tree: commonly \(O(N\log N+E)\);
- brute-force pair search: \(O(N^2)\).

No end-to-end linear claim may exclude a rebuild that is actually performed.

## 5. Memory

### No-gradient inference

At fixed widths and ranks,

\[
M_{\rm infer}=O(N+E).
\]

This includes node state, graph metadata, geometry, and bounded layer
workspace.

### Training without activation checkpointing

Autograd may retain node and edge activations for every depth:

\[
M_{\rm train}=O\left(L(N+E)\right)
\]

up to fixed channel/rank factors.

### Refinement training

Backpropagating through all \(S\) refinement steps without recomputation gives a
conservative bound

\[
M_{\rm refine,train}
=
O\left((S+1)L(N+E)\right).
\]

Outer-step checkpointing or truncated-gradient policies must be reported when
used.

## 6. Why wall-clock need not be perfectly linear

Big-O arithmetic does not imply exact proportional device time. Relevant effects
include:

- kernel launch and Python dispatch overhead at small sizes;
- changing GPU occupancy;
- tensor-core utilization;
- memory-bandwidth saturation;
- allocator and cache behavior;
- degree skew in sparse reductions;
- graph padding and bucketing;
- autograd saved tensors;
- optional graph reconstruction.

Measured wall-clock linearity must be established for the target hardware and
batch regime.

## 7. Required benchmark sweeps

### Node count

Hold depth and degree fixed and sweep

\[
N\in\{128,512,2048,8192,32768\}.
\]

Fit

\[
\alpha_N
=
\frac{d\log t}{d\log N}.
\]

For a fixed-degree graph, the expected arithmetic slope is approximately one.

### Depth

Hold \(N\) and degree fixed and sweep

\[
L\in\{2,4,8,16,32\}.
\]

Expected arithmetic slope:

\[
\alpha_L\approx1.
\]

### Degree

Hold \(N,L\) fixed and sweep

\[
k\in\{8,16,32,64,128\}.
\]

The sparse-local contribution should be approximately affine in

\[
E=kN.
\]

The global linear-attention contribution remains independent of degree.

### Context

Compare the same trained architecture with:

```text
context absent
invariant condition only
semantic order only
condition + order
```

Record latency and peak memory. The context-free path must not execute learned
conditioner modulation.

### Refinement

Sweep

\[
S\in\{0,1,2,4,8\}
\]

with and without graph reconstruction. Record ELA time and rebuild time
separately.

### Batch shape

Compare:

- one large graph;
- uniform small graphs;
- strongly ragged graphs;
- skewed degree distributions.

## 8. Measurement protocol

For each setting record:

- warmup count;
- repeat count;
- median and p90 latency;
- forward-only and forward+backward separately;
- peak allocated and reserved device memory;
- node and edge counts;
- graph count and degree distribution;
- dtype and autocast policy;
- whether graph packing is included;
- whether neighbor discovery or rebuild is included;
- git SHA and environment receipt.

Prepared-model and end-to-end measurements must be labeled separately.

## 9. Safe public statement

> For a precomputed directed candidate graph with \(N\) nodes and \(E\) edges,
> fixed architectural widths and ranks, the ELA stack has
> \(O(L(N+E))\) forward arithmetic and does not materialize an \(N\times N\)
> attention tensor. It is linear in node count when \(E=O(N)\). Neighbor
> discovery is excluded unless explicitly included in the reported measurement.

For refinement add:

> With \(S\) outer coordinate-update steps, ELA stack arithmetic is
> \(O((S+1)L(N+E))\), excluding any separately reported neighbor reconstruction.
