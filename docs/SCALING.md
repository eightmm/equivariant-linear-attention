# Scaling contract and benchmark protocol

This document states exactly when the repository may claim linear complexity.
Algorithmic order, measured wall-clock time, neighbor construction, and training
activation memory are separate claims.

## 1. Base equivariant linear attention

For

- \(N\) nodes;
- \(E\) directed candidate edges;
- \(L\) layers;
- fixed hidden width, head count, radial rank, local rank, and multipole rank;

one layer contains an \(O(N)\) exact finite-feature global operator and an
\(O(E)\) sparse local operator. Therefore

\[
T_{\rm base}=O\left(L(N+E)\right).
\]

This does **not** by itself imply node-linear scaling. Node-linear scaling
requires a candidate family with

\[
E=O(N).
\]

Examples include fixed-degree kNN, bounded-density radius graphs, and fixed
valence meshes. A complete candidate graph has \(E=\Theta(N^2)\), yielding

\[
T_{\rm base}=O(LN^2).
\]

Edges outside the smooth cutoff still incur candidate geometry and score costs
if they are present in the prepared graph.

## 2. Coordinate updates

With fixed candidate topology, coordinate refinement recomputes graph centers,
normalized coordinates, edge geometry, RBF values, cutoff weights, and node
multipoles after each layer. This is \(O(N+E)\) per layer and does not change the
asymptotic order:

\[
T_{\rm coordinate}=O\left(L(N+E)\right).
\]

Neighbor-list rebuild is external. Its cost depends on the provider:

- cell/Verlet list: expected \(O(N+E)\) under bounded density;
- spatial tree: commonly \(O(N\log N+E)\);
- brute-force pair search: \(O(N^2)\).

No end-to-end linear claim may omit that distinction.

## 3. Attention Residuals

With \(B\) retained block-level depth sources, the two depth routers per layer
add an \(O(LBN)\) term:

\[
T_{\rm AttnRes}
=
O\left(L(N+E)+LBN\right).
\]

If \(B\) is fixed independently of depth, the model remains linear in \(L\). If
\(B=\Theta(L)\), depth routing contributes

\[
O(L^2N).
\]

Inference cache adds \(O(BN)\) hidden storage. Training activation memory can
contain an \(O(LBN)\) term unless checkpointing or recomputation removes it.

## 4. Edge-free implicit spatial kernel

For finite feature rank \(F\), transported value width \(D\), \(A\)
applications, \(G\) graphs, and bounded node chunk size \(C\),

\[
T_{\rm implicit}=O(ANFD),
\]

\[
M_{\rm implicit}
=
O\left(N(F+D)+GFD+CFD\right).
\]

The \(GFD\) term stores one sufficient statistic per graph. The \(CFD\) term is
the bounded chunked outer-product workspace. The implementation does not create
a full \(NFD\) temporary. For fixed \(F,D,C\), arithmetic is node-linear and
memory is linear in total input size. The claim is about a smooth low-rank
spatial-kernel approximation, not exact hard-cutoff neighborhoods.

## 5. Memory statements

The phrase “persistent state is \(O(N)\)” refers to one inference hidden state.
More complete bounds are:

### Base no-grad inference

\[
M_{\rm infer}=O(N+E).
\]

### Base training without activation checkpointing

\[
M_{\rm train}=O\left(L(N+E)\right)
\]

up to fixed channel/rank factors.

### AttnRes inference

\[
M_{\rm infer,AttnRes}=O(N+E+BN).
\]

### AttnRes training

A conservative activation upper bound is

\[
M_{\rm train,AttnRes}
=
O\left(L(N+E)+LBN\right).
\]

## 6. Why wall-clock may not look linear

Big-O arithmetic does not imply exact proportional GPU time. Relevant effects
include:

- kernel launch and Python dispatch overhead at small sizes;
- changing occupancy and tensor-core utilization;
- padded or bucketed graph schedules;
- memory-bandwidth saturation;
- allocator and cache behavior;
- degree skew in sparse reductions;
- graph-count growth in per-graph sufficient statistics;
- autograd saved tensors and recomputation.

Measured wall-clock linearity must therefore be established empirically.

## 7. Benchmark harness

Use

```bash
uv run python scripts/benchmark_scaling.py \
  --modes base,attnres,implicit \
  --nodes 256,512,1024,2048,4096,8192 \
  --depths 4,8,16,32 \
  --blocks 4,8 \
  --degree 32 \
  --warmup 10 \
  --repeats 30 \
  --device cuda \
  --dtype bfloat16 \
  --output artifacts/scaling.json
```

The harness records:

- prepared-model forward or forward+backward latency;
- optional CSR graph-pack-inclusive latency;
- peak allocated CUDA memory;
- \(N,E,L,B\), feature rank, and symbolic formula;
- log-log node-size slope.

It deliberately records

```text
neighbor_discovery_included = false
```

because the synthetic benchmark starts from a fixed-degree candidate topology.
`--include-graph-pack` adds CSR packing but still does not perform geometric
neighbor discovery.

## 8. Required sweeps

### Node scaling

Hold \(L,k,B\) fixed and sweep

\[
N\in\{128,512,2048,8192,32768\}.
\]

Fit

\[
\alpha_N=\frac{d\log t}{d\log N}.
\]

Expected arithmetic slope for fixed-degree base and fixed-rank implicit kernels:

\[
\alpha_N\approx1.
\]

### Depth scaling

Hold \(N,k\) fixed and sweep

\[
L\in\{2,4,8,16,32\}.
\]

Expected:

- base: \(\alpha_L\approx1\);
- AttnRes with fixed \(B\): \(\alpha_L\approx1\);
- AttnRes with \(B=L\): \(\alpha_L\approx2\) once routing dominates.

### Degree scaling

Hold \(N,L\) fixed and sweep

\[
k\in\{8,16,32,64,128\}.
\]

The local term should be approximately affine in \(E=kN\), while the global
term remains unchanged.

### Batch-shape scaling

Compare:

- one large graph;
- many uniform small graphs;
- strongly ragged graph batches;
- skewed degree distributions.

This identifies padding, bucketing, graph-statistic, and segment-reduction
overhead that a simple Big-O formula does not expose.

## 9. Promotion language

A safe public statement is:

> For a precomputed directed candidate graph with \(N\) nodes and \(E\) edges,
> fixed architectural widths and ranks, the base equivariant linear-attention
> stack has \(O(L(N+E))\) forward arithmetic and does not materialize an
> \(N\times N\) attention tensor. When \(E=O(N)\), it is linear in node count.
> Neighbor construction is excluded.

For AttnRes add:

> With \(B\) retained block sources, total arithmetic is
> \(O(L(N+E)+LBN)\). Depth linearity requires \(B\) to remain bounded
> independently of \(L\).

For the implicit kernel add:

> The edge-free spatial approximation has \(O(ANFD)\) arithmetic at fixed
> finite feature rank and uses chunked per-graph sufficient statistics; it
> approximates a smooth isotropic kernel rather than an exact radius graph.
