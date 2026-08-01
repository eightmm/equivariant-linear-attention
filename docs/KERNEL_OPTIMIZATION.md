# Kernel optimization plan

ELA keeps a PyTorch numerical reference and promotes custom kernels only after
correctness and end-to-end performance gates pass. Triton or CUDA is an execution
backend, not a second architecture or public model option.

## 1. First optimize the execution boundary

Before writing a custom kernel:

1. use packed flat node tensors;
2. prepare and reuse `Prepared3DGraph`;
3. exclude neighbor discovery and CSR packing from the timed layer unless the
   benchmark explicitly measures end-to-end ingestion;
4. try `torch.compile` on the prepared-graph path;
5. profile forward, backward, saved tensors, kernel launches, and peak memory.

```python
graph = model.prepare_graph(pos, batch=batch, edge_index=edge_index)
compiled = torch.compile(model, mode="reduce-overhead")
out = compiled(x, pos, graph)
```

The convenience API remains outside the hot-path contract. It is acceptable for
`model(batch_dict)` or automatic radius construction to perform Python-side
normalization; performance claims must use the prepared flat path.

## 2. Optimization priority

### Priority 1: fused receiver-major local operator

This is the highest-value custom kernel.

For receiver `i`, local rank `r`, and edge `j -> i`, the PyTorch reference
computes:

\[
w_{ijr}
=
f_c(r_{ij})
\exp\left[3\tanh(a_{ijr}/3)\right],
\]

\[
m_{ir}=\sum_j w_{ijr},
\]

and multiple normalized sufficient statistics:

\[
S_{ir}^{f}
=
\frac{\sum_j w_{ijr}\rho_{ijr}^{f}z_{jr}^{f}}
{1+m_{ir}}.
\]

A fused kernel should consume:

```text
positions
receiver CSR row_ptr
sender index
optional relation ID
normalized node states
radial parameters
projection weights
```

and write receiver-level outputs only. It should fuse:

1. displacement and squared distance;
2. C2 cutoff and radial basis;
3. scalar/vector/axial/tensor invariants;
4. positive rank-R score;
5. receiver mass and squared mass;
6. scalar, polar, axial, relative-vector, even/odd tensor, and chiral direction
   accumulators;
7. normalization by `1 + mass`.

The production kernel should not materialize persistent tensors shaped like:

```text
[E, R]
[E, R, D]
[E, R, 3]
[E, R, 5]
```

One program may own one receiver or one `(receiver, rank-block)` pair. Degree
buckets already present in `PackedNeighborGraph` can select low-, medium-, and
high-degree launch configurations without changing the mathematical path.

### Priority 2: local backward with recomputation

The backward should avoid saving all edge intermediates. Save compact node
states, CSR metadata, and the minimum normalization statistics; recompute edge
geometry and scores during backward.

Suggested split:

- receiver-side gradients: forward CSR;
- sender-side gradients: reverse CSR or atomics;
- coordinate gradients: recomputed displacement path;
- FP16/BF16 accumulation: FP32;
- FP64 reference lane: PyTorch fallback until a verified FP64 kernel exists;
- double backward: PyTorch fallback unless explicitly implemented.

### Priority 3: exact on-the-fly radius candidates

Automatic radius construction currently performs exact chunked pair tests with
quadratic arithmetic inside each graph. A production path should use a cell
list or Verlet list.

Recommended pipeline:

```text
positions
→ graph-aware cell key
→ sort/bucket by key
→ cell offsets
→ adjacent-cell traversal
→ exact distance test
→ fused local accumulation
```

The user still supplies no `edge_index`, but neighbor discovery remains real.
Under bounded density and fixed cutoff, expected work is `O(N+E)`; worst-case
work remains quadratic when all points occupy one cell.

Sorting and cell-list construction may be better implemented with PyTorch/CUB or
a C++/CUDA extension, while Triton handles adjacent-cell traversal and message
accumulation. A pure Triton implementation should be promoted only if it beats
that hybrid on the target GPUs.

### Priority 4: small pointwise fusion

EqRMSNorm, branch routing, bounded gates, and per-copy LayerScale are candidates
for Inductor fusion. They should be left to `torch.compile` before introducing
hand-written kernels.

### Lower priority: global ELA

The global branch is already expressed primarily as GEMM/BMM and graph-level
sufficient-statistic contractions. Vendor BLAS and Inductor generally have a
better optimization opportunity than a custom hand-written kernel. A custom
global kernel is justified only by profiling evidence such as excessive padding
or launch overhead in highly ragged batches.

## 3. Backend policy

The default installation remains PyTorch-only.

A custom backend must be:

- optional and import-safe;
- selected automatically from device and capability;
- semantically identical to the reference;
- bypassed for unsupported dtype/device/gradient order;
- invisible to `ELAConfig` and checkpoint schemas;
- overrideable through an execution/debug environment variable, not an
  architecture option.

Suggested internal policy:

```text
ELA_KERNEL_BACKEND=auto      # default
ELA_KERNEL_BACKEND=torch     # numerical/debug reference
ELA_KERNEL_BACKEND=triton    # fail if unavailable or unsupported
```

This environment variable is an execution control, not a model hyperparameter.
It must not change parameters, state dicts, or mathematical outputs.

## 4. Triton suitability

Triton is a good fit for the local operator because it combines irregular CSR
row traversal, elementwise geometry, and several reductions that currently
require many kernel launches and edge intermediates.

It is less attractive for:

- framework-independent CPU support;
- graph sorting and dynamic memory allocation;
- very small graphs dominated by launch overhead;
- FP64 and higher-order derivative reference paths;
- standard GEMMs already handled by optimized libraries.

The official Triton project currently targets Linux, NVIDIA GPUs with compute
capability 8.0 or newer, and AMD GPUs with ROCm 6.2 or newer; CPU support remains
under development. Therefore Triton cannot be the only ELA backend.

## 5. Correctness gates

Every custom kernel must be compared with the PyTorch reference on the same
weights and graph metadata.

Required fixtures:

```text
empty edge set
singleton graph
zero-neighbor receiver
self edges
uniform degree
highly skewed degree
multiple batched graphs with overlapping coordinates
relation IDs and narrowed cutoffs
proper and improper O(3) transforms
node permutation
FP32, BF16, and supported FP64
```

Required comparisons:

1. forward output for every message family;
2. node-feature gradients;
3. coordinate gradients;
4. parameter gradients;
5. first-order training update;
6. graph isolation;
7. deterministic behavior when requested;
8. fallback correctness;
9. double-backward capability receipt.

Initial tolerances should be predeclared, for example:

```text
FP64 forward:          atol/rtol <= 1e-9
FP64 first gradients:  atol/rtol <= 1e-8
FP32 forward:          atol/rtol <= 2e-5
BF16:                  finite plus task-specific relative tolerance
```

## 6. Performance gates

Benchmark at least:

```text
N:          128, 512, 2k, 8k, 32k
mean degree:8, 16, 32, 64, 128
topology:   uniform, ragged, hub/skewed
batching:   one graph, uniform many-graph, strongly ragged
dtype:      FP32, BF16
mode:       forward, backward, optimizer-inclusive step
```

Record separately:

- neighbor discovery;
- COO/CSR packing;
- local operator;
- complete ELA layer;
- complete stack;
- peak allocated and reserved memory;
- saved tensor bytes;
- kernel launch count;
- compile time and steady-state time.

A reasonable promotion target is:

```text
>= 20% lower forward+backward layer time
or
>= 25% lower training peak memory
```

without a task-metric regression outside the predeclared tolerance. Small-graph
regression should remain below 10%; otherwise keep a size-based PyTorch fallback.

## 7. Existing benchmark entry points

Input and graph-preparation overhead:

```bash
uv run python scripts/benchmark_input_pipeline.py \
  --graphs 8 \
  --nodes-per-graph 64 \
  --degree 16 \
  --width 64 \
  --depth 4 \
  --device cuda \
  --dtype bfloat16 \
  --compile-prepared \
  --output artifacts/input-pipeline.json
```

Canonical model overhead:

```bash
uv run python scripts/benchmark_canonical_ela.py \
  --nodes 4096 \
  --degree 32 \
  --width 128 \
  --depth 8 \
  --device cuda \
  --dtype bfloat16 \
  --output artifacts/canonical-overhead.json
```

A future Triton benchmark must use the same prepared graph and same model state,
then report reference/custom ratios rather than isolated kernel throughput only.

## 8. Recommended implementation order

1. Run the input-pipeline and canonical benchmarks.
2. Profile the prepared flat path with `torch.profiler` and Nsight Systems.
3. Enable `torch.compile`; retain only stable gains.
4. Implement a fused local forward kernel behind an internal capability check.
5. Add reference-equivalence and BF16 CUDA tests.
6. Implement recompute backward.
7. Add a graph-aware cell-list builder only after local fusion is stable.
8. Promote to `auto` only after multi-GPU-architecture benchmarks.

Until those gates pass, the PyTorch path remains canonical and custom kernels
remain execution optimizations rather than advertised model features.
