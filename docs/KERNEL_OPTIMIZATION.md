# ELA kernel optimization

ELA has one mathematical implementation and multiple execution backends. The
PyTorch path is the numerical reference; Triton is an optional acceleration path
and never changes model parameters, checkpoints, or architecture.

## 1. Prepared execution boundary

Performance measurements use one prepared `ELABatch`:

```python
batch = model.prepare(batch)
output = model.forward_prepared(batch)
```

This excludes radius discovery, edge validation, COO sorting, CSR construction,
and graph-layout planning from the timed stack. End-to-end ingestion must be
reported separately when it is measured.

`torch.compile` should be tested on this path:

```python
compiled = torch.compile(model.forward_prepared, mode="reduce-overhead")
output = compiled(batch)
```

## 2. Implemented backend

The current Triton backend accelerates receiver-major CSR reductions.

For edge payload `x_e` and receiver row pointers:

\[
y_i
=
\sum_{e=\operatorname{ptr}_i}^{\operatorname{ptr}_{i+1}-1}x_e.
\]

The Triton kernel:

- assigns programs to receiver rows and feature blocks;
- accumulates FP16/BF16/FP32 payloads in FP32;
- supports int32 and int64 row pointers;
- buckets maximum degree by powers of two to limit compile variants;
- falls back for unsupported device, dtype, graph size, or degree;
- uses a differentiable receiver-gradient gather in backward.

Backend policy:

```bash
ELA_KERNEL_BACKEND=auto
ELA_KERNEL_BACKEND=torch
ELA_KERNEL_BACKEND=triton
```

Dispatch controls:

```bash
ELA_TRITON_MIN_EDGES=256
ELA_TRITON_MAX_DEGREE=2048
```

`auto` uses PyTorch below the minimum edge count to avoid launch-overhead
regression. `triton` fails closed when the requested execution is unsupported.

## 3. Local payload grouping

The canonical local operator computes positive receiver-normalized sufficient
statistics:

\[
w_{ijr}
=
f_c(r_{ij})
\exp\left[3\tanh(a_{ijr}/3)\right],
\]

\[
S_{ir}^{f}
=
\frac{\sum_j w_{ijr}\rho_{ijr}^{f}z_{jr}^{f}}
{1+\sum_j w_{ijr}}.
\]

The Triton path keeps projections and edge-score equations in ordinary PyTorch
autograd, then reduces five lifetime-compatible payload groups:

1. mass and squared mass;
2. scalar and pseudoscalar values;
3. polar and axial vector values;
4. even and odd symmetric-traceless tensors;
5. three direction moments used for chirality.

This avoids one giant concatenated `[E,F_all]` payload. Its temporary memory is
bounded by the largest group:

\[
M_{\rm payload}
=
O\left(E\max_k F_k\right)
\]

instead of the sum of every message family. The trade-off is several CSR kernel
launches rather than one.

## 4. Correctness contract

Every backend comparison uses the same model weights, prepared graph, and input
values.

Required checks include:

- empty edge set and zero-neighbor receivers;
- uniform and skewed degree;
- multiple graph segments;
- relation IDs;
- proper and improper O(3) transforms;
- node permutation and edge-order invariance;
- FP32 and BF16;
- node-feature, coordinate, and parameter gradients;
- required double backward through the PyTorch reference;
- forced-backend fail-closed behavior.

Current focused tests:

```bash
uv run pytest -q tests/test_triton_ops.py
uv run pytest -q tests/test_triton_ops_cuda.py
```

## 5. Benchmark

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

Record:

- PyTorch/Triton output maximum error;
- feature-gradient error;
- coordinate-gradient error;
- local-parameter-gradient error;
- inference latency and peak allocated memory;
- forward/backward latency and peak allocated memory;
- node, edge, degree, width, depth, and dtype;
- exclusion of neighbor discovery.

A backend should be promoted for a graph regime only when it improves complete
stack time or memory without violating numerical tolerances.

## 6. Automatic radius construction

The dependency-free radius builder uses:

- chunked exact dense pair tests for small graphs;
- an exact 3D cell list plus final distance filtering for larger graphs.

Under bounded density and fixed cutoff, cell-list work is expected to scale as
`O(N+E)`. Worst-case work remains quadratic when many nodes occupy one cell.

The current builder is not periodic. Periodic and triclinic systems should supply
explicit candidates until minimum-image cell-list support is implemented and
validated.

## 7. Remaining high-value optimization

The current Triton path accelerates reductions but still materializes edge-level
scores, gates, and message groups in PyTorch. The next high-value step is a fused
local forward that consumes node states, positions, CSR metadata, and parameters,
and writes receiver statistics directly.

A production fused operator should combine:

1. displacement and squared distance;
2. cutoff and radial basis;
3. scalar/vector/axial/tensor invariants;
4. positive rank-R scores;
5. all receiver sufficient statistics;
6. normalization by `1 + mass`.

It should avoid persistent intermediates shaped like:

```text
[E, R]
[E, R, D]
[E, R, 3]
[E, R, 5]
```

The backward should recompute compact edge quantities rather than save every
edge activation. This larger kernel must not replace the existing reference
until feature, coordinate, parameter-gradient, equivariance, BF16, and memory
benchmarks pass.

## 8. Lower-priority targets

Global ELA is primarily GEMM/BMM plus graph sufficient-statistic contractions.
Vendor libraries and Inductor generally have better optimization opportunities
there than a handwritten sparse kernel.

EqRMSNorm, branch routing, bounded gates, and LayerScale should also be left to
`torch.compile` unless profiling identifies a stable unfused bottleneck.

## 9. Performance policy

Recommended sweeps:

```text
N:           128, 512, 2k, 8k, 32k
mean degree: 8, 16, 32, 64, 128
topology:    uniform, ragged, hub/skewed
batching:    one graph, uniform many-graph, strongly ragged
dtype:       FP32, BF16
mode:        inference, forward/backward, optimizer-inclusive step
```

A reasonable promotion target is at least one of:

```text
>= 20% lower forward/backward stack time
>= 25% lower training peak memory
```

with no numerical or task-metric regression outside the predeclared tolerance.
Small-graph regression should remain below 10%; otherwise retain a size-based
PyTorch fallback.
