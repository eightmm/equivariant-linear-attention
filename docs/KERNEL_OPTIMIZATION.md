# ELA kernel optimization

ELA has one mathematical implementation and multiple execution backends. The
PyTorch path is the numerical reference; Triton is an optional acceleration path
and never changes model parameters, checkpoints, or architecture.

## 1. Execution boundary

Users always execute the public graph contract:

```python
out = model(graph)
```

Kernel microbenchmarks in this repository may time the private packed/CSR path to
exclude radius discovery, edge validation, COO sorting, CSR construction, and
graph-layout planning. Those private helpers are not a second public API and may
change without compatibility guarantees. End-to-end ingestion is reported
separately whenever it is measured.

ELA compilation uses the inference helper:

```python
from equivariant_linear_attention.inference import prepare_for_inference

compiled = prepare_for_inference(
    model,
    device="cuda",
    compile_model=True,
    compile_mode="reduce-overhead",
)
with torch.inference_mode():
    out = compiled(graph)
```

The helper deliberately compiles only the prepared numerical core. Public
validation, cache lookup, radius construction, pooling, and `ELAGraph` wrapping
stay eager. Recognized Dynamo/Inductor lowering failures emit a warning and
permanently return to the exact eager core; unrelated runtime errors remain
visible.
Direct `torch.compile(model)` is not the documented ELA path.

## 2. Implemented backend

The current Triton backend accelerates receiver-major CSR reductions.

For edge payload `x_e` and receiver row pointers:

$$
y_i =
\sum_{e=\text{ptr}_i}^{\text{ptr}_{i+1}-1}x_e.
$$

The Triton kernel:

- assigns programs to receiver rows and feature blocks;
- accumulates FP16/BF16/FP32 payloads in FP32;
- supports int32 and int64 row pointers;
- iterates to each row's runtime degree rather than a graph-wide padded maximum;
- copies non-contiguous raw row pointers and validates arbitrary raw CSR input;
- falls back for unsupported device or dtype;
- uses direct receiver-gradient gathers and CSR row broadcasts for ordinary
  first-order backward;
- falls back to differentiable PyTorch gathers for `create_graph=True` or unused
  grouped outputs.

Backend policy is selected by exporting exactly one of `auto`, `torch`, or
`triton`. For example:

```bash
export ELA_KERNEL_BACKEND=triton
```

The same policy can be selected task-locally without changing `os.environ`:

```python
from equivariant_linear_attention.kernels import kernel_backend

with kernel_backend("triton"):
    output = model(graph)
```

`auto` currently selects the PyTorch reference for every regime. `triton` fails
closed when the requested execution is unsupported. Automatic Triton selection
will be enabled only for a hardware/runtime-specific regime that passes the
complete-stack promotion gate below.

## 3. Local payload grouping

The canonical local operator computes positive receiver-normalized sufficient
statistics:

$$
w_{ijr} =
f_c(r_{ij})
\exp\left[3\tanh(a_{ijr}/3)\right],
$$

$$
S_{ir}^{f} =
\frac{\sum_j w_{ijr}\rho_{ijr}^{f}z_{jr}^{f}}
{1+\sum_j w_{ijr}}.
$$

The Triton training path keeps projections and edge-score equations in ordinary
PyTorch autograd, then reduces five lifetime-compatible groups with one compact
mass reduction and four specialized fused kernels:

1. mass and squared mass;
2. scalar and pseudoscalar values;
3. polar and axial vector values;
4. even and odd symmetric-traceless tensors;
5. three direction moments used for chirality.

The four value kernels perform sender gather, invariant/radial weighting, edge
carrier construction, and receiver reduction internally. This avoids one giant
concatenated `[E,F_all]` payload, the previous per-group `torch.cat`, and all
expanded `[E,R,C]` value payloads. Their custom autograd wrappers save compact
node/edge inputs and recompute exact differentiable gathers/scatters in backward,
retaining first and double backward.

Forward transport and saved activation memory are bounded by compact score/gate
inputs and node/output carriers:

$$
M_{\text{transport}} =
O(ER + NRC)
$$

at fixed carrier width `C`, rather than by every expanded `E x R x C` message
family. First-order backward deliberately rematerializes differentiable
edge-carrier products and can therefore have an `O(ERC)` transient. The CUDA
training profiler measures that actual forward/backward peak; the smaller bound
must not be quoted as a whole-training memory guarantee. The five semantic
launches remain; their separate pointers preserve tensor lifetimes and autograd
ownership.

Training and inference share the same fused boundary:

```text
sender gather x invariant weight x radial gate -> receiver CSR sum
```

For the tensor lane, the kernel additionally constructs the symmetric-traceless
edge-direction carrier; for the three chiral moments it applies the sender
direction gate and edge vector internally. Every Cartesian component receives
the same invariant scalar coefficient, so fusion changes execution only, not
the represented irrep or parity.

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
- required double backward through both PyTorch and forced Triton;
- FP16 with int64 CSR as well as BF16 with int32 CSR;
- forced-backend fail-closed behavior.

Current focused tests:

```bash
uv run pytest -q tests/test_kernel_triton.py
uv run pytest -q tests/test_kernel_triton_cuda.py
uv run pytest -q tests/test_triton_equivariance_cuda.py
```

The forced-Triton layer contract activates every `l<=2` parity sector and covers
generic improper O(3), translation, node and edge permutation, typed relations,
ragged graphs with an isolated receiver, coordinate refinement, all local
parameter VJPs, and conservative-force second derivatives.

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
stack time or memory without violating numerical tolerances. The benchmark uses
alternating Torch/Triton order, records both synchronized wall time and CUDA
events, and excludes cold Triton/Inductor compilation.

### Current promotion status

Bounded BF16 prepared-stack measurements on 2026-08-02 after the backward and
inference fusions found no latency-promotable regime:

| N | degree | E | inference Triton/Torch | F+B Triton/Torch | training peak Triton/Torch |
|---:|---:|---:|---:|---:|---:|
| 512 | 32 | 16,384 | 1.018 | 1.020 | 0.880 |
| 512 | 64 | 32,768 | 1.039 | 1.080 | 0.867 |
| 4,096 | 32 | 131,072 | 1.012 | 1.006 | 0.839 |
| 4,096 | 64 | 262,144 | 1.003 | 0.998 | 0.845 |

These are short single-machine diagnostics, not performance claims. They justify
keeping `auto` on PyTorch. The exact commands and mechanical receipts are in
`docs/EXPERIMENTS.jsonl`; neighbor discovery was excluded. Training medians were
noisy under shared-workstation load, so ratios near one are treated as ties.
Peak allocation was stable: forced Triton used `0.839--0.880x` training memory
and `0.843--0.956x` inference memory. Neither result crosses the documented
promotion threshold.

### Compiled prepared path

The same `N=512, k=32, width=64, depth=2` BF16 model was also measured with
the repository-private compiled prepared stack:

| path | eager | compiled | compiled/eager |
|---|---:|---:|---:|
| inference | 22.714 ms | 5.464 ms | 0.241 |
| forward/backward | 97.522 ms | 21.472 ms | 0.220 |

Cold compilation was about 6.9 seconds. Output maximum absolute difference was
`0.03125` at output magnitude `2.640625`; maximum feature, coordinate, and one
local-parameter gradient differences were `1.94e-5`, `3.62e-3`, and `6.11e-5`.
This is a strong static-shape speed diagnostic, not a general ragged-graph claim.
The default compiled backward rejected double backward because Inductor donated
non-empty buffers, so conservative-force/HVP workloads remain on eager execution.
A repeat with new tensor inputs and a live parameter perturbation matched eager
within `0.02344` and `0.03125`; the perturbation changed the compiled output by
`0.015625`, ruling out constant capture. A generic improper O(3) transform plus
translation changed the compiled scalar output by at most `0.015625` in BF16.

## 6. Automatic radius construction

The dependency-free radius builder uses:

- chunked exact dense pair tests for small graphs;
- an exact 3D cell list plus final distance filtering for larger graphs.

Under bounded density and fixed cutoff, cell-list work is expected to scale as
`O(N+E)`. Worst-case work remains quadratic when many nodes occupy one cell.

Graph-major collation forwards its already validated graph counts into batched
radius discovery and private CSR construction, avoiding a second batch scan and
post-discovery COO repack. The current cell-list discovery still performs one
stable `E`-sized receiver grouping before CSR offsets are formed; direct CSR
moves/removes packaging work but does not yet remove that discovery sort. The
27-cell traversal also contains host-controlled branches, so GPU latency is an
empirical benchmark question rather than an asymptotic speed claim.

The current builder is not periodic. Periodic and triclinic systems should supply
explicit candidates until minimum-image cell-list support is implemented and
validated.

## 7. Ragged global grouped GEMM

Highly ragged BF16 CUDA inference packs graph/head token segments once and uses
native grouped matrix multiplication for both `K^T V` summaries and `Q S`
application. It does not pad a node dimension or materialize node-wise outer
products. CPU, FP32, training, and higher-order gradients use the exact tiled
segmented fallback; that fallback is not described as a GEMM and retains
ordinary PyTorch double backward.

Balancing masses and the final normalized division stay in FP32. Only the two
grouped matrix products run in BF16, either for BF16 inputs or inside CUDA BF16
autocast. Native dispatch additionally requires inference/no-gradient operands;
ordinary training never enters this lane.

## 8. Remaining high-value optimization

The current Triton path removes graph-wide degree padding and payload
concatenation and fuses every local value family in training and inference. It
still materializes edge-level scores and gates in PyTorch, while its custom
backward recomputes edge gathers/scatters with ordinary PyTorch operations. The
next high-value step is a deterministic reverse-CSR training backward followed
by score/gate fusion, contingent on measured complete-stack benefit.

A production fused operator should combine:

1. displacement and squared distance;
2. cutoff and radial basis;
3. scalar/vector/axial/tensor invariants;
4. positive rank-R scores;
5. all receiver sufficient statistics;
6. normalization by `1 + mass`.

Learned `Linear` projections should remain in PyTorch/vendor GEMM. The fused
elementwise/reduction operator should avoid persistent intermediates shaped like:

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
