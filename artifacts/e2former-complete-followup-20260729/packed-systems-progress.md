# Packed/global systems final receipt

Date: 2026-07-29

This is the final S3 systems receipt for the implemented software mechanics. It
records equation-preserving execution paths and their executable contracts. It
does **not** promote a backend, establish target-GPU speed or memory gains, or
claim downstream model quality.

## Packed sparse topology

- `PackedNeighborGraph.to(same_device)` returns the validated object itself.
  Cross-device transfer uses a trusted constructor and retains receiver CSR,
  optional reverse CSR, relation IDs, degree statistics, and ELL metadata
  without repeating semantic validation.
- Receiver-major and reverse sender-major plans have independent builders.
  Reverse metadata is requested only by routes that need sender-side
  balancing; receiver-only sparse residual and streamed routes do not pay for
  it.
- Addressable indices remain int32, with an explicit int64 overflow fallback.
  Stable receiver order, zero-degree rows, max degree, fixed degree buckets,
  histogram, and skew are validated metadata.
- Optional ELL is a lossless view of the same directed edge list and is guarded
  by maximum-degree, padding-ratio, and element-budget limits. Reverse CSR
  changes traversal order only; it never applies a semantic reverse-relation
  mapping.
- Typed relation IDs and per-relation scalar cutoffs/biases share one sparse
  edge list. Overlapping distance bands are additive residuals, not a partition
  of unity.

## Exact global layout and dispatch

- `PackedGraphLayout` is cached on `GraphBatch` and reused by the model.
  Supplied metadata avoids repeated graph-size extraction, sorting, and host
  synchronization in the forward path.
- `global_reduction_backend="auto"` deterministically selects direct GEMM,
  padded/bucketed BMM, ragged grouped GEMM, or an extreme-ragged/small-work
  outer/scatter fallback from static layout metadata.
- Hardware-width padding is exact zero padding. Every lane evaluates the same
  finite global feature map, numerator, and denominator; dispatch changes
  execution only, not attention mathematics.
- If layout metadata is omitted, constructing it remains caller-visible
  forward cost. The receipt therefore does not erase graph packing or
  discovery from resource accounting.

## Geometry, local streaming, and cache policy

- Edge admission compares normalized squared distance against one, avoiding
  `sqrt` and unsafe raw float32 cutoff squaring. RBF centers/widths are
  nonpersistent buffers, and receiver sorting preserves RBF/relation alignment.
- Geometry cache modes `full`, `compact`, `recompute`, and deterministic
  `auto` are explicit. Coordinate-updating models refresh geometry after each
  coordinate step instead of consuming stale cached displacement.
- Receiver-streamed CSR and ELL PyTorch references implement both positive
  mass-damped transport and receiver-local softmax. Sensitive reductions use
  FP32 for BF16/FP16 inputs, and gradgrad requests have an explicit PyTorch
  fallback receipt.
- The streaming implementations are correctness references. No fused
  CUDA/Triton forward, custom reverse-CSR backward, or measured
  kernel-residency/load claim is part of this packet.

## Neighbor providers and execution receipts

- Precomputed, deterministic reference-radius, external-adapter, and reference
  Verlet providers expose capability receipts rather than silently claiming
  production behavior.
- Reference radius construction is deterministic but quadratic. Reference
  Verlet implements skin/rebuild semantics; neither it nor the external
  adapter is represented as a production cell list or PBC implementation.
- `ExecutionMetadata` records requested and effective global, local, cache,
  provider, symmetry, and fallback choices. A deterministic receipt proves
  routing identity, not performance superiority.
- The optional ELL view is admitted only when its padding ratio is within the
  configured bound. In `auto`, excessive skew now keeps receiver CSR and falls
  back to the streamed PyTorch route instead of converting a safe dispatch
  choice into an exception. Explicit `ell` remains fail-closed.

## Representative CUDA matrix

- `architecture-matrix-cuda-representative.json` completed 72/72 rows on the
  NVIDIA RTX PRO 6000 Blackwell Max-Q: `N={128,512}`, `k={4,32}`, all
  uniform/skew/ragged topologies, and the six registered architecture arms.
- Every topology is a simple directed graph with exactly one self edge per
  node and exactly `E=kN`; construction/packing time is reported separately
  and excluded from both forward and optimizer-inclusive train-step timing.
- The matrix deliberately uses one timed repeat and only a bounded discrete
  width search. Five non-reference arms miss the 1% parameter-matching target,
  so raw latency and memory values are mechanics diagnostics, not fair
  architecture rankings or promotion evidence.
- The sparse-residual arms execute the PyTorch streamed CSR reference and are
  substantially slower in this small/medium regime. This is evidence that a
  fused sparse kernel is still needed, not evidence for a sparse-hybrid
  accuracy or efficiency advantage.

## Evidence

- Packed topology and metadata:
  `tests/test_packed_neighbors.py`, `tests/test_typed_relations.py`.
- Exact global layout/dispatch:
  `tests/test_graph_layout.py`, `tests/test_global_feature_gemm.py`,
  `tests/test_execution_metadata.py`.
- Geometry/cache/streaming:
  `tests/test_geometry_cache_modes.py`, `tests/test_local_streaming.py`,
  `tests/test_streamed_sparse_integration.py`,
  `tests/test_local_radial_basis.py`.
- Provider contracts:
  `tests/test_neighbor_providers.py`.
- Integrated real-data wiring receipts:
  `real-lba-cpu-smoke.json`, `real-lba-cuda-bf16-smoke.json`,
  `real-lba-cuda-strict-smoke.json`.
- Representative CUDA resource receipt:
  `architecture-matrix-cuda-representative.json`.

## Explicit remaining evidence gaps

- No fused custom CUDA/Triton kernel, custom streamed backward, or production
  cell-list/PBC neighbor builder is implemented.
- The representative target-GPU subset is executed, but the full K04
  `N x k x topology` grid through `N=8192,k=128` has not been executed.
- Parameter and optimizer-state matching, train-step matching, downstream
  accuracy, and the registered speed/memory promotion thresholds remain
  experimental gates.
- S7 is complete: independent local release review returned GO after the
  reproduced blockers were fixed; `scripts/check.sh fast` passed 1,182 tests
  at 86.87% coverage, and `scripts/check.sh gpu` passed BF16 and FP32 CUDA
  smokes. See `release-review.md`. External-provider review remains explicitly
  unclaimed.
