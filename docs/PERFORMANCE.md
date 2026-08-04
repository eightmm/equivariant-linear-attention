# Performance

ELA separates graph preparation from tensor-only numerical work:

```text
ELAGraph validation and topology discovery
    -> private packed receiver-major CSR
    -> ELA numerical stack
    -> ELAGraph output
```

An `ELAGraph` may retain a private prepared topology. Explicit edges can be
reused after coordinates move because distances are recomputed. Automatic
radius topology records cutoff, neighbour cap, relation schema, skin, and
reference coordinates; incompatible or stale caches are rebuilt.

Automatic radius graphs exclude self edges. `max_neighbors=k` therefore means
up to `k` non-self distance shells, with all exact ties at the boundary kept to
preserve permutation equivariance.

PyTorch is the canonical backend. Triton is an explicit, memory-oriented
experimental backend; it is not selected automatically because recorded
complete-stack timings did not establish a stable latency gain. Static prepared
workloads may benefit from `torch.compile`, but force, HVP, and other
double-backward paths should remain eager until that contract is supported.
Highly ragged BF16 CUDA inference uses PyTorch's native grouped matrix multiply;
training and unsupported dtype/device combinations use the exact tiled
segmented reduction fallback.

Benchmarks must state whether they include:

- graph ingestion and radius discovery;
- cold compilation;
- forward only, backward, or an optimizer step;
- static explicit topology, cold radius build, skin reuse, or moving-coordinate
  rebuild;
- latency percentiles and peak allocated memory.

`scripts/benchmark_ela.py` preserves the prepared numerical-stack benchmark by
default. Add `--include-end-to-end` to measure the complete public API:

```bash
uv run python scripts/benchmark_ela.py \
  --nodes 4096 --degree 32 \
  --include-end-to-end --e2e-nodes 512 --e2e-degree 16 \
  --device cuda --dtype bfloat16 \
  --output artifacts/ela-complete-stack.json
```

The end-to-end section contains four independently timed lanes: cold automatic
radius discovery, repeated trusted reuse of a primed immutable prepared cache,
cold explicit COO topology including CSR conversion, and
`update_positions=True` stagewise
coordinate execution. Every lane records `graph_ingestion_included`; each lane
also truthfully records whether neighbor discovery ran in the timed region.
The moving lane reports the exact number of interstage radius refreshes
(`depth - 1`); at depth one coordinates move but no final, unused topology is
rebuilt.
The section-level `neighbor_discovery_included=true` means at least one lane
measures it. Latency output includes p50/p90/p95/p99 and raw samples. CUDA runs
also report peak allocated bytes; CPU peak memory is explicitly unavailable.

For the current optimized paths, run the bounded CUDA evidence suite:

```bash
uv run --locked python scripts/capture_source_manifest.py \
  artifacts/source-manifest.json
uv run --locked python scripts/run_gpu_gate.py \
  --source-manifest artifacts/source-manifest.json \
  --output artifacts/gpu-gate-receipt.json
uv run --locked python scripts/profile_gpu_completion.py \
  --source-manifest artifacts/source-manifest.json \
  --output artifacts/gpu-completion-profile.json
```

The gate wrapper leaves a source-bound `running`, `passed`, or `failed` receipt;
the final adjudicator does not accept a caller-supplied success code. The GPU
gate and profiler must identify the same manifest path, file hash, combined
source hash, and file count.

It measures explicit immutable prepared-cache reuse plus safe unsealed DLPack
invalidation, native grouped-MM against the same BF16 inference fallback,
automatic-radius/direct-CSR ingestion, complete
Torch-versus-forced-Triton forward/backward, and private-core compilation. The
JSON retains raw timings, medians, p95, incremental/absolute peak allocation,
numerical errors, compiler fallback/recompile telemetry, and explicit acceptance
thresholds. A failed threshold makes the corresponding lane and process fail;
it is not rewritten as a speedup claim. Every timed lane also fails at an
absolute peak allocation of 16 GiB or greater.

The profiler alternates like-for-like inference lanes. Its training setup clears
retained gradients before taking the allocation baseline. The compilation lane
requires a compiler graph to be observed, rejects steady or topology-only
recompilation, and compares warm compiled execution against the same public
prepared eager path. Forced Triton evidence records all scalar/vector, tensor,
and directional fused primitive dispatches in addition to comparing outputs and
input/parameter gradients. Direct-CSR evidence requires both canonical topology
and final-output parity.

Ordinary public `ELAGraph` tensors are never admitted by identity/version alone.
They can be exported through DLPack and changed without advancing the original
tensor's mutation counter. Safe-default reuse therefore compares public topology
content exactly before entering the internally trusted packed path. This avoids
rebuilding CSR/radius topology, but validation remains O(E) for explicit edges
or O(N) for radius positions.

For a genuinely E-independent cache hit, call `graph.assume_immutable()` before
the first execution. It clones all topology-bearing storage and enables a
version/schema fast path that reuses the previously packed carrier itself. This
is an explicit lifetime promise: mutating or
exporting mutable aliases of the returned `pos`, `edge_index`, `batch`,
`edge_type`, or `group` tensors violates the contract. The profiler tests both
this trusted explicit-edge lane and DLPack invalidation on the unsealed
safe-default lane. Call `assume_immutable()` before `torch.inference_mode()`;
otherwise unavailable version counters cause a safe exact-validation fallback.

Bounded QM9 and ATOM3D-LBA architecture checks are documented in
[`REALDATA_VALIDATION.md`](REALDATA_VALIDATION.md). They include graph ingestion
inside each training step but deliberately label sequential-arm latency as
observational.

Historical measurements and their limitations are recorded in
`docs/KERNEL_OPTIMIZATION.md` and `docs/EXPERIMENTS.jsonl`.

The fast verification gate additionally builds a wheel, installs it without
dependencies into an isolated temporary `uv` environment, imports it from
outside the source tree, and verifies that its declared package-root surface is
exactly `ELA` and `ELAGraph`.
