# Release review receipt

Date: 2026-07-29 KST

Scope: blocking correctness, equivariance, scientific/data contracts,
provenance, and evidence claims for the generic 3D architecture follow-up.

## Independent local review

An independent release-review agent inspected the integrated source and
executable contracts. It returned **GO** after the following findings were
reproduced and fixed:

1. transient `l=3` now executes before every selected layer, consumes only
   same-graph supplied edges inside local/relation cutoffs, and reaches the
   final scalar readout;
2. cached Clebsch--Gordan tensors cannot be mutated through returned values,
   and the reference tensor-product executor survives dtype/device moves and
   strict state round trips;
3. sparse backend receipts no longer claim nonexistent custom/segment
   operators, and `auto` ELL falls back to streamed CSR on excessive padding;
4. real-LBA O(3) smoke uses a determinant-`-1` transform, a finite
   dtype-aware hard tolerance, true per-update medians, and train-only labels;
5. SBDD graph receipt v2 hashes all model-visible annotations, split receipts
   reject duplicate/conflicting membership, screening metrics reject flattened
   multi-screen inputs, and affinity labels must match prediction direction;
6. the architecture matrix reports live feature/value widths, separates
   transient-workspace execution from sparse-local execution, and uses simple
   directed exact-`E=kN` topologies.

The integrated focused suite passed 86 tests and Ruff after these corrections.

## Final repository gates

- `bash scripts/check.sh fast`: passed, 1,182 tests, 86.87% coverage, Ruff,
  compileall, and CPU float64 ML smoke.
- `bash scripts/check.sh gpu`: passed, CUDA BF16 and FP32 ML smokes.
- real ATOM3D-LBA train-only receipts: CPU, seeded CUDA BF16, and strict CUDA
  completed with determinant-`-1` O(3) checks.
- representative CUDA architecture matrix: 72/72 rows completed after the
  skew/ELL auto-fallback correction.

## External-provider boundary

`oms peer-review` was attempted with Codex, Claude, and Antigravity providers.
The sandboxed attempt produced 0/3 registered results because its artifact lock
was outside the writable roots. An escalated retry was rejected because it
could transmit the private repository diff to external destinations without
separate explicit disclosure approval. No workaround was attempted, and this
receipt does **not** claim external-provider review.

## Claim boundary

The review supports the implemented and tested mechanics through the bounded
transient `l=3` path. It does not establish arbitrary-high-`l` numerical
stability, downstream accuracy, a fused sparse-kernel speedup, the full K04
resource grid, or architecture superiority.
