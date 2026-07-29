# E2Former-informed vNext implementation protocol

Date: 2026-07-29

## Question

Can the current exact factorized equivariant attention be made more homogeneous,
more extensible in irreps, and cheaper in its dominant global/local reductions
without changing the incumbent default function or introducing a dense
node-by-node tensor?

## Admitted implementation packet

1. Add an opt-in homogeneous block path:
   every layer retains all exact global heads and selected layers add an
   edge-state-free, separable, rank-`R` sparse residual.
2. Separate legacy local and global key-balancing controls while preserving
   `use_key_balancing` as the default inherited setting.
3. Add an opt-in exact explicit-feature GEMM reduction for the ordinary global
   factorized kernel. It must reproduce scalar, constant, linear, quadratic,
   and optional distinct query/key spatial feature blocks.
4. Add receiver-CSR neighbor metadata and an explicit receiver
   `segment_reduce` path with int32 indices where safe.
5. Add a behavior-neutral generic irrep layout/tensor-product planning layer.
   The numerical production backend remains the canonical Cartesian
   `0e/1o/2e` fast path.

All new numerical paths are explicit and default-off. Existing LGL routes,
state dictionaries, RNG consumption, and default outputs remain unchanged.

## Acceptance criteria

- No hot-path tensor has both node axes `(N, N)`.
- Sparse residual work is `O(E R D_head)` plus node projections and
  fixed-coordinate equivariant lanes (`O(E R)` at fixed head width), with no
  persistent edge state or edge MLP.
- Every homogeneous block has `global_head_count == num_heads`; the sparse
  residual schedule is independent of global rank and supports arbitrary
  positive depth.
- Reflection-inclusive O(3), translation, node permutation, edge order, batch
  isolation, finite coordinate/feature gradients, and two-step parameter
  wake-up are checked for the sparse residual.
- Explicit-feature GEMM matches the incumbent outer/scatter implementation in
  forward values and gradients across balanced/unbalanced, fixed/inverse
  graph-size baselines, multiple graphs, distinct spatial query/key features,
  and widened value channels.
- CSR packing round-trips, preserves int32 metadata, supplies reverse receiver
  metadata when requested, and receiver segment sums match index-add in values
  and gradients. The model consumes packed receiver/reverse plans directly;
  cutoff restriction must not re-sort them.
- Highly ragged feature-GEMM groups nodes once and uses graph-offset slices,
  without a graph-by-node rescan or repeated full-size output copy.
- Generic irrep planning parses arbitrary nonnegative `l` and parity, applies
  triangle/parity rules, and fails explicitly when a requested numerical path
  has no registered executor.
- `scripts/check.sh fast` passes.

## Bounded performance evidence

Measure CPU forward/backward mechanics for:

- incumbent outer/scatter versus exact feature GEMM;
- prepacked COO index-add versus receiver CSR segment-reduce, explicitly
  excluding CSR construction;
- incumbent all-global block versus the enabled sparse residual at fixed state.

These are diagnostic mechanics only. No CUDA, downstream accuracy, QM9, LBA,
PDBBind, or production crossover claim will be made without a separately
authorized and preregistered run.

## Explicit non-goals

- No Triton kernel, EAAS/Wigner implementation, ELL backend, threshold-based
  auto-dispatch, or unmeasured default promotion.
- No persistent/public `l >= 3`, arbitrary Clebsch-Gordan numerical executor,
  spherical-harmonic dependency, or claim of a production-generic irrep model.
- No pocket/ligand-specific semantics in the core architecture.
- No claim that the new architecture improves downstream accuracy until
  real-data confirmation is run.
