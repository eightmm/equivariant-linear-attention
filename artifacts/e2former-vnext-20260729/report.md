# E2Former-informed equivariant linear-attention vNext

## Outcome

The admitted architecture packet is implemented as opt-in, backward-compatible
mechanics for generic 3D point clouds and sparse 3D graphs. The core remains an
exact edge-free global factorized attention model; sparse geometry is now an
additive low-rank residual instead of a head replacement.

This is a mechanics result, not an accuracy promotion. No QM9, LBA, PDBBind,
point-cloud benchmark, CUDA profile, or test split was run in this packet.

## Implemented architecture

1. **Homogeneous global plus sparse block.** Every selected block retains all
   global heads. A rank-`R`, edge-state-free local residual can be refreshed on
   any explicit layer schedule. It transports scalar, vector, relative
   direction, traceless rank-2 tensor, radial trace, and optional persistent
   `2e` value information. General scalar transport is `O(E R D_head)` and
   becomes `O(E R)` at fixed head width.
2. **Exact feature-GEMM global backend.** The configured constant, scalar,
   linear, quadratic, and optional distinct spatial query/key terms are lifted
   to explicit features. Per-graph summaries are then matrix multiplications;
   no node-pair matrix is formed.
3. **Independent balancing controls.** Local and global key balancing can now be
   selected independently. Leaving them unset inherits the legacy
   `use_key_balancing` value and preserves the incumbent state/function.
4. **Packed sparse execution.** Stable receiver-major CSR, optional
   sender-major reverse CSR, and safe int32 metadata support segment reductions
   and reverse receiver accumulation without an `N x N` hop matrix. The model
   consumes these plans directly and restricts them after cutoff filtering
   without sorting again.
5. **Generic static irrep planning.** `IrrepLayout` represents arbitrary
   nonnegative `l` and parity; `TensorProductPlan` applies triangle and parity
   selection rules and binds known executors. The production numerical backend
   remains the fast Cartesian `0e/1o/2e` path and fails explicitly for
   unregistered paths.

## Verification

The repository fast gate passed:

- 775 tests;
- 88.86% line coverage;
- Ruff and `compileall`;
- CPU float64 ML smoke.

Focused tests cover exact global value and input-gradient equivalence,
fixed/inverse graph-size baselines, balanced/unbalanced kernels, multiple
graphs, widened values, distinct spatial query/key features, O(3) including
reflection, translation, node and edge permutation, batch isolation, finite
gradients, two-step zero-init wake-up, arbitrary-depth schedules, packed
adjacency round trips, CSR/reverse-CSR reductions, compact index dtypes, and
generic irrep planner failure boundaries.

## Bounded CPU diagnostics

Two post-review same-recipe one-thread PyTorch 2.12.1 diagnostics were run with
five warmup and 21 measured forward/backward calls. Global and reduction
microbenchmarks differentiate every tensor input. Hybrid timings include full
parameter and input backward but exclude the optimizer.

| Comparison | First ratio | Repeat ratio | Interpretation |
| --- | ---: | ---: | --- |
| feature GEMM / outer-scatter, `N=288,H=4,Fsp=40,V=37` | 0.1130 | 0.1155 | Directionally stable CPU speedup; max float32 forward difference `5.36e-7` |
| prepacked receiver reduction / index-add, `N=4096,k=32` | 0.6367 | 0.6303 | Reduction-only speedup; packing and cutoff work excluded; exact forward equality |
| homogeneous sparse / gated LGL, `N=256,E=2048,R=4`, two of three layers | 1.1305 | 1.1312 | Sparse candidate is slower at this point |
| homogeneous sparse / all-global | 1.4393 | 1.4368 | Added geometry has a real CPU cost |

The homogeneous residual saved-tensor payload was `0.9547x` gated LGL but
`1.5259x` all-global, and state bytes were `1.0808x` all-global. Zero-initialized
forward difference was exactly zero. These numbers support backend mechanics
only; they do not establish a GPU or end-to-end crossover.

## Independent-review corrections

The first independent review found two blocking execution-contract defects:

1. packed neighbors were unpacked to COO and then sorted again;
2. the highly ragged feature-GEMM fallback scanned all nodes once per graph and
   repeatedly copied a full-size output.

Both were corrected before completion. Packed geometry now consumes and
linearly filters its receiver/reverse plans, with a regression test forbidding
sorting and row-pointer reconstruction. Ragged GEMM now groups once, slices
contiguous graph ranges, concatenates once, and restores node order once; an
unsorted many-singleton backward test forbids per-graph `nonzero` rescans.
Reverse CSR construction now also validates that sender rows agree with the
declared reverse pointer. The reviewer additionally caused timing labels to be
narrowed and hybrid measurements to include parameter backward.

## Feedback items intentionally deferred

- arbitrary-`l` Clebsch-Gordan/Wigner numerical execution;
- ELL, Triton, or fused custom CUDA kernels;
- graph-size auto-dispatch thresholds;
- persistent edge state or an edge-width hidden MLP;
- pocket/ligand semantics in the architecture core;
- default promotion of any new path.

Those choices need either a missing functional requirement or a measured CUDA
and downstream bottleneck. Implementing them now would enlarge the state and
maintenance surface without evidence that they solve the current limitation.

## Decision

Admit the packet as experimental, opt-in infrastructure. The feature-GEMM and
CSR backends are promising CPU implementations, but no default changes are
justified yet. The homogeneous residual is architecturally cleaner and more
expressive than replacing global heads, but its measured CPU cost means its
next gate must be utility per resource on a real 3D task, not another synthetic
feature addition.
