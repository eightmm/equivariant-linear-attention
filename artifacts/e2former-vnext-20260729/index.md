# Scientific run: e2former-vnext-20260729

**Question:** Can the current exact factorized equivariant attention be made more homogeneous, more extensible in irreps, and cheaper in its dominant global/local reductions without changing the incumbent default function or introducing a dense node-by-node tensor?
**Review:** passed

## Plan

- **completed:** Audit the shared feedback against the current implementation and freeze the admitted packet.
- **completed:** Create behavioral and mathematical contract tests.
- **completed:** Implement the homogeneous residual, execution backends, static planner, and documentation.
- **completed:** Run focused validation, two bounded CPU diagnostics, and the repository fast gate.
- **completed:** Package provenance, independently review the bounded claims, resolve findings, and close the run.

## Claims

- **C-VNEXT-1:** The explicit-feature GEMM backend computes the same configured factorized global attention function as the incumbent outer/scatter backend for the covered scalar, spatial, batching, graph-size, and value-width cases. — status=supported; inference=Bounded software and numerical-equivalence claim; not a universal floating-point identity proof.; supports=3
  - Uncertainty: Mixed precision, CUDA kernels, and configurations outside the tested feature construction remain unmeasured.
  - Next action: Profile CUDA train steps before considering automatic backend selection.
- **C-VNEXT-2:** The opt-in homogeneous sparse low-rank residual preserves the covered O(3), translation, permutation, batching, compatibility, and gradient contracts while keeping all global heads active. — status=supported; inference=Implementation-contract claim for the registered Cartesian 0e/1o/2e numerical path.; supports=2
  - Uncertainty: The source and tests establish the implemented mechanics, not downstream usefulness or an asymptotic lower-bound proof.
  - Next action: Run a matched real-task ablation only after a CUDA train-step resource gate.
- **C-VNEXT-3:** Packed receiver CSR and optional reverse CSR are consumed by the model without resorting, reproduce the covered COO receiver reductions in values and gradients, and retain compact int32 metadata when the graph range permits it. — status=supported; inference=Bounded implementation and representation claim.; supports=3
  - Uncertainty: CUDA segment-reduction speed and very large graphs requiring int64 are not measured.
  - Next action: Measure external packing cost and a CUDA forward/backward crossover before default promotion.
- **C-VNEXT-4:** The generic IrrepLayout and TensorProductPlan can describe arbitrary nonnegative angular momentum and parity and statically select triangle/parity-compatible paths, while unavailable numerical executors fail explicitly. — status=supported; inference=Static metadata and planning claim only.; supports=2
  - Uncertainty: This is not a production-generic arbitrary-l equivariant numerical model and does not execute l&gt;=3 tensor products.
  - Next action: Add numerical executors only with a concrete task and resource justification.
- **C-VNEXT-5:** On the two post-review one-thread CPU diagnostics, feature GEMM and prepacked receiver reduction were faster than their comparison operators, while the homogeneous sparse residual was slower than both all-global and gated-LGL controls and only slightly reduced saved-tensor payload versus gated LGL. — status=supported; inference=Descriptive, workload-specific CPU operator evidence.; supports=2
  - Uncertainty: The receiver result excludes CSR construction; hybrid timing includes parameter/input backward but no optimizer. Two same-machine diagnostics do not establish GPU, large-N, mixed-precision, production, or end-to-end performance.
  - Next action: Use synchronized CUDA forward/backward/peak-allocation profiles if compute is authorized.
- **C-VNEXT-6:** This run does not establish improved QM9, LBA, PDBBind, point-cloud, or other downstream accuracy, nor a CUDA latency or memory advantage. — status=supported; inference=Explicit evidence-boundary statement.; supports=2
  - Uncertainty: Downstream utility remains unknown rather than negative.
  - Next action: Treat all new numerical paths as opt-in until a preregistered real-data and CUDA gate is completed.

## Evidence graph

- Nodes: 13; edges: 14; supports=14
- Graph file: [evidence-graph.json](evidence-graph.json)

## Visual results

No raster image artifacts recorded.

## Files

- [protocol.md](protocol.md) — protocol; SHA-256 `f7ef7b0aa097d5b8af23adb3a67daa79c2be29671d13de9b6d94e8174c02caf4`
- [shared-feedback-summary.md](shared-feedback-summary.md) — decision-log; SHA-256 `9af6f6cd031fdf2150c7bd345b107fa17b18d50c33a5970c3eeca91319b732b0`
- [claim-register.json](claim-register.json) — claim-register; SHA-256 `4c0bf2e1969ac7ed9630403ff38d64055d21be3404a23a8eee5ab6e49fae5fc2`
- [evidence-graph.json](evidence-graph.json) — evidence-graph; SHA-256 `78ab937c3546708d99e5d270e280477ecfae0bb76ecc50d15822b2c1ad889a71`
- [compute-environment.json](compute-environment.json) — environment; SHA-256 `8740745734bbfaaeded90fb9ae52c4fe6e8c25f7a8dfd9ec6eb58d771b26cf7e`
- [source-files.json](source-files.json) — code-provenance; SHA-256 `1549a2bd1ee15501e163f497c7b6ebd0690f8be0a8b82f145a48a01089ff7c6b`
- [execution-log.json](execution-log.json) — execution-log; SHA-256 `5f08e05327938fcf5bc24eebe42a6a489514bd8a54e655c5f1f759ca64e2a34c`
- [cpu-backend-probe.json](cpu-backend-probe.json) — metrics; SHA-256 `849ada867ddf50d0153785e4940b27abd59e578793baddd7e14de87ca28758bf`
- [cpu-backend-probe-final.json](cpu-backend-probe-final.json) — metrics; SHA-256 `167efd8afb353325f653d611e3ea3d9407d7e9fed6b366e47b9c88a94d6ffe5c`
- [cpu-backend-probe-post-review.json](cpu-backend-probe-post-review.json) — metrics; SHA-256 `71d98b8deb8cce8982855222949a4fcee5eb4621d7a1d478a890f8aa54f87fef`
- [cpu-backend-probe-post-review-repeat.json](cpu-backend-probe-post-review-repeat.json) — metrics; SHA-256 `eba1e4263a74c0709f9f553af2d634c71599e7c3da69cc328cbc97eb37d6655d`
- [reference-use-ledger.json](reference-use-ledger.json) — reference-use-ledger; SHA-256 `08d65f6ed738a88c33c7724b036cf1bacefd657b4a70a1d7d8a8940148c3b30f`
- [report.md](report.md) — report; SHA-256 `06243c5f18fca454f112525ba32efdc19c211956c59c0b921aa0f9be1600d1ff`
- [review-task.json](review-task.json) — review-task; SHA-256 `6d90d06b1a0ea4c53ec160969bb0a3ae97a83fbe0bafa3a24b872ecf77dea4bf`
- [reviewer-response.json](reviewer-response.json) — reviewer-response; SHA-256 `e034721ec9c85b944baf5b3295f5b837d29136703841b52151c177d3b0aa53fe`
- [review-receipt-v2.json](review-receipt-v2.json) — review-receipt-v2; SHA-256 `624b204e7436a9e22d2f9fd4fdaf8a3ed470274766b364ea8088e43e0b71d8c4`
- [checkpoint-review.json](checkpoint-review.json) — checkpoint-review; SHA-256 `088506478deacdeca3b46e7a37982f643e98ac90ac9fce93a7a9fda0c14f26d8`

_Generated from `manifest.json` and validated hashed sidecars; this index is a derived view, not evidence._
