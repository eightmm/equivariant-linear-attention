# Scientific run: edge-free-spatial-linear-20260723

**Question:** Can the existing all-global equivariant factorized attention add a fixed-rank multi-scale Euclidean spatial kernel without edges or pair tensors, preserve symmetry and coordinate updates, and improve forward latency and peak memory against private static EGNN as E=kN grows?
**Review:** passed

## Plan

- **completed:** Freeze the edge-free kernel, models, grid, resource ceiling, falsifiers, and inference boundaries.
- **completed:** Write and retain intentional RED tests for the feature map, dense equivalence, symmetry, coordinates, benchmark schema, and nonfinite output.
- **completed:** Implement the opt-in spatial sufficient statistics, public configuration, coordinate-compatible plumbing, and benchmark entrypoint.
- **completed:** Pass focused regressions, the 396-test fast gate, and CUDA bf16/fp32 smoke.
- **completed:** Run and retain the initial grid, optimize disclosed spatial-only overhead, rerun the authoritative grid, and confirm the high-density crossover.
- **completed:** Validate the hash-bound provenance bundle, obtain independent record/method/source review, resolve findings, and publish the reviewed record.

## Claims

- **C-301:** The opt-in ten-feature/head spatial kernel is positive, O(3)-invariant, translation-invariant after graph normalization, permutation-equivariant, and exactly factorized without an edge list or node-pair tensor. — status=supported; inference=the implemented feature map, source inspection, dense-reference tests, and full-model symmetry tests; supports=3
  - Uncertainty: the finite degree-two feature map is a soft truncated spatial kernel, not an exact Gaussian or hard neighborhood cutoff
  - Next action: retain the fixed-rank map until a separate accuracy packet compares ranks or learned scales
- **C-302:** The spatial option is backward-compatible when disabled, adds no learned or persistent state when enabled, and supports the existing coordinate updater with finite gradients. — status=supported; inference=matched initialization and train-path tests for the public model; supports=3
  - Uncertainty: checkpoint interoperability was tested through state schema and tensors, not a historical production checkpoint archive
  - Next action: keep the option false by default
- **C-303:** On the recorded RTX PRO 6000 FP32 one-graph forward workload at N=8192, static edge-free spatial attention is faster and uses less measured working-plus-edge memory than private static EGNN at k=64 and k=128. — status=supported; inference=this hardware, software state, model shapes, synthetic input, forward-only protocol, and confirmed edge multipliers; supports=3
  - Uncertainty: the k=64 speed advantage is 2.7% and is hardware, dtype, shape, implementation, and timing-protocol dependent
  - Next action: treat k=128 as the robust systems win and remeasure k=64 after any environment or kernel change
- **C-304:** Coordinate-updating edge-free spatial attention is faster than private static EGNN at N=8192,k=64. — status=unsupported; inference=none; this registered prediction was tested and failed; contradicts=2
  - Uncertainty: dynamic attention was 3.6% slower at k=64 using EGNN latency as the denominator, but faster at exploratory k=80 and registered k=128
  - Next action: do not promote a dynamic k=64 crossover; optimize coordinate work only if accuracy later justifies it
- **C-305:** Edge-free spatial attention is universally faster or lower-memory than EGNN across small, sparse, and dense 3D workloads. — status=unsupported; inference=none; the universal systems claim is contradicted by the registered grid; contradicts=2
  - Uncertainty: different hardware, batching, compilation, precision, width, and graph construction can shift the crossover
  - Next action: select the edge-free path only for workloads whose measured N and density justify it
- **C-306:** The synthetic forward crossover establishes better molecular, protein, force, or point-cloud accuracy or preserves arbitrary omitted graph topology. — status=unsupported; inference=none; external-validity and representation boundary; contradicts=2
  - Uncertainty: no labels, task split, forces, chirality, graph-builder, backward pass, or domain data were evaluated
  - Next action: run a separate accuracy packet before any scientific-model promotion

## Evidence graph

- Nodes: 23; edges: 28; contradicts=6, derived\_from=6, supports=16
- Graph file: [evidence-graph.json](evidence-graph.json)

## Visual results

No raster image artifacts recorded.

## Files

- [analysis.json](analysis.json) — analysis; SHA-256 `7f436d7b71035a043acf015a0de8f05bdf5c83c663e48ccf284e35639a3f673a`
- [claim-register.json](claim-register.json) — claim-register; SHA-256 `5ec094349dc48a1618ec9994c5f1b6ae51247832926ff97e96b2dacdf2474391`
- [compute-approval.json](compute-approval.json) — compute-approval; SHA-256 `844a34e5e393c75c026a5a4514a37152902f616d9dd4db4bf81bdb265bc36549`
- [compute-environment.json](compute-environment.json) — environment; SHA-256 `89007c38cff0880a41e3070efb034204714373b2d12134299e1ebe61454960a9`
- [evidence-graph.json](evidence-graph.json) — evidence-graph; SHA-256 `3d2db854fecb31f1c236836af96887475b191b440cc46c0dc7fd4c423a782d5e`
- [execution-log.json](execution-log.json) — execution-log; SHA-256 `427e440ef368ceb0ed616c6cdec1029f826fdfc17079ff30519074b83187dd93`
- [gpu-benchmark.json](gpu-benchmark.json) — exploratory-benchmark-result; SHA-256 `d7f9243bdd68073b3181f1eb13d606038c5a7f57392bc980bf6bdf1782abd54a`
- [gpu-benchmark-optimized.json](gpu-benchmark-optimized.json) — benchmark-result; SHA-256 `1fbd92a15594d072a906a7a953fc069a7a178dd6ab3217b1c5f54ca2486afc71`
- [gpu-crossover-confirmation.json](gpu-crossover-confirmation.json) — benchmark-confirmation; SHA-256 `aab43e25532fa1446860a294f329a448f199d8e64c47119611b943f570940e04`
- [implementation-record.json](implementation-record.json) — implementation-record; SHA-256 `fff8115eee136c2f2f025ce9800e769b272ff3dada2567d29174030b1d758d32`
- [nonfinite-red-test.json](nonfinite-red-test.json) — test-first-record; SHA-256 `7a3c537a54f4d6a201df196cf549a14f0c46d3629f91e2ba0186a578035ce095`
- [red-tests.json](red-tests.json) — test-first-record; SHA-256 `e752e8faa957f0ef114511ab59d8d3cb5afee29883b4f90eb7341cc7545149da`
- [reference-use-ledger.json](reference-use-ledger.json) — reference-use-ledger; SHA-256 `2516935df462b1589455e2dab8155b06b24be15c4b393081fec7d3b91a58dce3`
- [report.md](report.md) — report; SHA-256 `d5a1f93fcb64a71e9653a709ce4af203481d6ebab7012d995e7497950c908be3`
- [reproduction-commands.json](reproduction-commands.json) — reproduction-command-record; SHA-256 `27e3f85a0b072e967654c59598a80d2083cc35a05579b3b2abdab3c4c3871b95`
- [review-resolution.json](review-resolution.json) — review-resolution; SHA-256 `69fa2e7f12409b74032a70ed6ce3f61dcda0f51d53dae7af8709f00c4e23787a`
- [review-task.json](review-task.json) — initial-review-task; SHA-256 `d6cc55e9ffdcf20be8d30fe4904f8e3bfa61c3e527972f39f490df492d3052f8`
- [reviewer-response.json](reviewer-response.json) — initial-review-response; SHA-256 `48081a67b0a5483630017340662f332208d426f63941028771854686a09203ef`
- [review-receipt-initial.json](review-receipt-initial.json) — initial-review-receipt; SHA-256 `471367a3389a6fba234383181b73c08eb0d1d8192dd2cd2901d30945b9be38f8`
- [review-task-rereview.json](review-task-rereview.json) — final-review-task; SHA-256 `a08f1eda83c7e3479325e3e2774a7b16a058863b2b24d6e0f18598fa433dd9ee`
- [reviewer-response-rereview.json](reviewer-response-rereview.json) — final-review-response; SHA-256 `523186c9a383e8c47c1e3d36c0f35922376edcd966d4436a74cce138eb9959b5`
- [review-receipt-v2.json](review-receipt-v2.json) — final-review-receipt; SHA-256 `a1e75faad3fd5c30b67d97843d8267226f264f4d80e103f6c7508e55f2779317`
- [review-checkpoint.json](review-checkpoint.json) — checkpoint-compatible-review-receipt; SHA-256 `b308f5a6ed781051b11971e5261b8fa1bec681ea40a5a15a24b4df1450bdd84b`
- [record-review.json](record-review.json) — deterministic-record-review; SHA-256 `901c63d7d00aa05be77f85b488662475ee888b1ce79414f4d60f515898faa0df`
- [scope.md](scope.md) — protocol; SHA-256 `cbed5f337efd06a13ae4ee3b7b212e1900c272bd7eb5e6d1ff8259034ccd9754`
- [source-changes.patch](source-changes.patch) — source-diff; SHA-256 `30afa9840add9c43460f1b275257de9946b878c4ff14b71c80a6fd54368a9d63`
- [verification-summary.json](verification-summary.json) — verification-summary; SHA-256 `eba1adb363dce6615c9cdcc61617233cee59da23a6b009de6032f850cd6ddc77`

_Generated from `manifest.json` and validated hashed sidecars; this index is a derived view, not evidence._
