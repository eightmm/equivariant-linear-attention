# Scientific run: train-step-scaling-20260723

**Question:** Does edge-free spatial-linear attention retain a large dense-graph efficiency advantage after backward and AdamW, and what explains any discrepancy from forward-only timing?
**Review:** passed

## Plan

- **completed:** P1
- **completed:** P2
- **completed:** P3
- **completed:** P4
- **completed:** P5
- **completed:** P6

## Claims

- **C1:** The benchmark measures an eager FP32 zero-grad, forward, synthetic MSE, backward, and AdamW step with isolated absolute peak CUDA allocation. — status=supported; inference=software and measurement contract on the recorded environment; supports=2
  - Uncertainty: Profiler overhead and model construction are excluded; contrary to one scope sentence, creation of the one-element constant target with torch.full\_like is included in every timed arm and is recorded as a protocol deviation
  - Next action: Preserve the same boundary in future comparisons
- **C2:** The preregistered unoptimized static model beats EGNN train-step latency at N=8192,k=128. — status=unsupported; inference=none; the registered hypothesis was falsified; contradicts=1
  - Uncertainty: Single GPU and synthetic loss
  - Next action: Retain the null result beside the post-outcome optimization
- **C3:** After the disclosed single-graph broadcast optimization, static edge-free attention is faster than private static EGNN at N=8192,k=64 and k=128. — status=supported; inference=post-outcome descriptive same-harness systems result; supports=1
  - Uncertainty: Not preregistered confirmation; no task accuracy or multi-graph claim
  - Next action: Confirm multi-graph batching separately
- **C4:** At N=8192,k=128, optimized static edge-free attention uses less absolute peak CUDA allocation than private static EGNN. — status=supported; inference=descriptive memory result under the recorded isolated-process accounting; supports=1
  - Uncertainty: Allocator and hardware specific
  - Next action: Retain absolute and delta memory fields

## Evidence graph

- Nodes: 7; edges: 5; contradicts=1, supports=4
- Graph file: [evidence-graph.json](evidence-graph.json)

## Visual results

No raster image artifacts recorded.

## Files

- [scope.md](scope.md) — protocol; SHA-256 `54ff1fb48d00bb47489993db83e6b1066f7ca0d8236f44b3b5084322c9304857`
- [optimization-scope.md](optimization-scope.md) — protocol; SHA-256 `02b8528f686ae5b85f0fbc1c93cb85accd9f0327911171e29dc721ebaafc64f9`
- [claim-register.json](claim-register.json) — claim-register; SHA-256 `06346123f753cf335adc4f2b967ba0684f9a06465d6cdba8d31749939363ce6c`
- [evidence-graph.json](evidence-graph.json) — evidence-graph; SHA-256 `dcc6776b6590a0b2c249fd6d36efa9968bb2661c4fa1b4166aedd560c4a4d8c7`
- [execution-record.json](execution-record.json) — execution-log; SHA-256 `295afbae4a087a1b4621f7a4e7a080478480e80f3424f8a5093e1b19927cb8cb`
- [protocol-deviation.json](protocol-deviation.json) — decision-log; SHA-256 `3bf00cdd7bfdb9986989649470754aa5f7fa779dcd065f9a6a6ca8a879121a43`
- [compute-environment.json](compute-environment.json) — environment; SHA-256 `075541dd24bf04b28097f385822ff36fbd53e055c3ce6a6a2ff382e804f1b36c`
- [reference-use-ledger.json](reference-use-ledger.json) — decision-log; SHA-256 `60082d4915095f47f7a5401245fc0b95e6a75f87ecee91c5af8e93857ed9cacd`
- [gpu-grid.json](gpu-grid.json) — metrics; SHA-256 `4f6c1ac0675268bafa237bca71ac416edd3581e54f0a134974cf20785f1dbe54`
- [gpu-grid-optimized.json](gpu-grid-optimized.json) — metrics; SHA-256 `a6ae787b91c0f8670f0458c122b4ca9e3a5d75ef0758f9219b38c45bf30feeda`
- [profile-spatial-static-n8192-k128.json](profile-spatial-static-n8192-k128.json) — profile; SHA-256 `20b0b576547814ac709394dd412172e86e49fd648870742b71f811d5d7fd32a1`
- [profile-static-egnn-n8192-k128.json](profile-static-egnn-n8192-k128.json) — profile; SHA-256 `cf82b875e0c81f01922d1a3ba9d6d0f0e1cc4b7b1b6cb11dd7eb4609bbb89d98`
- [profile-spatial-static-n8192-k128-optimized.json](profile-spatial-static-n8192-k128-optimized.json) — profile; SHA-256 `0a280d78badfd1008ecc41a72ec7355d8fe28b86aa917caf8a8bc505620a1a3c`
- [report.md](report.md) — report; SHA-256 `f2bf2e25e2a9384e4243b5c02efcfd66715f7dbd76cf8233b2fed10b84b77d71`
- [reviewer-response-initial.json](reviewer-response-initial.json) — decision-log; SHA-256 `6f3a1583f259c4e7c045eab7bb5df488302d321e6eae1386559eb658bb3ad53b`
- [review-task.json](review-task.json) — decision-log; SHA-256 `e58bff6e24270d11b4b577f77fdc148148ddd9b98b30b2d382fe8b61afdc6e47`
- [reviewer-response.json](reviewer-response.json) — decision-log; SHA-256 `5663f3bfbfcc0ab927408e80673b76fbd4e813a2edafcf29c67b44238447258d`
- [review-receipt-v2.json](review-receipt-v2.json) — review-receipt-v2; SHA-256 `1cd7179d00ead8b6ed5956ee8d67adb8308a2fc77cf74dd388f6c42b99d5215f`

_Generated from `manifest.json` and validated hashed sidecars; this index is a derived view, not evidence._
