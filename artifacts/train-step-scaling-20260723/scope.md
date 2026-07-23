# Full train-step scaling contract

Frozen before the registered GPU execution on 2026-07-23.

## Question and hypothesis

Question: does the edge-free spatial-linear model retain its forward-only
large/dense efficiency advantage after autograd and AdamW state are included?

Hypothesis: at `N=8192,k=128`, static spatial-linear attention has lower median
full train-step latency and lower absolute peak CUDA allocation than the
near-parameter-matched private static EGNN. The coordinate-updating candidate
is measured separately and is not allowed to rescue a failed static claim.

The disconfirming result is a static latency ratio at least `1.0`, a static
peak-memory ratio at least `1.0`, OOM/nonfinite execution, or an unavailable
registered cell.

## Frozen execution

- Models: edge-free spatial static, edge-free spatial coordinate-updating, and
  private static EGNN.
- Trainable parameters: approximately 153k per arm; exact values and initial
  state hashes must be recorded.
- Precision/runtime: eager FP32 on one local CUDA GPU.
- Grid: `N={512,2048,8192}`, `k={16,64,128}`, exact receiver-regular directed
  EGNN edges with one self edge per node.
- Seeds: graph and model seed `20260723`.
- Timing: five warmups and twenty synchronized repeats.
- Timed operation: `zero_grad(set_to_none=True)`, forward, synthetic
  single-graph MSE to `0.75`, backward, and `AdamW.step`.
- Optimizer: AdamW, learning rate `1e-4`, weight decay `0`.
- Cumulative wall ceiling: 1,200 seconds. A started step is not killed midway;
  the ceiling is checked before every cell/model.
- OOM, nonfinite loss/gradient, skipped cells, and failures remain in the raw
  result.

## Resource and accounting boundary

Model, optimizer, input, and target construction are excluded from step
latency. Optimizer state is initialized during warmup. Absolute peak CUDA
allocation is measured with one isolated model and includes parameters, AdamW
state, inputs, activations, gradients, and the EGNN edge tensor. Peak delta
from the post-warmup allocated baseline is also recorded.

Exact graph construction on CPU and host-to-device transfer are measured once
and reported separately. The primary ratio excludes graph construction; a
descriptive first-step system ratio adds the one-time graph setup to EGNN.

## Inference boundary

This is a synthetic systems benchmark. It makes no accuracy, convergence,
neighbor-search, arbitrary-topology, molecular, protein, point-cloud,
generalization, or public-EGNN-reproduction claim. No dataset or
validation/test label is accessed. The private EGNN is a same-harness control.
