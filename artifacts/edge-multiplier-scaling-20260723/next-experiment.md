# Next bounded experiment: remove fixed and temporary-tensor overhead

## Question

Can an equation-preserving fused/compiled EC-LGL execution path reduce the
low-density fixed cost and move the same-edge crossover below
`N=8192,k=64` without changing outputs, symmetry, parameters or memory safety?

## Falsifiable hypothesis

Repeated receiver/sender gathers, concatenations, pointwise transforms and
small global/reduction kernels cause enough launch and temporary-tensor cost
that an equation-preserving fused path will reduce EC-LGL synchronized forward
latency by at least 20% at `N=2048,k=16` and `N=8192,k=32`, while retaining at
least 95% of the current `k=128` memory advantage over EGNN.

## Smallest disconfirming run

- Freeze the current model state, FP32, RTX PRO 6000 environment, exact-degree
  graph seeds 20260723--20260725 and the existing same-edge baseline.
- Implement one intervention only: reuse gathered edge features and fuse or
  compile the pointwise/global/gather/scatter chain. Record compile time and
  recompilations separately; exclude them only from explicitly warm steady
  state.
- Require eager/fused output agreement at `atol=1e-5, rtol=1e-5`, unchanged
  parameter count, and the existing permutation/O(3) suite before timing.
- Measure `N={512,2048,8192}`, `k={8,16,32,64,128}`, three graph seeds, forward
  and backward separately, plus peak allocation and allocated temporary bytes.
- Reject the intervention if either registered sparse cell improves less than
  20%, any correctness/symmetry gate fails, dynamic shapes cause repeated
  compilation, or high-density peak memory grows more than 5%.

## Separate accuracy lane

Do not infer QM9 improvement from this systems run. The proposed
zero-initialized degree-normalized EC residual remains a separate registered
accuracy experiment with matched training seeds and the prior validation-only
stop rule.
