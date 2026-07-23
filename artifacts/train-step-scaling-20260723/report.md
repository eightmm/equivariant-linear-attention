# Full train-step scaling result

## Verdict

The preregistered first grid **failed** its static latency hypothesis even
though it passed the memory comparison. At `N=8192,k=128`, static spatial
attention was `1.755x` as slow as private static EGNN (`109.884` versus
`62.601 ms`) and used `0.141x` its peak CUDA allocation.

A disclosed post-outcome optimization removed an unnecessary duplicate-index
backward from the one-graph sufficient-statistic broadcast. The full grid was
then rerun unchanged. At `N=8192,k=128`, static attention became `2.46x`
faster (`25.471` versus `62.626 ms`) and used `7.26x` less peak allocation
(`1.223` versus `8.884 GB`). It also crossed at `N=8192,k=64` with latency
ratio `0.849` and peak-memory ratio `0.274`.

The coordinate-updating model crossed only at `N=8192,k=128`, with latency
ratio `0.580` and peak-memory ratio `0.141`. EGNN remained much faster for
every `N<=2048` cell and for `N=8192,k=16`.

## Causal diagnostic

The pre-optimization three-step static profile recorded `254.573 ms` aggregate
device time in 99 `IndexBackward` evaluations. The model was processing one
graph, so each `summary[batch]` used an all-zero index and paid for
duplicate-index gradient accumulation. Replacing only this expansion with a
stride-zero view preserved values and accumulated gradients. The optimized
profile records zero `IndexBackward` time and has `bmm` as its largest
operator. Static `N=8192,k=128` latency fell by `76.8%`.

## Verification

- Original registered grid: 9/9 completed, no OOM/nonfinite gradient.
- Optimized identical grid: 9/9 completed, no OOM/nonfinite gradient.
- Focused broadcast, spatial, single-layer, and profiler tests: 25 passed.
- Repository fast gate before optimized rerun: 454 passed, 88.64% coverage.
- CPU float64 ML smoke: passed.

## Boundary

This is a synthetic one-graph systems comparison on one RTX PRO 6000
Blackwell Max-Q GPU. Model/optimizer construction and data loading are
excluded. EGNN graph construction and transfer are reported separately.
Neither grid measures accuracy, convergence, public EGNN fidelity, arbitrary
topology preservation, molecular/protein/point-cloud generalization, or
multi-graph throughput.

Protocol deviation `PD-001`: the frozen resource paragraph stated that target
construction was excluded, but the implementation creates the one-element
constant target with `torch.full_like` inside the timed loss stage. This same
scalar operation is present in every arm and does not change the reported raw
measurements, but the runs are not an exact implementation of that sentence.
Future benchmark versions should pass a preconstructed target.
