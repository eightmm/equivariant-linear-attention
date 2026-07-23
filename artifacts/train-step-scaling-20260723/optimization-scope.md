# Post-outcome single-graph backward optimization

Frozen after the registered grid failed its latency hypothesis and after the
read-only profiler identified repeated `IndexBackward` as the dominant
spatial-linear operator, but before measuring the optimized GPU path.

## Observation

At `N=8192,k=128`, the registered static spatial model required
`109.8843 ms` per full train step versus `62.6008 ms` for private static EGNN.
Its peak allocation was `1.2515 GB` versus `8.8842 GB`. A three-step
post-outcome profile attributed `254.573 ms` aggregate device time to 99
`IndexBackward` evaluations in the static model.

The model input contains one graph. Structured sufficient statistics had shape
`(1, ...)` but were expanded to nodes with `summary[batch]`, where every batch
entry is zero. Its backward therefore used duplicate-index accumulation even
though an equivalent stride-zero broadcast view exists.

## Frozen intervention and prediction

Change only graph-summary expansion:

- one graph: `summary[0].unsqueeze(0).expand(N, ...)`;
- multiple graphs: retain `summary[batch]`.

Do not change equations, parameters, optimizer, model width/depth, kernel,
precision, grid, or EGNN. The prediction is that the static `N=8192,k=128`
median full-step latency improves by at least 20% relative to the retained
`109.8843 ms` result. Beating EGNN is not assumed; it is reported only if the
measured ratio is below `1.0`.

## Gates

- broadcast forward values and accumulated gradients match indexed semantics;
- edge-free spatial, single-layer, and profiler-focused tests pass;
- repository fast gate passes;
- rerun the complete registered nine-cell GPU grid with identical arguments;
- retain both original and optimized raw grids;
- any regression, OOM, nonfinite gradient, or missing cell remains visible.

This is a post-outcome systems optimization and cannot be described as a
preregistered confirmation of the original hypothesis.
