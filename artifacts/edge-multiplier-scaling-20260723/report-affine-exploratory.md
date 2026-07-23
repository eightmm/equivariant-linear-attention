# Exact `E = kN` same-edge scaling report

## Outcome

The comparison is now implemented over deterministic pseudo-random directed
graphs with exactly `E=kN` candidate edges, including one self edge per node.
The generator traverses the nonself pair universe without replacement in
`O(E)`, and each cell passes the identical edge tensor and hash to EC-LGL and
the private static EGNN. Edge construction is outside the timed region.

All 24 registered CUDA cells completed. The preregistered prediction was mixed:
the EC-LGL/EGNN latency ratio decreased from the lowest to the highest density
at every node size, but the prediction of no same-edge crossover was falsified.
At `N=8192`, the first measured crossover was `k=64`.

| N | k | candidate edges | EC-LGL | static EGNN | EC/EGNN |
|---:|---:|---:|---:|---:|---:|
| 8,192 | 32 | 262,144 | 8.556 ms | 5.484 ms | 1.560 |
| 8,192 | 64 | 524,288 | 11.270 ms | 11.745 ms | 0.960 |
| 8,192 | 128 | 1,048,576 | 16.737 ms | 25.715 ms | 0.651 |

The table is the 10-warmup/31-repeat confirmation with the crossover model
order reversed relative to the first grid where applicable. At `k=128`, EC-LGL
used 1.410 GB peak CUDA allocation delta versus 1.563 GB for EGNN.

## Why the crossover occurs

Over the confirmed high-density points, descriptive latency slopes were 10.407
ms per million candidate edges for EC-LGL and 25.857 ms for EGNN. The fitted
crossover is about 474,011 edges (`k≈57.9` at `N=8192`), consistent with the
observed `k=64` cell. The fit is not valid near zero edges: its EGNN intercept is
negative, so it is used only as a local high-density description.

This is an architecture-level systems crossover, not an identical-computation
microbenchmark. EC-LGL performs two local edge-conditioned stages and one exact
factorized global stage, whereas EGNN performs three edge-message stages. At
high `E`, replacing one edge stage with an `O(N)` global stage pays off. At low
`E`, EC-LGL's larger fixed/global/pointwise cost dominates.

The separate fixed-edge-count control makes that node cost visible. With exactly
262,144 edges, moving from `N=2048,k=128` to `N=8192,k=32` increased EC-LGL from
6.586 to 8.564 ms (`1.30x`), while EGNN stayed at 5.517/5.467 ms (`0.99x`).

## Is one optimization enough?

No. The profile at `N=8192,k=128` shows a distributed EC-LGL cost: gather
(`aten::index`), scatter (`index_add`), concatenation, GEMM, and elementwise
operations all contribute. Optimizing only `index_add` cannot remove the broad
fixed and temporary-tensor cost. The priorities are:

1. Fuse or compile the pointwise/global/gather/scatter chain and remove repeated
   edge gathers and concatenations. This should move the crossover below
   `N=8192,k=64`.
2. Reduce or chunk temporary edge features. EC-LGL already uses less memory at
   the confirmed crossover, but still reaches 1.410 GB at one million edges.
3. Independently repair accuracy with the proposed zero-initialized,
   degree-normalized EC residual. The previous QM9 screen failed; speed does not
   establish a better molecular inductive bias.
4. Add a production cell-list/radius or bounded-kNN builder and measure it
   separately. This packet proves model-forward behavior, not end-to-end graph
   construction.

## Boundaries

The synthetic one-graph workload is a systems probe, not a molecule, protein,
or point-cloud accuracy benchmark. Forward equations differ even though edges
are identical. Backward/training time, graph construction, multi-graph batching,
chirality, force quality, and domain transfer are not inferred. Synchronized
medians in `gpu-grid.json` and `gpu-crossover-confirm.json` are authoritative;
profiler timings are diagnostic only.
