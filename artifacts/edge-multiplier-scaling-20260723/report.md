# Exact `E = kN` same-edge scaling report

## Outcome

The comparison now runs on deterministic receiver-regular pseudo-random
directed graphs with exactly `E=kN` candidate edges, including one self edge
per node. Every receiver has exactly `k` candidates, construction is `O(E)` and
outside timing, and one edge tensor/hash is passed to EC-LGL and the private
static EGNN in each cell. Graph seeds vary independently while model seed
`20260723` and the resulting model-state hashes remain fixed.

All 24 registered CUDA cells completed. The preregistered prediction was mixed:
the EC-LGL/EGNN latency ratio decreased from the lowest to highest density at
every node size, but the prediction of no same-edge crossover was falsified.
The 31-repeat high-density check reproduced the first crossover at `N=8192,
k=64` for all three topology seeds.

| N | k | candidate edges | EC-LGL mean | static EGNN mean | EC/EGNN mean | EC wins/seeds |
|---:|---:|---:|---:|---:|---:|---:|
| 8,192 | 32 | 262,144 | 8.615 ms | 5.400 ms | 1.595 | 0/3 |
| 8,192 | 64 | 524,288 | 11.408 ms | 11.559 ms | 0.987 | 3/3 |
| 8,192 | 128 | 1,048,576 | 17.127 ms | 25.365 ms | 0.675 | 3/3 |

Each table entry averages one synchronized median per topology seed, with 10
warmups and 31 repeats per model. The `k=64` win is real in this packet but
small: ratios ranged from 0.985 to 0.988. The `k=128` win is much stronger, with
ratios from 0.673 to 0.678. At graph seed 20260723, peak CUDA allocation delta at
`k=128` was 1.410 GB for EC-LGL and 1.563 GB for EGNN.

## Topology quality-control amendment

The first generator satisfied exact count, uniqueness and self coverage, but a
new RED test found that one globally flattened affine traversal could create
severe receiver-degree skew (`N=512,k=8` ranged from 1 to 32). That can mix
scatter contention with edge-count scaling. The initial measurements remain as
`*-affine-exploratory.*`; they were not overwritten. The authoritative rerun
uses an independent seeded affine sender permutation per receiver, guaranteeing
exact degree `k`. The grid, models, inputs, timing protocol, hypothesis and
resource ceiling stayed fixed. Because this amendment followed inspection of
the first timings, both the amendment and the retained exploratory outputs are
part of the provenance record.

## Why the crossover occurs

Over `k={32,64,128}` at graph seed 20260723, descriptive latency slopes were
10.832 ms per million candidate edges for EC-LGL and 25.413 ms for EGNN. The
local fit crosses near 495,887 edges (`k≈60.5` at `N=8192`), consistent with the
confirmed `k=64` cell. Its EGNN intercept is negative, so this is a local
high-density description and must not be extrapolated to zero edges.

This is an architecture-level systems crossover, not an identical-computation
microbenchmark. EC-LGL performs two local edge-conditioned stages and one exact
factorized `O(N)` global stage; EGNN performs three edge-message stages. At high
`E`, replacing one edge stage with the global factorized stage pays off. At low
`E`, EC-LGL's larger fixed/global/pointwise cost dominates.

A fixed-total-edge control exposes the node cost. At exactly 262,144 candidate
edges, moving from `N=2048,k=128` to `N=8192,k=32` raised EC-LGL from 6.722 to
8.611 ms (`1.281x`), while EGNN changed from 5.466 to 5.391 ms (`0.986x`).

## Is one optimization enough?

No. The `N=8192,k=128` profile distributes EC-LGL time across gather
(`aten::index`), scatter (`index_add`), concatenation, GEMM and elementwise
operations. Optimizing only `index_add` cannot remove the fixed, per-node and
temporary-tensor costs. The priorities are:

1. Fuse or compile the pointwise/global/gather/scatter chain and eliminate
   repeated edge gathers and concatenations. This targets the poor low-density
   regime and should move the crossover to smaller `N` or `k`.
2. Reduce or chunk temporary edge features. EC-LGL already uses less memory at
   the confirmed high-density cells, but still reaches about 1.410 GB at one
   million candidate edges.
3. Independently repair accuracy using a zero-initialized, degree-normalized EC
   residual. The previous QM9 screen failed; speed does not establish a better
   molecular inductive bias.
4. Add a production cell-list/radius or bounded-kNN builder and measure its
   construction, transfer, forward and backward costs separately.

## Boundaries

These single-graph synthetic workloads are systems probes, not molecule,
protein or point-cloud accuracy benchmarks. Forward equations differ even when
edges are identical. Backward/training time, graph construction, multi-graph
batching, chirality, force quality and domain transfer are not inferred. The
full grid uses one topology seed per cell; only the `N=8192,k={32,64,128}`
confirmation uses three seeds. Synchronized medians are authoritative;
profiler timings are diagnostic only.
