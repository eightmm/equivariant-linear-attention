# Project contract

ELA exposes exactly one public workflow:

```text
ELAGraph -> ELA -> ELAGraph
```

The architecture is completely edge-free. `ELAGraph` contains node irreps,
coordinates, sample IDs, optional interaction-component IDs, optional invariant
context, targets, and an optional coordinate mask. It contains no topology.

The canonical layer is fixed:

```text
adaptive exact relative moments through order four
+ self-adjoint content/Mercer/atlas relation
+ chart-recentered degree-2 truncated-Gaussian local relation
+ three-term invariantly orthogonal Krylov filter
+ parity-complete l<=2 closure with transient l=3/l=4 carriers
+ equivariant feed-forward update
```

The local relation recenters truncated-Gaussian Mercer features at per-head
soft chart centers in absolute coordinates (`length_scale` units, default 10),
so contact-scale pair bandwidths are expressible without edges or dense pairs.
Charts are anchored to equivariant soft farthest-point seeds computed once per
geometry, with the assignment sharpness expressed in units of chart spacing;
learned logits alone leave chart centers clustered near the centroid.
`num_local_charts=0` disables the whole sector. Radial gates additionally receive
absolute-scale invariants; per-sample RMS normalization alone no longer
removes absolute segment scale from the core.

Coordinate refinement uses an SPD atlas metric and an `SE(3)` quotient
translation/rotation/shape decomposition.

## Non-negotiable invariants

- no explicit or inferred edges;
- no dense `N x N` matrix;
- no pair or tuple hidden state;
- no sparse gather/scatter execution path;
- no topology cache or checkpoint migration layer;
- O(3), translation, and permutation equivariance;
- exact interaction-component isolation;
- node-linear memory at fixed representation order;
- one PyTorch implementation without PyG, DGL, Triton, or custom kernels.

## Current empirical validation

The current real-data validation is ATOM3D Protein Structure Ranking (PSR),
using the official split-by-year data. The prediction unit is one CASP decoy;
the immutable grouping key is its CASP target ID; the target is `GDT_TS`.
Checkpoint selection uses validation mean per-target Spearman correlation.

Inference receives the decoy's heavy-atom coordinates and atom features. The
`element-local` diagnostic additionally receives 3.5/5.0 Angstrom heavy-atom
neighbor counts computed from those same coordinates. Those counts use an
offline dense distance calculation and are not part of the edge-free ELA core.

The PSR test split has already been evaluated for multiple development
variants. Its current results are descriptive and must not be treated as an
untouched final benchmark. See `docs/PSR_RESULTS.md` and
`docs/EXPERIMENTS.jsonl` for the full retrospective record and limitations.
