# Equivariant Linear Attention

A focused PyTorch prototype of one O(3)-equivariant local/global attention
architecture for 3D graphs. The model keeps scalar and polar-vector states and
transports relative first and symmetric-traceless second moments through a
bounded degree-2 kernel. Global heads use exact structured factorization; local
heads use same-graph cutoff edges.

The repository intentionally exposes one implementation,
`EquivariantAttention`, with routing, multi-memory (HEMM), and radial-trace
settings on `EquivariantAttentionConfig`. Earlier dense, rich, and
backend-dependent model families were removed so mathematical changes can be
tested against one stable contract.

## Install and verify

Core installation requires only PyTorch:

```bash
uv sync --locked
scripts/check.sh fast
```

QM9 tooling is optional:

```bash
uv sync --locked --extra qm9
```

The train-only ATOM3D-LBA/PDBBind overfit loader is also optional:

```bash
uv sync --locked --extra qm9 --extra pdbbind
```

## Example

```python
import torch

from equivariant_attention import EquivariantAttention, EquivariantAttentionConfig

model = EquivariantAttention(
    EquivariantAttentionConfig(
        node_dim=16,
        hidden_irreps="64x0e + 4x1o",
        output_irreps="1x0e + 1x1o + 1x2e",
        num_layers=3,
        num_heads=4,
        # Public defaults: global/global/global, one memory, no interaction.
        local_head_counts=None,
        global_transport_mode="learned",
        global_memory_count=1,
        use_memory_interaction=False,
        use_radial_trace=False,
        # Optional no-edge multi-scale Euclidean kernel; default off.
        use_multiscale_spatial_kernel=False,
        # Optional latent-coordinate refinement; disabled by default.
        coordinate_updates=False,
        # External sparse candidates + moving coordinates must choose a policy.
        coordinate_neighbor_policy="error",  # error | fixed | rebuild
        # Optional invariant receiver/sender/RBF local content; default off.
        use_pairwise_local_content=False,
        pairwise_residual_scale_init=0.1,
        # Optional per-local-layer invariant edge filters; default off.
        use_edge_conditioned_local_transport=False,
        # mean | sum | ligand-pocket interaction residual.
        readout_mode="mean",
    )
)

node_feats = torch.randn(12, 16)
pos = torch.randn(12, 3)
batch = torch.tensor([0] * 5 + [1] * 7)
out = model(node_feats, pos, batch=batch)

print(out["graph_scalars"].shape)  # (2, 1)
print(out["graph_vectors"].shape)  # (2, 1, 3)
print(out["graph_tensors"].shape)  # (2, 1, 3, 3)
```

Adding hidden channels such as `hidden_irreps="64x0e + 4x1o + 4x2e"` carries a
persistent symmetric-traceless Cartesian rank-2 state through every block.
Invariant gates couple it to scalar/vector updates while preserving O(3),
translation, batching, and node-permutation behavior. This opt-in path supports
only reflection-even `2e`; it is not a general spherical-harmonic or arbitrary
`l>2` implementation.

Setting `coordinate_updates=True` adds bounded, graph-centroid-preserving
updates between successive blocks and returns `out["node_positions"]`. Each
step is at most 0.25 Angstrom, and every later local/global geometry calculation
uses the updated positions. The default remains off, preserving the existing
six output keys and checkpoint schema. These are latent task coordinates, not
optimized molecular geometries or force predictions.

If moving coordinates are combined with an external sparse `edge_index`, the
default `coordinate_neighbor_policy="error"` rejects the ambiguous topology.
Choose `"fixed"` only when omitted pairs are intentionally forbidden from
entering later, or `"rebuild"` to ignore the supplied candidates and rebuild
complete same-graph candidates at every local stage. The latter is exact but
quadratic without a production neighbor-list backend.

For local routes, `use_pairwise_local_content=True` adds a small shared
receiver/sender/RBF edge MLP plus explicit smooth cutoff-mass/effective-degree
features to the scalar message. It excludes self edges only in this added
branch, keeps coordinates static unless separately enabled, and preserves the
default parameter/state schema when off. See
[the exact local equations](docs/LAYER_MATH.md#local-heads).

`use_edge_conditioned_local_transport=True` instead replaces each active local
attention transport with an independent invariant receiver/sender/RBF edge MLP
and equivariant scalar, sender-vector, relative-vector, and rank-2 sums. It is
registered only when every local block uses all heads and the hidden vector
channel count equals the head count. It cannot be combined with the legacy
pairwise-content or learned-radial controls. This path is opt-in: its scaling
contract passed, but its repeated 500-step QM9 accuracy screen failed.

`use_gated_local_transport=True` selects the newer same-feature local operator.
Its per-head edge MLP sees the current receiver/sender scalar states, existing
RBF distances, and invariant contractions of the current vectors with each
other and the relative direction. It emits gated scalar, receiver-vector,
sender-vector, relative-vector, and symmetric-traceless rank-2 messages,
aggregates them as `sum(f_cutoff * message) / sqrt(1 + sum(f_cutoff))`, and
exposes smooth cutoff mass/effective degree explicitly. The squared-cutoff
statistic remains a learned diagnostic rather than the divisor, so a singleton
message still decays to zero at the radial boundary.
`use_grouped_invariant_normalization=True`
separately standardizes scalar-message, angular, and persistent-tensor
invariant families before their existing shared update. Both switches are
disabled by default and add no raw atom, bond, residue, or label feature.
Their exact equations and matched real-data result are in
[the derivation](docs/LAYER_MATH.md#gated-same-feature-local-transport) and
[the evaluation record](docs/EVALUATION.md#same-feature-gated-hybrid-outcome-2026-07-24).
The current function-preserving hot-path refactor reduced the matched
`N=2048, k=64` train-profile peak allocation by `17.98%` and the frozen
ATOM3D-LBA candidate peak by `16.45%`; see the
[performance report](docs/PERFORMANCE_REFACTOR_20260724.md).
On the complete official ATOM3D-LBA ID30 train/validation split, the current
gated-plus-grouped LGL reached `1.550035 pK` validation RMSE versus `1.592008`
for the matched previous LGL and `1.692812` for the private static EGNN.
This passes the registered one-seed point gate, but the paired
candidate-versus-incumbent validation interval crosses zero and all arms clip
over 99% of updates, so the switches remain opt-in. See the
[full LBA report](docs/LBA_ID30_VALIDATION_20260724.md).

Architecture v3 adds opt-in exact quartic angular features, a second learned
`1o` axis per head, invariant non-scalar RMS normalization, and public
equivariant inputs:

```python
model = EquivariantAttention(
    EquivariantAttentionConfig(
        node_dim=node_dim,
        input_vector_dim=2,
        input_tensor_dim=1,
        hidden_irreps="64x0e + 4x1o + 4x2e",
        angular_feature_rank=2,
        use_quartic_kernel=True,
        use_irrep_rms_normalization=True,
    )
)
out = model(
    node_feats,
    pos,
    node_vectors=polar_vectors,          # (N, 2, 3)
    node_tensors=symmetric_traceless,    # (N, 1, 3, 3)
)
```

The two-axis option is two copies of `1o`, not a general `l=2` irrep. The
quartic term and compressed quadratic summaries remain exact and fixed width,
so global scaling is still `O(N)` without an `N x N` tensor. The implementation
and CPU contracts are complete; v3 QM9/CUDA and conditional LBA evidence is
pending because CUDA was unavailable at the recorded run boundary. See the
[v3 capability comparison](docs/ARCHITECTURE_V3_20260725.md) and
[exact equations](docs/LAYER_MATH.md#optional-architecture-v3-angular-features).

For protein-ligand affinity experiments, `readout_mode="interaction"` keeps the
ligand mean prediction as a zero-initialized residual baseline and adds
ligand, pocket, cross-interface, and parity-even triple-product features. It
requires a Boolean ligand `readout_mask`; the complement is treated as pocket.
The head is O(3)-invariant, translation invariant, and permutation consistent,
but it is an experimental parity-aware readout rather than a parity-complete
hidden backbone. The current 16-complex train-only diagnostic did not beat the
mean readout, so the public default remains `"mean"`.

For a route with local heads, callers may supply precomputed directed candidate
edges without adding a neighbor-library dependency:

```python
# edge_index[0] is receiver i, edge_index[1] is sender j for i <- j.
# Candidates must be same-graph, unique, and include every self edge.
local_model = EquivariantAttention(
    EquivariantAttentionConfig(node_dim=16, local_head_counts=(4, 0, 4))
)
edge_index = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
out = local_model(node_feats[:2], pos[:2], edge_index=edge_index)
```

The supplied path bypasses quadratic pair discovery, applies the same strict
distance cutoff, and uses O(E) candidate/retained storage. Omitting it preserves
the vectorized complete-candidate fallback exactly.

`GraphSample` and `GraphBatch` carry the same optional `edge_index`; collation
checks uniqueness/self coverage, offsets graph-local indices, and marks the
result for the validated hot path. Training then skips repeated content
validation inside the model; direct tensor callers remain fully validated by
default. `edge_index_is_validated=True` is a trusted assertion and must not be
set on unchecked external edges. The resulting fixed-width model forward is
`O(E_local + N)`; one-time validation and graph construction are separate.
The QM9 CLI flag `--precompute-local-edges` builds radius candidates once while
loading. That convenience builder still performs a dense per-graph radius scan
and is excluded from any linear-time preprocessing claim.

## Mathematical contract

- O(3) equivariant, including reflections; translation invariant and
  permutation consistent.
- Persistent `0e` scalars and polar `1o` vectors; transient `2e` moments by
  default, with opt-in persistent reflection-even `2e` hidden channels.
- `node_feats` contains invariant scalar (`0e`) channels only; coordinates are
  stored in float32+ independently of feature/model precision.
- `local_head_counts=None` gives the global/global/global (`ggg`) route. The
  registered three-block alternatives are `lgg=(H,0,0)`, `ggl=(0,0,H)`,
  `lgl=(H,0,H)`, and `lll=(H,H,H)`.
- `global_transport_mode=learned/uniform/none` isolates the learned kernel from
  exact graph-mean moment pooling and from a no-global-transport FFN control.
  All three allocate the same parameter schema; `none` also skips global
  geometry preprocessing.
- Global heads use exact graph-wise structured factorization without an
  `N x N` attention tensor. Local heads use directed raw-coordinate edges with
  a 2.5-Angstrom cosine cutoff of squared scaled distance and 16 Gaussian RBFs
  by default. Keyword-only `edge_index` accepts validated receiver/sender
  candidates for local routes and bypasses the fallback's quadratic discovery.
- Multi-memory interaction is off by default with `M=1`. The `M=1` path reduces
  exactly to incumbent global attention; `M=4/8` interaction arms are
  experimental and registered only for the middle global block of `lgl`. The
  current HEMM uses an M-shared invariant MLP router with fixed DCT slot codes
  and a shared-read/write symmetric low-rank pair gate, not persistent memory.
  The full frozen Stage-0 matrix still blocks M=4/8 promotion: radial coupling
  collapses and neither identity nor the registered residual mixtures produce
  sufficient transport/gradient activation across all aligned lanes.
  Constructing an interacting M=4/8 arm emits a Stage-0-blocked warning, and
  memory interaction is rejected with uniform or disabled global transport.
- Unit-normalized positive scalar content, unit-ball vector queries/keys,
  bounded linear/quadratic angular scales, and structured vector/3x3
  mass/denominator contractions avoid signed
  flattened-feature cancellation; signed value numerators remain unclamped.
- Turning off the alignment-linear term removes only `beta * (q dot k)` and
  retains the same `beta` constant. Key balancing is exactly one cycle.
- Geometry preprocessing is scale-first. Geometry squares, angular/ST
  features, moment reductions, and invariant normalization use float32 for
  fp16/bf16 features; float64 coordinates remain float64. Coordinates must be
  float32 or float64 and are never downcast through model precision.
- Integer, nonnegative, contiguous graph IDs starting at zero.

Scalar outputs are O(3)-invariant and therefore cannot distinguish
isolated mirror pairs. Chirality-sensitive prediction requires a future,
explicitly tested parity-complete extension.

See [the model contract](docs/MODEL.md) and [the derivation](docs/LAYER_MATH.md).

## Research boundary

Existing QM9 results are adaptive random-row warm-start probes, not evidence of
scaffold or cold-target generalization. Historical experiments remain in
`docs/EXPERIMENTS.jsonl`; removed architecture variants are not public APIs.
Any route containing a global head is not a size-consistent interatomic
potential. The intended role is a bounded-size property probe or a global
context block within the same local/global architecture.

The 2026-07-24 same-feature hybrid screen is the first bounded packet in which
the gated-local plus grouped-normalization package beat its matched incumbent:
QM9 seed-42/500-update validation MAE improved from `0.709287` to
`0.683609 eV`. On 16 train-only ATOM3D-LBA complexes, it reached
`0.10 pK` in 1,050 updates versus 1,800 for the incumbent; the private static
EGNN missed the threshold at 3,000. All three affinity arms consumed the same
153,029 candidates. This does not establish multi-seed QM9 robustness or
affinity generalization, and the candidate remained about `5.55x` slower per
step than EGNN on these small complexes, so the switches remain opt-in.

The earlier registered validation-only comparison found M=1 `lgl` better than
matched M=1 `ggg` on all three seeds (mean gap-MAE improvement 0.052774 eV),
while also passing the CUDA latency/memory ceiling. This does not change the
public default or admit the Stage-0-blocked M=4/M=8 interaction arms. See
[the benchmark record](docs/BENCHMARKS.md#registered-m1-routing-result-2026-07-17).
The subsequent five-seed mechanism check completed with mean validation MAE
0.515688/0.534776/0.691821 eV for learned/uniform/none. Learned beat uniform by
0.019088 eV on average and both global arms beat none on all five seeds, but
the frozen efficiency gate failed: learned used 1.359x the peak memory of
uniform, while learned/uniform took 1.512x/1.288x the elapsed time of none
(learned also used 1.414x its peak memory). Therefore no transport mode was
locked, no default changed, and the conditional private-EGNN comparison was
not run. See [the benchmark record](docs/BENCHMARKS.md#registered-transport-mechanism-result-2026-07-19).

The independently approved dynamic-coordinate packet then completed all six
500-step screen arms and twenty 2,000-step confirmation arms. The screen
selected `ggg` and admitted dynamic EGNN, but neither family passed the frozen
five-seed promotion rule. Attention static/dynamic mean validation MAE was
0.582946/0.585535 eV; EGNN static/dynamic was 0.408932/0.410428 eV. All dynamic
runs had active coordinate gradients and respected the centroid and 0.25
Angstrom per-layer bounds, but the accuracy gates failed and dynamic EGNN also
used 1.456x median elapsed time. Coordinate updates therefore remain opt-in and
the public default stays off. See [the coordinate-study record](docs/BENCHMARKS.md#registered-dynamic-coordinateegnn-result-2026-07-19).

The subsequent three-iteration EGNN-parity packet tested learned radial gating,
an opt-in 1,105-parameter receiver/sender/RBF plus degree/mass branch, and
exact-baseline staged initialization. Radial-only reached 0.499508 eV five-seed
mean versus its rerun private static EGNN at 0.421199 eV. Pairwise `alpha=0.1`
failed its 500-step screen; staged `alpha=0` passed the screen but reached
0.509008 eV versus rerun EGNN at 0.438268 eV. Both confirmations lost 0/5 paired
seeds, so no default or candidate is promoted. The packet stopped at its third
architecture iteration after 850.7 GPU-wall seconds with test evaluation off.
See [the completed parity record](docs/BENCHMARKS.md#registered-egnn-parity-result-2026-07-20)
and [the frozen evaluation contract](docs/EVALUATION.md#registered-egnn-parity-study-confirmed-2026-07-20).

The scaling-aware EC-LGL packet then separated identical-kernel factorization
from different-edge-regime model comparisons. For the exact same float64
kernel at 4096 nodes, factorized evaluation used 0 pair elements and 3.38 MB
peak CUDA delta versus 16,777,216 pair elements and 671.09 MB for the
materialized dense path; median latency was 0.761/3.374 ms and maximum output
error was `2.41e-15`. Full EC-LGL with degree 16 first crossed complete-edge
static EGNN at 512 nodes (5.243/5.577 ms, 11.71/391.59 MB), but remained slower
than EGNN when both used the same sparse edges. Its repeated QM9 screen mean was
0.802194 eV versus static LGL at 0.712178 eV, so the accuracy gate rejected
confirmation and the feature remains off by default. See
[the scaling record](docs/BENCHMARKS.md#scaling-aware-ec-lgl-result-2026-07-22).

The follow-up exact-`E=kN` benchmark controls the edge tensor directly on
receiver-regular pseudo-random graphs. Across `N={128,512,2048,8192}` and
`k={4,8,16,32,64,128}`, EC-LGL remained much slower at low density but crossed
the same-edge private static EGNN at `N=8192,k=64`. A 31-repeat confirmation
over three topology seeds with fixed model seed 20260723 gave mean
EC-LGL/EGNN ratios 1.595, 0.987 and 0.675
for `k=32,64,128`; the `k=64` advantage is only about 1.3%, while `k=128` is a
clearer high-density systems win. This compares different model equations in
forward-only execution and excludes neighbor construction and accuracy. See
[the exact-edge record](docs/BENCHMARKS.md#exact-ekn-same-edge-scaling-2026-07-23).

The edge-free spatial extension adds a ten-feature/head positive Euclidean
kernel to all-global learned attention without an edge list or `N x N` pair
tensor. On the same RTX PRO 6000, a 100-repeat `N=8192` confirmation measured
static spatial attention at 11.359 ms and 121.58 MiB peak CUDA delta. Private
static EGNN took 11.672 ms/753.00 MiB at `k=64` and
25.407 ms/1506.56 MiB at `k=128`; the spatial path was therefore 1.03x and
2.24x faster, with 6.19x and 12.39x lower measured working-plus-edge memory.
The coordinate-updating spatial path was 3.6% slower than EGNN at `k=64`, but
1.29x faster by `k=80` and 2.10x faster at `k=128`. Low-edge/small-node EGNN
remains much faster. These are forward-only synthetic systems results, not
topology preservation or molecule/protein/point-cloud accuracy evidence. See
[the edge-free record](docs/BENCHMARKS.md#edge-free-spatial-linear-scaling-2026-07-23).

```bash
uv run python scripts/train_compare.py --dataset synthetic --steps 10 --routing ggg
uv run python scripts/train_compare.py --dataset synthetic --steps 10 \
  --routing lgl --global-transport-mode uniform --bounded-diagnostics
uv run python scripts/train_compare.py --dataset synthetic --steps 10 \
  --benchmark-model internal_static_egnn_baseline --hidden-dim 91
uv run python scripts/train_compare.py --dataset synthetic --steps 10 \
  --routing lgl --coordinate-updates
uv run python scripts/train_compare.py --dataset synthetic --steps 10 \
  --benchmark-model internal_dynamic_egnn_baseline --hidden-dim 91
./scripts/run_bounded_control_screen.sh \
  artifacts/egnn-matched-baseline-development-20260718
uv run --locked python scripts/run_registered_transport_study.py \
  artifacts/transport-study-reproduction --dry-run
uv run --locked python scripts/run_registered_coordinate_study.py \
  artifacts/coordinate-study-reproduction --dry-run
uv run python scripts/bench_attention.py --graphs 1 8 32 --nodes-per-graph 16 32
uv run python scripts/benchmark_sparse_scaling.py --device cuda \
  --metrics-out artifacts/sparse-scaling.json
uv run python scripts/benchmark_sparse_scaling.py --edge-multiplier-grid \
  --sizes 128 512 2048 8192 --edge-multipliers 4 8 16 32 64 128 \
  --device cuda --seed 20260723 --model-seed 20260723 \
  --warmup 3 --repeats 7 \
  --metrics-out artifacts/exact-edge-grid.json
uv run python scripts/benchmark_sparse_scaling.py --edge-free-spatial-grid \
  --sizes 128 512 2048 8192 --edge-multipliers 4 16 64 128 \
  --device cuda --seed 20260723 --model-seed 20260723 \
  --warmup 5 --repeats 15 \
  --metrics-out artifacts/edge-free-spatial-grid.json
uv run python scripts/probe_memory_activation.py --memory-counts 4 8
```

GitHub Actions is manual-only (`workflow_dispatch`) to avoid spending hosted
runner time on every push. Run `scripts/check.sh fast` locally before merging;
the workflow can still be dispatched explicitly when an independent hosted
check is useful.

The training CLI exposes registered `--routing ggg/lgg/ggl/lgl/lll`,
`--global-transport-mode learned/uniform/none`,
`--memory-count 1/4/8`, `--memory-interaction`, `--radial-trace`, and opt-in
`--coordinate-updates`, `--edge-conditioned-local-transport`, and
`--precompute-local-edges` arms. Use
`--no-alignment-linear-term`, `--no-key-balancing`, and
`--kernel-floor-mode fixed/inverse_graph_size` only for matched ablations. Test
evaluation is disabled by default and requires explicit `--evaluate-test`.
Despite the compatibility name, `inverse_graph_size` scales the complete
shifted positive baseline `(c + beta + delta*beta*t)` by `1/N_g` and leaves the
content and `gamma*t^2` terms unchanged.

`--benchmark-model internal_static_egnn_baseline` and
`internal_dynamic_egnn_baseline` are private comparison paths, not second public
architectures. By default both use fully connected same-graph directed
non-self edges; they also accept the validated supplied-edge path for
controlled sparse scaling. Their messages use squared-distance scalars and the same local
split/normalization/optimizer/readout harness. The dynamic control additionally
uses invariant edge scalars to weight relative coordinate vectors, followed by
the same 0.25-Angstrom bound and graph-centroid correction. Neither is an
official-paper code reproduction. Their model-specific default width is 91;
factorized-only nondefault flags are rejected rather than ignored.

Except for the registered M=1 `lgl` probe above, these switches are implemented
capabilities rather than evidence that a non-default arm is better. Any further
promotion requires the preregistered matched, multi-seed validation and
resource comparison in [the evaluation contract](docs/EVALUATION.md).

The repository currently has no declared open-source license. Contact the owner
before redistributing substantial portions of the code.
