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
        # Optional latent-coordinate refinement; disabled by default.
        coordinate_updates=False,
        # Optional invariant receiver/sender/RBF local content; default off.
        use_pairwise_local_content=False,
        pairwise_residual_scale_init=0.1,
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

Setting `coordinate_updates=True` adds bounded, graph-centroid-preserving
updates between successive blocks and returns `out["node_positions"]`. Each
step is at most 0.25 Angstrom, and every later local/global geometry calculation
uses the updated positions. The default remains off, preserving the existing
six output keys and checkpoint schema. These are latent task coordinates, not
optimized molecular geometries or force predictions.

For local routes, `use_pairwise_local_content=True` adds a small shared
receiver/sender/RBF edge MLP plus explicit degree and smooth cutoff-mass
features to the scalar message. It excludes self edges only in this added
branch, keeps coordinates static unless separately enabled, and preserves the
default parameter/state schema when off. See
[the exact local equations](docs/LAYER_MATH.md#local-heads).

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

## Mathematical contract

- O(3) equivariant, including reflections; translation invariant and
  permutation consistent.
- Persistent `0e` scalars and polar `1o` vectors; transient `2e`
  symmetric-traceless moments.
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
uv run python scripts/probe_memory_activation.py --memory-counts 4 8
```

The training CLI exposes registered `--routing ggg/lgg/ggl/lgl/lll`,
`--global-transport-mode learned/uniform/none`,
`--memory-count 1/4/8`, `--memory-interaction`, `--radial-trace`, and opt-in
`--coordinate-updates` arms. Use
`--no-alignment-linear-term`, `--no-key-balancing`, and
`--kernel-floor-mode fixed/inverse_graph_size` only for matched ablations. Test
evaluation is disabled by default and requires explicit `--evaluate-test`.
Despite the compatibility name, `inverse_graph_size` scales the complete
shifted positive baseline `(c + beta + delta*beta*t)` by `1/N_g` and leaves the
content and `gamma*t^2` terms unchanged.

`--benchmark-model internal_static_egnn_baseline` and
`internal_dynamic_egnn_baseline` are private comparison paths, not second public
architectures. Both use fully connected same-graph directed non-self edges,
squared-distance scalar messages, and the same local
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
