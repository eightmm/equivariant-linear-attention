# Equivariant Linear Attention

A general-purpose PyTorch layer for parity-aware 3D data. The canonical model is
**equivariant linear attention**, not a task-specific molecular network and not
a menu of interchangeable graph backends.

\[
\boxed{
\text{exact global equivariant linear attention}
+
\text{exact sparse short-range residual}
}
\]

Global and local messages stay separate until an invariant,
identity-initialized router combines them. Positions remain a separate affine
geometry input; node inputs and outputs are declared with `input_irreps` and
`output_irreps`.

## Canonical model

```python
from equivariant_attention import ELA, ELAConfig, SparseGeometry

config = ELAConfig(
    input_irreps="32x0e + 4x1o + 1x1e",
    output_irreps="1x0e + 1x1o",
    width=128,
    depth=8,
    geometry=SparseGeometry(
        cutoff=6.0,
        num_rbf=16,
    ),
)
model = ELA(config)

graph = config.geometry.prepare(
    batch,
    edge_index,  # edge_index[0] receives from edge_index[1]
)
output = model(node_irreps, positions, graph)
```

The canonical public choices are deliberately limited to:

```text
input_irreps
output_irreps
width
depth
geometry
```

Attention heads, local rank, hidden parity sectors, normalization, residual
scales, tensor closure, and chirality construction are derived implementation
choices rather than public architecture-search knobs.

## One layer

\[
\bar h^\ell
=
\operatorname{EqRMSNorm}_{\rm attn}(h^\ell),
\]

\[
G^\ell
=
\operatorname{ExactGlobalELA}_{l\le2}(\bar h^\ell),
\qquad
L^\ell
=
\operatorname{ExactSparseLocal}_{l\le2}
(\bar h^\ell,x,\mathcal E).
\]

For each sector

\[
\tau\in\{0e,0o,1o,1e,2e,2o\},
\]

the model computes invariant branch statistics and positive routing weights:

\[
(w_{G,i}^{\tau},w_{L,i}^{\tau})
=
2\operatorname{softmax}
\left[
R_\tau
\left(
\bar h_i^{0e},
\log\operatorname{RMS}(G_i^\tau),
\log\operatorname{RMS}(L_i^\tau)
\right)
\right].
\]

The router output and branch-balance parameters are zero initialized, so
initially

\[
w_G^\tau=w_L^\tau=1,
\qquad
M_i^\tau=G_i^\tau+L_i^\tau.
\]

The model therefore starts from the admitted refined ELA equation. It learns a
branch preference only when gradients support it.

After fusion, one parity update, low-order tensor closure, residual, and
pointwise equivariant FFN are applied:

\[
\begin{aligned}
M^\ell &= \operatorname{Fuse}(G^\ell,L^\ell),\\
\widetilde h^\ell
&=h^\ell+
\operatorname{AttnResidual}
\left[
\operatorname{ParityUpdate}(M^\ell)
+
\operatorname{TPClosure}_{l\le2}
\right],\\
h^{\ell+1}
&=\widetilde h^\ell+
\operatorname{EqFFN}
\left(
\operatorname{EqRMSNorm}_{\rm ffn}
(\widetilde h^\ell)
\right).
\end{aligned}
\]

No `N x N` attention tensor or persistent edge hidden state is materialized.

## Representations

The optimized path accepts arbitrary multiplicities of

```text
0e  ordinary scalar
0o  pseudoscalar
1o  polar vector
1e  axial vector
2e  even symmetric-traceless tensor
2o  odd symmetric-traceless tensor
```

with `l <= 2`. Cartesian bases are

```text
l=1: [x, y, z]
l=2: [xx, yy, xy, xz, yz], where zz = -xx - yy
```

Helpers:

```python
from equivariant_attention import (
    matrix_to_st5,
    pack_irreps,
    split_irreps,
    st5_to_matrix,
)
```

Raw positions are not `1o` input features. They transform affinely,

\[
x_i\mapsto Rx_i+t,
\]

whereas a polar feature transforms homogeneously as `v -> Rv`.

## Invariant conditioning

Conditioning is an explicit wrapper, not a field in the minimal model config:

```python
from equivariant_attention.conditioning import (
    ConditionedELA,
    InvariantConditioningConfig,
)

conditioned = ConditionedELA(
    config,
    InvariantConditioningConfig(condition_dim=256),
)
output = conditioned(
    node_irreps,
    positions,
    graph,
    condition=time_embedding,
)
```

The condition is `0e`. Even scalars receive bounded shift and scale;
non-scalar sectors receive invariant copy-wise scale only. Conditioner output
projections are zero initialized, so shared ELA weights initially reproduce the
unconditioned function.

Vector or tensor conditions belong in `input_irreps`.

## Coordinate refinement

Coordinate mutation is not part of the canonical layer. Refinement, denoising,
registration, and learned relaxation use an outer-loop wrapper:

```python
from equivariant_attention import (
    CoordinateRefinementConfig,
    ELACoordinateRefiner,
)

refiner = ELACoordinateRefiner(
    model,
    CoordinateRefinementConfig(
        steps=4,
        max_step=0.2,
        centering="selected",
    ),
)

output = refiner(
    node_irreps,
    positions,
    graph,
    update_mask=movable_nodes,
    graph_rebuilder=optional_neighbor_rebuilder,
)
```

The zero-initialized equivariant head predicts a bounded `1o` displacement. A
caller-provided graph rebuilder makes neighbor-list policy explicit. For
conservative force fields, derive forces from scalar energy:

\[
F_i=-\nabla_{x_i}E.
\]

## Complexity

For `N` nodes, `E` directed candidate edges, and `L` layers, with fixed widths
and ranks,

\[
T=O\left(L(N+E)\right).
\]

The branch router adds `O(LN)` work and does not change the asymptotic order.
The model is linear in node count only when `E = O(N)`. Neighbor discovery is
outside the layer and must be measured separately.

## Architecture policy

Tracked evidence supports distinct operator roles:

```text
canonical:    exact global ELA + exact sparse local + branch-aware fusion
experimental: edge-free implicit smooth transport
experimental: block Attention Residuals for deep stacks
legacy:       historical configurable architectures
```

The implicit Gaussian--Taylor operator is useful for smooth spatial research,
but fixed-rank transport does not reproduce compact support, edge-axis routing,
typed relations, or sharp local interactions. Always-on and periodic hybrid
schedules are therefore not canonical options.

Experimental components:

```python
from equivariant_attention.experimental import ...
```

Compatibility models:

```python
from equivariant_attention.legacy import ...
```

`EquivariantLinearAttentionConfig` and `EquivariantLinearAttention` remain
available for existing experiments and historical state schemas. New work
should start from `ELAConfig` and compose a wrapper only when the task requires
one.

## Migration

```python
from equivariant_attention.migration import (
    canonical_config_from_advanced,
    load_advanced_ela_state,
)

minimal = canonical_config_from_advanced(old_config)
model = ELA(minimal)
receipt = load_advanced_ela_state(model, old_state_dict)
```

The migration fails closed when shapes or wrapper-only features differ. Missing
keys are allowed only for the new branch router.

## Install and validate

```bash
uv sync --locked
scripts/check.sh fast
scripts/check.sh gpu
```

GitHub Actions do not run automatically on pushes or pull requests. Validation
is explicit and local.

Focused canonical checks:

```bash
uv run pytest \
  tests/test_branch_fusion.py \
  tests/test_canonical_api.py \
  tests/test_canonical_migration.py \
  tests/test_conditioning_wrapper.py \
  tests/test_refinement_wrapper.py \
  tests/test_api_policy.py
```

Resource comparison:

```bash
uv run python scripts/benchmark_canonical_ela.py \
  --nodes 1024 \
  --degree 32 \
  --width 128 \
  --depth 8 \
  --device cuda \
  --dtype bfloat16 \
  --output artifacts/canonical-ela-overhead.json
```

## Design documents

- `docs/CANONICAL_ELA.md`
- `docs/API_POLICY.md`
- `docs/MIGRATION_TO_ELA.md`
- `docs/ARCHITECTURE_DECISION_20260731.md`
- `docs/SCALING.md`
- `docs/SPATIAL_OPERATOR_INDEX.md`

Unit tests establish shape, symmetry, finite-gradient, initialization, and
execution contracts. They do not by themselves establish downstream accuracy or
hardware speedup.
