# PROJECT.md

## Mission

Build one domain-agnostic, high-performance equivariant linear-attention layer
for sparse 3D data.

The same architecture must support molecules, proteins, protein--ligand
complexes, particles, meshes, and point clouds without task-specific vocabulary
inside the core.

## Canonical architecture

The public model and layer are:

```text
ELA
ELALayer
```

Every layer evaluates

\[
\boxed{
\text{exact global equivariant linear attention}
+
\text{exact sparse short-range residual}
+
\text{invariant global/local fusion}
}
\]

The hidden carrier is parity-complete over

```text
0e, 0o, 1o, 1e, 2e, 2o
```

with `l <= 2`. Input and output representations are declared with
`input_irreps` and `output_irreps`; positions remain a separate affine geometry
input.

## Public API

```python
from equivariant_attention import (
    ELA,
    ELAConfig,
    ELAContext,
    ELAFeatures,
    ELALayer,
    OrderContext,
    RefinementRequest,
    SparseGeometry,
)
```

No second backbone or architecture layer is public.

Optional functionality is allocated through

```python
ELAFeatures(
    condition_dim=0,
    order_dim=0,
    coordinate_refinement=False,
)
```

and activated per call through

```python
ELAContext(
    condition=None,
    order=None,
    refinement=None,
)
```

Absent context fields bypass their modules entirely. Semantic order is a
node-attached label such as residue rank or time, never the tensor row index.

## Architecture policy

Canonical:

- exact finite-feature graph-global linear attention;
- exact compact-support sparse local interaction;
- branch-aware invariant fusion;
- equivariant RMS normalization;
- parity-valid gated nonlinearities;
- low-order tensor closure and aggregate chirality;
- optional invariant context and semantic-order PE in the same layer;
- optional coordinate refinement through the same model entry point.

Not canonical:

- alternate `Unified`, `Advanced`, `LGL`, AttnRes, implicit-only, or hybrid
  backbone selection;
- full-state implicit Gaussian--Taylor residual;
- architecture schedules such as `implicit_every` or AttnRes block count;
- a separate conditioned or coordinate-updating model class.

Historical numerical implementation modules and tracked artifacts may remain as
internal provenance while canonical ELA or checkpoint migration depends on them.
They are not public model choices.

## Complexity contract

For `N` nodes, `E` directed candidate edges, and `L` layers at fixed widths and
ranks,

\[
T=O\left(L(N+E)\right).
\]

The model is node-linear only for graph families with

\[
E=O(N).
\]

A refinement request with `S` outer steps evaluates the stack approximately
`S+1` times. Neighbor discovery and graph reconstruction are external costs and
must be reported separately.

## Current validation gates

Required focused contracts:

- one package-root architecture and one layer;
- exact zero-initialized global/local additive behavior;
- proper and improper O(3), translation, node permutation, edge order, graph
  isolation, and batching;
- all supported input/output parity sectors;
- semantic-order permutation consistency and disabled-node isolation;
- condition path neutrality at initialization and true context-free bypass after
  training;
- coordinate-refinement identity initialization, mask, centering, bound, and
  equivariance;
- feature and coordinate first gradients and required double backward;
- CUDA BF16 finite forward/backward;
- latency and memory measured without hiding graph-construction costs.

Run:

```bash
ELA_SUITE_MODE=full \
ELA_SUITE_DEVICE=cuda \
ELA_SUITE_DTYPE=bfloat16 \
  bash scripts/run_canonical_ela_suite.sh \
  artifacts/canonical-ela/final
```

Automated push/PR CI remains disabled. Validation is local and explicit.

## Evidence boundary

Tests prove contracts, not architecture superiority. Accuracy, resource, and
scaling claims require paired seeds, frozen splits and data revisions, exact
commands, environment receipts, and explicit neighbor-construction accounting.

No alternate mechanism is promoted by a single task, seed, overfit probe, or
operator microbenchmark.
