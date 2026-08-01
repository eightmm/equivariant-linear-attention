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

The default path is deliberately small:

```python
from equivariant_attention import ELA

model = ELA.scalar(node_dim=32, cutoff=6.0)
out = model(x, pos)
```

Generic irreps use:

```python
model = ELA(
    input_irreps="32x0e + 4x1o",
    output_irreps="1x0e + 1x1o",
    width=128,
    depth=8,
    cutoff=6.0,
)
```

The advanced config types remain available for reproducibility:

```python
from equivariant_attention import (
    ELAConfig,
    ELAContext,
    ELAFeatures,
    OrderContext,
    RefinementRequest,
    SparseGeometry,
)
```

No second backbone or architecture layer is public.

## Dependency-free data policy

Core execution and graph batching depend only on PyTorch. Public input forms are:

```text
flat single graph                 [N,D], [N,3]
flat packed batch                 [N_total,D], batch[N_total]
padded batch                      [B,M,D], [B,M,3], mask[B,M]
plain mapping DataLoader batch    model(batch_dict)
```

`ELA.collate` concatenates variable-size graph dictionaries and offsets
per-graph edges. `ELA.forward` accepts flat COO, padded COO, ragged COO lists, or
boolean adjacency. When graph metadata is omitted, it constructs exact radius
candidates as a convenience reference path.

Automatic radius discovery performs quadratic pair tests inside each graph.
Repeated and performance-sensitive workloads must prepare and reuse a graph or
use a cell-list/Verlet provider.

## Optional functionality

Condition, semantic order, and coordinate refinement use the same model class.
They may be passed directly as forward keywords or packaged in `ELAContext`.
Absent fields bypass their modules entirely. Semantic order is a node-attached
label such as residue rank or time, never the tensor row index.

## Architecture policy

Canonical:

- exact finite-feature graph-global linear attention;
- exact compact-support sparse local interaction;
- branch-aware invariant fusion;
- equivariant RMS normalization;
- parity-valid gated nonlinearities;
- low-order tensor closure and aggregate chirality;
- optional invariant context and semantic-order PE in the same layer;
- optional coordinate refinement through the same model entry point;
- flat, padded, and mapping input facades over one packed execution path.

Not canonical:

- alternate `Unified`, `Advanced`, `LGL`, AttnRes, implicit-only, or hybrid
  backbone selection;
- full-state implicit Gaussian--Taylor residual;
- architecture schedules such as `implicit_every` or AttnRes block count;
- a separate conditioned, padded, or coordinate-updating model class;
- PyG/DGL as a core execution dependency;
- kernel backend as a model hyperparameter.

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

The built-in reference radius builder has

\[
O\left(\sum_g N_g^2\right)
\]

pair-test arithmetic. A refinement request with `S` outer steps evaluates the
stack approximately `S+1` times. Neighbor discovery and graph reconstruction
must be accounted for separately.

## Kernel policy

The PyTorch prepared-graph path is the numerical contract. `torch.compile`,
Triton, C++/CUDA, or other kernels may optimize execution without creating a new
model class or checkpoint schema.

Priority order:

1. fuse receiver-major local geometry, score, and reductions;
2. use recomputation for local backward;
3. implement exact on-the-fly cell-list candidates;
4. leave global GEMM/BMM to Inductor/vendor libraries unless profiling proves a
   specific deficit.

Every custom kernel requires a PyTorch fallback, numerical/equivariant gradient
checks, and measured end-to-end latency or memory benefit.

## Current validation gates

Required focused contracts include:

- one package-root architecture and one layer;
- exact zero-initialized global/local additive behavior;
- proper and improper O(3), translation, node permutation, edge order, graph
  isolation, and batching;
- all supported input/output parity sectors;
- flat/padded/mapping batch equivalence;
- padded mask and output restoration semantics;
- dependency-free radius graph agreement with a dense reference;
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
