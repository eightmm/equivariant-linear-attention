# PROJECT.md

## Mission

Build one domain-agnostic, high-performance equivariant linear-attention layer
for sparse 3D data.

The same architecture supports molecules, proteins, protein--ligand complexes,
particles, meshes, and point clouds without task-specific vocabulary inside the
core.

## Canonical architecture

The public model, layer, and graph container are:

```text
ELA
ELALayer
ELABatch
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

with `l <= 2`. Input and output representations are declared only with
`input_irreps` and `output_irreps`; positions remain a separate affine geometry
input.

## Public API

```python
from equivariant_attention import ELA, ELABatch

model = ELA(
    input_irreps="32x0e + 4x1o",
    output_irreps="1x0e + 1x1o",
    width=128,
    depth=8,
    cutoff=6.0,
)

batch = ELABatch(
    node_irreps=x,
    positions=pos,
    edge_index=edge_index,
)

output = model(batch)
```

Scalar-only features use ordinary scalar irreps such as `"32x0e"`. There is no
parallel `node_dim`, `output_dim`, or scalar-model API.

Advanced config types remain available for reproducibility:

```python
from equivariant_attention import (
    ELAConfig,
    ELAFeatures,
    OrderContext,
    RefinementRequest,
    SparseGeometry,
)
```

No second backbone or architecture layer is public.

## Dependency-free data policy

Core execution and graph batching depend only on PyTorch. External layouts are
normalized before model execution:

```python
batch = ELA.batch(x, pos, batch=batch_index, edge_index=edge_index)
batch = ELA.padded(x_padded, pos_padded, mask=node_mask)
batch = ELA.collate(samples)
```

The model always receives one packed `ELABatch`. `ptr` is canonical graph
membership, and optional edges use receiver/sender COO.

When edges are omitted, ELA constructs exact radius candidates. Small graphs use
a chunked dense reference; larger graphs use an exact 3D cell list followed by
distance filtering. Repeated workloads should call `model.prepare(batch)` and
reuse the prepared execution metadata.

## Optional functionality

Condition, semantic order, and coordinate refinement are fields of the same
`ELABatch`. Absent fields bypass their modules entirely. Semantic order is a
node-attached label such as residue rank or time, never tensor row index.

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
- one packed ELABatch execution path.

Not canonical:

- alternate Unified, Advanced, LGL, AttnRes, implicit-only, or hybrid backbone
  selection;
- full-state implicit Gaussian--Taylor residual;
- schedules such as `implicit_every` or AttnRes block count;
- separate scalar, conditioned, padded, or coordinate-updating model classes;
- PyG/DGL as a core dependency;
- kernel backend as a model hyperparameter.

Historical numerical modules and tracked artifacts may remain private while the
canonical implementation or checkpoint migration depends on them. They are not
public model choices.

## Complexity contract

For `N` nodes, `E` directed candidates, and `L` layers at fixed widths and ranks:

\[
T=O\left(L(N+E)\right).
\]

Node-linear execution requires

\[
E=O(N).
\]

The exact cell-list radius path has expected `O(N+E)` work under bounded density
and fixed cutoff, but worst-case work can remain quadratic. A refinement request
with `S` outer steps evaluates the stack approximately `S+1` times. Neighbor
discovery and reconstruction must be accounted for separately.

## Kernel policy

The PyTorch prepared path is the numerical contract. Triton currently accelerates
receiver-major CSR reductions and groups local payloads to reduce peak temporary
memory. It is an optional execution backend with automatic PyTorch fallback.

Further optimization priority:

1. fuse local geometry, score, and receiver reductions;
2. use recomputation for local backward;
3. add periodic/minimum-image neighbor construction when required;
4. leave global GEMM/BMM to Inductor and vendor libraries unless profiling proves
   a specific deficit.

Every custom kernel requires PyTorch equivalence, equivariance, feature,
coordinate, and parameter-gradient checks, plus measured complete-stack latency
or memory benefit.

## Current validation gates

Required contracts include:

- one package-root architecture, layer, and graph container;
- irreps-only representation configuration;
- exact zero-initialized global/local additive behavior;
- proper and improper O(3), translation, node permutation, edge order, graph
  isolation, and batching;
- all supported input/output parity sectors;
- packed, padded-ingestion, and mapping-collate behavior;
- exact dense/cell-list radius agreement;
- semantic-order permutation consistency and disabled-node isolation;
- condition neutrality and context-free bypass;
- coordinate-refinement identity, mask, centering, bound, and equivariance;
- feature and coordinate first gradients and required double backward;
- CUDA FP32/BF16 and PyTorch/Triton agreement;
- latency and memory measured separately from graph construction.

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
