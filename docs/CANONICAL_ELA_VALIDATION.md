# Canonical ELA validation

Validation is organized around one public contract:

```text
ELAGraph -> ELA -> ELAGraph
```

## API contract

Tests verify that:

- the package root exports exactly `ELA` and `ELAGraph`;
- `ELA.forward` accepts one `ELAGraph` argument;
- the return value is an `ELAGraph`;
- node, graph-mean, graph-sum, coordinate, and displacement outputs occupy fixed
  fields on that object;
- public edges use source-to-target order;
- `ELAGraph.collate` is the only public batching path;
- coordinate updates are declared on the model.

## Symmetry contract

Randomized tests cover:

- proper and improper O(3) transformations;
- translations of Cartesian positions;
- node permutations;
- scalar, pseudoscalar, polar, axial, and symmetric-traceless sectors;
- coordinate-update equivariance and update bounds;
- relation-conditioned local kernels;
- chiral channels.

## Differentiation contract

CPU reference tests cover first derivatives and double backward. Energy and
force tests verify that scalar energy differentiation produces equivariant
conservative forces.

## Topology contract

Regression tests cover:

- no automatic self edges;
- graph and interaction-component isolation;
- complete nearest-distance shells at a neighbor cap;
- explicit-topology reuse after coordinate changes;
- radius-topology invalidation after incompatible changes;
- exact safe reuse inside a configured skin;
- cutoff and relation-schema compatibility.

## Representation contract

Tests validate packing and splitting of irreps and the physical Frobenius metric
for compact symmetric-traceless tensor coordinates.

## Backend contract

PyTorch is the reference backend. CUDA, BF16, Triton, and compiled execution are
checked separately when those runtimes are available. A backend is not promoted
by latency alone; it must preserve outputs, gradients, equivariance, and memory
safety for its declared workload.

## Required local gate

The canonical local gate is:

```bash
bash scripts/check.sh fast
uv run python examples/flow_matching_velocity.py
```

CUDA and Triton checks must additionally run on compatible NVIDIA hardware before
a release that changes kernels or mixed-precision behavior:

```bash
bash scripts/check.sh gpu
uv run pytest -q tests/test_triton_equivariance_cuda.py
```
