# Canonical Equivariant Linear Attention

## Public surface

Canonical ELA has exactly two root objects:

```text
ELA       one model
ELAGraph  one input-and-output graph
```

Execution is always:

```python
out = model(graph)
```

where both `graph` and `out` are `ELAGraph`.

## Layer equation

For hidden state $h^l$, every layer evaluates two exact branches:

$$
\bar h^l = \text{EqRMSNorm}(h^l),
$$

$$
G^l = \text{GlobalELA}(\bar h^l),
\qquad
L^l = \text{SparseLocal}(\bar h^l, x, \mathcal E),
$$

and uses the canonical fixed fusion

$$
M^l = G^l + L^l.
$$

The message enters parity-valid updates, low-order tensor closure, a residual,
and an equivariant feed-forward block. No branch materializes a dense
$N\times N$ attention matrix and no persistent edge hidden state is stored.

## Representation

Inputs and outputs are declared only through irreps. Supported sectors are:

```text
0e  ordinary scalar
0o  pseudoscalar
1o  polar vector
1e  axial vector
2e  even symmetric-traceless tensor
2o  odd symmetric-traceless tensor
```

Arbitrary multiplicities are supported for $l\le 2$. Positions are separate from
`1o` features because positions transform affinely under translation.

The hidden carrier is deterministic from `width`. Scalar and geometric capacity
both grow with width; users do not select heads, ranks, or hidden irreps.

## Global branch

The global branch forms positive, parity-valid low-rank query and key features
from:

- invariant scalar state;
- polar and axial alignment;
- even and odd tensor alignment;
- paired pseudoscalar routing;
- a small fixed set of graph-centered radial shells.

It aggregates sufficient statistics in node-linear memory and time for fixed
representation size.

## Local branch

The local branch uses exact sparse edges, radial basis features, smooth cutoffs,
unit-direction angular invariants, relation-conditioned radial modulation, and
relation-conditioned value gates. First and second directional moments provide
polar, axial, tensor, and chiral carriers without explicit edge triples.

## Geometry

Public edges are source-to-target. The private core converts once to a
receiver-major CSR layout. When edges are omitted, an exact radius graph is built
without self edges.

Prepared topology records its source, cutoff, neighbor policy, relation schema,
skin, and reference positions. This provenance prevents a stale radius topology
from being silently reused.

## Coordinate updates

Coordinate updates are a model property:

```python
model = ELA(..., update_positions=True)
```

The update is a bounded polar-vector displacement, optionally masked by the
input graph. After each advanced outer update, geometry is rebuilt or reused only
when preparation provenance permits it. Fixed-position and moving-position models
share the same `ELAGraph -> ELA -> ELAGraph` call.

## Complexity

For $N$ nodes, $E$ directed candidates, and $L$ layers at fixed hidden size:

$$
T = O\left(L(N+E)\right).
$$

Node-linear execution requires $E=O(N)$, as obtained under bounded spatial
density and fixed cutoff. Degenerate dense neighborhoods can still make the local
path quadratic.

## Non-goals

Canonical ELA does not expose:

- multiple backbones;
- user-selected hidden irreps, head count, or local rank;
- PyG or DGL as a core dependency;
- a separate output class;
- public packed-batch or prepared-graph classes;
- a coordinate refiner callback protocol;
- automatic Triton selection without a numerical contract.
