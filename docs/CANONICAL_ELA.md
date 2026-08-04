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

The same receiver-CSR edge set also carries two smooth radial scales:
relation-aware C2 envelopes at the full cutoff and at half the cutoff. Learned score mixing
acts per local rank, while learned positive value mixing acts per rank and value
lane. Both residuals are zero initialized, so the added multiscale lane is an
exact identity for migrated checkpoints until trained. Edges outside a
relation-specific cutoff remain exactly masked at every scale; no second graph
or dense pair matrix is built. A private layer switch exists only for paired
architecture ablations and is not a public model variant.

## Low-order tensor closure

The pointwise closure retains every parity-valid output with $l\le 2$ from
$1\otimes1$, $2\otimes2$, and the newly completed $1\otimes2$ coupling. For a
vector $v$ and symmetric-traceless tensor $T$, the two added Cartesian
Clebsch--Gordan projections are

$$
C_1(v,T)=Tv,
\qquad
C_2(v,T)=\mathrm{ST}\!\left([v]_\times T-T[v]_\times\right).
$$

Here $[v]_\times w=v\times w$. Polar/even and axial/odd inputs route to the
polar $l=1$ and odd $l=2$ sectors; axial/even and polar/odd inputs route to the
axial $l=1$ and even $l=2$ sectors. The $l=3$ component of $1\otimes2$ is
intentionally omitted by the public $l\le2$ representation contract. Separate
zero-initialized output maps keep this added lane identity-safe; a private
closure switch exists only for paired architecture ablations.

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
input graph. The public switch updates at every layer boundary while preserving
the hidden state; updated geometry is then consumed by the following layer.
Advanced `ELAConfig.coordinate_updates=K` spreads K distinct update boundaries
deterministically across the depth and requires `K <= depth`. Geometry is
rebuilt between stages or reused only when preparation provenance permits it.
The cumulative displacement is bounded by `max_coordinate_step`. Fixed-position
and moving-position models share the same `ELAGraph -> ELA -> ELAGraph` call.

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
