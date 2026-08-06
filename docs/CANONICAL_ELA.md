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

For hidden state $h^l$, every layer first normalizes the irrep carrier,

$$
\bar h^l = \text{EqRMSNorm}(h^l),
$$

then constructs one invariant factorized relation operator $R^l$ and reuses it
at three Krylov orders:

$$
Z_1^l = R^lV^l,
\qquad
Z_2^l = R^lZ_1^l,
\qquad
Z_3^l = R^lZ_2^l.
$$

The implicit global message is

$$
G^l = Z_1^l + g_2^l\odot Z_2^l + g_3^l\odot Z_3^l,
$$

where $g_2^l$ and $g_3^l$ are invariant `0e` gates initialized to zero. An
explicit sparse residual is evaluated only when `edge_index` is supplied:

$$
L^l = \mathbf 1_{\{E>0\}}\text{SparseLocal}(\bar h^l,x,\mathcal E),
\qquad
M^l = G^l + L^l.
$$

The message enters parity-valid updates, low-order tensor closure, a residual,
and an equivariant feed-forward block. No branch materializes a dense
$N\times N$ attention matrix, a pair representation, or persistent edge hidden
state.

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

## Edge-free relative geometry

When public edges are omitted, ELA performs no radius search. It centres and
normalizes positions per interaction component and computes graphwise sufficient
statistics for several radial lanes. Receiver corrections recover the exact
selected relative moments

$$
P_i=\sum_{j\ne i}w_j(x_j-x_i),
$$

and

$$
Q_i=\sum_{j\ne i}w_j
\text{ST}\!\left((x_j-x_i)(x_j-x_i)^\top\right)
$$

without constructing pairs. These provide `1o` and `2e` geometry. Cross-lane
vector products, triple products, and tensor closure provide axial and chiral
`1e`, `0o`, and `2o` carriers.

The identities are exact for the chosen separable source weights; what is
compressed is individual pair identity, not the resulting first/second moment
sum.

## Implicit Krylov global branch

The global branch forms positive, parity-valid low-rank query and key features
from:

- invariant scalar state;
- polar and axial alignment;
- even and odd tensor alignment;
- paired pseudoscalar routing;
- a small fixed set of graph-centred radial features.

The same query/key features define $R^l$ for all three order applications inside
a layer. Thus $R^2V$ represents all one-intermediate-node relational paths and
$R^3V$ all two-intermediate-node paths, while remaining contracted to node
states. This is relational closure rather than an explicit triangle or polygon
tensor.

The first-order path is the initialization-time function. Higher-order gates are
zero initialized and may be absent from historical checkpoints; migration keeps
their deterministic zero initialization.

## Optional explicit sparse residual

Explicit edges are reserved for topology supplied by the task: covalent bonds,
meshes, temporal transitions, curated contacts, metal coordination, or other
semantic relations. The residual uses receiver-major CSR, radial basis features,
smooth cutoffs, unit-direction angular invariants, relation-conditioned radial
modulation, and relation-conditioned value gates.

The same explicit edge set carries two smooth radial scales: relation-aware C2
envelopes at the full cutoff and half cutoff. Learned score mixing acts per local
rank, while learned positive value mixing acts per rank and value lane. Edges
outside a relation-specific cutoff remain exactly masked. No second graph or
dense pair matrix is built.

Omitting `edge_index` sets this residual to zero and does not infer a replacement
edge set.

## Low-order tensor closure

The pointwise closure retains every parity-valid output with $l\le 2$ from
$1\otimes1$, $2\otimes2$, and $1\otimes2$ coupling. For a vector $v$ and
symmetric-traceless tensor $T$, the Cartesian Clebsch--Gordan projections include

$$
C_1(v,T)=Tv,
\qquad
C_2(v,T)=\text{ST}\!\left([v]_\times T-T[v]_\times\right).
$$

Here $[v]_\times w=v\times w$. Polar/even and axial/odd inputs route to the
polar $l=1$ and odd $l=2$ sectors; axial/even and polar/odd inputs route to the
axial $l=1$ and even $l=2$ sectors. The $l=3$ component of $1\otimes2$ is
intentionally omitted by the public $l\le2$ representation contract. Separate
zero-initialized output maps keep added closure lanes identity-safe.

## Geometry and topology

Public edges are source-to-target. Explicit edges are converted once to a
receiver-major CSR layout and are validated against interaction components.
Prepared explicit topology records relation schema and storage provenance, which
prevents stale or externally mutated topology from being silently reused.

Edge-free inputs carry an empty internal topology marker only to preserve the
single numerical-core interface. There is no discovered adjacency, cutoff
search, neighbor cap, or coordinate-dependent topology cache.

## Coordinate updates

Coordinate updates are a model property:

```python
model = ELA(..., update_positions=True)
```

The update is a bounded polar-vector displacement, optionally masked by the
input graph. The public switch updates at every layer boundary while preserving
the hidden state; the following layer recomputes relative moments from the
updated positions. Advanced `ELAConfig.coordinate_updates=K` spreads K distinct
update boundaries deterministically across the depth and requires `K <= depth`.
The cumulative displacement is bounded by `max_coordinate_step`.

When explicit edges are present, topology remains fixed and geometric values are
recomputed from current coordinates. Fixed-position and moving-position models
share the same `ELAGraph -> ELA -> ELAGraph` call.

## Complexity

For $N$ nodes, hidden width $C$, and $L$ layers, the fixed three-order edge-free
path has

$$
T=O(LNC^2),
\qquad
M=O(NC+C^2).
$$

At fixed representation size this is node-linear. Explicit $E$-edge topology
adds approximately

$$
O(LEC)
$$

work and `O(E)` packed metadata. No term scales as $N^2$ in stored attention or
pair state.

## Non-goals

Canonical ELA does not expose:

- multiple public backbones;
- user-selected hidden irreps, head count, or relation rank;
- automatic radius-neighbor discovery;
- exact arbitrary per-pair hidden states;
- PyG or DGL as a core dependency;
- a separate output class;
- public packed-batch or prepared-graph classes;
- a coordinate refiner callback protocol;
- automatic Triton selection without a numerical contract.

The retained sparse compatibility engine and radius utilities remain internal
for explicit experiments and regression coverage; they are not the package-root
model contract.
