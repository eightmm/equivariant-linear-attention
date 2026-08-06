# Edge-free relative moments and Krylov relation closure

Canonical ELA does not discover a radius graph when `edge_index` is absent. It
keeps every persistent state on nodes, uses graphwise coordinate moments for
`l <= 2` geometry, and composes one implicit all-irrep relation operator without
materializing pair states.

## 1. Representation contract

For one component with node state

$$
H_i=\bigoplus_{\tau=(l,p)}H_i^\tau
$$

and positions `x_i`, ELA retains only `O(NC)` node tensors plus small factorized
attention summaries. It never constructs

$$
A\in\mathbb R^{N\times N}
\quad\text{or}\quad
Z\in\mathbb R^{N\times N\times C}.
$$

Positions are centred and normalized per interaction component before they enter
geometric moments. Translation therefore cancels exactly, while rotations and
reflections act through the declared O(3) irreps.

## 2. Edge-free relative geometric moments

For one learned or deterministic source lane with scalar weight `w_j`, define
component sums

$$
S_0=\sum_j w_j,
\qquad
S_1=\sum_j w_jx_j,
\qquad
S_2=\sum_j w_j\text{ST}(x_jx_j^\top).
$$

The receiver-centred first moment is recovered without pair enumeration:

$$
P_i
= S_1-w_ix_i-(S_0-w_i)x_i
=\sum_{j\ne i}w_j(x_j-x_i).
$$

Likewise, using the compact five-component symmetric-traceless representation,

$$
Q_i
=\sum_{j\ne i}w_j
  \text{ST}\!\left((x_j-x_i)(x_j-x_i)^\top\right)
$$

is assembled from `S_0`, `S_1`, and `S_2` with receiver corrections. These are
exact algebraic identities for the selected separable weights, not sampled edge
approximations.

The resulting sectors are

- scalar mass: `0e`;
- first relative moment: `1o`;
- second symmetric-traceless moment: `2e`;
- cross-lane first moments: `1e` axial carriers;
- triple products and vector-tensor closure: `0o`/`2o` chiral carriers.

Several deterministic radial profiles provide independent lanes. Existing
pointwise tensor closure mixes these lanes with learned node features and keeps
every parity-valid output through `l=2`.

## 3. One implicit relation, three orders

Each layer forms positive invariant query/key features from all retained sectors:

- `0e` scalar state;
- paired `0o` routing;
- `1o` and `1e` alignment;
- `2e` and `2o` alignment;
- graph-centred radial features.

They define an implicit normalized relation operator `R`. Its matrix is never
constructed; applying it to any irrep value bank uses the factorized linear
attention contraction.

Within one layer, query and key features are frozen and reused:

$$
Z_1^\tau=RV^\tau,
\qquad
Z_2^\tau=RZ_1^\tau=R^2V^\tau,
\qquad
Z_3^\tau=RZ_2^\tau=R^3V^\tau.
$$

`R²` aggregates all one-intermediate-node relation paths, while `R³` aggregates
all two-intermediate-node paths. This is an implicit relational closure, not a
stored triangle or quadrilateral tensor: individual pair and tuple identities
are intentionally contracted into node states.

A node/head-wise invariant gate produces

$$
G_i^\tau=Z_{1,i}^\tau
+g_{2,i}^{0e}Z_{2,i}^\tau
+g_{3,i}^{0e}Z_{3,i}^\tau.
$$

The higher-order gate is initialized to zero. Migrated checkpoints therefore
start from the established first-order global operator and can learn to use
higher Krylov orders without an initialization-time function jump.

## 4. Optional explicit sparse residual

Supplying `edge_index` activates the retained typed sparse geometric operator:

$$
M_i^\tau=G_i^\tau+L_i^\tau.
$$

This path is intended for topology that is part of the data rather than inferred
from distance, such as covalent bonds, meshes, temporal transitions, metal
coordination, or curated contact candidates. Explicit edges use receiver-major
CSR packing, radial cutoffs, relation types, and exact topology provenance.

Omitting `edge_index` gives `L=0`; no radius search or neighbor rebuild occurs.
The public `cutoff` and `max_neighbors` arguments therefore affect only explicit
sparse processing and retained compatibility utilities.

## 5. Equivariance

The relation operator is an invariant scalar operator on the node axis. For an
irrep `tau` and O(3) transformation `D_tau(g)`, it commutes with the irrep action:

$$
R\bigl(V^\tau D_\tau(g)^\top\bigr)
=(RV^\tau)D_\tau(g)^\top.
$$

The same holds for every power `R^k`. Relative coordinate moments transform as
`1o` and `0e + 2e`; the existing Cartesian Clebsch--Gordan closure then routes
all outputs according to angular degree and parity.

## 6. Complexity

Let `N` be nodes, `C` hidden width, `L` layers, and keep the relation feature
rank proportional to `C`. A fixed three-order edge-free layer costs

$$
O(NC^2)
$$

and uses

$$
O(NC+C^2)
$$

working memory. The constant is larger than one first-order linear-attention
application because the same operator is applied three times, but sequence
length remains linear. An explicit sparse residual adds approximately `O(EC)`
work and `O(E)` packed topology.

## 7. What is preserved and what is not

Preserved without edges:

- graph/component isolation;
- all declared node irreps and parity;
- exact selected first/second relative moment aggregates;
- one-, two-, and three-step implicit relation actions;
- coordinate-update equivariance and second-order autograd;
- node-linear memory in sequence length.

Not represented explicitly:

- arbitrary per-pair hidden states;
- exact hard nearest-neighbor or radius adjacency;
- individual triangle identities;
- bond types unless supplied as explicit edges;
- singular or sharply truncated pair potentials.

Tasks requiring exact local topology should supply edges. Tasks dominated by
long-range representation learning can use the edge-free default directly.

## 8. Validation contract

The focused tests verify:

1. first and second relative moment results against explicit pair enumeration;
2. absence of radius-graph discovery on the default path;
3. cutoff independence when no explicit edges are supplied;
4. identity-safe but trainable higher Krylov orders;
5. activation of the sparse residual only for explicit edges.

The broader suite retains independent tests for the sparse compatibility engine,
O(3) actions, translation, permutations, graph isolation, checkpoint migration,
coordinate double backward, and explicit-topology provenance.
