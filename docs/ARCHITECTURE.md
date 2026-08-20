# Canonical architecture

This repository implements one model: `TriELA`, an exact-dense, pair-centric
equivariant architecture for token-level 3D graphs and point clouds.

```text
ELAGraph + optional BiomolecularPairContext -> TriELA -> ELAGraph
```

There is no legacy ELA execution path, alternate approximation backend, or
silent fallback. The full equations, shapes, masking rules, and symmetry proof
are in [TRI_ELA_ARCHITECTURE.md](TRI_ELA_ARCHITECTURE.md).

## 1. Persistent state

For node `i` and ordered pair `(i, j)`, TriELA carries:

$$
H_i\in\bigoplus_{\ell\le2,p} \mathbb R^{m_{\ell,p}}\otimes V_{\ell,p},
\qquad
Z_{ij}\in\mathbb R^{C_z},
\qquad
X_i\in\mathbb R^3.
$$

- `H` is O(3)-equivariant and parity-aware.
- `Z` is O(3)-invariant, persistent, dense within an interaction segment, and
  ordered: the model does not assume `Z_ij = Z_ji`.
- `X` contains Cartesian coordinates.

The pair stream is relational memory. A node operator such as `R^2 V` can
summarize multi-hop paths, but it cannot preserve an arbitrary independent
state for every endpoint pair. It is therefore not a substitute for `Z`.

## 2. Fixed stage schedule

The default model has three stages. Every stage performs one pair refresh,
four pair/global blocks, and two local blocks:

```text
NodeGeometryToPair

4 x (
  TriangleMultiplication[outgoing]
  TriangleMultiplication[incoming]
  PairTransition
  PairToNodeSummary
  PairContextInjection
  GlobalELA
  EquivariantTransition
)

2 x (
  PairConditionedLocalELA
  EquivariantTransition
  optional CoordinateUpdate
)
```

This ordering gives pair memory several self-refinement steps before its
constraints are converted into sharp local geometric interactions. Pair
context still reaches the node stream after every pair block.

## 3. Exact pair core

Each direction has independent normalized projections and gates. With masked
operands `A` and `B`, the exact contractions are:

$$
M^{\mathrm{out}}_{ijc}=\sum_k A_{ikc}B_{jkc},
\qquad
M^{\mathrm{in}}_{ijc}=\sum_k A_{kjc}B_{kic}.
$$

Equivalent PyTorch references are:

```python
outgoing = torch.einsum("bikc,bjkc->bijc", a, b)
incoming = torch.einsum("bkjc,bkic->bijc", a, b)
```

The contraction is normalized, projected, gated, masked, and added through a
residual. Output projections begin at zero. A pre-normalized SwiGLU pair
transition follows the two directional updates.

This is exact dense PairMixer-style triangle multiplication. It is not the old
Krylov relation power, a low-rank factorization, sparse triangle closure, or a
linear-triangle approximation.

## 4. Pair construction and isolation

Dense storage uses a padded layout per interaction segment:

```text
z            [B, Nmax, Nmax, Cz]
node_mask    [B, Nmax]
pair_mask    [B, Nmax, Nmax]
packed_batch [N]
packed_slot  [N]
lengths      [B]
```

The layout guarantees:

- no pair between different samples or interaction groups;
- exact-zero padded slots and padded residuals;
- exact packed-to-padded-to-packed round trips;
- consistent permutation of both pair axes;
- an explicit `max_pair_tokens` failure before oversized allocation.

`group` means interaction isolation. Chain and entity identifiers are ordinary
pair metadata and never isolate interactions. This distinction preserves
cross-chain and biomolecular-interface pairs.

Initial pair features may include left/right even scalars, norms and invariant
inner products of matching irreps, distance RBFs, relative token indices,
chain/entity/molecule identity, sequence adjacency, bonds, external invariant
features, and recycled pair state. Raw vector or tensor components cannot be
inputs to a scalar pair MLP.

## 5. Pair-to-node coupling

An arbitrary pair-dependent logit bias would destroy the separability of
Global ELA. TriELA instead computes gated, normalized row and column summaries:

$$
c_i^{\mathrm{out}}
=
\frac{\sum_j M_{ij}\,\sigma(g_o(Z_{ij}))\odot v_o(Z_{ij})}
{\epsilon+\sum_j M_{ij}\,\sigma(g_o(Z_{ij}))},
$$

$$
c_i^{\mathrm{in}}
=
\frac{\sum_j M_{ji}\,\sigma(g_i(Z_{ji}))\odot v_i(Z_{ji})}
{\epsilon+\sum_j M_{ji}\,\sigma(g_i(Z_{ji}))}.
$$

The projected concatenation is an invariant node context. It provides an even
scalar residual, conditions Global ELA, and gates multiplicity channels in
each irrep sector. It never mixes a geometric axis with a learned matrix.

## 6. Global and local responsibilities

`GlobalELA` transports equivariant node values with invariant routing. It does
not build all-pair geometry, materialize pair-specific attention logits, or
update coordinates.

`PairConditionedLocalELA` gathers `Z_ij` only on truncated geometric support. An
invariant pair-derived gate modulates parity-valid messages built from node
irreps, relative displacements, radial features, and fixed equivariant
contractions. This is the only part of the trunk allowed to update positions.

If coordinates are fixed, the stage reuses geometry and support. If a local
coordinate update is enabled, it applies a bounded step, respects
`update_mask`, and rebuilds geometry and support before the next local block.

## 7. Equivariance contract

Let `Q` be any orthogonal matrix, `t` a translation, and `P` a node
permutation. The canonical implementation satisfies, up to floating-point
error:

$$
H^{(\ell,p)}(QX+t)=D^{(\ell,p)}(Q)H^{(\ell,p)}(X),
$$

$$
Z(QX+t)=Z(X),
$$

$$
X_{\mathrm{out}}(QX+t)=QX_{\mathrm{out}}(X)+t,
$$

and

$$
H(PX)=P H(X),
\qquad
Z(PX)=P Z(X) P^\top.
$$

These statements follow because pair construction and pair-to-node routing use
only invariant scalars, learned maps act only on multiplicity axes, local
directional operations are parity-valid equivariant contractions, and pair
axes are packed from the same node permutation.

Every exact distance tie at the local truncation radius is retained. This is
necessary because selecting an arbitrary tied subset would make the result
depend on packed order. Pair row dropout preserves permutation symmetry in
distribution during training; the equations above are pointwise statements
for deterministic evaluation or `pair_dropout=0`.

The pair latent is not symmetrized. A symmetric head may use
`(Z + Z.transpose(1, 2)) / 2` without changing the directed trunk state.

## 8. Numerical contract

The following rules are part of the architecture rather than implementation
details:

- mask both operands before every triangle contraction;
- normalize contracted features before output projection;
- zero-initialize triangle, pair-transition, node-to-pair, and pair-to-node
  output projections;
- initialize local pair conditioning near the identity;
- reapply `pair_mask` after every pair residual;
- accumulate sensitive denominators in FP32 or use a dtype-safe epsilon;
- keep padded outputs exactly zero;
- avoid in-place mutation of tensors needed by autograd;
- support FP32 and BF16 forward/backward without NaN or Inf.

## 9. Complexity boundary

For batch size `B`, padded token count `Nmax`, pair width `Cz`, triangle width
`Ch`, and generic local degree `k`:

$$
\text{pair memory}=O(BN_{\max}^2C_z),
$$

$$
\text{exact triangle compute}=O(BN_{\max}^3C_h),
$$

$$
\text{local message storage}=O(kN)\ \text{generically}.
$$

Tie-complete local support has an `O(N^2)` worst case when many points are
exactly coincident or equidistant at the cutoff. This is an intentional
symmetry-preserving exception to the nominal `local_points` target.

Global ELA remains node-linear at fixed feature order, but that does not make
the complete model linear. Dense pair state belongs at token or coarse level,
not at full protein all-atom resolution. Size bucketing is a required practical
measure for ragged batches.

## 10. Public surface

The canonical package exports only the contracts needed by this model family:

```text
TriELA
TriELAConfig
TriELAOutput
ELAGraph
BiomolecularPairContext
DensePairState
```

The direct constructor is the canonical ergonomic API. Serializable setups use
the equivalent explicit factory:

```python
model = TriELA("4x0e", "1x0e", width=128, pair_width=64)

config = TriELAConfig(
    input_irreps="4x0e",
    output_irreps="1x0e",
    width=128,
    pair_width=64,
)
same_model_shape = TriELA.from_config(config)
```

`TriELA.forward` returns an `ELAGraph`. `forward_with_aux` returns the graph,
ordered final pair state, masked distogram logits, and tensor diagnostics.
Prediction loss weights remain in training code.

## 11. Evidence boundary

The architecture is accepted only after exact-reference, masking, isolation,
directed-pair, derivative, mixed-precision, symmetry, benchmark, and ablation
checks defined in [TRI_ELA_EXPERIMENTS.md](TRI_ELA_EXPERIMENTS.md).

QM9, LBA, and ATOM3D PSR records in this repository were generated by older
architectures. They are historical context only and provide no accuracy,
runtime, memory, or equivariance evidence for canonical TriELA.
