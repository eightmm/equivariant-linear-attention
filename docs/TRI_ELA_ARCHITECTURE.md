# TriELA equations and implementation contract

This document is the normative mathematical contract for the canonical model.
It describes the exact dense production path only. Pair attention,
linear-triangle approximations, chunked triangle multiplication, low-rank pair
factors, sparse global pair memory, and legacy wrappers are not production
backends.

## 1. Notation

| Symbol | Meaning | Shape |
|---|---|---|
| `N` | packed nodes in a batch | scalar |
| `B` | interaction segments after `(batch, group)` packing | scalar |
| `N_b` | nodes in segment `b` | scalar |
| `Nmax` | `max_b N_b` | scalar |
| `H` | packed equivariant node state | sector dependent |
| `Z` | padded ordered pair state | `[B,Nmax,Nmax,Cz]` |
| `X` | packed positions | `[N,3]` |
| `M` | valid pair mask | `[B,Nmax,Nmax]` |
| `Cz` | pair width | scalar |
| `Ch` | triangle hidden width | scalar |
| `Cctx` | pair-to-node context width | scalar |

Indices `i` and `j` denote ordered pair endpoints; `k` is the contracted
triangle centre. Elementwise channel multiplication is written `odot`.

## 2. Node and pair representations

Each node carries real O(3) irreps through `l = 2`:

$$
H_i=
H_i^{0e}\oplus H_i^{0o}\oplus
H_i^{1o}\oplus H_i^{1e}\oplus
H_i^{2e}\oplus H_i^{2o}.
$$

A sector has shape `[N,m_l,p,2l+1]`, with arbitrary multiplicity `m_l,p`.
Learned linear maps act on multiplicity axes only. Fixed equivariant maps or
Clebsch-Gordan contractions are required whenever geometric axes interact.

Each ordered pair carries an invariant scalar vector:

$$
Z_{ij}\in\mathbb R^{C_z},
\qquad
Z_{ij}\not\equiv Z_{ji}.
$$

The pair stream has no vector or tensor geometric axis. Directional information
enters it only through invariant summaries. This keeps triangle operations
ordinary channelwise tensor algebra while retaining an O(3) proof.

## 3. Dense pair layout

Nodes are first partitioned by `(batch, group)`. Without `group`, each sample
is one interaction segment. A layout records:

```text
node_mask     bool  [B,Nmax]
pair_mask     bool  [B,Nmax,Nmax]
packed_batch  long  [N]
packed_slot   long  [N]
lengths       long  [B]
```

For the default complete pair support:

$$
M_{bij}=m_{bi}m_{bj},
$$

where `m` is the node mask. Invalid and padded pair values are exact zero:

$$
Z\leftarrow M[...,\mathrm{None}]\odot Z.
$$

Packing and unpacking use `packed_batch` and `packed_slot`, so a node
permutation changes both pair axes consistently. No dense storage exists
between segments. `max_pair_tokens` is checked before allocation, and ragged
batches should be size-bucketed.

Chain and entity identifiers are not part of this isolation key. They remain
features so the model can represent interfaces across chains and molecule
types.

## 4. Pair embedding

Pair initialization concatenates invariant endpoint, geometry, and metadata
features:

$$
u_{ij}=
\left[
L(H_i^{0e}),
R(H_j^{0e}),
\nu(H_i),
\nu(H_j),
\chi(H_i,H_j),
\operatorname{RBF}(\lVert X_j-X_i\rVert),
e_{ij}^{\mathrm{meta}},
e_{ij}^{\mathrm{external}}
\right].
$$

Here:

- `nu` contains per-sector invariant norms;
- `chi` contains normalized inner products between matching `(l,p)` sectors;
- metadata may encode relative token index, same chain/entity, molecule and
  residue types, sequence adjacency, directed bonds, or symmetry identifiers;
- external pair features must already be invariant and are ordered.

The initial state is:

$$
Z_{ij}=M_{ij}\,W_z u_{ij}.
$$

Raw Cartesian vector or tensor components cannot enter `W_z`. The endpoint
order is retained, the diagonal is valid by default, and `Z` is never averaged
with its transpose.

## 5. Stage-level node/geometry-to-pair refresh

At the start of every stage, current node and coordinate invariants refresh the
persistent pair memory:

$$
\Delta Z^{\mathrm{n2p}}_{ij}
=M_{ij}\,W_{\mathrm{n2p}}
\left[
H_i^{0e},H_j^{0e},
\nu(H_i),\nu(H_j),
\chi(H_i,H_j),
\operatorname{RBF}(d_{ij}),
e_{ij}^{\mathrm{meta}}
\right],
$$

$$
Z\leftarrow M\odot\left(Z+\Delta Z^{\mathrm{n2p}}\right).
$$

`W_n2p` begins at zero. Refreshing once per stage lets the pair stream retain
its own relational memory across several triangle blocks instead of merely
echoing the latest node noise after every block.

## 6. Exact gated triangle multiplication

Outgoing and incoming directions are separate modules with independent
parameters. For direction `d`, pre-normalize the pair state:

$$
\widehat Z=\operatorname{LayerNorm}(Z).
$$

Build gated operands:

$$
A^{d}=M\odot\left(W_A^d\widehat Z\right)
\odot\sigma\left(W_{gA}^d\widehat Z\right),
$$

$$
B^{d}=M\odot\left(W_B^d\widehat Z\right)
\odot\sigma\left(W_{gB}^d\widehat Z\right).
$$

The exact outgoing contraction preserves endpoint `(i,j)` while composing
relations through common target `k`:

$$
T^{\mathrm{out}}_{bijc}
=\sum_k A^{\mathrm{out}}_{bikc}B^{\mathrm{out}}_{bjkc}.
$$

The exact incoming contraction composes relations entering the endpoints:

$$
T^{\mathrm{in}}_{bijc}
=\sum_k A^{\mathrm{in}}_{bkjc}B^{\mathrm{in}}_{bkic}.
$$

Reference implementations are:

```python
t_out = torch.einsum("bikc,bjkc->bijc", a_out, b_out)
t_in = torch.einsum("bkjc,bkic->bijc", a_in, b_in)
```

The contraction is normalized over its hidden feature statistics before the
output projection. For either direction:

$$
\Delta Z^d
=M\odot\sigma(W_{go}^d\widehat Z)
\odot W_o^d\operatorname{CenterNorm}(T^d).
$$

`W_o^d` and its bias begin at zero; the output-gate bias begins near one. The
mask is applied before contraction and after projection. The directional
residuals execute sequentially:

$$
Z\leftarrow M\odot
\left(Z+\operatorname{RowDropout}(\Delta Z^{\mathrm{out}})\right),
$$

$$
Z\leftarrow M\odot
\left(Z+\operatorname{RowDropout}(\Delta Z^{\mathrm{in}})\right).
$$

The rowwise dropout mask broadcasts over the second pair axis, for example
`[B,Nmax,1,Cz]`. There are no Python loops over `i`, `j`, or `k` and no
in-place mutation of an autograd dependency.

This stochastic mask is permutation-equivariant in distribution, not
pointwise under a reused RNG stream. Exact pathwise permutation checks use
evaluation mode or `pair_dropout=0`.

## 7. Pair transition and pair block

The pair transition is pre-normalized SwiGLU:

$$
(a,b)=W_{\mathrm{in}}\operatorname{LN}(Z),
$$

$$
\Delta Z^{\mathrm{ffn}}
=W_{\mathrm{out}}\left(\operatorname{SiLU}(a)\odot b\right),
$$

$$
Z\leftarrow M\odot\left(Z+\Delta Z^{\mathrm{ffn}}\right).
$$

`W_out` begins at zero. The canonical block is exactly:

```text
GatedTriangleMultiplication(outgoing)
GatedTriangleMultiplication(incoming)
PairTransition
```

Pair attention and approximation schedules are not part of this production
block.

## 8. Pair-to-node summary

The dense pair state cannot appear as an arbitrary additive bias in a
linear-attention logit: a general `b(Z_ij)` is not separable in `i` and `j`.
TriELA instead forms invariant summaries.

For valid outgoing pairs:

$$
g^{o}_{ij}=M_{ij}\sigma(W_g^o Z_{ij}),
\qquad
v^{o}_{ij}=W_v^o Z_{ij},
$$

$$
c_i^{o}=
\frac{\sum_j g^o_{ij}\odot v^o_{ij}}
{\operatorname{clamp}(\sum_jg^o_{ij},\epsilon)}.
$$

Incoming context uses the transposed endpoint orientation:

$$
g^{i}_{ji}=M_{ji}\sigma(W_g^i Z_{ji}),
\qquad
c_i^{i}=
\frac{\sum_j g^i_{ji}\odot W_v^i Z_{ji}}
{\operatorname{clamp}(\sum_jg^i_{ji},\epsilon)}.
$$

The packed invariant context is:

$$
c_i=W_c[c_i^o,c_i^i].
$$

Sensitive numerator and denominator reductions use FP32 accumulation under
mixed precision.

## 9. Pair-context injection

The context modifies the node state only through invariant operations. The
even scalar sector receives a zero-initialized residual:

$$
H_i^{0e}\leftarrow H_i^{0e}+W_s c_i.
$$

Every sector may receive a multiplicity-channel gate:

$$
H_i^{\ell,p}\leftarrow
H_i^{\ell,p}+
\gamma_{\ell,p}(c_i)\odot W_{\ell,p}H_i^{\ell,p}.
$$

`W_s` and gate output projections begin at zero, making initial injection a
near no-op. The gate broadcasts over the geometric axis; no learned matrix
mixes `2l+1` components.

## 10. Global ELA

Global ELA provides all-segment node transport after each pair update. Its
routing features are invariant; its values retain their declared irreps. A
generic positive-feature linear-attention form for sector `(l,p)` is:

$$
S_g^{\ell,p}=\sum_{j\in g}\phi(k_j)\otimes V^{\ell,p}_j,
\qquad
z_g=\sum_{j\in g}\phi(k_j),
$$

$$
Y_i^{\ell,p}=
\frac{\phi(q_i)^\top S_g^{\ell,p}}
{\phi(q_i)^\top z_g+\epsilon}.
$$

`q`, `k`, and any pair-context modulation are invariant. Therefore the same
scalar coefficients multiply every geometric component and the value transport
is equivariant. Global ELA:

- does not construct all-pair displacement geometry;
- does not add an arbitrary dense `Z_ij` logit bias;
- does not update coordinates;
- does not substitute a self-adjoint node relation for pair memory.

An equivariant node transition follows each Global ELA update.

## 11. Pair-conditioned Local ELA

The local operator uses truncated support `E`. For directed local edge `j -> i`,
gather the matching pair slot:

$$
z_{ij}^{E}=Z_{b(i),s(i),s(j)}.
$$

Pair and radial context produce an invariant edge gate:

$$
g_{ij}=\sigma\left(
f_g[z_{ij}^{E},\operatorname{RBF}(\lVert r_{ij}\rVert),e_{ij}]
\right),
\qquad
r_{ij}=X_j-X_i.
$$

A parity-valid local message has the general form:

$$
m_{ij}^{\ell,p}
=g_{ij}\odot
\sum_{\ell_1,p_1,L}
f_{\ell_1,p_1,L}(d_{ij},z_{ij}^{E})
\operatorname{CG}_{\ell_1,L\rightarrow\ell}
\left(H_j^{\ell_1,p_1},Y_L(\widehat r_{ij})\right),
$$

where parity obeys `p = p1 * (-1)^L`. Receiver aggregation is normalized by a
valid soft or discrete degree before its residual update. Pair-derived
coefficients remain invariant scalars.

The canonical local implementation may initially gate a complete existing
equivariant local delta rather than every tensor-product path separately. That
does not alter the symmetry contract, provided the gate is invariant and
initializes near identity.

Support construction retains every exact tie at the `local_points` boundary.
Arbitrary top-k tie breaking would violate permutation equivariance. Generic
positions therefore use at most `local_points` sources per receiver, while a
degenerate coincident/equidistant configuration can expand to all tied points.

An equivariant node transition follows every local update.

## 12. Coordinate update contract

Coordinates can change only after a local block. A typical polar update is:

$$
\Delta X_i=
\sum_{j\in\mathcal N(i)}
\alpha_{ij}(H_i,H_j,z_{ij}^{E},d_{ij})
(X_j-X_i),
$$

where `alpha_ij` is invariant. Thus `Delta X` transforms as `1o`.

The update must:

- respect `update_mask` exactly;
- bound the total displacement by `max_coordinate_step`;
- divide that budget across all enabled local updates;
- leave pair/global blocks coordinate-free;
- rebuild relative geometry and local support immediately afterward;
- reuse geometry/support when coordinates remain fixed.

This avoids interpreting global dense pair relations as direct physical forces.

## 13. Stage and model recurrence

For stage `s`:

$$
Z^{s,0}=Z^{s-1}+\operatorname{NodeGeometryToPair}(H^{s-1},X^{s-1}),
$$

then for `b = 1..4`:

$$
Z^{s,b}=\operatorname{PairBlock}_{s,b}(Z^{s,b-1}),
$$

$$
c^{s,b}=\operatorname{PairToNode}(Z^{s,b}),
$$

$$
H^{s,b}=\operatorname{NodeTransition}_{s,b}
\left(
\operatorname{GlobalELA}_{s,b}
\left(
\operatorname{Inject}(H^{s,b-1},c^{s,b}),c^{s,b}
\right)
\right).
$$

Finally, for `l = 1..2`:

$$
(H^{s,l},X^{s,l})=
\operatorname{LocalStep}_{s,l}(H^{s,l-1},X^{s,l-1},Z^{s,4}).
$$

The default model repeats this stage three times.

## 14. Distogram and auxiliary output

The directed trunk remains asymmetric. A symmetric distogram head alone uses:

$$
Z^{\mathrm{sym}}=\tfrac12(Z+Z^\top),
$$

$$
L^{\mathrm{dist}}=W_d\operatorname{LN}(Z^{\mathrm{sym}}).
$$

Logits and losses exclude padded pairs. Distance bins are configurable and the
loss weight belongs to the training harness, not the model. Contact or
interface heads may use the same masked-head interface without changing the
latent state.

`forward` returns an `ELAGraph`. `forward_with_aux` returns:

```text
TriELAOutput
  graph
  pair_state
  distogram_logits
  diagnostics
```

## 15. Symmetry proof sketch

For orthogonal `Q` and translation `t`:

1. Distances, norms, matching-irrep inner products, metadata, and pair gates
   are invariant, hence `Z` and pair context are invariant.
2. Triangle operations act only on invariant pair channels, so every updated
   pair state remains invariant.
3. Global coefficients are invariant and multiply equivariant values, so
   global outputs transform in the value irrep.
4. Local spherical/tensor products use fixed parity-valid equivariant
   contractions, and invariant pair gates cannot change their transformation.
5. Coordinate deltas are invariant scalar combinations of relative polar
   vectors, hence transform as polar vectors and are translation independent.

For permutation matrix `P`, layout construction applies the same permutation
to both pair endpoints. Sums over `j` and `k` are index contractions, giving:

$$
H' = PH,
\qquad
Z'=PZP^\top,
\qquad
X'=PX.
$$

No ordering-dependent truncation is permitted inside the dense pair core.

## 16. Complexity and memory

One pair activation in dtype size `s` bytes uses approximately:

$$
sBN_{\max}^2C_z\ \text{bytes}.
$$

For BF16, `Cz = 64`, and `B = 1`, this is about 32 MiB at `N = 512` and
128 MiB at `N = 1024`, before saved backward activations.

One exact triangle direction costs:

$$
O(BN_{\max}^3C_h),
$$

and pair transitions cost:

$$
O(BN_{\max}^2C_z^2).
$$

Pair-to-node summaries cost `O(B Nmax^2 Cctx)`. Global ELA is linear in packed
nodes at fixed feature width/order. Local execution uses `O(kN)` message
storage for generic positions after support construction. Tie-complete support
has an `O(N^2)` degenerate worst case, preserving permutation symmetry instead
of selecting an order-dependent tied subset.

Activation checkpointing may trade recomputation for pair-block saved-memory
reduction, but it does not change asymptotic pair memory or triangle compute.
The complete model must never be described as linear-scaling.

## 17. Production exclusions

The canonical path excludes:

- legacy edge-free model wrappers and compatibility loaders;
- pair-free implicit triangle approximations;
- selected dense pair-attention blocks;
- linear triangular attention;
- chunked, low-rank, or sparse global pair backends;
- silent backend fallback;
- Triton or custom-kernel requirements;
- dense all-atom protein pair state;
- global or pair-driven coordinate steps;
- unconditional pair symmetrization.

These may be future research questions only after the exact dense reference has
its own verified accuracy and systems evidence.
