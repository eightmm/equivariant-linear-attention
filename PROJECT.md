# Project contract

This repository implements one canonical pair-centric equivariant architecture
for token-level 3D graphs and point clouds:

```text
ELAGraph + optional BiomolecularPairContext -> TriELA -> ELAGraph
```

`TriELA` has three state streams:

- an O(3)-equivariant node state `H` with persistent `l <= 2` parity sectors;
- an O(3)-invariant, ordered dense pair state `Z_ij`;
- Cartesian coordinates `X`.

The pair state is persistent and may be asymmetric. It is the relational memory
of the model; the old self-adjoint node relation is not treated as a substitute
for pair memory.

## Canonical stage

The model has three stages by default. Every stage executes exactly:

```text
Node/Geometry -> Pair refresh once

repeat four times:
  exact gated outgoing triangle multiplication
  exact gated incoming triangle multiplication
  pair SwiGLU transition
  gated outgoing/incoming Pair -> Node summary
  invariant pair-context injection
  Global ELA
  equivariant node transition

repeat twice:
  pair-conditioned Local ELA
  equivariant node transition
  optional coordinate update
```

The production pair operator is the exact dense PairMixer core. There is no
runtime backend choice, approximation backend, legacy ELA wrapper, silent
fallback, checkpoint migration path, or diagnostic model route. Selected pair
attention, chunked triangle multiplication, low-rank/sparse pair factors, and
distillation are research proposals, not alternate production paths.

## Mathematical contracts

For normalized pair operands `A` and `B`, every pair block applies

```text
M_out[i,j] = sum_k A[i,k] * B[j,k]
M_in [i,j] = sum_k A[k,j] * B[k,i]
```

with independent projections and gates for the two directions. Operands are
masked before contraction, contracted features are normalized, output
projections are zero-initialized, and padded residuals are masked back to exact
zero.

Pair-to-node coupling uses normalized gated row and column summaries. It never
adds an arbitrary `Z_ij` bias to global linear-attention logits, because that
would destroy separability. Pair-derived quantities that gate equivariant node
sectors are invariant scalars.

The local operator may gather `Z_ij` on its truncated geometric support. It uses
relative displacements and invariant pair gates to produce parity-valid
equivariant messages. Coordinates may change only after a local block. Geometry
and support are rebuilt after such a change and otherwise reused.

Pair embedding may use left/right even scalars, norms and same-irrep inner
products, distance RBFs, relative token index, chain/entity/molecule metadata,
bond metadata, and external invariant pair features. Raw Cartesian vector or
tensor components never enter a scalar MLP.

## Symmetry and isolation

- O(3), translation, and node-permutation equivariance are exact up to floating
  point error in deterministic evaluation. Training-time row dropout is
  equivariant in distribution, as in standard stochastic regularization.
- Pair states and pair-derived node contexts are O(3)-invariant.
- Permuting nodes permutes both pair axes consistently.
- Different batch samples never share pair slots.
- `group` is an interaction-isolation mask inside a sample.
- Chain and entity identifiers are features, never isolation masks; cross-chain
  and protein-ligand pairs are preserved.
- `Z_ij` is ordered and is not symmetrized. Only symmetric prediction heads,
  such as a distogram head, may symmetrize their input.

## Scale and intended domain

The model is intended for residue/token/coarse-grained 3D graphs and point
clouds. Dense all-atom protein pair state is outside the default contract; an
all-atom decoder should use a separate sparse local representation.

The complexity claim is deliberately narrow:

- persistent pair memory: `O(B * N_max^2 * C_z)`;
- exact triangle multiplication: `O(B * N_max^3 * C_h)`;
- Global ELA node transport: linear in node count at fixed feature order;
- Local ELA support: `O(kN)` for generic positions; exact cutoff ties are all
  retained to preserve permutation equivariance, giving an `O(N^2)` degenerate
  worst case.

The complete architecture is not a linear-scaling model. `max_pair_tokens`
must reject accidental oversized dense layouts, and size bucketing is expected
for ragged batches.

## Public surface

The canonical package exports:

```text
TriELA
TriELAConfig
TriELAOutput
ELAGraph
BiomolecularPairContext
DensePairState
```

Input and output irreps use the existing parity-aware `l <= 2` layout. The
model returns `ELAGraph` from `forward`; `forward_with_aux` additionally returns
the final ordered pair state, masked distogram logits, and tensor diagnostics.
Pair-head loss weights belong to training code, not the model.

`ELAGraph.x` and `ELAGraph.pos` must share a floating-point dtype and device.
Structural index tensors are immutable after validation. No PyG, DGL, Triton,
or custom kernel dependency is part of the canonical implementation.

## Evidence gate

Completion requires CPU verification of:

- exact outgoing/incoming algebra against naive einsum references;
- pair packing, padding, transpose, graph/group isolation, and token guard;
- ordered/asymmetric pair behavior;
- finite first and second derivatives in float64;
- FP32 and CPU BF16 autocast forward/backward without NaN or Inf;
- O(3), reflection, translation, and permutation behavior of node, pair, and
  coordinate outputs;
- distogram masking and head-only symmetrization;
- executable benchmark and component-ablation scripts;
- the repository fast gate.

Historical QM9/LBA/PSR results describe earlier architectures and are not
evidence for this model. New accuracy claims require new runs tied to the exact
source revision and data split.
