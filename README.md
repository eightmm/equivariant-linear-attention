# TriELA: Pair-Centric Equivariant Attention

TriELA is one canonical architecture for token-level 3D graphs and point
clouds. It combines persistent pair reasoning with O(3)-equivariant node
features and local geometric refinement:

```text
ELAGraph + optional BiomolecularPairContext -> TriELA -> ELAGraph
```

The model has three persistent state streams:

- `H`: parity-aware node irreps through `l = 2`;
- `Z`: an ordered, O(3)-invariant dense pair state;
- `X`: Cartesian coordinates.

`Z_ij` and `Z_ji` are intentionally distinct. Every pair block applies exact
gated outgoing and incoming triangle multiplication, a pair transition, and a
pair-to-node update. Global ELA then transports equivariant node values, while
a tie-complete truncated-support Local ELA turns pair context and relative geometry into
physical local messages. Coordinates can change only after a local block.

There is no legacy model wrapper, approximate production backend, runtime
fallback, or silent checkpoint migration. The production pair operator is the
exact dense PairMixer core.

## Intended scale

TriELA is intended for residue, nucleotide, ligand-atom, or other
coarse/token-level representations. Dense pair state is not appropriate as an
all-atom representation of a large protein; use a sparse local atom decoder for
that stage.

The complete model is not linear-scaling:

| Component | Complexity |
|---|---:|
| Dense pair memory | `O(B N_max^2 C_z)` |
| Exact triangle multiplication | `O(B N_max^3 C_h)` |
| Global ELA node transport | linear in `N` at fixed feature order |
| Local ELA messages | `O(kN)` generically; `O(N^2)` for exact cutoff ties |

`max_pair_tokens` rejects accidental oversized layouts. Ragged batches should
be bucketed by token count to avoid excessive padding.

Local support retains every point tied at its truncation radius. This avoids
an ordering-dependent top-k choice and preserves permutation equivariance for
coincident point clouds. It can exceed `local_points` only in such degenerate
ties; dense pair memory already bounds the complete model at `O(N^2)` or more.

## Install

```bash
git clone https://github.com/eightmm/equivariant-linear-attention.git
cd equivariant-linear-attention
uv sync --locked
```

Python 3.12 or newer is required. The canonical package does not require PyG,
DGL, Triton, or a custom CUDA kernel.

## Minimal use

```python
import torch

from equivariant_linear_attention import ELAGraph, TriELA

model = TriELA(
    "32x0e",
    "1x0e",
    width=128,
    pair_width=64,
    triangle_hidden=64,
    num_stages=3,
    pair_blocks_per_stage=4,
    local_blocks_per_stage=2,
    max_pair_tokens=512,
)

graph = ELAGraph(
    x=torch.randn(96, 32),
    pos=torch.randn(96, 3),
    batch=torch.arange(3).repeat_interleave(32),
)

out = model(graph)
node_prediction = out.x
```

`batch` defaults to one sample. `ELAGraph.x` and `ELAGraph.pos` must share a
floating-point dtype and device. Structural index tensors are validated as
immutable graph metadata; construct a new graph when membership changes.

`forward` returns an `ELAGraph`. To inspect the final pair memory, masked
distogram logits, and tensor diagnostics, use `forward_with_aux`:

```python
aux = model.forward_with_aux(graph)
out = aux.graph
pair = aux.pair_state
distogram_logits = aux.distogram_logits
```

For a serializable configuration, construct the same model explicitly:

```python
from equivariant_linear_attention import TriELAConfig

config = TriELAConfig(
    input_irreps="32x0e",
    output_irreps="1x0e",
    width=128,
    pair_width=64,
    triangle_hidden=64,
)
model = TriELA.from_config(config)
```

The package exports one model family and its data contracts:

```text
TriELA
TriELAConfig
TriELAOutput
ELAGraph
BiomolecularPairContext
DensePairState
```

## Irreps and symmetry

Input and output features use real O(3) irreps:

```python
model = TriELA(
    "16x0e + 4x1o + 2x2e",
    "2x0e + 1x1o + 1x2e",
)
```

The persistent node carrier supports arbitrary multiplicities of:

```text
0e  scalar                 0o  pseudoscalar
1o  polar vector           1e  axial vector
2e  even ST tensor         2o  odd ST tensor
```

Learned maps act on multiplicity axes. Pair features, pair gates, attention
coefficients, and per-irrep channel gates are invariant scalars. Consequently:

- rotating or reflecting the input transforms node outputs in their declared
  irreps;
- translating coordinates translates coordinate outputs but leaves invariant
  predictions unchanged;
- permuting nodes permutes both pair axes and all node outputs consistently;
- pair state and pair-derived context remain O(3)-invariant.

These are deterministic/evaluation-mode statements. The default row-wise pair
dropout is permutation-equivariant in distribution during training; set
`pair_dropout=0` when pathwise equality under a shared RNG stream is required.

## Batching, groups, and biomolecular metadata

`batch` separates samples. `group` optionally creates independent interaction
segments inside a sample; no pair slot or node update crosses a group boundary.

Chain and entity identifiers are not groups. They are ordered pair features, so
cross-chain, protein-ligand, and protein-nucleic-acid interactions remain
available. Optional metadata is supplied through `BiomolecularPairContext`:

```python
from equivariant_linear_attention import BiomolecularPairContext

context = BiomolecularPairContext(
    token_index=token_index,
    chain_id=chain_id,
    entity_id=entity_id,
    molecule_type=molecule_type,
    bond_index=bond_index,
    bond_type=bond_type,
)

out = model(graph, pair_context=context)
```

`context.to(...)` moves floating pair features while preserving integer index
dtypes. `BiomolecularPairContext.collate(...)` concatenates node metadata,
offsets bond indices, and block-packs external ordered pair features alongside
an `ELAGraph.collate(...)` batch.

Raw Cartesian vector or tensor components never enter the scalar pair MLP.
Geometry contributes through distance bases and invariant contractions of
matching irreps. The latent pair state is never symmetrized; only a symmetric
prediction head, such as the distogram head, may symmetrize its input.

## Canonical stage

Every stage executes one fixed schedule:

```text
Node/Geometry -> Pair refresh once

repeat four times:
  exact gated outgoing triangle multiplication
  exact gated incoming triangle multiplication
  pair SwiGLU transition
  normalized outgoing/incoming Pair -> Node summary
  invariant pair-context injection
  Global ELA
  equivariant node transition

repeat twice:
  pair-conditioned Local ELA
  equivariant node transition
  optional local-only coordinate update
```

With fixed coordinates, geometry and local support are reused. When positions
change, geometry and support are rebuilt immediately after that local update.
Global or pair blocks never move coordinates.

## Validation and evidence

```bash
bash scripts/check.sh fast
```

The acceptance contract includes exact dense triangle references, graph/group
isolation, directed-pair behavior, first- and second-order derivatives, FP32
and CPU BF16 execution, O(3)/translation/permutation checks, distogram masking,
and executable benchmark and ablation entry points.

See:

- [architecture overview](docs/ARCHITECTURE.md)
- [full equations and contracts](docs/TRI_ELA_ARCHITECTURE.md)
- [experiment and evidence plan](docs/TRI_ELA_EXPERIMENTS.md)

Historical QM9, LBA, and ATOM3D PSR measurements were produced by pre-TriELA
architectures. They do not validate this model and must not be compared as if
they came from the canonical pair-centric implementation. Any new accuracy,
runtime, or memory claim requires a run tied to the exact source revision,
configuration, data split, and hardware record.
