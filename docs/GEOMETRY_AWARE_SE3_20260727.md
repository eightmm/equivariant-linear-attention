# Sparse Geometry-Aware O(3)/SE(3) Local Attention

> **Historical:** the reproduction commands below invoke retired experiment
> runners that are not shipped by the current architecture-only package. See
> `docs/EXPERIMENTS.md` for why and how to reproduce from the recorded Git
> revision.

## Decision

The EquiFlex-inspired geometry path is implemented as an opt-in capability and
is **not promoted to the default architecture**. It is mathematically active,
keeps sparse linear complexity, and supports an explicit `O3`/`SE3` symmetry
choice, but a matched 20-epoch ATOM3D-LBA ID30 validation screen found no
accuracy gain over the current gated-plus-grouped LGL.

This is a rejection of the tested configuration, not a claim that
geometry-aware or SE(3)-only models are generally unhelpful.

## What was transferred from EquiFlex

The useful idea in the supplied presentation was the ordering of local
computation:

1. pair-conditioned transport creates local geometric `1o` and `2e` moments;
2. those node moments participate in the attention score;
3. scalar, vector, relative-vector, and rank-2 values are transported together.

The dense pair tensor, triangle updates, dense pair bias, and quadratic global
attention were deliberately not transferred. They conflict with this
project's large-biomolecule objective.

The implementation reuses the incumbent gated-local aggregation as the
bootstrap:

```text
v*_ih = v_ih + m_vector_ih + m_relative_ih
T*_ih = m_tensor_ih (+ T_persistent_ih when present)
```

Bounded node gates produce query/key carriers, and retained sparse edges use

```text
ell_ijh =
    a0_h b_pair(h_ijh)
  + a1_h <q1_ih, k1_jh>
  + a2_h <q2_ih, k2_jh>_F

alpha_ijh =
    receiver_softmax_j(softclip(ell_ijh), cutoff_ij).
```

The attended values are accumulated only on the supplied sparse edge list.
There is no `N x N` tensor. At fixed channel width, the local path remains
`O(E)` and the exact factorized global route remains `O(N)`.

## Symmetry contract

`symmetry_group="O3"` is the safe default. All transported vectors are polar
and reflection covariance is required.

`symmetry_group="SE3"` permits an additional proper-rotation-equivariant axial
value:

```text
a_ijh = vee(T*_jh Q_ij - Q_ij T*_jh)
Q_ij = ST(d_ij d_ij^T).
```

This is the `l=1` component of `2e x 2e`. Under full `O(3)` it is axial
(`1e`), not polar (`1o`), so it is never mixed into the polar vector carrier
in `O3` mode. In `SE3` mode polar/axial parity is intentionally not
distinguished because only proper rotations and translations are contracted.

For biomolecules this option is semantically reasonable: biological structures
are chiral and mirror reflection is not normally a label-preserving
augmentation. It is nevertheless optional because many tasks and datasets
still benefit from the stronger `O(3)` inductive bias.

## Public configuration

```python
EquivariantAttentionConfig(
    ...,
    symmetry_group="SE3",
    use_gated_local_transport=True,
    use_geometry_aware_local_attention=True,
    use_se3_axial_tensor_product=True,
    geometry_aware_local_layers=(0,),
)
```

- `symmetry_group`: `"O3"` or `"SE3"`.
- `use_geometry_aware_local_attention`: enables sparse `0e/1o/2e` score
  refinement.
- `use_se3_axial_tensor_product`: enables the axial value and requires
  `symmetry_group="SE3"`.
- `geometry_aware_local_layers`: statically selects local layers; `None`
  selects every local layer.

All controls default to the incumbent behavior. Enabling the geometry path
requires gated local transport. Invalid symmetry combinations and non-local
layer indices fail during construction.

## Verification

Focused CPU checks cover:

- exact disabled-path compatibility and common seeded parameters;
- stable receiver-normalized sparse softmax;
- cutoff continuity and finite nonzero gradients;
- proper rotations, translations, node permutations, and edge-order
  permutations;
- full reflection covariance for `O3`;
- the expected reflection separation of the SE(3)-only axial tensor product;
- configuration and LBA builder wiring.

On a real cached LBA batch with 7,378 nodes and 153,029 retained edges, the
isolated train-step profile was:

| arm | parameters | step ratio | peak allocation ratio |
| --- | ---: | ---: | ---: |
| current candidate | 168,815 | 1.000x | 1.000x |
| geometry O(3) | 169,591 | 1.443x | 1.154x |
| geometry SE(3) | 169,624 | 1.480x | 1.163x |

The isolated SE(3) profile measured nonzero finite gradients for every geometry
parameter tensor. The axial gate itself had a nonzero gradient, but its
post-clipping L2 norm was only `4.45e-10`; this foreshadowed its negligible
empirical effect.

The end-to-end training loop amortized part of the isolated overhead:

| arm | best validation RMSE (pK) | delta vs current | median step | peak allocation |
| --- | ---: | ---: | ---: | ---: |
| current candidate | 1.602722 | — | 1.000x | 1.000x |
| geometry O(3) | 1.606076 | +0.003354 | 1.180x | 1.157x |
| geometry SE(3) | 1.610371 | +0.007649 | 1.202x | 1.164x |

The comparison used the official ID30 train/validation split, all 3,507 train
and 466 validation complexes, identical atom features, a single shared
32,303,245-edge topology, seed 41, 20 epochs / 2,940 updates per arm, and
strict CUDA determinism. Test labels were structurally inaccessible.

The O(3) arm was effectively tied with the candidate, while the SE(3)-only
axial lane added neither fitting capacity nor validation accuracy at this
budget. Both geometry arms also raised the mean pre-clip gradient norm
(`9.80 -> 11.63/11.51`) while clipping remained about 98%.

## Interpretation and next architecture step

The implementation resolves the requested functional gap: the model can now
select full `O(3)` or proper-rotation-only `SE(3)` behavior and can fuse local
`0e/1o/2e` scores without dense attention. The screen does not justify paying
its cost by default.

The higher-value EquiFlex transfer for a future packet is a **persistent sparse
pair state**, not another node-value branch. A fixed-width state on retained
protein/ligand edges can carry chemistry and interaction context across LGL
blocks in `O(E)` memory, while a dense pair tensor or triangle enumeration
cannot. That change should be evaluated separately because it changes the
layer interface and checkpoint schema.

## Reproduction

```bash
uv run python -u scripts/profile_lba_train_step.py \
  artifacts/geometry-aware-20260727/lba-profile-v4.json \
  --device cuda --batch-size 16 --warmup 2 --repeats 1 \
  --timing-repeats 20 --model-seed 41 \
  --arms candidate geometry_o3 geometry_se3

uv run python -u scripts/train_lba_id30.py \
  artifacts/geometry-aware-20260727/lba-screen-seed41 \
  --device cuda --arms candidate geometry_o3 geometry_se3 \
  --batch-size 24 --max-epochs 20 --min-epochs 20 --patience 20 \
  --warmup-epochs 3 --model-seed 41 --order-seed 71 \
  --budget-seconds 900
```
