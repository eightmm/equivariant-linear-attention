# Architecture decision: canonical ELA v1

Date: 2026-07-31

Status: implemented as the canonical API. Mathematical and migration safeguards
passed. The 2026-07-31 resource/LBA gates rejected empirical promotion of
learned branch routing; it remains identity-initialized and experimental, with
no accuracy or hardware-efficiency advantage claimed.

## Decision

The repository has one canonical architecture:

\[
\boxed{
\text{exact global equivariant linear attention}
+
\text{exact sparse short-range local residual}
+
\text{invariant branch-aware fusion}
}
\]

The canonical public model is `ELA(ELAConfig(...))`.

Edge-free implicit spatial transport, periodic implicit schedules, always-on
hybrid transport, block Attention Residuals, and in-layer coordinate mutation
are not canonical options.

## Evidence boundary

The tracked real-data spatial comparison showed different task roles rather
than one universally dominant spatial operator.

### QM9 one-seed screen

At seed 42 and 500 updates, validation MAE was:

```text
explicit  0.671843 eV
implicit  0.627661 eV
hybrid    0.638748 eV
```

Implicit smooth transport was useful on this small smooth-property screen, but
this was one seed and did not establish a general replacement.

### LBA train-only capacity

On the frozen 16-complex training subset:

```text
explicit  0.051845 pK
implicit  0.286117 pK
hybrid    0.198044 pK
```

Only explicit local geometry passed the registered `0.10 pK` capacity gate.
This was memorization capacity, not validation performance, but it directly
showed that fixed-rank smooth implicit transport did not replace sharp local
interaction capacity.

### Scheduled implicit follow-up

Running implicit transport only at zero-based layer `[0]` reduced workspace in
some settings but did not recover LBA local capacity. The schedule introduced a
new architecture option without resolving the operator-role mismatch.

These receipts motivate a structural decision, not a claim that every recorded
metric is noise-free:

- in this fixed-rank comparison, only exact local geometry passed the
  16-complex capacity gate;
- the negative results are consistent with partial overlap between implicit
  smooth transport and the existing global sufficient-statistic path, but do
  not directly measure that overlap;
- treating the two full-state global-like paths as redundant is therefore a
  design hypothesis, not an empirical identity;
- a periodic schedule controls frequency, not mathematical role.

## Why branch-aware fusion

The admitted refined layer computed global and local messages separately but
added them before the parity update:

\[
M^\tau=G^\tau+L^\tau.
\]

This erased branch identity and made relative scale depend on graph degree,
projection width, and optimization history.

Canonical ELA replaces only that fusion step. It computes invariant message RMS
statistics and routes each irrep sector with positive node-wise weights.

The router is zero initialized:

\[
(w_G^\tau,w_L^\tau)=(1,1),
\]

so the initial function remains the admitted `G + L` model. This is a safer
architectural extension than adding a third full-state transport branch.

## Why implicit is not integrated

The current implicit Gaussian--Taylor operator is valuable as:

- a smooth spatial reference;
- an edge-memory research lane;
- a possible low-dimensional conditioner;
- a future global query/key feature augmentation;
- a long-range component for tasks that permit nonlocal coupling.

It is not integrated into canonical v1 because:

1. it does not reproduce compact support or edge-axis routing at fixed rank;
2. it duplicates graph sufficient-statistic work already performed by global
   ELA;
3. always-on and scheduled hybrid receipts did not satisfy the combined task and
   resource gates;
4. graph-centering plus finite truncation raises fragment/size-locality concerns;
5. integrating it now would retain `implicit_every`, scale, order, normalization,
   and workspace options in the canonical API.

If future multi-seed downstream evidence supports it, the preferred integration
is a small spatial condition or concatenated global feature map, not another
full-state residual pass.

## Why AttnRes is not canonical

Block Attention Residuals may help deep stacks, but the current canonical use
case does not require a depth-cache mechanism at every model size.

It remains experimental because:

- its strongest external motivation is deep language-model optimization;
- it adds a block-count axis and `O(LBN)` depth-routing work;
- EqRMSNorm, bounded residuals, per-copy LayerScale, and branch fusion already
  define the base stability contract;
- it should be evaluated only on a fixed deep-stack preset, not exposed as a
  routine architecture option.

## Why coordinate updates moved out

Coordinate mutation combines four separate policies:

- displacement head;
- update mask and centroid constraint;
- step-size control;
- neighbor topology reuse or rebuild.

Those policies do not belong in every static property layer. Canonical ELA keeps
coordinates read-only and uses `ELACoordinateRefiner` for explicit outer-loop
refinement.

## Public option reduction

Previous advanced configuration exposed architecture, training, geometry,
conditioning, and refinement controls together.

Canonical v1 exposes only:

```text
input_irreps
output_irreps
width
depth
geometry
```

Derived or fixed:

```text
num_heads
local_rank
hidden irreps
normalization
residual scale
tensor closure
chirality construction
```

Moved outside core:

```text
coordinate refinement
neighbor discovery/rebuild
task readout and masks
training dropout/DropPath
experimental implicit transport
experimental AttnRes
```

## Validation required for performance claims

### Mechanics

- branch fusion is exactly `G + L` at initialization;
- learned routing remains O(3)-equivariant;
- all sectors and router parameters receive finite gradients;
- full model passes translation, proper/improper rotation, permutation, graph
  isolation, and edge-order tests;
- input and coordinate double backward remain finite;
- CUDA BF16 forward/backward passes;
- external refiner is identity at initialization and bounded/equivariant when
  activated.

### Compatibility

- existing advanced and legacy imports remain valid;
- historical models retain their state schemas;
- canonical ELA receives a new schema rather than silently loading incompatible
  checkpoints;
- regression and task adapters use the canonical API explicitly.

### Resource

- parameter count and forward/backward latency are recorded versus the refined
  `EquivariantLinearAttention` control;
- branch router overhead is reported separately;
- peak memory does not include neighbor discovery unless stated;
- models are profiled one at a time;
- no speedup is claimed without a measured result.

### Downstream

At minimum:

- one smooth/general 3D task;
- one sharp interaction or molecular/protein task;
- paired seeds and frozen splits;
- no test-set use for model selection;
- task metric, latency, peak memory, clipping, and instability counts.

The gates were run in
`CANONICAL_BRANCH_FUSION_STUDY_20260731.md`. QM9 showed a one-seed
`0.023484 eV` validation-MAE improvement, but the resource packet failed and
trainable routing failed the LBA train-only capacity gate (`0.331238 pK` versus
the `0.10 pK` ceiling). Canonical ELA remains the supported design and API;
learned routing is not an admitted empirical advantage and no multi-seed
confirmation is authorized.

## Superseded selection policy

The following are no longer competing canonical arms:

```text
explicit
implicit
hybrid
implicit_every=N
AttnRes blocks=B
in-layer coordinate_updates=True/False
```

They remain experiment or compatibility labels only.
