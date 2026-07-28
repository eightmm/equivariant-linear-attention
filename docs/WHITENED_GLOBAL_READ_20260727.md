# Whitened global read: linear attention as ridge regression (2026-07-27)

## What was conceptually missing

This project's global path is an exact factorized kernel attention. Written with
the feature map that makes it linear, the incumbent read is

```text
o_i = phi_i^T S / phi_i^T m,     S = sum_j psi_j v_j^T,   m = sum_j psi_j
```

so the output is an *affine functional of the first moment* of the value
distribution. Two registered measurements say what that costs:

- the kernel is numerically uniform: on a cached QM9 graph, normalized entropy
  over `log N` was `0.999759`, mean maximum weight `1.05x` uniform, and the
  selectivity-bearing alignment and quadratic terms spanned `0.3%` of the kernel;
- the transport study found learned transport (`0.515688 eV`) barely beat an
  exact uniform mean broadcast (`0.534776 eV`), and both far beat no transport
  (`0.691821 eV`).

Every previous repair attacked the **weights**: differential attention, a
bounded output gain, a quartic angular map, rank-two query/key axes, persistent
`2e` kernels, a geometry-aware local score. All were rejected. Read together
they falsify one particular hypothesis — that the global path needs a *sharper
kernel* — and they leave a different axis untouched.

The untouched axis is the **metric of the read**. A positive kernel with a
constant floor puts almost all of its mass in one direction of feature space, so
pooling along that direction returns the graph mean no matter how the remaining
directions are shaped. The fix is not to sharpen `psi`; it is to stop reading in
an unwhitened metric.

## The mechanism

Replace the pooled read by the ridge-regression read

```text
o_i = phi_i^T (G + lambda I)^-1 S,
G   = (1/N_g) sum_j psi_j psi_j^T,
S   = (1/N_g) sum_j psi_j v_j^T,
lambda = ridge * tr(G) / F.
```

This is linear attention read as *kernel regression*: fit a linear map from key
features to values across the graph, then evaluate it at the query. The
equivalent attention matrix is `A = Phi (G + lambda I)^-1 Psi^T / N_g`, which is
never materialized.

Three properties follow:

1. **It divides out the dominant direction.** If keys are nearly constant,
   `G ~ N sigma^2 uu^T + lambda I`, so the `u` component is shrunk by
   `1/(N sigma^2 + lambda)` while the residual variation is shrunk only by
   `1/lambda`. The read becomes contrastive: it answers "what distinguishes this
   node from the graph's key covariance" instead of "what is the graph mean".
2. **Its large-shrinkage limit is the incumbent numerator.** As `ridge` grows,
   `(G + lambda I)^-1 -> I/lambda`, so the read becomes a scaled copy of the
   *unnormalized* kernel moment `phi_i^T S`. It is **not** a scaled copy of the
   incumbent read `phi_i^T S / phi_i^T m`; that denominator is query dependent
   and, measured on this packet's own float64 fixture, varies by a factor of
   `1.73--2.25` across nodes of a single graph, so no rescaling turns the limit
   into the incumbent function. An independent Codex review caught this packet
   overstating the relationship as "strictly generalizes"; the exact statement
   that does hold is the disabled and zero-initialized equivalence below.
3. **It stays linear in nodes.** Cost is `O(N F^2)` for the moments plus
   `O(G H F^3)` for the factorization, with `F = 26` for the current LBA
   candidate. No `N x N` tensor and no pair state.

`ridge` is dimensionless because the applied shrinkage is `ridge * tr(G)/F`. The
same value therefore means the same amount of whitening at any feature scale, and
`G + lambda I` is positive definite because the constant kernel block keeps the
trace strictly positive.

## The isometry requirement

The lane is exactly `O(3)` equivariant, but only in an isometric feature basis,
and that is a real constraint rather than a formality.

Under a rotation `R`, the key features transform by a block matrix
`M(R) = diag(I, 1, R, D(R))`, where `D(R)` represents `S -> R S R^T` on the
symmetric rank-2 block. Then `G -> M G M^T`, and

```text
phi^T (G + lambda I)^-1 psi
  -> phi^T M^T (M G M^T + lambda I)^-1 M psi
```

which equals the untransformed value **iff `M` is orthogonal**, because only then
does `(M G M^T + lambda I)^-1 = M (G + lambda I)^-1 M^T`. The incumbent
numerator compresses `(q.k)^2` asymmetrically, carrying off-diagonal products at
`1x` on one side and `2x` on the other. That pairs to the same kernel but is not
norm preserving, so it would break the whitened read. The lane therefore uses the
isometric basis `[x^2, y^2, z^2, sqrt2 xy, sqrt2 xz, sqrt2 yz]` on both sides,
which reproduces `(x.y)^2` exactly *and* makes `M(R)` orthogonal. `tr(G)` is
invariant under the same action, so the shrinkage normalization does not weaken
the contract. `tests/test_whitened_global_read.py` asserts both directions: the
isometric basis is rotation consistent and the compressed basis is not.

## Public configuration

```python
EquivariantAttentionConfig(
    ...,
    use_whitened_global_read=True,
    whitened_global_ridge=0.1,
)
```

- The lane is added to the incumbent global message through per-head
  zero-initialized `whitened_scalar_mix` and `whitened_vector_mix`, so an
  enabled model is the exact incumbent function at initialization while both
  mixes receive gradient. Only `2 * num_heads` parameters are added, and only on
  stages that have at least one global head.
- No RNG is consumed by the new parameters, so an enabled model shares every
  incumbent weight draw. Measured on the LBA candidate: `168,823` versus
  `168,815` parameters, identical shared state dict, and a bit-identical
  initial forward.
- Construction rejects combinations whose key feature map or metric the lane
  cannot reproduce exactly: key balancing, `kernel_floor_mode="inverse_graph_size"`,
  the multiscale spatial kernel, memory interaction, and non-learned transport.
- The lane rewrites only the `0e` scalar and `1o` vector value lanes. Relative
  position and `2e` moment reconstruction stay on the incumbent normalized read.

## Verification

`scripts/check.sh fast`: 670 tests, 88.88% coverage. `scripts/check.sh gpu`: ok.

`tests/test_whitened_global_read.py` (29 tests) covers the isometric pairing,
exact reproduction of the incumbent dense kernel by the feature map, exact
agreement with an explicit dense ridge reference at `atol=1e-10` in float64, the
large-shrinkage kernel-moment limit *and* its separation from the normalized
incumbent read, full `O(3)` including reflection, the
compressed-basis counterexample, permutation equivariance, graph isolation,
linearity in the values, degenerate and single-node graphs, the padded and
per-node reduction paths agreeing to `1e-12`, exact incumbent equivalence at
initialization, gradient flow, and every rejected configuration.

Strict deterministic CUDA was checked directly, since an unsupported
deterministic operator would be a terminal failure for this project's screens.
With `configure_reproducibility(mode="strict")`, six AdamW steps on 512 nodes
gave mix gradients growing `0 -> 4.8e-5 -> ... -> 6.8e-3` and a mix magnitude of
`1.97e-2`, and two runs agreed bitwise on the final loss, the whole gradient
sequence, and the mix magnitude. Step zero has zero gradient for every arm of
this project because the regression readout is zero initialized.

## Measured activity on real data

The mechanism's claim is falsifiable without any accuracy outcome: it must
measurably de-uniformize the read. On the first cached ATOM3D-LBA train complex
(331 nodes, real atom features, seed-41 candidate, head 0, `F = 26`,
`tr(G)/F = 0.0789`), reconstructing both dense row distributions gave

| read | row CV | max/mean | entropy / log N | negative weights |
| --- | ---: | ---: | ---: | ---: |
| incumbent kernel | 0.00298 | 1.0073 | 0.9999992 | 0 |
| whitened, ridge 1.0 | 0.0791 | 1.246 | 0.99947 | 0 |
| whitened, ridge 0.5 | 0.153 | 1.478 | 0.99801 | 0 |
| whitened, ridge 0.1 | 0.677 | 3.118 | 0.96429 | 0.066 |
| whitened, ridge 0.05 | 1.213 | 4.782 | 0.95166 | 0.221 |
| whitened, ridge 0.01 | 3.954 | 12.92 | 0.95358 | 0.401 |

Two things are worth stating plainly. First, the uniformity pathology is *worse*
on a 331-node biomolecular complex than on the 20-atom QM9 graph where it was
first measured, and the key Gram matrix has condition number `3.3e9`, which is
the quantitative reason a whitened read has something to work with. Second, the
knob behaves monotonically, so `ridge` is a genuine dial between the incumbent
limit and full whitening.

Evidence: `artifacts/whitened-global-read-20260727/probe-lba.json`, reproduced by
`scripts/probe_whitened_global_read.py`.

## Resource cost

On the real training path (256 train and 64 validation complexes, batch 16,
two epochs, 32 matched updates per arm, strict CUDA FP32):

| ratio versus candidate | value |
| --- | ---: |
| parameters | 1.0000474x |
| median synchronized train step | 1.1138x |
| peak CUDA allocation | 1.0179x |

Evidence: `artifacts/whitened-global-read-20260727/lba-training-smoke.json`.

The first implementation was more expensive. A CUDA operator profile attributed
its cost to per-node `(N, H, F, F)` and `(N, H, F, V)` moment materialization and
the scatter/gather backward around them, not to the factorization: `index_put`,
`indexing_backward_kernel`, and `index_add` grew by `1.0`, `0.9`, and `0.7 ms`.
Reducing per-graph moments as batched matrix products on a padded graph-major
layout removed those intermediates and cut peak allocation from `1.1416x` to
`1.0169x`. The padded layout declines itself under extreme graph-size skew and
falls back to the per-node reduction, and both paths are tested to agree.

One measurement disagreement is disclosed rather than resolved by choosing the
favorable number. The isolated profiler, which replays one identical 16-complex
batch back to back, reports `1.4234x` median step latency for the same code that
the training path measures at `1.1138x`. The padded layout needs one host read of
the maximum graph size per forward, which stalls CPU run-ahead; the training loop
already synchronizes each step, so the stall is hidden there and exposed in the
tight replay loop. The training-path number is the one the frozen resource gates
compare, but the profiler number is the honest upper bound for a pipeline that
never synchronizes.

## Claim boundaries

- **One-seed exploratory accuracy evidence only.** The seed-44 screen above is a
  rejection screen, not a confirmation. No default changed, no checkpoint is
  published, and no test label was read. Promotion requires the seeds 41--43
  packet with its own frozen thresholds.
- **This is not a probability-weighted attention.** `(G + lambda I)^-1` is
  indefinite in its action on the kernel, so the equivalent row weights are
  signed: `6.6%` are negative at `ridge=0.1` and `40%` at `ridge=0.01`. The read
  is bounded, since `||(G + lambda I)^-1|| <= 1/lambda`, and exactly `O(3)`
  equivariant, but it must not be described as a sparse or convex attention
  distribution.
- **The registered QM9 prior runs against this mechanism.** The differential
  attention packet concluded that for QM9 `gap` the useful content of the global
  path *is* the near-uniform graph mean, and that making the global message node
  dependent cost accuracy. This lane makes the global message node dependent. The
  reason to test it anyway is that the prior was established at `N ~ 18`, where
  the graph mean is a strong summary, while the deciding task has mean `~460` and
  maximum `~2000` nodes per complex and a `6 Angstrom` local cutoff, so
  mid-range structure currently has no pathway other than a near-uniform mean.
  That asymmetry is a prediction, not a result.
- Excluded from this packet: key balancing, inverse-graph-size baselines, the
  multiscale spatial kernel, memory interaction, a learnable ridge, whitening the
  relative-position and `2e` moment lanes, and any local-path whitening.

## Screen result (2026-07-28): the registered prediction was wrong

The seed-44 ridge screen ran to completion in 868.1 s of runner wall time, strict
deterministic FP32 CUDA, all 3,507 train and 466 validation complexes, one shared
candidate list at the frozen topology
`57f40fb1...` (32,302,952 edges), and exactly 4,400 updates per arm.

| arm | last RMSE | best RMSE (epoch) | last MAE | Pearson | delta last vs candidate |
| --- | ---: | ---: | ---: | ---: | ---: |
| candidate | 1.638053 | 1.626270 (11) | 1.322775 | 0.5773 | — |
| whitened, ridge 0.5 | 1.629871 | 1.624551 (17) | 1.314239 | 0.5827 | **+0.008183** |
| whitened, ridge 0.1 | **1.586308** | 1.586308 (20) | 1.284472 | 0.6110 | **+0.051745** |
| whitened, ridge 0.01 | 1.588391 | **1.576846** (15) | **1.265750** | 0.6109 | **+0.049663** |

All three ridges improved and all three passed the frozen screen rule, so
`ridge=0.1` is selected and advances to a seeds 41--43 confirmation. Resource
ratios were `1.138--1.144x` median step latency, `1.0171x` peak allocation, and
`1.0000474x` parameters, all inside the `1.25x` ceilings.

**The preregistered prediction was that no ridge would improve by more than
`0.020 pK`.** It is falsified: the selected arm improved `0.051745 pK`, about
2.5x the paired multi-seed effect that the current candidate itself earned over
the previous incumbent (`0.021099 pK`). A prediction failing in the favorable
direction is exactly when to tighten, not relax, the reading of the evidence.

Four things about this result should temper it:

- **One seed, and this project has been burned before.** Distance-spacing RBFs
  improved `0.029220 eV` at 500 QM9 updates and lost `0.011491 eV` across five
  seeds; the first LBA one-seed candidate gate passed by `0.041973 pK` while its
  paired bootstrap interval still crossed zero. Nothing here is confirmed.
- **The selected arm never peaked.** Its best epoch is the last one, so at this
  budget it was still improving while the candidate peaked at epoch 11 and then
  drifted. Part of the last-epoch gap is therefore a difference in *when* each
  arm degrades. The best-checkpoint comparison is the conservative one, and it
  still favors whitening by `0.039961 pK` at `ridge=0.1` and `0.049423 pK` at
  `ridge=0.01`.
- **The two strong ridges are not separable here.** `0.1` wins on last RMSE while
  `0.01` wins on best RMSE and MAE, and their Pearson correlations are equal to
  four digits. One seed cannot rank them; the confirmation packet should not
  silently treat `0.1` as established.
- **The lane was used, with a larger vector gate.** Best-checkpoint gate
  magnitudes were `0.031--0.053` scalar and `0.160--0.324` vector. This proves
  activity but not causal benefit. Equal clip fractions also do not establish
  equal optimization: pre-clip norms and effective clip scales differ.

The baseline is not comparable to earlier LBA numbers. The seed-44 candidate here
reached `1.638053` on the repaired topology, while the clipping packet's seed-44
clip-1 arm reached `1.628645` on the pre-repair candidate list. Those differ by
both topology and run, so the useful comparison is strictly within this screen.

Primary output:
`artifacts/whitened-ridge-screen-20260728/seed44/summary.json`, SHA-256
`c74bd7a42f10527b3cb35070791c268c431f11c7d3879cfe3e52ecdb1d6c6178`.

## Original preregistered screen

Thresholds fixed before any outcome is inspected. Following the CTP precedent,
QM9 is a bounded regression smoke and ATOM3D-LBA ID30 is the deciding task.

1. **QM9 safety smoke.** Seed 42, 500 updates, strict CUDA, arms: candidate and
   whitened at `ridge=0.1`. The lane must stay finite and within `0.020 eV` of
   the candidate. This is a safety check, not an admission gate; the registered
   prediction is that whitening does not help QM9 `gap`.
2. **LBA ridge screen.** Fixed seed 44, unused by the confirmation set, matched
   20 epochs, batch 16, one shared and hashed topology; arms: candidate and
   whitened at `ridge` in `{0.5, 0.1, 0.01}`. A ridge advances only if it
   improves last-epoch validation RMSE over the candidate with no regression
   larger than `0.05 pK` and within `1.25x` latency and peak allocation.
   Executed by `scripts/run_whitened_ridge_screen.py`, which trains all four
   arms in one process on one in-memory candidate list, aborts before training
   if the topology hash differs from the frozen identity, and records the
   best-checkpoint magnitude of each arm's zero-initialized lane gates so an
   inert lane cannot be mistaken for an unhelpful one.
3. **LBA confirmation.** Seeds 41--43, matched epochs, identical batches, one
   process and one topology. Promotion requires mean paired improvement at least
   `0.020 pK`, at least two of three paired wins, worst paired regression at most
   `0.050 pK`, and the same resource ceilings. Because the screen could not
   separate `ridge=0.1` from `ridge=0.01`, and because the selected arm had not
   peaked at 20 epochs, the confirmation should carry both ridges and report
   best-checkpoint as well as fixed-budget metrics.

Every arm consumes the frozen topology
`57f40fb157e6416558db5507d95c3a5e4f828881e0bc92e142e1b85de802dc6c`
(`docs/TOPOLOGY_CONTRACT_20260727.md`). Failures and nulls are recorded as
authoritative results.

The original step-3 design was superseded before execution after the QM9 run
showed that raw-candidate and whitened arms do not share the same numerical
training graph even at zero mix. The schema-matched repair and its completed
result are recorded below.

## Schema-matched safety repair and three-seed result (2026-07-28)

The registered follow-up found and repaired an evaluation confound before
interpreting the LBA confirmation.

### What failed first

The original two-arm QM9 safety smoke failed: ridge-0.1 whitening reached
`0.696559 eV` validation MAE versus `0.640651 eV` for the raw candidate, a
`+0.055908 eV` regression against the `0.020 eV` bound. Post-outcome ridge
diagnostics did not repair it: ridge `0.5/1.0` reached
`0.726088/0.686130 eV`.

A rank-based graph gate was then tested. The first form
`max(0, n-F)/n` remained unsafe (`0.687010 eV`). Tightening the covariance
aspect-ratio requirement to two samples per feature made the learned mixes stay
exactly zero on QM9, yet the result was still `0.720716 eV`. This exposed the
confound: zero mix preserves the mathematical forward function, but evaluating
the extra whitening branch changes the autograd graph and floating-point
gradient accumulation order. Under `~91%` QM9 clipping, those small changes
produce a different deterministic training trajectory. Therefore the old
raw-candidate comparison and its seed-44 LBA gain cannot isolate whitening
causally.

### Repair

Two changes make the comparison and small-graph behavior explicit:

1. The optional finite-sample reliability gate is
   `rho_g = max(0, n_g - 2F) / n_g`, where `F` is the actual kernel-feature
   dimension. It is invariant to node permutation and O(3), adds only graphwise
   scalar work, retains linear node complexity, and approaches one on large
   biomolecular graphs.
2. If every graph in a batch has `n_g <= 2F`, the model skips the whitening
   branch rather than computing a zero contribution. The small-graph path is
   therefore the incumbent computation, not merely an algebraically equal
   branch.

For scientific comparisons, the control now has the identical whitening
modules, full initial state, ridge, rank gate, forward graph, and optimizer
schema. Gradient hooks hold only the eight scalar/vector mix parameters at zero.
The active arm differs only by allowing those gradients to update.

The corrected strict-CUDA QM9 safety run gave both arms exactly
`0.6406513190 eV` MAE and `0.8343833076 eV` RMSE after 500 updates. Step latency
ratio was `1.013299`, peak allocation ratio `1.0`, both full final-state hashes
matched, and the test split was not evaluated.

### ATOM3D-LBA ID30 confirmation

The deciding run used seeds 41--43, 20 epochs and 4,400 updates per arm, batch
16, strict FP32 CUDA, one `32,302,952`-edge candidate list with frozen hash
`57f40fb1...`, and the official train/validation split (`3,507/466`). Each pair
started from the same full state.

| seed | frozen control last RMSE | active last RMSE | paired improvement | control best | active best |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 41 | 1.599205 | 1.586143 | +0.013061 | 1.595748 | 1.585772 |
| 42 | 1.602016 | 1.614437 | -0.012421 | 1.602016 | 1.613715 |
| 43 | 1.611601 | 1.615970 | -0.004369 | 1.597070 | 1.598024 |

Mean last-epoch improvement was `-0.001243 pK` with sample standard deviation
`0.013026 pK`; only one of three seeds improved. Mean best-checkpoint effect was
`-0.000892 pK`. Thus the registered `+0.020 pK`, two-win accuracy gate failed.
All completion, finiteness, full-initial-state, update-count, topology,
worst-regression, latency, memory, and no-test criteria passed. Active/control
maximum step-latency and peak-allocation ratios were `0.998024` and `1.0`.

Clipping remained `0.9827--0.9859` in every arm. Learned vector mix magnitude
was largest on seed 42 (`0.343637`), precisely the seed that regressed most.
Gate magnitude is therefore evidence that a lane is active, not evidence that
the lane caused an accuracy gain. The prior seed-44 wording that the gain came
mostly through the vector route is withdrawn.

The result rejects promotion. `use_whitened_global_read` and
`whitened_global_rank_gate` remain off by default. Because the LBA primary
failed, the preregistered clipping interaction and scalar-only/vector-only
ablations were not executed.

Primary artifacts:

- `artifacts/whitened-global-followup-20260728/qm9-rank-safety/summary.json`
  (`37bee694...`)
- `artifacts/whitened-global-followup-20260728/lba-rank-confirmation/summary.json`
  (`9ece52a5...`)
- `artifacts/whitened-global-followup-20260728/conditional-followups.json`
  (`dadd3fbd...`)

## Reproduction

```bash
uv run pytest tests/test_whitened_global_read.py -q

uv run python scripts/probe_whitened_global_read.py \
  artifacts/whitened-global-read-20260727/probe-lba.json \
  --dataset lba --complexes 2

uv run python -u scripts/profile_lba_train_step.py \
  artifacts/whitened-global-read-20260727/lba-profile.json \
  --device cuda --batch-size 16 --warmup 3 --repeats 1 \
  --timing-repeats 40 --model-seed 41 --arms candidate whitened

uv run python -u scripts/run_whitened_ridge_screen.py \
  artifacts/whitened-ridge-screen-20260728/seed44 --device cuda \
  --batch-size 16 --epochs 20 --warmup-epochs 5 \
  --model-seed 44 --order-seed 44 --budget-seconds 2400

uv run --locked python -u scripts/run_whitened_qm9_smoke.py \
  artifacts/whitened-global-followup-20260728/qm9-rank-safety \
  --device cuda --steps 500 --rank-gated-schema-control

uv run --locked python -u scripts/run_whitened_confirmation.py \
  artifacts/whitened-global-followup-20260728/lba-rank-confirmation \
  --device cuda --batch-size 16 --epochs 20 --warmup-epochs 5 \
  --rank-gated-schema-control \
  --qm9-safety-summary \
  artifacts/whitened-global-followup-20260728/qm9-rank-safety/summary.json
```
