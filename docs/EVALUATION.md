# Evaluation

The public architecture remains one `EquivariantAttention` class. Routing,
global transport, memory, and coordinate-update flags select registered arms of
that class. The runner also contains explicitly private static/dynamic EGNN
selectors for a same-data/training comparison; neither is exported or
described as an official-paper code reproduction.

```bash
uv run python scripts/train_compare.py \
  --dataset synthetic --steps 10 --split-seed 42 --model-seed 42 \
  --routing ggg
```

The script reports train loss, validation MAE/RMSE, split hashes, model seed,
parameter counts, source hash, optimizer settings, target normalization, the
full routing/kernel/memory run configuration, and whether test evaluation
occurred. Test evaluation is disabled by default and requires explicit
`--evaluate-test`; adaptive architecture work selects only on validation.

## Bounded-content and persistent-`2e` kernel outcome (2026-07-23)

Two exact-factorization-compatible options were added behind disabled-by-
default flags:

- `scalar_content_mode=bounded` retains a bounded norm signal after the
  positive feature map;
- `tensor_product_kernel=true` adds a nonnegative shifted Frobenius product of
  persistent symmetric-traceless `2e` query/key state.

The tensor term is appended as a finite feature map, so global transport stays
exactly factorized at fixed width without an `N x N` tensor. Focused algebra,
default-compatibility, O(3)/reflection, translation, permutation, batch-
isolation, and strict-CUDA finite-gradient checks pass.

The original tensor-enabled run was retained but not used for component
attribution after finding that optional module construction advanced the CPU
RNG and shifted common incumbent initialization. The repaired v2.1 source
constructs those modules inside a CPU RNG fork. Its common state entries match
the persistent-`2e` control bitwise under the registered construction path.
Original and repaired results are not pooled.

The repaired strict-CUDA QM9 `gap` screen used seed 42, 500 updates,
train/validation sizes 110,000/10,000, and no test evaluation:

| arm | validation MAE | delta vs incumbent |
|---|---:|---:|
| unit/no tensor-kernel incumbent | 0.709287 eV | 0 |
| bounded/no tensor kernel | 0.722743 eV | +0.013456 eV |
| unit + persistent `4x2e` tensor package | 0.811461 eV | +0.102174 eV |
| bounded + persistent `4x2e` tensor package | 0.716811 eV | +0.007524 eV |

No candidate met the frozen `0.010 eV` improvement rule, so the five-seed
confirmation and conditional private-EGNN run were not executed. The
four-arm screen evaluates packages rather than a clean tensor-only ablation;
no tensor-component benefit is claimed.

On frozen ATOM3D-LBA train rows 0--15, the repaired combined candidate,
incumbent, and near-parameter private static EGNN all missed the registered
`train MAE <= 0.10 pK` capacity threshold within 3,000 updates. Their best
observed train MAEs were `0.184002`, `0.143626`, and `0.191400 pK`,
respectively. The candidate used 336.3 MiB peak CUDA versus EGNN's 743.0 MiB,
but its median measured train step was 42.24 ms versus 5.31 ms and it also
lost to the incumbent on accuracy, time, and memory. Validation and test were
not accessed, so this provides no affinity-generalization claim.

The implementation is retained as an experimental opt-in mathematical
capability; defaults and performance claims are unchanged. The complete
source hashes, raw results, failure record, and decision are in
`artifacts/architecture-v2-positive-tensor-20260723/`.

## Registered ATOM3D-LBA overfit outcome (2026-07-23)

The train-only capacity packet froze the first 16 rows of the pinned
ATOM3D-LBA train Parquet, retained pocket plus ligand nodes, and pooled only
ligand nodes. Both FP32 arms used identical features, targets, cyclic batches
of two, AdamW at `1e-3`, zero weight decay, clipping at 1.0, and at most 3,000
updates. Attention used no edge tensor; the private static EGNN used 382,530
directed 6-Angstrom candidates including the self candidates removed inside
the baseline.

| arm | parameters | best train MAE | final train MAE | CPU median step | elapsed |
|---|---:|---:|---:|---:|---:|
| edge-free GGG + persistent `4x2e` | 167,115 | 0.151798 pK @ 2,950 | 0.199863 pK | 0.117964 s | 373.890 s |
| private static EGNN, width 92 | 167,260 | 0.163536 pK @ 2,850 | 0.163732 pK | 0.252309 s | 823.408 s |

These medians come from one sequential CPU run after excluding the first ten
train steps. The timed interval covers `train_regression_step` only; radius
construction, batch-index selection/collation, and periodic full-train
evaluation are outside it. It is not a repeated end-to-end benchmark.

Neither CPU arm reached the `0.10 pK` threshold. The CPU fallback
instantiation is therefore rejected, but it cannot formally substitute for the
frozen CUDA protocol: the exact preregistered C3 remains not verified.
Attention's lower best observed MAE and `2.139x` CPU step-rate ratio are
descriptive post-outcome observations, not promotion criteria. The late-step
curves oscillated, making a separately registered learning-rate-decay
diagnostic the smallest next experiment; changing the optimizer after seeing
this result would not repair the CPU fallback.

The registered CUDA run was not executed: CUDA was unavailable inside the
sandbox and external GPU execution was rejected by the current Codex usage
limit. Consequently CUDA C2 execution, C3 overfit, and C4 speed/peak-memory
evidence remain not verified. Because C4's frozen falsifier includes
unavailable or non-comparable measurements, C4 is also contractually
unfulfilled/rejected as registered; this permits no performance conclusion.
`validation_evaluated=false` and `test_evaluated=false` in both result arms.

## Registered comparison sequence

All adaptive QM9 arms use target `gap` in eV, random-row split seed 42, FP32,
batch size 64, three blocks, and matched optimizer/schedule settings. Stages
1--3 use model seeds 41/42/43. Stage 3a and the subsequent EGNN comparison use
seeds 41--45; attention width 64 is matched by EGNN width 91.

0. Before any interacting M=4/M=8 training arm, run the fixed activation gate:

   ```bash
   uv run python scripts/probe_memory_activation.py \
     --memory-counts 4 8 --device cpu --dtype float64
   ```

   It uses one fixed 16-node graph, seed 401, an identical state dictionary,
   and no labels. Every head must pass the frozen assignment, occupancy,
   coupling, pair-gate, and full-output thresholds. The current M=4/8
   implementation fails because `C` and `G` are numerically constant; step 4
   is blocked until a separately preregistered redesign passes Stage 0.

1. On `ggg`, compare the 2x2 alignment-term/key-balancing arms. Turning off
   alignment removes only `beta * (q dot k)` and retains the `beta` constant.
2. With the selected normalization, compare `fixed` with global row-only
   `inverse_graph_size`, which scales `(c + beta + delta*beta*t)` by `1/N_g`
   while leaving content and `gamma*t^2` unchanged. It requires balancing off.
3. With memory interaction and radial trace off, compare `ggg` with `lgl`.
   `lll` is a mechanistic local-only control, not a performance candidate.
4. Decompose the route and global mechanism. A seed-42/500-step numerical
   screen runs `ggg/lgg/ggl/lgl` with learned transport and `lgl` with exact
   uniform and no global transport. The confirmation runs
   `lgl learned/uniform/none` at seeds 41--45 and 2,000 steps. This distinguishes
   learned query/key selectivity from global moment pooling and from local/FFN
   effects. Local diagnostics report receiver degree, entropy over log degree,
   effective support, maximum weight, and cutoff-distance quantiles for every
   active local layer/head over at most 32 deterministically selected
   validation graphs.
5. Only after transport is locked, compare the selected attention arm with the
   private width-91, three-layer static EGNN at the same seeds/update budget.
   It keeps PyG features, split, train-only target normalization, MSE, AdamW,
   cyclic batches, and LayerNorm-node-linear-graph-mean readout fixed. Its
   complete directed edges exclude self edges and use raw squared distance.
5a. The independently approved coordinate study first screens static/dynamic
   `ggg`, static/dynamic `lgl`, and static/dynamic private EGNN at seed 42 for
   500 updates. It then confirms one admitted attention route and EGNN with
   static/dynamic pairs at seeds 41--45 for 2,000 updates. Every dynamic run
   must report active nonzero coordinate gradients, per-layer steps at most
   0.25 Angstrom, and graph-centroid drift below `1e-6` Angstrom. Test labels
   remain disabled and the entire packet has a 1,500 GPU-second ceiling.
6. On `lgl`, compare interacting `M=1,4,8`; `M=1` is the exact incumbent
   reduction. Memory count with interaction off is not an expressivity arm.
   The current Stage-0 result blocks M=4/M=8, so this step cannot run without a
   separately preregistered mechanism repair.
7. Change only radial trace off/on for the selected routing and memory setting.

The non-adaptive Stage-2 size probe is reproducible without a dataset:

```bash
uv run python scripts/probe_kernel_scaling.py \
  --sizes 16 32 64 128 512 2048 --device cpu
```

It compares `fixed` and corrected `inverse_graph_size` with key balancing off
and reports JSON-safe mean row maximum weight, entropy/log-N, synchronized
runtime, `beta`/`gamma` and value gradient norms, and a small output-probe norm.
Exact row statistics are streamed in query blocks and the differentiable probe
uses only a fixed number of query rows. The script therefore retains no full
attention matrix and deliberately does not compute effective rank.

Inspect the exact dynamic-coordinate packet without running it:

```bash
uv run --locked python scripts/run_registered_coordinate_study.py \
  artifacts/coordinate-study-reproduction --dry-run
```

A 500-step run is a numerical screen. The coordinate confirmation requires
paired mean validation MAE improvement of at least 0.01 eV, improvement in at
least three of five seeds, worst-seed regression no larger than 0.02 eV, and
median latency and peak-memory increases no larger than 20%. Earlier registered
three-seed decisions retain their original two-of-three rule. A joint-only gain
is recorded as an interaction, not attributed independently to either
mechanism.

## Registered Stage-3a outcome (2026-07-19)

The validation-only runner completed all 6 seed-42/500-step screen arms and all
15 `lgl learned/uniform/none` seed-41--45/2,000-step confirmation arms in 819.2
GPU-wall seconds. Data/source/split/state-schema and matched initialization
hashes passed, all values were finite, and `test_evaluated=false` for every
run.

Mean validation MAE was 0.515688 eV for learned, 0.534776 eV for uniform, and
0.691821 eV for none. Learned beat uniform by 0.019088 eV on average in four of
five seeds, but its median peak-memory ratio was 1.359, above the 1.20 ceiling.
Learned and uniform each beat none on all five seeds by 0.176133 and 0.157044
eV on average, respectively, but learned exceeded both resource ceilings
(elapsed 1.512, memory 1.414) and uniform exceeded the elapsed ceiling (1.288).

The complete promotion rule therefore failed. This is evidence of validation
accuracy benefit at this update budget, but not an admitted accuracy/efficiency
mechanism lock. The public `ggg learned` default remains unchanged and the
conditional private-EGNN comparison was not run.

## Registered dynamic-coordinate outcome (2026-07-19)

The independent validation-only packet completed six seed-42/500-step screen
arms and twenty seed-41--45/2,000-step confirmation arms in 944.3 GPU-wall
seconds, below the frozen 1,500-second ceiling. The screen admitted `ggg`
attention and private dynamic EGNN. Source/data/split/state and paired-base
initialization hashes validated, all values were finite, and every run recorded
`test_evaluated=false`.

| seed | attention static | attention dynamic | improvement | EGNN static | EGNN dynamic | improvement |
|---:|---:|---:|---:|---:|---:|---:|
| 41 | 0.512575 | 0.510856 | 0.001718 | 0.429862 | 0.395129 | 0.034733 |
| 42 | 0.586021 | 0.605571 | -0.019550 | 0.431229 | 0.417954 | 0.013275 |
| 43 | 0.623774 | 0.629016 | -0.005241 | 0.421278 | 0.372225 | 0.049053 |
| 44 | 0.598871 | 0.589326 | 0.009545 | 0.396939 | 0.449499 | -0.052560 |
| 45 | 0.593489 | 0.592905 | 0.000584 | 0.365354 | 0.417335 | -0.051981 |
| mean | 0.582946 | 0.585535 | -0.002589 | 0.408932 | 0.410428 | -0.001496 |

Positive improvement favors dynamic coordinates. Attention improved in three
of five seeds and passed the worst-seed, elapsed (`1.179x`), and peak-memory
(`1.010x`) gates, but failed the required `0.010 eV` mean gain. EGNN also
improved in three seeds, but failed mean gain, worst regression (`-0.052560
eV`), and elapsed (`1.456x`) gates; its peak-memory ratio was `1.008x`.
Neither coordinate path is promoted.

All ten dynamic confirmation arms had active coordinate displacements and
nonzero coordinate-parameter gradients. Across them, maximum per-layer step
was `0.25000003 Angstrom` (float32 tolerance), maximum graph-centroid drift was
`4.92e-7 Angstrom`, and displacement RMS ranged from `0.0640` to `0.2943
Angstrom`. These validate the registered numerical contract only; they do not
establish relaxed physical geometries, forces, dynamics, or energy
conservation.

## Registered EGNN-parity study (confirmed 2026-07-20)

The next scientific question is whether the attention gap comes primarily from
its local pairwise representation rather than coordinate motion. The user
confirmed this packet for at most three architecture iterations and 3,600
cumulative GPU-seconds on one local GPU. Test evaluation remains disabled.

- Observation: private static EGNN reaches 0.408932 eV five-seed mean
  validation MAE, compared with 0.582946 eV for static GGG and the historical
  0.515688 eV for static LGL learned transport at 2,000 updates.
- Hypothesis: adding receiver/sender/distance-conditioned invariant local edge
  content and exposing neighborhood degree or attention mass to the scalar
  updater will reduce LGL validation MAE by at least 0.050 eV. A learned radial
  gate alone may help but is not expected to close the full 0.106756 eV LGL-to-
  EGNN descriptive gap.
- Primary baseline: private static EGNN width 91. Attention baseline: static LGL
  width 64, learned global transport, coordinate updates off.
- Changed variables, in order: (A) enable the already allocated learned local
  radial gate; (B) add a parameter-bounded local edge-content MLP over receiver,
  sender, and RBF distance features plus explicit degree/mass invariants; (C)
  select one topology or optimization repair only after A--B diagnostics. Do
  not combine unrelated repairs.
- Optimization diagnostic: record a fixed train-probe MAE, validation MAE,
  pre-clip gradient norm and clip fraction, and residual-scale trajectories at
  fixed intervals. Only then consider an isolated residual-scale, learning-rate,
  or gradient-clip change.
- Development screen: seed 42 and at most 500 updates may reject nonfinite,
  inactive, or clearly regressing arms; it cannot promote an architecture.
- Confirmation: matched seeds 41--45 and 2,000 updates, unchanged QM9 data and
  random-row warm split, train-only target normalization, FP32, batch size 64,
  and test evaluation disabled.
- Parity threshold: candidate mean validation MAE at most 0.408932 eV. Promotion
  threshold: mean at most 0.398932 eV, at least three of five paired seeds beat
  static EGNN, and worst paired regression no larger than 0.020 eV.
- Efficiency claim: report parameters, synchronized elapsed, and peak memory.
  Practical promotion additionally requires either a comparable QM9 resource
  envelope or a preregistered large-graph crossover where sparse-local plus
  factorized-global execution offsets the small-graph overhead.
- Abandonment rule: if the pairwise edge-content branch fails to improve mean
  validation MAE by 0.050 eV, do not spend a new long-run budget on coordinate
  updates, multi-memory, or simple scaling as substitutes for the failed local
  representation hypothesis.
- Packet stop: stop on the first robust promotion pass, after three architecture
  iterations, or at 3,600 cumulative GPU-seconds, whichever comes first. A
  failed source/data/split/equivariance/precision check stops the packet rather
  than substituting an unregistered arm.

The executable iteration runner freezes the data, split, update count, static
coordinates, candidate switches, private-EGNN control, provenance validation,
and cumulative packet timer. Inspect iteration A without spending GPU time:

```bash
uv run --locked python scripts/run_registered_egnn_parity_iteration.py \
  artifacts/egnn-parity-20260720/iteration-1-radial \
  --candidate radial --iteration 1 --dry-run
```

Each completed update contributes to a run-wide pre-clip gradient-norm summary
and clip fraction. Post-training metrics also include validation and the fixed
first 256 rows of the frozen train split. Candidate-specific gradient counters
must show an active learned radial or pairwise path before confirmation is
admitted. The runner writes progress after every arm so a failed or interrupted
iteration still consumes its measured share of the 3,600-second packet.

After iterations A--B, the evidence-selected third repair is staged pairwise
initialization: change only the pairwise residual scale from `0.1` to `0.0`.
Iteration B activated all 1,105 pairwise parameters but worsened both the fixed
train probe and validation while changing the 500-step clip fraction only from
`91.2%` to `92.2%`. This points to immediate residual injection rather than an
inactive branch or a uniquely binding clip threshold. Complete-local topology
was therefore not selected within this packet.

### Outcome (completed 2026-07-20)

All three permitted iterations completed in 850.7 cumulative GPU-wall seconds,
well below the time ceiling but exactly at the architecture-iteration ceiling.
No test metric was evaluated.

| iteration | seed-42/500 baseline | candidate | candidate minus baseline | disposition |
|---|---:|---:|---:|---|
| learned radial gate | 0.778593 | 0.759655 | -0.018938 | confirm |
| pairwise, `alpha=0.1` | 0.767847 | 0.840664 | +0.072816 | reject |
| pairwise, `alpha=0` | 0.738009 | 0.712453 | -0.025556 | confirm |

The radial confirmation averaged 0.499508 eV against rerun static EGNN at
0.421199 eV. The staged-pairwise confirmation averaged 0.509008 eV against its
rerun EGNN at 0.438268 eV. Both candidates lost all five paired seeds and had
worst paired improvements of -0.118947 and -0.134977 eV, respectively. Thus
every frozen promotion criterion failed and all public defaults remain off.

The staged branch was active, not dead: all 1,105 pairwise parameters had
nonzero finite final gradients, and its learned `alpha` ended at
`[-0.1055, 0.0888, -0.1393, -0.0956, -0.0945]` across seeds. The sign change
and wide validation spread show that this additive repair is seed-unstable at
2,000 updates. Gradient clipping affected roughly 97.3% of candidate updates,
but the private EGNN was also heavily clipped, so this packet does not identify
the clip threshold as the causal gap.

One reproducibility warning is now explicit: the iteration-1 and iteration-2
screen baselines had identical source, data/split, command semantics, and
initial-state hashes, yet ended at 0.778593 and 0.767847 eV. The seeded CUDA
lane is therefore not bitwise deterministic. Future accuracy work should first
add a deterministic/repeated-run gate before interpreting effects near 0.01 eV.

This remains an adaptive random-row interpolation study. Even a pass is only a
same-harness validation result against a private EGNN-style control; an official
baseline reproduction, similarity/scaffold split, and untouched final test are
separate gates.

## Scaling-aware EC-LGL study (completed 2026-07-22)

The confirmed packet required three distinct conclusions rather than one
aggregate speed claim:

1. exact dense and factorized implementations of one finite kernel must agree
   below `1e-10` and demonstrate the storage/runtime boundary;
2. full EC-LGL and static EGNN must be compared both on the same precomputed
   sparse edges and in the intended capped-local versus complete-edge regime;
   timed model forwards use already content-validated edges, while validation
   and neighbor construction are reported outside the `O(E_local + N)` claim;
3. a repeated seed-42/500-step EC-LGL screen may enter five-seed confirmation
   only when its mean is no more than `0.020 eV` worse than static LGL.

The exact kernel passed with `2.406e-15` maximum error and crossed the dense
implementation at 4096 nodes. Fixed-degree EC-LGL crossed complete-edge EGNN at
512 nodes but never beat EGNN on the same sparse edges. The accuracy screen
failed: EC-LGL/static-LGL repeated means were `0.802194/0.712178 eV`. Therefore
the 15 confirmation arms were not run, no test labels were accessed, and the
EC feature remains opt-in. A future accuracy packet should preregister one
bounded message normalization or staged residual intervention and retain the
repeated-run gate; these results do not authorize selecting among several
repairs after inspecting validation labels.

## EC-LGL receiver-degree normalization (completed 2026-07-23)

The opt-in intervention divides every edge-conditioned non-self local message
sum by the square root of the receiver's incoming candidate count. It adds no
parameter and leaves the unnormalized sum as the default. Float64 reference,
zero-degree, disabled-state identity, O(3), translation, permutation,
edge-order, graph-isolation, finite-gradient, and CLI provenance tests pass.

The strict-CUDA paired seed-42 screen held source, initialization, state schema,
data, split, precomputed radius candidates, optimizer, and evaluation fixed.
Clipping changed from `460/500` (`0.920`) to `458/500` (`0.916`), failing the
preregistered `0.05` absolute-reduction gate. Mean and maximum pre-clip norms
also increased from `6.154/44.101` to `6.726/53.507`, so simple receiver-degree
normalization is not a sufficient clipping repair.

Validation MAE improved from `0.744964` to `0.715997 eV` (`-0.028967 eV`).
This is a descriptive one-seed result on the adaptively reused random-row
validation split. Because the primary clipping gate failed, it does not
authorize a default change or accuracy claim. A later multi-seed experiment
requires a separately frozen contract. No test labels or EGNN comparison were
used in this packet.

Record common initialization hashes, total and nonzero-gradient parameters,
synchronized latency, peak CUDA memory, node-count strata, bounded kernel
scales, mass/denominator quantiles, entropy over log node count, maximum weight,
effective support, column CV, gradient/residual norms, and per-graph/per-head
HEMM occupancy, assignment marginal/conditional entropy and normalized mutual
information, center spread, off-diagonal center distances, coupling, and
effective pair-gate min/p01/median/p99/max, CV, centered-Frobenius ratio, and
tolerance-defined nonconstant fraction. The Stage-0 probe also records
middle-message, post-middle-state, scalar/vector/position-gradient, and
full-output symmetric relative RMS. Effective-rank computation is opt-in and
size-bounded; row or column scaling does not change exact matrix rank.

QM9 numbers use a random-row warm split. They do not measure scaffold,
protein-target, temporal, or cold-complex generalization. Historical test access
also means the split is not a pristine confirmatory holdout. A five-seed,
10,000-step test comparison is a separate approval gate after architecture
lock. No non-default route, memory count, radial trace, or floor mode is promoted
without the registered experiment evidence.
