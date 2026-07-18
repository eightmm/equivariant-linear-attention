# Evaluation

The public architecture remains one `EquivariantAttention` class. Routing,
global transport, and memory flags select registered arms of that class. The
runner also contains one explicitly private `internal_static_egnn_baseline`
selector for a same-data/training comparison; it is not exported or described
as an official-paper reproduction.

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

A 500-step run is a numerical screen. The transport confirmation requires
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
