# Explicit, implicit, and hybrid spatial-operator comparison

This document is the canonical execution and reporting protocol for deciding how
the edge-free spatial kernel should be used in equivariant linear attention.

## 1. Compared arms

| Arm | Exact global ELA | Explicit sparse local | Implicit residual |
|---|---:|---:|---:|
| `explicit` | yes | yes | no |
| `implicit` | yes | no | yes |
| `hybrid` | yes | yes | yes |

All arms instantiate the same module tree and load the same initial state dict.
The arm changes execution only. Parameter count, initialization, hidden/output
irreps, optimizer, and compute budget are paired.

The comparison model intentionally disables coordinate updates so every arm sees
one frozen geometry. Coordinate-refinement comparison is a later task-specific
stage.

### Timing boundary

The implicit arm runs the common backbone with a precomputed empty explicit
CSR graph. This removes all edge-dependent sparse work while retaining the same
module schema and some dormant node-side local projections. Its timing is
therefore a **conservative schema-matched implicit timing**, not the final speed
of a purpose-built implicit-only model.

## 2. What the protocol can establish

It can test:

- sharp short-range and edge-axis fidelity;
- smooth long-range spatial modeling;
- value of explicit-short-range plus implicit-long-range hybridization;
- optimization stability, clipping, latency, and peak memory;
- exact initial state and parameter-schema matching;
- implicit independence from explicit edge metadata;
- measured scaling against the documented complexity contracts;
- fragment and graph-size locality risk.

It cannot establish downstream superiority from synthetic data alone. A
synthetic pass only advances an arm to real-task validation.

## 3. Audit invariants

For every task and seed, the runner records and validates:

1. identical train and validation data hashes across arms;
2. identical initial state-dict hash across arms;
3. identical parameter count and parameter schema;
4. identical optimizer, learning rate, step budget, clipping threshold, and
   evaluation schedule;
5. validation labels are not used for training;
6. no-edge graph metadata is prepared outside the timed implicit forward;
7. zero-initialized hybrid output equals explicit output before training;
8. implicit output is unchanged when explicit edge metadata is removed, while
   graph membership is held fixed;
9. exactly three arms and exactly one audit exist for every declared task/seed
   pair.

Default numerical audit tolerance:

```text
max absolute error <= 1e-7
```

Use tighter FP64 tolerances for double-precision audits.

## 4. Synthetic tasks

Every graph contains scalar features `a_i`, one polar input vector `v_i`, and
positions `x_i`.

### 4.1 Local directional target

For one unordered pair `i<j`,

\[
r_{ij}=\lVert x_j-x_i\rVert,
\qquad
\widehat r_{ij}=\frac{x_j-x_i}{r_{ij}},
\]

and `f_c` is the compact C2 cutoff. The target is

\[
y_{\rm local}
=
\frac1N\sum_{i<j}
f_c(r_{ij}/R_c)
\left[
 a_{i0}a_{j0}
 +\frac14a_{i1}a_{j1}
 +\frac12
 (v_i^T\widehat r_{ij})
 (v_j^T\widehat r_{ij})
\right].
\]

This probes hard locality and edge-axis anisotropy. `explicit` is the reference.

### 4.2 Smooth Gaussian target

\[
y_{\rm smooth}
=
\frac1N\sum_{i<j}
\exp\left[-\frac{r_{ij}^2}{2\sigma^2}\right]
\left[
 a_{i0}a_{j0}
 +\frac14a_{i1}a_{j1}
 +\frac15v_i^Tv_j
\right].
\]

This probes a graph-global smooth field and is the favorable lane for the
implicit kernel.

### 4.3 Mixed target

\[
y_{\rm mixed}=y_{\rm local}+\frac12y_{\rm smooth}.
\]

This probes whether hybridization preserves local fidelity while adding smooth
long-range information.

Targets are normalized with train statistics only.

## 5. One-command workflow

Smoke:

```bash
SPATIAL_SUITE_MODE=smoke \
  bash scripts/run_spatial_operator_suite.sh \
  artifacts/spatial-operator-comparison/smoke
```

Full CUDA evaluation:

```bash
SPATIAL_SUITE_MODE=full \
SPATIAL_SUITE_DEVICE=cuda \
SPATIAL_SUITE_DTYPE=bfloat16 \
  bash scripts/run_spatial_operator_suite.sh \
  artifacts/spatial-operator-comparison/full-$(date +%Y%m%d-%H%M%S)
```

See `SPATIAL_OPERATOR_SUITE.md` for environment overrides and artifact details.

## 6. Direct comparison command

Minimal CPU smoke:

```bash
uv run python scripts/compare_spatial_operators.py \
  --tasks local_directional,smooth_gaussian,mixed \
  --seeds 0 \
  --train-graphs 8 \
  --validation-graphs 4 \
  --nodes-per-graph 8 \
  --hidden-dim 16 \
  --layers 1 \
  --heads 4 \
  --local-rank 2 \
  --candidate-skin 0 \
  --steps 5 \
  --evaluation-interval 1 \
  --profile-warmup 1 \
  --profile-repeats 2 \
  --device cpu \
  --dtype float64 \
  --output artifacts/spatial-operator-comparison/smoke/result.json
```

Recommended paired CUDA screen:

```bash
RUN=artifacts/spatial-operator-comparison/$(date +%Y%m%d-%H%M%S)
mkdir -p "$RUN"

uv run python scripts/compare_spatial_operators.py \
  --tasks local_directional,smooth_gaussian,mixed \
  --seeds 0,1,2,3,4 \
  --train-graphs 64 \
  --validation-graphs 32 \
  --nodes-per-graph 24 \
  --hidden-dim 64 \
  --layers 4 \
  --heads 4 \
  --local-rank 4 \
  --cutoff 1.75 \
  --candidate-skin 0 \
  --gaussian-scale 2.5 \
  --implicit-scales 2,4,8 \
  --implicit-scale-init 0 \
  --steps 1000 \
  --evaluation-interval 50 \
  --profile-warmup 10 \
  --profile-repeats 30 \
  --device cuda \
  --dtype bfloat16 \
  --output "$RUN/result.json"
```

`candidate_skin=0` is required for the canonical operator attribution: the
explicit model cutoff exactly equals the local-target cutoff. Wider candidate
skins belong in a separate neighbor-list robustness experiment.

The default implicit scales are intentionally larger than the explicit cutoff,
so the hybrid arm evaluates a longer-range complement instead of duplicating the
short-range lane.

## 7. Result validation and report

```bash
uv run python scripts/validate_spatial_operator_result.py \
  "$RUN/result.json"

uv run python scripts/report_spatial_operator_comparison.py \
  "$RUN/result.json" \
  --output "$RUN/report.md" \
  --decision-json "$RUN/decision.json"
```

The report contains:

- protocol violations;
- zero-init and edge-independence audits;
- per-task accuracy and resources;
- paired seed differences versus explicit;
- synthetic gate checks;
- required downstream follow-up.

The result bundle follows
`schemas/spatial_operator_comparison.schema.json`.

## 8. Default synthetic gates

### Hybrid candidate

```text
minimum paired seeds          3
local MAE regression         <= 2%
smooth MAE improvement       >= 5%
mixed MAE improvement        >= 2%
train-time overhead          <= 25%
inference-time overhead      <= 25%
training-memory overhead     <= 25%
```

### Implicit replacement candidate

```text
minimum paired seeds          3
local MAE regression         <= 2%
smooth MAE improvement       >= 5%
inference time               <= explicit
training memory              <= explicit
```

These are screening thresholds, not universal scientific constants. Override
them explicitly when the intended workload has a different resource budget.

## 9. Kernel approximation diagnostic

```bash
uv run python scripts/evaluate_implicit_spatial.py \
  --nodes 512 \
  --value-width 32 \
  --scales 2,4,8 \
  --cutoff 1.75 \
  --topk 32 \
  --output "$RUN/implicit-accuracy.json"
```

This records exact-Gaussian and compact-cutoff kernel/message relative errors and
top-k overlap. A poor cutoff approximation does not automatically reject the
hybrid path; it confirms that the implicit operator is acting as a smooth
long-range complement rather than a hard local replacement.

## 10. Fragment and size-locality diagnostic

```bash
uv run python scripts/evaluate_fragment_locality.py \
  --base-nodes 64 \
  --fragment-nodes 16 \
  --fragment-distance 20 \
  --value-width 16 \
  --cutoff 1.75 \
  --scales 2,4,8 \
  --output "$RUN/fragment-locality.json"
```

This separates:

- implicit original-pair kernel drift caused by graph centering and truncation;
- zero-value fragment message drift;
- random-value long-range coupling;
- explicit compact-cutoff drift, which should be zero beyond the cutoff.

See `FRAGMENT_LOCALITY.md`. For molecular and extensive/conservative tasks,
repeat over several fragment sizes and distances.

## 11. Scaling diagnostic

```bash
uv run python scripts/benchmark_scaling.py \
  --modes base,attnres,implicit \
  --nodes 256,512,1024,2048,4096,8192 \
  --graphs 1,8,32 \
  --depths 4,8,16,32 \
  --blocks 4,8 \
  --degrees 8,16,32,64 \
  --warmup 10 \
  --repeats 30 \
  --device cuda \
  --dtype bfloat16 \
  --output "$RUN/scaling.json"
```

Interpret slopes under `SCALING.md`:

\[
T_{\rm explicit}=O(L(N+E)),
\]

\[
T_{\rm AttnRes}=O(L(N+E)+LBN),
\]

\[
T_{\rm implicit}=O(ANFD).
\]

A measured slope is device- and regime-specific evidence, not an unconditional
runtime proof.

## 12. Correctness gates

```bash
uv run pytest \
  tests/test_spatial_ablation.py \
  tests/test_spatial_benchmarks.py \
  tests/test_spatial_comparison.py \
  tests/test_fragment_locality.py \
  tests/test_implicit_spatial.py \
  tests/test_implicit_spatial_chunks.py \
  tests/test_implicit_spatial_gradients.py \
  tests/test_implicit_spatial_permutation.py \
  tests/test_implicit_spatial_residual.py \
  tests/test_implicit_spatial_validation.py \
  tests/test_scaling_contract.py \
  tests/test_scaling_memory.py
```

CUDA:

```bash
uv run pytest tests/test_implicit_spatial_cuda.py
```

Repository gates:

```bash
scripts/check.sh fast
scripts/check.sh gpu
```

## 13. Artifact layout

```text
artifacts/spatial-operator-comparison/<run-id>/
  result.json
  report.md
  decision.json
  implicit-accuracy.json
  fragment-locality.json
  scaling.json
  environment.txt
  git.txt
  manifest.json
  *.log
```

Record Python, dependencies, GPU, git SHA, worktree state, and exact command.
Large raw traces should remain outside `main`; commit compact summaries, hashes,
commands, and a manifest.

## 14. Downstream validation matrix

A synthetic pass must be followed by both lanes.

### General 3D lane

Choose at least one:

- point-cloud classification or segmentation;
- registration/deformation;
- particle or field prediction;
- mesh property prediction.

### Molecular/protein lane

Choose at least one:

- scalar molecular property with a frozen split;
- force/energy prediction with derivative and fragment tests;
- protein--ligand affinity with protein-cluster or target-disjoint validation;
- pose or coordinate refinement with geometry metrics.

For each lane retain:

- data revision and split hashes;
- sample/node/edge counts;
- initial state and parameter hashes;
- task metric and paired seed differences;
- train/inference latency and peak memory;
- clipping and numerical-instability counts;
- neighbor discovery/rebuild time where applicable.

## 15. Promotion policy

### Keep explicit canonical

This is the default unless another arm demonstrates stable value. Explicit local
is the reference for sharp directional contact and typed relations.

### Advance hybrid

Allowed only after the synthetic hybrid gate passes. Canonical promotion still
requires one general 3D and one molecular/protein validation with acceptable
overhead and stable paired-seed improvement.

If hybrid repeatedly helps, the cleaner final implementation is likely to
concatenate the positive implicit spatial features into the global linear-
attention feature map, avoiding a second full value-transport pass. That change
requires mathematical equivalence and performance validation.

### Replace explicit local

Requires the strongest evidence: local directional fidelity, force/contact and
stereochemical tasks, fragment/additivity tests, resource benefit, and both
real-task lanes. Until then, implicit-only remains experimental.

## 16. Handoff checklist

- [ ] focused CPU tests pass;
- [ ] CUDA BF16 test passes;
- [ ] repository fast/GPU gates pass;
- [ ] protocol validator passes;
- [ ] explicit/hybrid zero-init audit passes;
- [ ] implicit edge-independence audit passes;
- [ ] at least three paired seeds per task;
- [ ] approximation diagnostic attached;
- [ ] fragment-locality diagnostic attached;
- [ ] scaling result and slopes attached;
- [ ] report and decision JSON generated;
- [ ] downstream matrix complete, or the feature remains explicitly
      experimental;
- [ ] no claim exceeds the evidence boundary.
