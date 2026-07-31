# Explicit, implicit, and hybrid spatial-operator comparison

This document is the execution and reporting protocol for deciding how the
edge-free spatial kernel should be used in equivariant linear attention.

It compares three resource-matched arms:

| Arm | Global operator | Explicit sparse local | Implicit spatial residual |
|---|---|---:|---:|
| `explicit` | exact equivariant linear attention | yes | no |
| `implicit` | exact equivariant linear attention | no | yes |
| `hybrid` | exact equivariant linear attention | yes | yes |

All three models instantiate the same module tree and load the same initial
state dict. The arm changes execution only. This prevents parameter count,
initialization, or output-layout differences from being mistaken for operator
effects.

## 1. What this protocol can establish

The protocol can establish:

- whether the edge-free kernel retains short-range directional fidelity;
- whether it improves smooth long-range targets;
- whether hybridization adds value beyond explicit local interactions;
- paired optimization stability, latency, and memory differences;
- whether implicit execution is genuinely independent of explicit edge metadata;
- whether the measured scaling is consistent with the documented complexity.

It cannot establish general downstream superiority from synthetic tasks alone.
A synthetic pass only advances an arm to real-task validation.

## 2. Audit invariants

For every task and seed, the runner records and checks:

1. identical train and validation data hashes across arms;
2. identical initial state-dict hash across arms;
3. identical parameter count and parameter schema;
4. identical optimizer, learning rate, step budget, clipping threshold, and
   evaluation schedule;
5. validation labels are not used for training;
6. no-edge CSR metadata is prepared before the timed implicit forward;
7. zero-initialized hybrid output equals explicit output before training;
8. implicit output is unchanged when the supplied explicit edge metadata is
   replaced, provided graph membership is unchanged.

The last two numerical gates default to

```text
max absolute error <= 1e-7
```

and should be tightened to FP64-scale tolerances for double-precision audits.

## 3. Synthetic tasks

Every graph contains scalar node features `a_i`, one polar input vector `v_i`,
and positions `x_i`.

### 3.1 Local directional target

For one unordered pair `i<j`, let

\[
r_{ij}=\lVert x_j-x_i\rVert,
\qquad
\widehat r_{ij}=\frac{x_j-x_i}{r_{ij}},
\]

and let `f_c` be the C2 compact cutoff. The target is

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

This target probes sharp locality and edge-axis anisotropy. The explicit arm is
the reference.

### 3.2 Smooth Gaussian target

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

This target probes a graph-global smooth spatial field. It is the favorable lane
for the implicit kernel.

### 3.3 Mixed target

\[
y_{\rm mixed}=y_{\rm local}+\frac12y_{\rm smooth}.
\]

This target probes whether hybridization can preserve local fidelity while
adding smooth long-range context.

Targets are normalized using train statistics only. Validation is never used to
select normalization.

## 4. Model comparison runner

Quick CPU smoke:

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
  --steps 5 \
  --evaluation-interval 1 \
  --profile-warmup 1 \
  --profile-repeats 2 \
  --device cpu \
  --dtype float64 \
  --output artifacts/spatial-operator-comparison/smoke/result.json
```

Recommended CUDA screen:

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
  --candidate-skin 0.25 \
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

The implicit scales are intentionally longer than the compact local cutoff in
the hybrid screen. This reduces role duplication between exact short-range and
smooth long-range paths.

## 5. Report generation

```bash
uv run python scripts/report_spatial_operator_comparison.py \
  "$RUN/result.json" \
  --output "$RUN/report.md" \
  --decision-json "$RUN/decision.json"
```

The report contains:

- protocol violations;
- zero-init and edge-independence audits;
- per-task accuracy and resource tables;
- paired per-seed differences versus explicit;
- synthetic promotion gate checks;
- required downstream follow-up.

Default hybrid synthetic gate:

```text
local MAE regression <= 2%
smooth MAE improvement >= 5%
mixed MAE improvement >= 2%
train-time overhead <= 25%
inference-time overhead <= 25%
training-memory overhead <= 25%
minimum seeds = 3
```

Default implicit replacement gate additionally requires:

```text
local MAE regression <= 2%
smooth MAE improvement >= 5%
inference time <= explicit
training memory <= explicit
```

These are screening thresholds, not universal scientific constants. Override
them explicitly in the report command when the target workload has different
resource constraints.

## 6. Kernel approximation diagnostics

Before interpreting model performance, quantify the kernel approximation:

```bash
uv run python scripts/evaluate_implicit_spatial.py \
  --nodes 512 \
  --value-width 32 \
  --scales 2,4,8 \
  --cutoff 1.75 \
  --topk 32 \
  --output "$RUN/implicit-accuracy.json"
```

This reports:

- Gaussian-kernel relative Frobenius error;
- Gaussian message relative error;
- Gaussian top-k overlap;
- compact C2 cutoff relative error;
- cutoff message relative error;
- cutoff top-k overlap.

The implicit kernel is not expected to approximate a sharp cutoff well at low
rank. That result is informative rather than automatically a failure: the
hybrid arm is intended to use the kernel as a longer-range complement.

## 7. Scaling and memory diagnostics

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

Interpret the fitted slopes under the contracts in `docs/SCALING.md`:

\[
T_{\rm explicit}=O(L(N+E)),
\]

\[
T_{\rm AttnRes}=O(L(N+E)+LBN),
\]

\[
T_{\rm implicit}=O(ANFD).
\]

A wall-clock slope near one is empirical evidence for that device and regime,
not a proof for arbitrary graph shapes.

## 8. Focused correctness tests

```bash
uv run pytest \
  tests/test_spatial_ablation.py \
  tests/test_spatial_comparison.py \
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

Repository gate:

```bash
scripts/check.sh fast
scripts/check.sh gpu
```

## 9. Artifact layout

Every completed comparison should use one immutable directory:

```text
artifacts/spatial-operator-comparison/<run-id>/
  result.json
  report.md
  decision.json
  implicit-accuracy.json
  scaling.json
  environment.txt
  git.txt
```

Recommended provenance:

```bash
python --version > "$RUN/environment.txt"
uv pip freeze >> "$RUN/environment.txt"
nvidia-smi >> "$RUN/environment.txt" 2>/dev/null || true
git status --short > "$RUN/git.txt"
git rev-parse HEAD >> "$RUN/git.txt"
git diff --stat >> "$RUN/git.txt"
```

Do not commit large repeated raw traces to `main`. Keep compact summaries,
commands, hashes, and a manifest in Git; use release assets or external artifact
storage for large bundles.

## 10. Downstream validation matrix

A synthetic pass must be followed by at least two task families.

### General 3D lane

Choose at least one:

- point-cloud classification or segmentation;
- registration/deformation;
- particle or field prediction;
- mesh property prediction.

### Molecular/protein lane

Choose at least one:

- QM9-like scalar property with a fixed split;
- force/energy prediction with smooth derivative checks;
- protein--ligand affinity with protein-cluster or target-disjoint validation;
- pose or coordinate refinement with geometry metrics.

For each lane compare the same three arms and retain:

- data revision and split hashes;
- sample/node/edge counts;
- initial state hash;
- parameter count;
- task metric and paired seed differences;
- train/inference time and peak memory;
- gradient clipping and instability counts;
- neighbor construction/rebuild time where applicable.

## 11. Promotion policy

### Keep explicit canonical

Default outcome unless another arm demonstrates stable value. Explicit local is
the reference for sharp directional contact and typed relations.

### Promote hybrid as an optional long-range mode

Allowed after:

1. synthetic hybrid gate passes;
2. at least one general 3D and one molecular/protein validation improve or stay
   within their predeclared tolerance;
3. overhead remains inside the workload budget;
4. benefit is stable across seeds.

The clean final implementation should preferably merge the implicit spatial
feature into the global linear-attention feature map rather than execute a full
second transport pass, provided mathematical equivalence and speed are verified.

### Replace explicit local with implicit

Requires the strongest evidence:

1. implicit synthetic replacement gate passes;
2. no meaningful loss on local directional, force, contact, or stereochemical
   tasks;
3. measured latency or memory benefit on intended hardware;
4. real-task results across both workload families;
5. the approximation limitation is acceptable for the target use case.

Until those conditions hold, implicit-only remains experimental.

## 12. Handoff checklist

Before merging or promoting:

- [ ] focused CPU tests pass;
- [ ] CUDA BF16 test passes;
- [ ] `scripts/check.sh fast` passes;
- [ ] explicit/hybrid zero-init error passes;
- [ ] implicit edge-independence error passes;
- [ ] at least three seeds are present for every arm/task;
- [ ] approximation diagnostic is attached;
- [ ] scaling result and fitted slopes are attached;
- [ ] report and decision JSON are generated;
- [ ] real-task follow-up is complete or the feature remains explicitly
      experimental;
- [ ] no claim exceeds the evidence boundary.
