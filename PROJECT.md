# PROJECT.md

## Status
- State: confirmed

## Interview
- Stage 1 intent:
  - Data/domain: atom, molecule, or protein 3D graph data with node features, edge features, and coordinates.
  - First target: minimal prototype with equivariance tests.
  - Output tasks: support multiple task heads.
- Stage 2 scope:
  - Symmetry: SE(3) equivariance plus permutation handling.
  - Internal representation: cuEquivariance-first geometry backend; default `linear_sh` attention with l<=2 geometry and rank-2 tensor moments, plus dense/local variants and a rich explicit-irreps API.
  - API: `model(node_feats, pos, edge_feats=None, batch=None) -> dict`.
  - Graph connectivity: linear SH-enhanced global attention by default; local radius/top-k and dense all-pairs variants are available.
- Stage 3 execution:
  - Initial heads: graph/node scalar, vector, and rank-2 tensor outputs; rich API supports `0/1/2` irreps-like hidden/output specs.
  - Dependencies: PyTorch, e3nn, and cuEquivariance-family acceleration as the default where available; use a graceful e3nn fallback when cuEquivariance is unavailable.
  - Verification: unit tests for rotation, translation, and permutation behavior with error below 1e-6.
- Open decisions:
  - None for the prototype.

## Project
- Name: equivariant-attention
- Type: PyTorch ML prototype
- Goal: Build an attention-based architecture for 3D structural graph data that satisfies equivariance constraints while offering a faster linear-attention path.
- Users/workflow: Research/development use from Python scripts and tests.
- Scope: Minimal implementation plus focused equivariance tests.
- Non-goals: Full training pipeline, datasets, benchmarks, or production packaging until the model contract is confirmed.

## Commands
- Setup: `uv sync`
- Test: `uv run pytest -q`
- Run: import and call the PyTorch module directly.
- Lint/typecheck: `uv run ruff check .`

## Paths
- Data: none for prototype.
- Config: none for prototype.
- Outputs/logs: none for prototype.
- Checkpoints: none for prototype.

## Verification
- Success criteria: model forward supports graph and node scalar/vector heads; scalar outputs are invariant, vector outputs are SE(3)-equivariant, and all outputs are permutation-consistent.
- Required checks: `scripts/check.sh fast`; focused pytest tests for rotation, translation, and permutation with max error below 1e-6.
- Baseline/metric: no training metric; numerical equivariance error only.

## Experiment Pre-Registration
- Question: On the common QM9 110k/10k/10k split, does `rich_local` stay competitive with the EGNN baseline under the same probe budget?
- Hypothesis: `rich_local` test MAE will be within 10% of EGNN test MAE on target index 4 after 2,000 CUDA steps.
- Prediction: `rich_local` may train slower early, but should not be more than 10% worse if the richer equivariant attention path is usable.
- Baseline: `egnn`, same split, seed, batch size, hidden size, layer count, optimizer, and target normalization.
- Metric: test MAE/RMSE in original target units after train-only target normalization inverse transform.
- Abandon threshold: if `rich_local` is >10% worse than EGNN on test MAE at this scale, prioritize training stabilization or architecture ablation before longer QM9 runs.

### Moment Linear Probe

- Question: Can transient l=2 moments and invariant vector routing improve the
  `rich_linear_light` speed/accuracy tradeoff on the same QM9 probe?
- Hypothesis: `moment_linear` reaches test MAE <= 0.65 without exceeding the
  39-second `rich_linear_light` three-layer runtime by more than 10%.
- Prediction: richer invariant contractions improve MAE while removal of
  persistent tensor channels keeps runtime near the light linear model.
- Baseline: `rich_linear_light`, three layers, hidden size 64, four heads, same
  split, seed, batch size, optimizer, target normalization, and 2,000 steps.
- Smallest falsifying run: finite CPU/GPU train smoke followed by the standard
  2,000-step CUDA probe only after symmetry and backward checks pass.
- Primary metrics: test MAE and wall time; secondary metric: test RMSE.

### Enhanced Moment Linear Ablation

- Question: Which exact equivariant additions improve the validation accuracy of
  `moment_linear` without losing its linear-in-node attention complexity?
- Hypothesis: radial trace and fuller Gram contractions recover useful invariant
  information; shifted angular features and a learnable balancing exponent may
  further improve routing stability.
- Candidates, in order: exact radial second-moment trace, full state/message Gram
  invariants, shifted-square angular kernel, and per-head balancing exponent.
- Protocol: fixed QM9 split seed 42 and model seed 42, target index 4, 130,000
  samples, 110,000/10,000/10,000 split, hidden size 64, three layers, four
  heads, batch size 64, and the existing optimizer settings. Screen each candidate for 500
  CUDA steps against the current incumbent and carry it forward only when its
  validation MAE improves. Use validation only; skip test evaluation throughout
  architecture selection.
- Primary metric: validation MAE. Secondary metric: wall time; flag additions
  exceeding 10% of the baseline runtime.
- Follow-up: run the selected configuration for 2,000 steps on validation only.
  A future frozen multi-seed run is required before reporting an unbiased test
  comparison.
- Data contract: target index 4 is the QM9 HOMO-LUMO gap in eV. The current
  seeded random-row split is a warm-start architecture probe, not a scaffold or
  cold-molecule generalization estimate; see `docs/QM9_CONTRACT.md`.

### Equivariant FFN Bypass Ablation

- Question: Is the limiting factor the shared global transport rather than
  missing angular or moment features?
- Hypothesis: a pointwise gated equivariant FFN that bypasses global transport
  improves 500-step validation MAE while preserving exact equivariance and
  linear node scaling.
- Modification: after each moment-attention residual, apply a scalar SwiGLU to
  normalized scalar state plus per-vector-channel squared norms, and apply a
  scalar-gated channel mix to vector state. No attention kernel, moment basis,
  optimizer, split, width, or training schedule changes.
- Prediction: validation MAE improves from `0.78290` to at most `0.77000` and
  measured model/train/validation time remains within 15% of `7.3375s` at 500
  CUDA steps.
- Protocol: QM9 random-row warm split 110k/10k/10k, target `gap` in eV, split
  seed 42, model seed 42, hidden size 64, three layers, four heads, batch 64,
  500 steps, and validation only. Promote to a 2,000-step follow-up only if the
  prediction is met.
- Failure diagnostics: finite gradients and residual scales; if available,
  inspect scalar/vector FFN residual magnitude to distinguish a dead bypass
  from a genuinely null result.

#### Compact FFN Follow-up

- Observation: full-width FFN reached validation MAE `0.76659`, but its
  `8.855s` runtime exceeded the registered `8.438s` limit.
- Single change: reduce only the FFN hidden ratio from `1.0` to `0.25`
  (`64 -> 16` channels); retain the same equations and every other setting.
- Prediction: validation MAE remains at most `0.77000`, runtime is at most
  `8.438s`, and parameter count is below 95,000 at 500 steps.
- Decision: run 2,000 validation-only steps only if all three conditions hold.

#### Performance-Priority Follow-up

- Priority revision: runtime is now secondary to predictive performance.
- Provenance gate: add explicit reflection tests and record a content hash over
  source, training entry point, `PROJECT.md`, layer/data-contract documentation,
  project metadata, and dependency lock. Metrics must capture every run-defining
  argument and whether test evaluation occurred.
- Reproduction: rerun the full-width FFN for 500 validation-only steps on the
  hashed tree. Continue only if validation MAE is at most `0.77000`; runtime is
  recorded but is no longer a rejection criterion.
- 2,000-step hypothesis: full-width FFN improves the unchanged historical
  validation MAE `0.618763` under the same split/model seed and training budget.
  Test evaluation remains disabled.
- Outcome: the matched no-FFN baseline was `0.61710`; FFN ratio 1 reached
  `0.57584`, ratio 2 reached `0.51632`, and ratio 4 failed its 500-step screen.
  Promote ratio-2 equivariant FFN to the default moment-linear block. Runtime is
  secondary; the measured 2,000-step increase was about 8.3%.

### Exact Factorized Radial-Kernel Ablation

- Question: Does making pair distance part of the attention weights improve the
  ratio-2 FFN `moment_linear` model without materializing pairwise edges?
- Hypothesis: multiplying the existing content kernel by the exact positive
  factor `a_h - ||y_i-y_j||^2 / L_g^2`, with `a_h > 1` and
  `L_g = 2 max_i ||y_i||`, supplies a missing locality bias while preserving
  exact O(3) equivariance and linear node scaling.
- Prediction: across model seeds 41/42/43, the radial variant lowers mean QM9
  gap validation MAE by at least `0.01 eV` from the existing `0.56704 eV`
  default mean and improves at least two of three paired seeds.
- Baseline: the existing data-verified ratio-2 FFN artifacts for seeds 41/42/43.
  Hold split seed 42, target, data, width, depth, heads, optimizer, batch size,
  2,000-step budget, target normalization, and all other moment features fixed.
- Primary metric: paired validation MAE difference in eV. Secondary metrics:
  runtime, denominator finiteness, and parameter count. Test evaluation remains
  disabled because this is adaptive architecture selection on the random-row
  warm validation split.
- Smallest decisive run: mathematical factorization and symmetry tests followed
  by the three paired 2,000-step validation runs. Reject promotion if the mean
  improvement is below `0.01 eV` or fewer than two seeds improve.

### Invariant-Conditioned Dynamic-Moment Routing Ablation

- Question: Can node-conditioned invariant routing use the existing transient
  moments better than the fixed per-head mixture without adding pairwise edges?
- Hypothesis: replacing `B + c_R R + c_T Tq` with zero-initialized invariant
  corrections over `B`, `R`, `Tq`, and normalized `ST(T^2)q` improves QM9 gap
  validation MAE while preserving exact O(3) equivariance and linear node
  scaling.
- Prediction: across model seeds 41/42/43, dynamic routing lowers mean
  validation MAE by at least `0.01 eV` from the current same-tree static mean
  `0.54958 eV` and improves at least two of three paired seeds.
- Baseline: rerun dynamic-off on the final source tree for each seed. Hold split
  seed 42, target, data, width, depth, heads, ratio-2 FFN, optimizer, batch size,
  2,000-step budget, target normalization, and all other moment flags fixed.
- Initialization control: the dynamic residual projection is zero initialized,
  so dynamic-on and dynamic-off produce the same initial function under the
  same model seed. Added capacity and runtime are recorded explicitly.
- Primary metric: paired validation MAE difference in eV. Secondary metrics:
  runtime, parameter count, routing residual magnitude, and finite gradients.
  Test evaluation remains disabled on the adaptive random-row warm split.
- Smallest decisive run: exact tensor-polynomial and initialization-equivalence
  tests followed by three same-tree paired 2,000-step runs. Reject promotion if
  mean improvement is below `0.01 eV` or fewer than two seeds improve.

### Iterative Factorized-Sinkhorn Ablation

- Question: Does a second exact factorized Sinkhorn balancing cycle reduce
  attention sinks enough to improve the ratio-2 FFN `moment_linear` model
  without constructing pairwise attention?
- Hypothesis: alternating one additional `K^T u` and `K v` normalization uses
  the existing low-rank feature map more evenly and improves QM9 gap validation
  MAE while preserving exact O(3) equivariance and linear node scaling.
- Implementation contract: `sinkhorn_iterations=1` must be numerically identical
  to the current key-balance plus row-normalization path. For iteration count
  `S > 1`, every `K v` and `K^T u` product must use graph-wise factorized
  reductions; no `N x N` tensor or new learned parameter is allowed.
- Screen: final-tree paired 500-step validation-only runs with split seed 42,
  model seed 42, QM9 target `gap`, 110k/10k/10k random-row warm split, hidden
  size 64, three layers, four heads, batch 64, ratio-2 FFN, and all optional
  moment/radial/dynamic flags disabled. Change only Sinkhorn iterations `1 -> 2`.
- Prediction: two iterations lower validation MAE by at least `0.01 eV`.
  Runtime is secondary but must be recorded.
- Follow-up: only if the screen passes, run paired 2,000-step seeds 41/42/43.
  Promote only when mean paired improvement is at least `0.01 eV` and at least
  two of three seeds improve. Test evaluation remains disabled throughout.
