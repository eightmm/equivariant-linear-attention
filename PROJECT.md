# PROJECT.md

## Status

- State: confirmed
- Direction confirmed: 2026-07-15
- Evidence basis: repository tests and the external review shared by the user.

## Project

- Name: equivariant-linear-attention
- Type: PyTorch ML research prototype
- Goal: develop one global factorized moment-attention architecture for 3D
  structural graphs, with a narrow and testable O(3) mathematical contract.
- Users/workflow: research and development from Python scripts and tests.
- Non-goals: multiple attention families, dense/local/rich compatibility APIs,
  production training infrastructure, or chirality-sensitive prediction.

## Single-Layer Contract

- Public implementation: `EquivariantAttention` with
  `EquivariantAttentionConfig`.
- Internal state: persistent scalar (`0e`) and polar-vector (`1o`) channels;
  symmetric-traceless rank-2 (`2e`) moments are transient within each block.
- Connectivity: global factorized attention with linear node scaling at fixed
  width, head count, depth, and one balancing cycle.
- Symmetry: O(3), including reflections; translation invariant and permutation
  consistent. Scalar outputs cannot distinguish enantiomers.
- Outputs: tensor-only dictionary with plural keys `node_scalars`,
  `node_vectors`, `node_tensors`, `graph_scalars`, `graph_vectors`, and
  `graph_tensors`.
- Batch IDs: integer, nonnegative, contiguous IDs starting at zero. Empty graphs
  are not represented.
- Numerical policy: geometry squares, angular/ST feature construction, moment
  reductions, and invariant normalization use at least float32 for fp16/bf16
  inputs; float64 remains float64. Low-precision rank-2 outputs remain float32.
- Feature/geometry policy: `node_feats` are invariant `0e` scalars and may use
  model precision; coordinates remain float32+ and are never downcast to the
  feature dtype before geometry preprocessing.

## Architecture Baseline

- Unit-normalized positive scalar content plus finite-precision unit-ball
  vector queries/keys and bounded linear/squared-dot angular terms.
- Linear vector and quadratic 3x3 PSD summaries compute masses and
  denominators without graph-summed signed flattened outer features. Value
  numerators use analogous signed summaries and are not clamped.
- Exact graph-wise factorization of relative vector and symmetric-traceless
  second moments; no `N x N` attention tensor.
- Row normalization with one-cycle key balancing enabled by default; the
  no-balancing lane exists only for the registered normalization ablation.
- Pointwise O(3)-equivariant ratio-2 FFN after each attention residual.
- Experimental radial, dynamic-routing, alternate-Sinkhorn, dense, local, and
  rich paths are removed from the active implementation. Future additions must
  change one mathematical mechanism at a time and add a dense or symbolic
  reference test first.

## Verification

- Mathematical threshold: maximum float64 O(3), translation, and permutation
  error below `1e-6`.
- Required edge cases: singleton graphs, coincident coordinates, finite
  forward/backward, large valid fp16 graphs with extreme finite coordinates,
  invalid batch dtype/IDs, kernel endpoint bounds, factorized-vs-dense
  equality, alignment sign, moment collision, cluster-normalization,
  reflection, and coordinate-gradient equivariance.
- Required project check: `scripts/check.sh fast` with at least 80% coverage.
- GPU precision checks are required before making fp16/bf16 CUDA claims.

## Evaluation Boundary

- Existing QM9 runs are seeded random-row warm-start probes on target `gap`
  (eV), not scaffold/cold-molecule generalization evidence.
- Test evaluation remains frozen during adaptive architecture work.
- Historical experiments remain in `docs/EXPERIMENTS.jsonl`; they do not expand
  the current public architecture contract.
- Graph-wide centroid/RMS normalization does not provide cluster decomposition
  or extensive size consistency; standalone force-field claims are excluded.

## Commands

- Setup: `uv sync --locked`
- Fast verification: `scripts/check.sh fast`
- GPU smoke: `scripts/check.sh gpu`
- Training probe: `uv run python scripts/train_compare.py --dataset synthetic`
- Benchmark: `uv run python scripts/bench_attention.py`

## Paths

- Data: `data/` (untracked)
- Outputs/logs: `outputs/` (untracked)
- Scientific run records: `artifacts/` (untracked control/evidence bundles)
- Source: `src/equivariant_attention/`
- Tests: `tests/`
