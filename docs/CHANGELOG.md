# CHANGELOG

Version log for data, model, training, and eval. Newest on top.

Format: `[component] vX.Y -> vX.Z (date) — change. impact.`

Components: `data`, `model`, `training`, `eval`, `ckpt`, `config`.

---

## Unreleased

- [model] v0.7 -> v0.8 (2026-07-16) — normalize positive scalar content,
  promote the bounded degree-2 kernel with a linear angular term, factorize its
  vector summary exactly, add a controlled balance-off lane and executable P2
  counterexamples, keep auto-inference parameters in FP32, and pin the working
  setup-uv action release. A dimension/dtype inward normalization margin closes
  a general-direction float32 rounding overshoot. impact: establishes a finite pair-kernel bound,
  repairs alignment-sign blindness when the linear term is enabled, and makes
  remaining moment/cluster/parity limits testable without claiming they are
  solved.
- [model] v0.6 -> v0.7 (2026-07-15) — bound vector queries/keys and the
  quadratic angular scale, replace signed flattened angular reductions with
  structured 3x3 PSD mass/denominator contractions plus signed numerator
  summaries, preserve float32+ coordinate storage, and
  reuse graph metadata. impact: removes the reproduced orthogonal-feature
  cancellation and mixed-coordinate downcast while retaining linear node
  scaling and the single quadratic-kernel mechanism.
- [model] v0.5 -> v0.6 (2026-07-15) — consolidate the public implementation to
  one factorized-moment `EquivariantAttention`, remove base/rich/local/dense
  variants, enforce strict graph IDs, and accumulate low-precision attention
  and squared geometry/moment paths in float32. impact: narrows the mathematical
  contract and removes reproduced fp16 denominator and large-graph geometry
  overflows. This version deliberately changed the third geometry scalar from
  normalized radius-square to `log1p(normalized radius-square)`; it is an
  architecture change, not refactor-equivalent to earlier QM9 probes.
- [model] v0.4 -> v0.5 (2026-07-13) — add `EquivariantMomentAttention` with persistent scalar/vector states, transient five-component l=2 moments, squared-vector linear routing, and key-mass balancing. impact: tests a richer global linear path without persistent tensor storage.
- [training] v0.2 -> v0.3 (2026-07-03) — add `rich_linear_light`, empty-neighbor fast path, and `--amp-dtype bf16` probe option. impact: gives a faster scalar/vector-only rich regression path; bf16 is available but not always faster on small QM9 graphs.
- [model] v0.3 -> v0.4 (2026-07-03) — add `rich_linear` regression option and vectorize batched rich linear attention with segment sums. impact: enables neighbor-free linear rich QM9 probe and removes per-graph Python loop overhead.
- [model] v0.2 -> v0.3 (2026-07-03) — stabilize rich attention with bounded geometry, bounded vector/tensor messages, scalar output normalization, and residual layer scale. impact: prevents rich-local default-LR divergence on QM9 probe while preserving equivariance tests.
- [training] v0.1 -> v0.2 (2026-07-03) — add EGNN baseline, graph regression smoke harness, and optional QM9 loader. impact: enables controlled EGNN vs rich-local comparison setup.
- [model] v0.1 -> v0.2 (2026-07-02) — add optional rich local vector-edge attention bias. impact: richer invariant routing without changing default behavior.

## v0.1.0 — initial

- [data] v0.1: initial schema and splits
- [model] v0.1: baseline architecture
- [training] v0.1: AdamW constant-learning-rate probe
- [eval] v0.1: primary metric + frozen test split
- [ckpt] v0.1: schema with data/model version pins
- [config] v0.1: default config keys
