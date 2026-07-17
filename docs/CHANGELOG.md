# CHANGELOG

Version log for data, model, training, and eval. Newest on top.

Format: `[component] vX.Y -> vX.Z (date) — change. impact.`

Components: `data`, `model`, `training`, `eval`, `ckpt`, `config`.

---

## Unreleased

- [eval] v0.4 -> v0.5 (2026-07-17) — run the registered clean-source M=1
  `ggg`/`lgl` CUDA and three-seed QM9 gap comparison with test evaluation off.
  impact: `lgl` passes with 0.052774 eV mean validation-MAE improvement, 3/3
  improving seeds, and lower measured step latency; this remains a bounded
  validation result and does not admit Stage-0-blocked M=4/M=8.
- [model] v0.10 -> v0.11 (2026-07-17) — replace the functionally inactive
  one-dimensional HEMM router with an M-shared invariant MLP and fixed DCT slot
  codes, add exact residual-coupling counterfactuals, and vectorize batched local
  candidate construction. impact: preserves M=1 dispatch and route/memory
  parameter schema, makes the proposed coupling repair falsifiable, and removes
  the per-graph GPU launch loop from local geometry.
- [eval] v0.3 -> v0.4 (2026-07-17) — extend frozen Stage-0 to four graph roles,
  widths 16/64, seeds 401--403, entropy/MI/center diagnostics, actual transport,
  post-state, and input gradients. impact: rejects every registered M=4/M=8
  residual-coupling candidate without moving thresholds and keeps those arms
  out of QM9 performance studies.
- [eval] v0.2 -> v0.3 (2026-07-16) — add per-graph/per-head effective HEMM
  pair-gate statistics, connect them to the bounded runtime diagnostic, and add
  a matched-state M=1/4/8 Stage-0 activation probe. impact: detects that the
  current M=4/8 coupling and pair gate are numerically constant despite healthy
  assignment occupancy, blocks broader memory arms without moving thresholds,
  and records the current HEMM as a symmetric low-rank gate rather than
  persistent memory.
- [model] v0.9 -> v0.10 (2026-07-16) — keep one
  `EquivariantAttention` class while adding per-block local/global head routing,
  raw-coordinate 2.5-Angstrom local edges with 16 RBFs, an optional exact
  radial trace, and invariant multi-memory gating for the middle global block
  of `lgl`. `ggg`, `M=1`, interaction off, and radial trace off remain the
  public defaults. impact: makes local/global and HEMM hypotheses executable
  without claiming that an experimental arm improves accuracy or performance;
  `M=1` reduces exactly to the incumbent and interaction-off memories are
  algebraically degenerate.
- [model] v0.8 -> v0.9 (2026-07-16) — isolate the alignment ablation so it
  removes only `beta * (q dot k)` while retaining the `beta` constant, enforce
  exactly one balancing cycle, add fixed and global row-only graph-size-scaled
  shifted-baseline modes, and preprocess extreme coordinates scale-first in
  float32/float64. impact: repairs the ablation confound, exposes positive
  baseline dilution as a testable lane, and prevents direct large-coordinate
  reduction overflow.
- [eval] v0.1 -> v0.2 (2026-07-16) — make test evaluation opt-in through
  `--evaluate-test`, register `ggg/lgl/lll`, `M=1/4/8`, memory-interaction, and
  radial-trace CLI arms, and add pure bounded diagnostic helpers. impact:
  keeps adaptive work validation-only by default and makes expensive
  effective-rank diagnostics explicitly size-bounded.
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
