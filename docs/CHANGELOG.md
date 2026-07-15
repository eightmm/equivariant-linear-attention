# CHANGELOG

Version log for data, model, training, and eval. Newest on top.

Format: `[component] vX.Y -> vX.Z (date) — change. impact.`

Components: `data`, `model`, `training`, `eval`, `ckpt`, `config`.

---

## Unreleased

- [model] v0.4 -> v0.5 (2026-07-13) — add `EquivariantMomentAttention` with persistent scalar/vector states, transient five-component l=2 moments, squared-vector linear routing, and key-mass balancing. impact: tests a richer global linear path without persistent tensor storage.
- [training] v0.2 -> v0.3 (2026-07-03) — add `rich_linear_light`, empty-neighbor fast path, and `--amp-dtype bf16` probe option. impact: gives a faster scalar/vector-only rich regression path; bf16 is available but not always faster on small QM9 graphs.
- [model] v0.3 -> v0.4 (2026-07-03) — add `rich_linear` regression option and vectorize batched rich linear attention with segment sums. impact: enables neighbor-free linear rich QM9 probe and removes per-graph Python loop overhead.
- [model] v0.2 -> v0.3 (2026-07-03) — stabilize rich attention with bounded geometry, bounded vector/tensor messages, scalar output normalization, and residual layer scale. impact: prevents rich-local default-LR divergence on QM9 probe while preserving equivariance tests.
- [training] v0.1 -> v0.2 (2026-07-03) — add EGNN baseline, graph regression smoke harness, and optional QM9 loader. impact: enables controlled EGNN vs rich-local comparison setup.
- [model] v0.1 -> v0.2 (2026-07-02) — add optional rich local vector-edge attention bias. impact: richer invariant routing without changing default behavior.

## v0.1.0 — initial

- [data] v0.1: initial schema and splits
- [model] v0.1: baseline architecture
- [training] v0.1: Muon+AdamW + WSD trapezoidal schedule
- [eval] v0.1: primary metric + frozen test split
- [ckpt] v0.1: schema with data/model version pins
- [config] v0.1: default config keys
