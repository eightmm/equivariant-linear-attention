# Spatial operator evaluation index

Start here when continuing the explicit-versus-implicit spatial-operator work.

## Fastest entry point

Run the complete smoke suite:

```bash
SPATIAL_SUITE_MODE=smoke \
  bash scripts/run_spatial_operator_suite.sh \
  artifacts/spatial-operator-comparison/smoke
```

Then read, in order:

1. generated `report.md`;
2. generated `decision.json`;
3. `docs/SPATIAL_OPERATOR_COMPARISON.md`;
4. `docs/IMPLICIT_SPATIAL_HANDOFF.md`.

## Documents by purpose

| Question | Document |
|---|---|
| How do I run everything? | `SPATIAL_OPERATOR_SUITE.md` |
| How are explicit, implicit, and hybrid compared? | `SPATIAL_OPERATOR_COMPARISON.md` |
| What does the edge-free kernel compute? | `IMPLICIT_SPATIAL_KERNEL.md` |
| What scaling claims are allowed? | `SCALING.md` |
| Does adding a distant fragment change existing interactions? | `FRAGMENT_LOCALITY.md` |
| How should an exact edge-free API discover neighbors internally? | `ON_THE_FLY_NEIGHBOR_KERNEL.md` |
| What was implemented, and what remains? | `IMPLICIT_SPATIAL_HANDOFF.md` |

## Code by purpose

| Purpose | File |
|---|---|
| Edge-free Gaussian--Taylor transport | `src/equivariant_attention/implicit_spatial.py` |
| Zero-init implicit residual | `src/equivariant_attention/implicit_spatial_residual.py` |
| Matched three-arm model | `src/equivariant_attention/spatial_ablation.py` |
| Synthetic local/smooth/mixed tasks | `src/equivariant_attention/spatial_benchmarks.py` |
| Protocol validation, paired deltas, gates, report | `src/equivariant_attention/spatial_comparison.py` |
| Complexity contracts and slope fitting | `src/equivariant_attention/scaling_contract.py` |

## Scripts by output

| Output | Script |
|---|---|
| `result.json` | `scripts/compare_spatial_operators.py` |
| `report.md`, `decision.json` | `scripts/report_spatial_operator_comparison.py` |
| protocol pass/fail | `scripts/validate_spatial_operator_result.py` |
| `implicit-accuracy.json` | `scripts/evaluate_implicit_spatial.py` |
| `fragment-locality.json` | `scripts/evaluate_fragment_locality.py` |
| `scaling.json` | `scripts/benchmark_scaling.py` |
| complete artifact directory | `scripts/run_spatial_operator_suite.sh` |

## Decision boundary

The default canonical model remains

\[
\text{exact equivariant global linear attention}
+
\text{explicit sparse short-range local}.
\]

The edge-free implicit kernel is experimental. A synthetic gate can advance
`hybrid` or `implicit` to downstream evaluation, but cannot make either one the
canonical path. Real-task validation must include at least one general 3D lane
and one molecular/protein lane, with paired splits, initialization, optimization,
latency, memory, and leakage controls.
