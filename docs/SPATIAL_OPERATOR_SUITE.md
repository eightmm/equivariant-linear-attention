# Spatial operator suite

This is the operational entry point for the explicit/implicit/hybrid evaluation
packet described in `SPATIAL_OPERATOR_COMPARISON.md`.

## Smoke mode

```bash
SPATIAL_SUITE_MODE=smoke \
  bash scripts/run_spatial_operator_suite.sh \
  artifacts/spatial-operator-comparison/smoke
```

Smoke mode runs:

1. focused CPU correctness tests;
2. CUDA BF16 test when CUDA is selected;
3. one-seed, five-step synthetic comparison;
4. result protocol validation;
5. report and decision generation;
6. Gaussian/cutoff approximation diagnostic;
7. fragment-locality diagnostic;
8. a minimal scaling sweep.

The synthetic decision is expected to be `insufficient_synthetic_evidence`
because smoke mode intentionally uses one seed.

## Full mode

```bash
SPATIAL_SUITE_MODE=full \
SPATIAL_SUITE_DEVICE=cuda \
SPATIAL_SUITE_DTYPE=bfloat16 \
  bash scripts/run_spatial_operator_suite.sh \
  artifacts/spatial-operator-comparison/full-$(date +%Y%m%d-%H%M%S)
```

Full mode additionally runs the repository fast gate and GPU gate and defaults
to five paired seeds, 1,000 training steps, and wider scaling sweeps.

The expensive defaults may be overridden:

```text
SPATIAL_SUITE_SEEDS
SPATIAL_SUITE_TRAIN_GRAPHS
SPATIAL_SUITE_VALIDATION_GRAPHS
SPATIAL_SUITE_NODES_PER_GRAPH
SPATIAL_SUITE_HIDDEN_DIM
SPATIAL_SUITE_LAYERS
SPATIAL_SUITE_LOCAL_RANK
SPATIAL_SUITE_STEPS
SPATIAL_SUITE_EVAL_INTERVAL
SPATIAL_SUITE_PROFILE_WARMUP
SPATIAL_SUITE_PROFILE_REPEATS
SPATIAL_SUITE_SCALE_NODES
SPATIAL_SUITE_SCALE_DEPTHS
SPATIAL_SUITE_SCALE_BLOCKS
SPATIAL_SUITE_SCALE_DEGREES
SPATIAL_SUITE_APPROX_NODES
SPATIAL_SUITE_FRAGMENT_BASE_NODES
SPATIAL_SUITE_FRAGMENT_NODES
```

## Generated artifacts

```text
<run-dir>/
  result.json
  report.md
  decision.json
  implicit-accuracy.json
  fragment-locality.json
  scaling.json
  environment.txt
  git.txt
  manifest.json
  focused-tests.log
  protocol-validation.log
  comparison.log
  report.log
  implicit-accuracy.log
  fragment-locality.log
  scaling.log
  cuda-tests.log      # CUDA only
  fast-gate.log       # full mode
  gpu-gate.log        # full CUDA mode
```

`result.json` follows
`schemas/spatial_operator_comparison.schema.json`. The suite also executes
`validate_spatial_operator_result.py`; incomplete arm sets, unpaired hashes,
parameter-count mismatches, missing audits, or contaminated timing contracts
stop the suite before report generation.

## Canonical comparison settings

The suite sets `candidate_skin=0` so the explicit model cutoff is exactly the
synthetic local-target cutoff. Wider candidate-skin behavior should be tested in
a separate neighbor-list robustness experiment, not mixed into the operator
attribution.

The hybrid implicit scales default to `2,4,8`, larger than the explicit cutoff
`1.75`. This is deliberate: the hybrid arm evaluates a smooth longer-range
complement rather than duplicating the exact local lane.

## Result interpretation

- `retain_explicit_as_canonical`: neither candidate passed the bounded synthetic
  gate.
- `hybrid_passes_synthetic_candidate_gate`: run real-task validation; do not
  promote yet.
- `implicit_passes_synthetic_replacement_gate`: replacement passed the synthetic
  screen but still requires the strongest real-task and fragment-locality
  evidence.
- `insufficient_synthetic_evidence`: missing tasks, seeds, audits, or protocol
  compliance.

A successful suite is not itself a canonical architecture decision. Promotion
requires the downstream matrix and evidence boundary in
`SPATIAL_OPERATOR_COMPARISON.md`.
