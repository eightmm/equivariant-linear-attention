# Canonical ELA validation

## Smoke

```bash
ELA_SUITE_MODE=smoke \
  bash scripts/run_canonical_ela_suite.sh \
  artifacts/canonical-ela/smoke
```

Smoke mode runs the focused canonical tests and a small same-weight resource
comparison between the admitted refined `EquivariantLinearAttention` control and
branch-aware `ELA`.

The runner pins `PYTHONPATH` to the current repository, verifies the imported
package file, rejects a nonempty artifact directory, and uses the locked
environment. Smoke receipts are diagnostic; their tiny CPU shapes are not a
resource-promotion gate.

## Full

```bash
ELA_SUITE_MODE=full \
ELA_SUITE_DEVICE=cuda \
ELA_SUITE_DTYPE=bfloat16 \
  bash scripts/run_canonical_ela_suite.sh \
  artifacts/canonical-ela/full-$(date +%Y%m%d-%H%M%S)
```

Full mode additionally runs:

```text
scripts/check.sh fast
scripts/check.sh gpu
```

and a larger same-weight BF16 safety benchmark. Full mode requires a clean
worktree, but this single ordered run is not a resource-promotion decision.
Promotion uses the separately preregistered FP32 CUDA, five-seed AB/BA matrix
for both task-like shapes.

## Focused contracts

The suite checks:

- zero-init branch fusion reproduces `G + L`;
- shared refined-ELA weights reproduce the full old function before routing
  learns;
- learned fusion remains reflection-equivariant;
- learned fusion remains equivariant under generic proper and improper O(3);
- router and branch-balance parameters receive gradients;
- minimal config derives heads/rank deterministically;
- forward, input gradients, and coordinate gradients are finite;
- graph isolation, edge-order invariance, and input/coordinate double backward;
- canonical CUDA FP32 and BF16 forward/backward;
- advanced checkpoint migration fails closed;
- conditioned wrapper is neutral at initialization and trainable;
- coordinate refiner is neutral at initialization, bounded, and centered;
- regression adapter and public namespace contracts work.

## Resource receipt

`overhead.json` records:

- exact source path, common-state/input hashes, and migration receipt;
- clean-tree git SHA plus GPU, PyTorch, and CUDA runtime fingerprint;
- node/graph output and input/common-parameter gradient equivalence;
- control and candidate parameter counts;
- branch-router parameter count;
- inference latency;
- optimizer-inclusive train-step latency;
- raw latency samples and IQR;
- peak allocated CUDA memory, including optimizer state for the train step;
- candidate/control ratios;
- graph size and degree;
- exclusion of neighbor discovery.

The benchmark does not establish downstream accuracy. It isolates the cost of
the canonical branch router while holding the admitted backbone weights fixed.

## Artifact layout

```text
artifacts/canonical-ela/<run-id>/
  overhead.json
  environment.txt
  git.txt
  source-provenance.txt
  manifest.json
  focused-tests.log
  cuda-focused.log # CUDA
  benchmark.log
  fast-gate.log    # full
  gpu-gate.log     # full CUDA
```

Store downstream receipts separately. Do not infer an architecture promotion
from the overhead benchmark alone.
