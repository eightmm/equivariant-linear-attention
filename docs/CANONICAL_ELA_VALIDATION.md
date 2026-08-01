# Canonical ELA validation

## Smoke

```bash
ELA_SUITE_MODE=smoke \
  bash scripts/run_canonical_ela_suite.sh \
  artifacts/canonical-ela/smoke
```

Smoke mode runs the focused canonical tests and a small same-weight numerical
comparison between the internal pre-router reference and public `ELA`.

The runner pins `PYTHONPATH` to the current repository, verifies the imported
package file, rejects a nonempty artifact directory, and uses the locked
environment. Tiny CPU smoke receipts are functional diagnostics, not resource or
accuracy promotion evidence.

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
worktree. The benchmark still does not establish downstream superiority.

## Focused contracts

The suite checks:

- package root exposes one backbone `ELA` and one architecture layer
  `ELALayer`;
- zero-init branch fusion reproduces exact `G + L`;
- shared internal reference weights reproduce the full pre-router function;
- learned fusion remains proper/improper O(3)-equivariant;
- router and branch-balance parameters receive gradients;
- minimal config derives heads and local rank deterministically;
- all supported input/output parity sectors transform correctly;
- forward, feature gradients, and coordinate gradients are finite;
- graph isolation, edge-order invariance, node permutation, and required double
  backward hold;
- configured condition and semantic-order PE are neutral at initialization;
- a context-free call bypasses a trained conditioner entirely;
- semantic-order labels and enable masks follow the node permutation contract;
- disabled-node order labels have no effect;
- coordinate refinement is identity initialized, bounded, masked, centered, and
  equivariant;
- canonical CUDA FP32/BF16 forward and backward remain finite;
- historical checkpoint migration fails closed;
- the canonical regression adapter works.

## Resource receipt

`overhead.json` records:

- exact source path, common-state/input hashes, and migration receipt;
- git SHA plus GPU, PyTorch, and CUDA runtime fingerprint;
- node/graph output and input/common-parameter gradient equivalence;
- public ELA and internal numerical-reference parameter counts;
- branch-router parameter count;
- inference latency;
- optimizer-inclusive train-step latency;
- raw latency samples and IQR;
- peak allocated CUDA memory, including optimizer state for the train step;
- candidate/reference ratios;
- graph size and degree;
- exclusion of neighbor discovery.

Context and coordinate-refinement overhead require separate sweeps because they
are runtime capabilities rather than a second architecture.

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

Store downstream receipts separately. Do not infer architecture promotion from
contract or overhead tests alone.
