# Logging contract

The ELA library is tracker-neutral. Importing or running the model does not
initialize a network client, create a run, print metrics, or mutate a global
logging configuration. Training applications own their logger and tracker.

The shipped benchmark writes a JSON result when `--output` is supplied:

```bash
uv run python scripts/benchmark_ela.py \
  --nodes 4096 --degree 32 --device cuda --dtype bfloat16 \
  --output artifacts/ela-kernels.json
```

The canonical suite records its environment, Git state, focused-test output,
benchmark output, and a manifest in one run directory:

```bash
bash scripts/run_canonical_ela_suite.sh artifacts/canonical-ela/smoke
```

Downstream training logs should at minimum identify the source revision,
configuration, data/split identity, seed, update count, primary validation
metric, elapsed time, and peak allocated memory. Test metrics belong only in a
final evaluation, not adaptive architecture selection.

Historical Weights & Biases conventions are not a package dependency or current
default. Retained experiment evidence is indexed by
[`EXPERIMENTS.md`](EXPERIMENTS.md).
