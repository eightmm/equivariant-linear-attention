# Evaluation

The repository evaluates one `EquivariantAttention` class. Routing and memory
flags select registered arms of that class; there is no model-family selector.

```bash
uv run python scripts/train_compare.py \
  --dataset synthetic --steps 10 --split-seed 42 --model-seed 42 \
  --routing ggg
```

The script reports train loss, validation MAE/RMSE, split hashes, model seed,
parameter counts, source hash, optimizer settings, target normalization, the
full routing/kernel/memory run configuration, and whether test evaluation
occurred. Test evaluation is disabled by default and requires explicit
`--evaluate-test`; adaptive architecture work selects only on validation.

## Registered comparison sequence

All adaptive QM9 arms use target `gap` in eV, random-row split seed 42, model
seeds 41/42/43, FP32, batch size 64, width 64, three blocks, four heads, and
matched optimizer/schedule settings.

1. On `ggg`, compare the 2x2 alignment-term/key-balancing arms. Turning off
   alignment removes only `beta * (q dot k)` and retains the `beta` constant.
2. With the selected normalization, compare `fixed` with global row-only
   `inverse_graph_size`, which scales `(c + beta + delta*beta*t)` by `1/N_g`
   while leaving content and `gamma*t^2` unchanged. It requires balancing off.
3. With memory interaction and radial trace off, compare `ggg` with `lgl`.
   `lll` is a mechanistic local-only control, not a performance candidate.
4. On `lgl`, compare interacting `M=1,4,8`; `M=1` is the exact incumbent
   reduction. Memory count with interaction off is not an expressivity arm.
5. Change only radial trace off/on for the selected routing and memory setting.

The non-adaptive Stage-2 size probe is reproducible without a dataset:

```bash
uv run python scripts/probe_kernel_scaling.py \
  --sizes 16 32 64 128 512 2048 --device cpu
```

It compares `fixed` and corrected `inverse_graph_size` with key balancing off
and reports JSON-safe mean row maximum weight, entropy/log-N, synchronized
runtime, `beta`/`gamma` and value gradient norms, and a small output-probe norm.
Exact row statistics are streamed in query blocks and the differentiable probe
uses only a fixed number of query rows. The script therefore retains no full
attention matrix and deliberately does not compute effective rank.

A 500-step run is a numerical screen. Promotion requires paired mean validation
MAE improvement of at least 0.01 eV, improvement in at least two of three seeds,
worst-seed regression no larger than 0.02 eV, and median latency and peak-memory
increases no larger than 20%. A joint-only gain is recorded as an interaction,
not attributed independently to either mechanism.

Record common initialization hashes, total and nonzero-gradient parameters,
synchronized latency, peak CUDA memory, node-count strata, bounded kernel
scales, mass/denominator quantiles, entropy over log node count, maximum weight,
effective support, column CV, gradient/residual norms, and HEMM occupancy and
assignment entropy. Effective-rank computation is opt-in and size-bounded; row
or column scaling does not change exact matrix rank.

QM9 numbers use a random-row warm split. They do not measure scaffold,
protein-target, temporal, or cold-complex generalization. Historical test access
also means the split is not a pristine confirmatory holdout. A five-seed,
10,000-step test comparison is a separate approval gate after architecture
lock. No non-default route, memory count, radial trace, or floor mode is promoted
without the registered experiment evidence.
