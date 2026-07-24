# Function-preserving performance refactor (2026-07-24)

## Outcome

The selected gated-plus-grouped LGL route now uses less CUDA memory and less
time without changing public configuration, parameters, state-dict keys, or the
layer equations. The main large-graph synthetic gate passed, and the frozen
train-only ATOM3D-LBA/PDBBind capacity probe still crossed `0.10 pK`.

## Implementation

The refactor changes execution rather than learned features:

1. An all-local gated layer computes only the query-vector state needed by the
   shared equivariant update. Scalar query/key, key vector, values, kernel
   scales, and legacy gates remain allocated for checkpoint compatibility but
   are not executed on that route.
2. A single message group is returned directly instead of concatenated.
3. The first gated edge-MLP linear map is exactly partitioned as

   ```text
   W [s_i, s_j, rbf_ij, I_ij] + b
     = W_i s_i + W_j s_j + W_r rbf_ij + W_I I_ij + b.
   ```

   Receiver/sender scalar projections are performed before edge gathering, so
   the `[E, H, 2d + RBF + 5]` input is never materialized.
4. Local geometry computes displacement once and reuses cached non-self
   indices, RBFs, cutoff weights, and symmetric-traceless direction features
   across repeated local stages.
5. Global numerator and denominator summaries use one augmented value
   `[value, 1]`.
6. Multi-output receiver aggregation uses two width-balanced packed reductions
   instead of one peak-memory-heavy packed edge tensor.

## Matched synthetic CUDA profile

Both runs used the same `gated_static` LGL model, `N=2048`, `k=64`, FP32,
model/graph seed `20260723`, supplied edge hash
`cfc11b8d83eb28b41132383eb094b42a2b4d62e059969ec136904e0c0de0fd4c`,
and initial-state hash
`21ca283b0536c7d75fceb24658374111df16f143b20f98b7bd311dc850f3023c`.
The final profiled loss stayed exactly `0.5434731841087341`.

| implementation | forward device time, 3 profiled steps | peak CUDA |
|---|---:|---:|
| base `28fc9d45` | 15.784 ms | 1,298,436,608 B |
| refactored | 12.418 ms | 1,064,950,784 B |
| change | -21.33% | -17.98% |

Operator-profiler time includes profiler overhead and is diagnostic, not a
standalone latency benchmark.

## Frozen ATOM3D-LBA/PDBBind train-only probe

The real-data rerun used the same 16 public train complexes, 153,029 identical
directed candidates, raw features, targets, optimizer, order, strict CUDA
determinism, and candidate initial-state hash
`8976068e842c7b677a6cd705095f1bc4be072f905c48ffeb600b2cda202fe094`.
Validation and test labels were not read.

| gated-plus-grouped LGL | threshold | train MAE at stop | median step | peak CUDA |
|---|---:|---:|---:|---:|
| preceding v2 | 950 / 25.03 s | 0.099318 pK | 23.56 ms | 423.7 MB |
| refactored | 700 / 17.64 s | 0.083786 pK | 22.61 ms | 354.0 MB |
| systems change | — | — | -4.03% | -16.45% |

The private same-harness EGNN remained faster at `4.45 ms` per step and used
`326.1 MB`; the refactored candidate was `5.08x` slower and used `1.086x` its
peak memory on these small complexes. EGNN ended at `0.116225 pK` after 3,000
steps in this capacity probe.

The earlier threshold crossing is not treated as an architecture-accuracy
gain. Exact algebraic refactors change FP32 accumulation order, which can
change a long optimizer trajectory despite matching initial state and
single-step contracts. The supported conclusions are preserved memorization
capacity and measured resource improvement in this run.

## Verification

- Focused mathematical, route, geometry, and gradient tests passed.
- `scripts/check.sh fast`: 513 passed, 88.75% coverage.
- `scripts/check.sh gpu`: BF16 and FP32 CUDA smoke passed.
- Public defaults and state schema are unchanged.
- No automatic GitHub Actions run was enabled.

Reproduction commands:

```bash
uv run --locked python scripts/profile_train_step.py \
  --model gated_static --nodes 2048 --edge-multiplier 64 \
  --device cuda --warmup 2 --repeats 3 \
  --output artifacts/performance-refactor-20260724/profile.json

uv run --locked python scripts/run_hybrid_local_global_pdbbind.py \
  artifacts/performance-refactor-20260724/pdbbind-overfit.json \
  --device cuda --max-steps 3000 --budget-seconds 600 \
  --candidate gated_grouped
```
