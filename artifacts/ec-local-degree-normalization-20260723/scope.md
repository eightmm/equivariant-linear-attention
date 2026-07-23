# EC local receiver-degree normalization packet

## Decision

Determine whether the unnormalized receiver sum in edge-conditioned local
transport materially drives the observed high gradient-clipping frequency
without degrading the matched bounded QM9 validation screen.

The prediction unit is one QM9 molecule and the label is `gap` in eV. The
implementation intervention affects only the two local stages of the registered
three-layer LGL route. The global factorized attention stage, data, features,
optimizer, clipping threshold, target normalization, readout, and evaluation
remain fixed.

## Claims

- C1: an opt-in receiver-degree normalization is implemented exactly as the
  edge-conditioned non-self sum divided by the square root of the incoming
  non-self candidate count.
- C2: enabling the option preserves O(3), translation, node permutation,
  edge-order, batching/isolation, and finite-gradient contracts.
- C3: omitting or disabling the option preserves the prior state schema,
  initialization, and outputs exactly.
- C4: on the frozen strict-CUDA screen, normalization reduces clipping
  frequency by at least 0.05 absolute without increasing validation MAE by more
  than 0.020 eV.

## Falsifiers and acceptance

- C1 fails if a float64 explicit reference differs by more than `1e-10`, if
  self edges enter the degree, or if zero-degree receivers become nonfinite.
- C2 fails if any float64 symmetry/permutation/isolation error exceeds `1e-9`,
  edge-order error exceeds `1e-12`, or gradients are nonfinite.
- C3 fails if the default parameter schema/hash or output differs from the
  explicit disabled option.
- C4 fails if
  `candidate_clip_fraction > baseline_clip_fraction - 0.05` or
  `candidate_val_mae > baseline_val_mae + 0.020 eV`.

Passing C1-C3 admits the implementation as an opt-in experimental feature.
Passing C4 admits only a later multi-seed confirmation proposal; it does not
change the default.

## Frozen execution

Both full arms use:

- cached QM9 with `num_samples=130000`;
- train/validation sizes `110000/10000`, random-row split seed 42;
- train-only target normalization and no test evaluation;
- batch size 64, 500 updates, AdamW defaults from the matched harness;
- hidden width 64, three layers, four heads, LGL route, memory count 1;
- edge-conditioned local transport and precomputed 2.5-Angstrom candidates;
- model seed 42, FP32, strict deterministic CUDA.

The baseline uses the existing receiver sum. The candidate adds only the
receiver square-root-degree normalization. A two-step candidate CUDA smoke
precedes one fresh baseline process and one fresh candidate process.

## Resource and stop boundary

- Existing locked environment and cached data; no dependency or download.
- One local GPU.
- Expected wall time below five minutes and hard cumulative GPU ceiling of
  600 seconds.
- Stop on unsupported deterministic operation, nonfinite loss/gradient,
  identity drift, data/split mismatch, or ceiling exhaustion.
- Preserve negative or null results; do not tune the threshold or normalization
  after inspecting the screen.

## Non-goals

- No test-set evaluation.
- No multi-seed accuracy or generalization claim.
- No comparison with EGNN in this packet.
- No claim that clipping causes the historical accuracy gap.
- No change to the default architecture.
- No higher-order kernel, HEMM, coordinate update, or dependency change.
