# Bounded QM9 screen decision

Both EC-LGL and the static LGL incumbent used the current source tree, the same
seed-42 110k/10k random-row split, 500 updates, batch size 64, FP32, precomputed
2.5-Angstrom radius candidates, and no test evaluation.  Each arm was launched
twice from an identical hashed initial state.

| arm | repetition 1 | repetition 2 | mean validation MAE |
| --- | ---: | ---: | ---: |
| EC-LGL | 0.743185 | 0.861202 | 0.802194 eV |
| static LGL | 0.708419 | 0.715938 | 0.712178 eV |

The candidate regressed by `0.090015 eV`, exceeding the preregistered
`0.020 eV` admission ceiling.  Its trainable-parameter ratio to static EGNN was
`1.04256`, which passed the `1.05` ceiling, and all correctness gates passed.
Accuracy therefore vetoed confirmation.

The EC-LGL repeat range (`0.118016 eV`) was much larger than the incumbent range
(`0.007518 eV`).  Both EC runs clipped gradients on 92% of updates.  Together
with the frozen unnormalized local `sum`, this suggests optimization
sensitivity and degree-dependent message magnitude, but the screen alone does
not prove that causal diagnosis.  A future packet should test a preregistered
bounded normalization or zero-initialized EC residual against this exact
failure, rather than selecting a repair after inspecting these labels.

Per-arm machine-readable metrics are under `screen/`; aggregate values and the
stop decision are in `screen-summary.json`.
