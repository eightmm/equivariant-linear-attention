# HEMM counterfactual and repair result

Decision: `block_interacting_memory_arms`
Test labels evaluated: no

The clean incumbent reproduction confirms radial-coupling collapse. Identity coupling is numerically nonconstant, but the registered-width worst heads remain below the frozen material-activation threshold. The preregistered shared invariant MLP router increases assignment diversity in some seeds but is not robust: no residual candidate passes all aligned width/seed/M arms, and the semantic-only identity diagnostic also fails.

| Coupling | Aligned arms passed | min I_slot | min gate D | min message R_sym | min post R_sym | min gradient R_sym | min output RMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| identity | 1/12 | 3.437e-05 | 1.795e-03 | 2.230e-05 | 6.807e-06 | 7.414e-05 | 5.026e-07 |
| lambda_0.10 | 0/12 | 3.437e-05 | 4.349e-05 | 1.833e-06 | 5.683e-07 | 5.936e-06 | 4.154e-08 |
| lambda_0.25 | 0/12 | 3.437e-05 | 1.245e-04 | 4.785e-06 | 1.482e-06 | 1.554e-05 | 1.083e-07 |
| lambda_0.50 | 0/12 | 3.437e-05 | 3.279e-04 | 1.027e-05 | 3.172e-06 | 3.351e-05 | 2.316e-07 |

The fixed candidates are therefore rejected without changing lambda or any threshold. M=4/M=8 accuracy runs and memory-performance claims remain blocked. This does not block the independent ggg-versus-lgl backbone study.

Raw evidence: `stage0-suite.json` (compressed for publication). Compact machine-readable evidence: `stage0-summary.json`.
