# Scientific run: qm9-strict-cuda-repeatability-20260723

**Question:** Does the current static LGL QM9 lane produce an exactly repeatable validation metric and final model state in five fresh strict CUDA processes?
**Review:** passed

## Plan

- **completed:** P1
- **completed:** P2
- **completed:** P3
- **completed:** P4
- **completed:** P5
- **completed:** P6

## Claims

- **C1:** The current static LGL FP32 QM9 path completes under strict deterministic CUDA controls on the recorded RTX PRO 6000 without fallback. — status=supported; inference=operator support for the exact recorded source, configuration, and environment; supports=2
  - Uncertainty: Other architectures, dtypes, hardware, drivers, and distributed execution are untested
  - Next action: Retain strict mode for matched architecture comparisons
- **C2:** Five fresh 500-update strict CUDA processes produced identical validation MAE and one canonical final-state hash. — status=supported; inference=same-seed bitwise repeatability for the exact recorded lane; supports=2
  - Uncertainty: One seed and one random-row split do not estimate multi-seed accuracy variance
  - Next action: Use this lane as the deterministic basis for bounded matched comparisons
- **C3:** The registered 0.005 eV same-seed noise threshold passes with measured validation-MAE span and sample standard deviation both equal to zero. — status=supported; inference=descriptive runtime noise result for this exact lane; supports=1
  - Uncertainty: The threshold is a decision rule, not a universal statistical guarantee
  - Next action: Do not transfer the zero-noise conclusion across configurations without measurement
- **C4:** Gradient clipping remains active on 456 of 500 updates in every recorded process. — status=supported; inference=diagnostic evidence of persistent high clipping frequency; supports=1
  - Uncertainty: This does not establish clipping as the cause of the accuracy gap
  - Next action: Test an opt-in receiver-degree normalization with clipping and validation gates

## Evidence graph

- Nodes: 7; edges: 6; supports=6
- Graph file: [evidence-graph.json](evidence-graph.json)

## Visual results

No raster image artifacts recorded.

## Files

- [scope.md](scope.md) — protocol; SHA-256 `37c66f31198e1cd8ed7ed613de281dccb08ea8048a22a1dc411cea7c2bc5cc6c`
- [claim-register.json](claim-register.json) — claim-register; SHA-256 `74941aadaeab900b5d3694d909a3208ae936969a47c759b3e3e285786eba851a`
- [evidence-graph.json](evidence-graph.json) — evidence-graph; SHA-256 `a524a840e524c6893e9445e288ccc41b9828c16516f4988a799d3e9dd8154e84`
- [execution-record.json](execution-record.json) — execution-log; SHA-256 `f695e72e30ab1aba3937f4c83cfe85740cca00634491d595fa9314eb9040187b`
- [compute-environment.json](compute-environment.json) — environment; SHA-256 `c16ff7aea5846f138a7cc5b71f31f8cd28634cbdeb655cba01d377c3fcdc5fef`
- [strict-cuda-smoke.json](strict-cuda-smoke.json) — metrics; SHA-256 `5583514e54c8d86152522ab369420b715e52bdd0e71b57827857feccd9401750`
- [strict-run-1.json](strict-run-1.json) — metrics; SHA-256 `1df348d12bd60f6ce566d120e44b53dd373758730c9619170be5deb67b89766f`
- [strict-run-2.json](strict-run-2.json) — metrics; SHA-256 `189b2dcdb7d13180ac846ec3d69cea9f15f13d1c34d7c8911264a40cc5e23f20`
- [strict-run-3.json](strict-run-3.json) — metrics; SHA-256 `d0caa088a664cb224a65c6ec127ee86bd4b3d6255c71a00a9a18d09fe73ca52d`
- [strict-run-4.json](strict-run-4.json) — metrics; SHA-256 `443db5b60a7486d0b4134de3bc3e14cf5e8757b176549670dc130bfc595ecfc0`
- [strict-run-5.json](strict-run-5.json) — metrics; SHA-256 `2f60554b6177f0b5716ac0667f759fbc5f264ac3dbda6bb481cd4b1fe1111d61`
- [strict-summary.json](strict-summary.json) — metrics; SHA-256 `77f0e602f4aa6773bb76a13a2f2fbe9cafdd644968c9a5b90e292d7734e6eba2`
- [reference-use-ledger.json](reference-use-ledger.json) — decision-log; SHA-256 `e043579af1acaa739cef0741ee0355b29f708fa9b5f30cac7582a6a883313804`
- [report.md](report.md) — report; SHA-256 `939d78e183e57a5eb6812b25fb41b7455c7dc1a5c098e31d40b0f93ce9a32163`
- [review-task.json](review-task.json) — decision-log; SHA-256 `20dba179cf3acb0d3e807b94e22e02de70c138067936e180f575d676a0bb66e8`
- [reviewer-response.json](reviewer-response.json) — decision-log; SHA-256 `e2921e458e136ae5d0b0c36c4c914f9bb17ae51a41d43af08a31e21d9f4df468`
- [review-receipt-v2.json](review-receipt-v2.json) — review-receipt-v2; SHA-256 `5c42499f39c00f587d2bc2a72537d92dee5a7e1994bc02e20bd377c082c473a0`

_Generated from `manifest.json` and validated hashed sidecars; this index is a derived view, not evidence._
