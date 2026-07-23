# Scientific run: reproducibility-hardening-20260723

**Question:** Can the matched training harness expose and record strict deterministic controls and reject near-threshold accuracy claims when identical same-seed runs exceed a frozen noise-floor gate?
**Review:** passed

## Plan

- **completed:** P1
- **completed:** P2
- **completed:** P3
- **completed:** P4
- **completed:** P5
- **completed:** P6

## Claims

- **C1:** The matched training harness exposes explicit seeded and strict determinism modes and records their effective runtime state. — status=supported; inference=software contract on the tested CPU environment; supports=1
  - Uncertainty: Strict CUDA operator support is not tested
  - Next action: Run the separately approved strict CUDA gate
- **C2:** The repeat summarizer rejects identity drift and reports finite metric spread, sample standard deviation, final-state uniqueness, and a machine-readable verdict. — status=supported; inference=software contract; supports=1
  - Uncertainty: The 0.005 eV threshold is a preregistered decision rule, not a universal statistical law
  - Next action: Apply it to five fresh QM9 CUDA processes
- **C3:** Five strict CPU synthetic runs produced identical validation MAE and one final-state hash. — status=supported; inference=descriptive CPU reference-lane repeatability only; supports=1
  - Uncertainty: Does not establish CUDA, QM9, mixed-precision, or cross-hardware determinism
  - Next action: Do not transfer this conclusion to CUDA without measurement

## Evidence graph

- Nodes: 5; edges: 3; supports=3
- Graph file: [evidence-graph.json](evidence-graph.json)

## Visual results

No raster image artifacts recorded.

## Files

- [scope.md](scope.md) — protocol; SHA-256 `c04f863995d30b1878d760ddd0199d8b5f71157527dc8547d689da96c61476b3`
- [claim-register.json](claim-register.json) — claim-register; SHA-256 `5555f7f23b84089cfe65284461757d4f8f48d37e8d2cea5b6537e6a32e893a37`
- [evidence-graph.json](evidence-graph.json) — evidence-graph; SHA-256 `8c81e6a20b14ad741aa54c8f350377d3b17501c54b8f7acb093054e974dace38`
- [execution-record.json](execution-record.json) — execution-log; SHA-256 `0ebf57f6e0758b12fcc278c478c38e8d52440d00b7c236fa9ada0e1f9b60dbf9`
- [compute-environment.json](compute-environment.json) — environment; SHA-256 `90d158ecc68ba5bfb83c7bda9ece2459de4b42d72b70112500c21ee5b2391746`
- [strict-cpu-smoke.json](strict-cpu-smoke.json) — metrics; SHA-256 `a41df884912ab2a3a0836f22acee1c73795f6323e0d7109b23354ba5269e7039`
- [strict-cpu-repeat-2.json](strict-cpu-repeat-2.json) — metrics; SHA-256 `a78c243a2a88a65cb604fb71adad1a8e21b0ae02ff18524d2bcfe47b119b850a`
- [strict-cpu-repeat-3.json](strict-cpu-repeat-3.json) — metrics; SHA-256 `c0e4aaf45c609f9e961643dadb6c73dc3f9f07fd5988adc0280eb823a747d4a8`
- [strict-cpu-repeat-4.json](strict-cpu-repeat-4.json) — metrics; SHA-256 `3130c676095e5830e694ee6fcca1aadf89819728d0d37a4b8fbb9e309e2a5624`
- [strict-cpu-repeat-5.json](strict-cpu-repeat-5.json) — metrics; SHA-256 `5ace0948978db9d57cddf673771c3ab2767456347ab6c454ab0d8fe608475728`
- [strict-cpu-repeat-summary.json](strict-cpu-repeat-summary.json) — metrics; SHA-256 `630caab5346172da2334a7ddc058f7ab33c98e6db67ff56334ed3a5f5ba64780`
- [reference-use-ledger.json](reference-use-ledger.json) — decision-log; SHA-256 `83dfaab0b027746a9211c121d3ab9b287ec2f4e9173ed000b870a940f41a680f`
- [report.md](report.md) — report; SHA-256 `9cf6e57b424e631aff11538be89c60580131ccde83831ac7a537ba05d05c1568`
- [reviewer-response-initial.json](reviewer-response-initial.json) — decision-log; SHA-256 `cd7bf3c81ec35235031254ea799082eaf6853bba9e89624e90c57b0d2a1fa3c6`
- [review-task.json](review-task.json) — decision-log; SHA-256 `17486e4306d6f8371c3d89a7e151eafcd82fcae05413f56389a3b10110e7da90`
- [reviewer-response.json](reviewer-response.json) — decision-log; SHA-256 `53a5656bf2bd4c73cb588fb76ed7f691e8ecadbdca20652bcb6fdf2ab19f7d78`
- [review-receipt-v2.json](review-receipt-v2.json) — review-receipt-v2; SHA-256 `8e3b5e3002226c6e872e4aa15c259e0bc49b059787e4ffd87a2e731282b8509a`

_Generated from `manifest.json` and validated hashed sidecars; this index is a derived view, not evidence._
