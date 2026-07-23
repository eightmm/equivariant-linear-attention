# Scientific run: ec-local-degree-normalization-20260723

**Question:** Does opt-in square-root receiver-degree normalization materially reduce EC-LGL gradient clipping without degrading a matched bounded QM9 validation screen?
**Review:** passed

## Plan

- **completed:** P1
- **completed:** P2
- **completed:** P3
- **completed:** P4
- **completed:** P5
- **completed:** P6

## Claims

- **C1:** The opt-in edge-conditioned local path divides every non-self receiver message sum by the square root of incoming candidate degree. — status=supported; inference=implemented operator contract; supports=1
  - Uncertainty: Candidate degree is not cutoff-mass normalization
  - Next action: Keep the equation and CLI provenance stable
- **C2:** The normalized option preserves the tested O(3), translation, permutation, edge-order, batch-isolation, and finite-gradient contracts. — status=supported; inference=software correctness on the tested float64 boundary; supports=2
  - Uncertainty: Tests do not prove correctness for every graph or dtype
  - Next action: Retain the tests as regression gates
- **C3:** The option is disabled by default and explicit disabling preserves the previous state schema, initialization, and outputs exactly. — status=supported; inference=backward-compatible default software contract; supports=3
  - Uncertainty: No historical checkpoint migration was needed because no parameter was added
  - Next action: Do not change the default from this packet
- **C4:** The normalized candidate reduces clipping by at least 0.05 absolute without more than 0.020 eV validation regression. — status=unsupported; inference=single-seed bounded diagnostic screen; contradicts=1
  - Uncertainty: One seed does not estimate accuracy or clipping variability across initialization
  - Next action: Reject clipping promotion; profile parameter-group gradients before another repair

## Evidence graph

- Nodes: 9; edges: 9; contradicts=1, supports=8
- Graph file: [evidence-graph.json](evidence-graph.json)

## Visual results

No raster image artifacts recorded.

## Files

- [scope.md](scope.md) — protocol; SHA-256 `65b8d15ee18fd98d9c95355b0026afc3cf510a19d457280cce92b059af156b15`
- [compute-environment.json](compute-environment.json) — environment; SHA-256 `b9f2ad2c8f50572f3bc559d8914b74131f1874394f10a4312a77945ba9c545a3`
- [candidate-smoke.json](candidate-smoke.json) — metrics; SHA-256 `a6926de8ea0458ada1e567c6ddc4c6c1d3eb5af1ee7f3f0d470a9c63f142a5ef`
- [baseline-sum.json](baseline-sum.json) — metrics; SHA-256 `d7026be35f33236e73a128075c346c3c734778f38f4ab35a200b90019f34246b`
- [candidate-sqrt-degree.json](candidate-sqrt-degree.json) — metrics; SHA-256 `10652105d18613ca931ed6d7a114e9baf438896dfab8e240d01ff4f0bbdf9416`
- [screen-summary.json](screen-summary.json) — metrics; SHA-256 `682a2363a566e34b8f1fb3cef57060c23718766e13a2a4e027cddebaa13c5050`
- [claim-register.json](claim-register.json) — claim-register; SHA-256 `a9f7986422020462e5ce1f9d043737c8f50493b5cf347b37aa79d804cde19030`
- [evidence-graph.json](evidence-graph.json) — evidence-graph; SHA-256 `bb6a88d1127823050df88248485dc29cab308fd2e42286362dec2242e504e540`
- [execution-record.json](execution-record.json) — execution-log; SHA-256 `ecf077604e82825c9baaecf52ef86451cb7226e599fa9b1ccce792bbafaa0ff9`
- [reference-use-ledger.json](reference-use-ledger.json) — decision-log; SHA-256 `618f13a89de5e5c69ab4fac9daea2b85ee6cc6e4bce6ebcfb794f9c16e1d7b3a`
- [report.md](report.md) — report; SHA-256 `6d14373706e423544923721eccf5ea7a4734b170bf26c1bd858ab899c10d7c52`
- [review-task.json](review-task.json) — review-task; SHA-256 `6301699ed3bfe67b02d36a69360e99617da739c29e8136f273d6321b18120ce7`
- [reviewer-response.json](reviewer-response.json) — review-response; SHA-256 `af2eebbbc84c0612f458ec5add830fd4a970b11bf8e588d636977704e8050849`
- [review-receipt-v2.json](review-receipt-v2.json) — review-receipt; SHA-256 `dd390952a0efe786c335945232ef0da284bbb571c58df501320918c694058286`

_Generated from `manifest.json` and validated hashed sidecars; this index is a derived view, not evidence._
