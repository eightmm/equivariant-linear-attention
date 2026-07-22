# Scaling-aware EC-LGL report

## Outcome

The implementation now supports an honest sparse-local plus factorized-global
execution path. Supplied receiver/sender edges survive loading, batching,
device transfer, training and both comparison models without forward-time
complete-pair discovery. The new per-local-layer edge MLP is permutation
consistent, O(3)-equivariant, translation invariant, graph isolated and active
in all four intended message branches under the tested contract.

The systems hypothesis passed in its narrow form. For the identical finite
global kernel, factorization agreed with dense materialization to `2.406e-15`,
removed the `N x N` pair tensor and crossed runtime at 4096 nodes while reducing
peak CUDA delta from 671.09 MB to 3.38 MB. With degree 16, full EC-LGL first
crossed complete-edge static EGNN at 512 nodes and used 11.71 MB rather than
391.59 MB. This is not a same-computation comparison: EC-LGL deliberately caps
local edges and retains factorized global context.

The constant-factor result is less favorable. Static EGNN was faster whenever
both models received the same sparse edges; at 4096 nodes it used 1.402 ms
versus EC-LGL at 5.595 ms. The current full attention block therefore becomes
useful only once avoided edge work is large enough to repay its fixed overhead.

The accuracy hypothesis was falsified. Repeated seed-42/500-step QM9 validation
means were 0.802194 eV for EC-LGL and 0.712178 eV for static LGL. The
`+0.090015 eV` regression exceeded the `+0.020 eV` admission ceiling, so the
five-seed confirmation and EGNN accuracy comparison were not run. Test labels
were not accessed and the feature remains opt-in.

## Interpretation

The result supports the user's core systems intuition: QM9 is too small and
sparse for linear global attention to amortize its launch/feature overhead,
whereas a complete-edge GNN eventually pays quadratic edge cost. It does not
support the stronger claim that this EC operator is already a better molecular
inductive bias or that EC-LGL is faster than a sparse GNN.

The strict model-forward complexity claim now refers to the trusted
content-validated path. Graph collation checks range, uniqueness and complete
self coverage before enabling it; direct model calls retain full validation.
The one-time validation and dense QM9 radius builder are excluded from
`O(E_local + N)`, so this is not an end-to-end neighbor-pipeline claim.

The candidate's 0.118016 eV repeat range and 92% clipping fraction are warning
signals. An unnormalized local sum is a plausible source of degree-dependent
magnitude and optimization sensitivity, but the current evidence is
diagnostic rather than causal. Protein, generic point-cloud, chirality, force,
dynamics and neighbor-construction claims remain outside the run.

## Decision

- Keep sparse edge plumbing and the reproducible scaling harness.
- Keep edge-conditioned local transport opt-in; do not promote it.
- Preserve the exact failed screen and stop decision on GitHub.
- Treat graph construction as a separate system component with its own cost.
- Test only a preregistered stability repair in a new packet; do not select a
  repair from these validation outcomes and report it as confirmatory.

The primary records are `same-kernel-results.json`, `scaling-results.json`,
`screen-summary.json`, `verification-summary.json`, and the four files under
`screen/`.
