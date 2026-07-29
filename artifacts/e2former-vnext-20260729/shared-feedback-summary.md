# Shared-feedback disposition

Source:
`https://chatgpt.com/share/6a68ca40-9730-83ee-8c49-1a122068d2af?ogimg=plain`

Accessed: 2026-07-29 KST

Cached response SHA-256:
`ebc484a186dc2c850a126c22f321c12dcaa9f7c08589d4086cebe87b76683c5f`

The shared discussion was used as design input, not as empirical evidence. Its
actionable proposals were checked against the current repository before
admission:

| Proposal | Disposition |
| --- | --- |
| Preserve exact edge-free global linear attention | Retained in every homogeneous block |
| Add sparse geometry without replacing global heads | Implemented as an opt-in additive rank-`R` local residual |
| Keep local work separable and edge-state-free | Implemented without persistent edge state or an edge-width hidden MLP |
| Make local refresh independent of global rank | Implemented as an explicit layer-index schedule |
| Expose the exact factorization as feature GEMM | Implemented as a second exact global reduction backend |
| Separate local and global balancing | Implemented with legacy inheritance for compatibility |
| Use packed receiver/reverse adjacency | Implemented with stable CSR and safe int32 metadata |
| Prepare for general irreps without slowing the current fast path | Implemented as static layout and tensor-product planning metadata |
| Add full arbitrary-`l` numerical execution now | Deferred; no registered production executor exists beyond the canonical Cartesian paths |
| Add Triton, ELL, auto-dispatch thresholds, or custom Wigner kernels now | Deferred until CUDA profiles identify an actual kernel bottleneck |
| Add pocket/ligand semantics to the core | Rejected; such semantics remain task adapters |

The resulting packet is deliberately narrower than the discussion: it
implements reusable 3D point-cloud/graph architecture mechanics while leaving
GPU kernel specialization and downstream accuracy promotion to separately
authorized experiments.
