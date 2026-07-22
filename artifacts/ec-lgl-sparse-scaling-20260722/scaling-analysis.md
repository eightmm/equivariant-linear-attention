# Scaling result interpretation

## Exact same kernel

The dense and factorized implementations evaluate the same finite normalized
kernel.  Across `N=32..4096` in float64, maximum absolute disagreement was
`2.4061e-15`.  At `N=4096`:

| implementation | median CUDA forward | peak CUDA delta | pair elements |
| --- | ---: | ---: | ---: |
| materialized dense | 3.3744 ms | 671,088,640 B | 16,777,216 |
| factorized | 0.7611 ms | 3,380,224 B | 0 |

The first measured runtime crossover was `N=4096`.  Over the last three
measured sizes, the dense runtime log-log slope was `1.5796`; the factorized
slope was `-0.0912`.  The negative finite-window slope should be interpreted as
GPU kernel/occupancy variation, not subconstant asymptotic work.  The source
formula and retained sufficient statistics establish linear fixed-width work;
the timings establish only this hardware-specific crossover.

## Full-model edge regimes

The timed model path used already content-validated synthetic edges. With
degree fixed at 16, EC-LGL changed from `5.1345 ms` at 32 nodes to
`5.5951 ms` at 4096 nodes (last-three-size slope `0.0396`). Static EGNN on the
same sparse edges remained faster in absolute time (`0.5987` to `1.4023 ms`),
so the new model does **not** beat EGNN when both receive the same bounded edge
set.

The intended systems crossover appears when EGNN retains a complete edge set
while EC-LGL caps local degree and retains global factorized transport.  At 512
nodes:

| system | local candidates | median CUDA forward | peak CUDA delta |
| --- | ---: | ---: | ---: |
| EC-LGL, degree 16 | 8,192 | 5.2430 ms | 11,706,368 B |
| static EGNN, complete | 262,144 | 5.5770 ms | 391,588,352 B |

This is the first measured descriptive crossover.  It is a systems and
inductive-bias comparison, not an identical-computation speedup.  At 512 nodes
and the same complete candidate set, EC-LGL was still slower (`6.6754` vs
`5.5609 ms`), although its peak delta was smaller (`350.2` vs `391.6 MB`).

## Decision

- Same-kernel factorization gate: **pass** (accuracy, no quadratic pair tensor,
  memory reduction and measured runtime crossover).
- Fixed-degree execution claim: **pass** for the tested precomputed-edge model
  path; neighbor construction remains excluded.
- Same-sparse-edge speed superiority over EGNN: **fail** on this hardware.
- Dense-edge-regime systems crossover: **observed at N=512**, with the explicit
  caveat that the models do different local work.
- Accuracy remains unestablished until the bounded QM9 screen.

Primary machine-readable records are `same-kernel-results.json` and
`scaling-results.json` in this run directory.
