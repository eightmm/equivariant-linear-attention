# Next bounded experiment (proposal only)

The current packet is closed scientifically after its failed accuracy screen.
A separate confirmed packet may test one frozen composite stability repair:

```text
m_local = m_incumbent
        + alpha_l / sqrt(max(1, degree_i)) * m_edge_conditioned,
alpha_l initialized to 0 independently in each local layer.
```

This preserves the incumbent first forward, retains `O(E_local)` work, bounds
the new branch's degree scaling, and lets the loss select its sign/magnitude.
It costs the legacy local kernel plus the edge MLP, so both same-edge latency
and dense-edge crossover must be remeasured; a predictive gain cannot be
traded silently for an unreported systems regression.

The next packet should reuse the exact data/split, two-run seed-42/500 screen,
parameter ceiling, no-test policy and 0.020 eV admission rule. If admitted,
confirmation must compare the candidate, incumbent LGL and static EGNN at
seeds 41--45. A positive result belongs to the whole composite; normalization
and staged residual effects require later matched ablations.

This proposal is not implemented or authorized by the completed run.
