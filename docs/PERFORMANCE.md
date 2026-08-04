# Performance

ELA separates graph preparation from tensor-only numerical work:

```text
ELAGraph validation and topology discovery
    -> private packed receiver-major CSR
    -> ELA numerical stack
    -> ELAGraph output
```

An `ELAGraph` may retain a private prepared topology. Explicit edges can be
reused after coordinates move because distances are recomputed. Automatic
radius topology records cutoff, neighbour cap, relation schema, skin, and
reference coordinates; incompatible or stale caches are rebuilt.

Automatic radius graphs exclude self edges. `max_neighbors=k` therefore means
up to `k` non-self distance shells, with all exact ties at the boundary kept to
preserve permutation equivariance.

PyTorch is the canonical backend. Triton is an explicit, memory-oriented
experimental backend; it is not selected automatically because recorded
complete-stack timings did not establish a stable latency gain. Static prepared
workloads may benefit from `torch.compile`, but force, HVP, and other
double-backward paths should remain eager until that contract is supported.

Benchmarks must state whether they include:

- graph ingestion and radius discovery;
- cold compilation;
- forward only, backward, or an optimizer step;
- static explicit topology, cold radius build, skin reuse, or moving-coordinate
  rebuild;
- latency percentiles and peak allocated memory.

Use `scripts/benchmark_ela.py` for the numerical stack and report graph-building
time separately when making end-to-end claims. Historical measurements and
their limitations are recorded in `docs/KERNEL_OPTIMIZATION.md` and
`docs/EXPERIMENTS.jsonl`.
