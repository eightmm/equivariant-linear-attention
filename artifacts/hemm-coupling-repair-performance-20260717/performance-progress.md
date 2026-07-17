# Performance progress

## CUDA protocol

- Device: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- PyTorch: 2.12.1+cu130
- Eager FP32, synchronized CUDA
- 64 graphs, 18 and 29 nodes per graph
- 20 warm-up and 50 measured iterations
- Five fresh processes per route and shape
- Statistic: median of process-level mean latency; median peak allocated memory
- Arms: `ggg`, M=1 versus `lgl`, M=1

## Unpinned preliminary diagnostic

The first run (`cuda-benchmarks.json`) failed the 20% runtime
ceiling. LGL forward/backward was 21.742% slower at 18 nodes and 22.098% slower
at 29 nodes. Forward-only latency was approximately 2.1x the GGG control. This
file did not record the model source hash or git state, so it is retained as a
preliminary diagnostic rather than causal before/after evidence.

Static inspection identified the graph-by-graph candidate construction in
`_local_geometry`: it launched `nonzero`, distance, mask, and indexing kernels
once per graph. The replacement uses a stable batch sort and one vectorized
receiver-major Cartesian expansion. Focused dense, gradient, permutation,
unsorted-batch, singleton, self-edge, and cutoff tests preserve semantics.

## Pinned final-source result

The five-process rerun (`cuda-benchmarks-vectorized.json`) passed every 20%
ceiling:

| nodes | pass | GGG ms | LGL ms | latency change | memory change |
|---:|---|---:|---:|---:|---:|
| 18 | forward | 6.499 | 6.039 | -7.078% | +2.395% |
| 18 | forward/backward | 21.196 | 18.286 | -13.729% | -18.758% |
| 29 | forward | 6.528 | 5.961 | -8.686% | +6.312% |
| 29 | forward/backward | 21.336 | 18.177 | -14.806% | -6.112% |

This establishes eligibility for the preregistered validation-only QM9
comparison. It does not establish an accuracy improvement and does not admit
the Stage-0-blocked interacting M=4/M=8 arms.

## Verification before implementation commit

- `scripts/check.sh fast`: 235 passed, 89.20% coverage, CPU smoke passed.
- `scripts/check.sh gpu`: bf16 and FP32 CUDA smoke passed.
- Test labels were not evaluated.
