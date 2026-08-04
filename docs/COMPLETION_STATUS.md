# ELA completion checklist

This page maps the final architecture/performance checklist to current source
and evidence. `Implemented` means the execution path exists and has focused CPU
contracts. It does not imply a CUDA speedup or downstream accuracy gain. Those
claims remain pending until the frozen GPU and real-data packet runs.

| # | Checklist item | Current implementation | Evidence state |
|---:|---|---|---|
| 1 | staging history and workflow cleanup | `main` is the only local branch; hosted Actions are manual-dispatch only | Complete repository-state check; no hosted CI run required |
| 2 | one working `ELA + ELAGraph` API | package root exports exactly `ELA` and `ELAGraph`; the package-root execution contract is `ELAGraph -> ELA -> ELAGraph` | Complete CPU public/API tests |
| 3 | import, wheel, public-surface tests | isolated wheel build/install/import is part of `scripts/check.sh fast` | Complete: frozen-source fast gate passed |
| 4 | end-to-end benchmark including ingestion | `scripts/benchmark_ela.py --include-end-to-end` separates cold radius, explicit-immutable trusted reuse, explicit COO ingestion, and moving coordinates | Complete CPU scope receipt; CUDA latency/memory pending |
| 5 | safe prepared cache and grouped fast path | ordinary graphs exactly revalidate aliasable content; `ELAGraph.assume_immutable()` clones topology storage before admitting an explicit E-independent version/schema path; graph counts enter grouped layout construction | CPU safe-default, in-place, NumPy, DLPack, clone, and trusted-admission tests complete; CUDA cache receipt pending |
| 6 | private numerical-core compile | `prepare_for_inference` compiles only `_execute_numerical`; validation, cache, radius, pooling, and output wrapping stay eager; recognized Dynamo/Inductor failures permanently fall back with a warning | CPU boundary/fallback tests complete; cold/warm/recompile/latency CUDA gate pending |
| 7 | batched GPU radius graph and direct CSR | many small grouped graphs use a bounded padded dense lane; larger grouped inputs use batched cell-list discovery; both emit receiver-major CSR directly and avoid a second explicit COO-to-CSR repack | CPU exact topology/selector contracts complete; CUDA topology/output/latency pending. One E-sized receiver-grouping sort remains where discovery order requires it |
| 8 | training local custom-autograd fusion | forced Triton fuses scalar, vector, l=2 tensor, and three directional/chiral local transports; backward recomputes from compact inputs and retains first/double backward | CPU reference/autograd contracts complete; full CUDA dispatch and gradient parity pending |
| 9 | ragged segmented global GEMM | BF16 CUDA inference can use two-pass `grouped_mm`; CPU, FP32, training, and double backward use the exact tiled segmented fallback | CPU fallback/integration tests complete; native grouped-MM numerical and speed gate pending |
| 10 | stagewise coordinate update | `update_positions=True` updates at every layer boundary, carries hidden state, refreshes invalid geometry between stages, and splits the displacement bound | CPU O(3), translation, permutation, mask, and double-backward tests complete; CUDA and QM9 functionality arm pending |
| 11 | relation-conditioned transport | invariant relation score, radial, and value modulation is identity-initialized and has a matched disable control | Synthetic paired mechanics reproduction complete; QM9 inapplicable, LBA attribution pending |
| 12 | explicit `1 x 2` CG closure | all parity-valid Cartesian vector/tensor closure outputs are implemented behind an identity-compatible lane | Synthetic rich-irrep mechanics reproduction complete; QM9/LBA attribution pending |
| 13 | multi-scale local lane | learned short/full-cutoff mixing shares one candidate CSR rather than constructing a second edge set | Synthetic same-CSR mechanics reproduction complete; QM9/LBA attribution pending |

## Frozen remaining evidence

The implementation is CPU-complete, but CUDA performance/memory and current
QM9/LBA receipts remain deliberately unexecuted while the workstation GPU is
occupied by another task. They are not release evidence and no speedup claim is
made from CPU tests.

The frozen packet contains five serialized local GPU/data jobs:

1. canonical GPU smoke and equivariance/autograd tests;
2. the alternating CUDA completion profiler;
3. a 100-update QM9 gap screen with the separate stagewise arm;
4. a 250-update, 16-complex LBA train-only capacity screen; and
5. a 220-update LBA ID30 train/validation screen.

Execution is split into three separately authorized phases by
`scripts/run_completion_packet.py`: `gpu` runs G1-G2 without network or data,
`data` runs G3-G5 only after source-bound G1/G2 receipts exist, and `finalize`
runs the CPU-only G6 adjudicator directly. There is intentionally no combined
`all` mode. The runner validates the frozen packet SHA-256 and canonical G1-G6
argv before any job is submitted, uses no shell, and stops before G2 when G1 is
nonzero, missing, malformed, or not source-bound. Ad-hoc pytest selection is not
part of this protocol. It snapshots each selected receipt before enqueue, then
requires a freshly rewritten G1-G5 terminal receipt with the expected status,
source, and command before submitting the next job. Thus queue-wrapper exit-code
behavior or a stale completed JSON cannot bypass the stop rule.

G1 itself runs through `scripts/run_gpu_gate.py`, which verifies the frozen
source bytes and always emits a real execution receipt instead of assuming a
zero exit code. G6 rejects missing, malformed, non-finite, source-mismatched,
over-budget, or incomplete receipts with exit 2.

The numerical, selected latency, and per-lane `<16 GiB` allocation gates are
recorded before execution in the run-local
`gpu-completion-plan-v2.json`. QM9/LBA runs are one-seed exploratory process and
attribution screens with no preregistered accuracy-win threshold. Null, slower,
and failed results are retained. The contaminated local LBA test holdout is not
resolved, opened, indexed, or evaluated.

The grouped-over-segmented and compiled-over-eager promotion ratios retain a
strict `<=1.0` median gate. This is deliberately zero-margin: scheduler noise or
a genuinely slower path may fail, and that failure is retained rather than
triggering an automatic rerun or a post-hoc threshold change.

## Promotion boundaries

- Triton is not automatically selected merely because the fused path exists.
  Promotion requires the recorded numerical gate and a measured winning regime.
- `grouped_mm` is an inference-only BF16 fast path; training remains exact
  segmented execution.
- Compiled execution is ordinary inference/first-order execution only where the
  backend succeeds. Force/Hessian or documented double-backward work stays eager.
- Synthetic ablations prove isolated wiring and learnability, not downstream
  utility or superiority.
- This project remains a general 3D point-cloud/sparse-graph architecture. QM9
  and LBA are validation harnesses, not the public model definition.
