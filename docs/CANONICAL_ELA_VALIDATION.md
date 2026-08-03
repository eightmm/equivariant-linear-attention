# Canonical ELA validation

Validation is explicit; automated push/PR CI is disabled.

## Smoke suite

```bash
ELA_SUITE_MODE=smoke \
  bash scripts/run_canonical_ela_suite.sh \
  artifacts/canonical-ela/smoke
```

Smoke mode runs focused CPU contracts and a small kernel-backend benchmark.

## Full suite

```bash
ELA_SUITE_MODE=full \
ELA_SUITE_DEVICE=cuda \
ELA_SUITE_DTYPE=bfloat16 \
  bash scripts/run_canonical_ela_suite.sh \
  artifacts/canonical-ela/full-$(date +%Y%m%d-%H%M%S)
```

Full mode additionally runs:

```text
scripts/check.sh fast
scripts/check.sh gpu
```

and a larger same-state PyTorch/Triton benchmark. Full mode requires a clean
worktree.

## Public API contracts

The suite checks:

- package root exposes one backbone `ELA`, one architecture layer `ELALayer`, and
  one graph container `ELABatch`;
- input and output representations are configured only with irreps;
- no `node_dim`, `output_dim`, or scalar-only model factory is exposed;
- `ELA` accepts one `ELABatch`;
- `ELA.batch`, `ELA.padded`, and `ELA.collate` normalize external data into the
  same packed representation;
- prepared execution matches ordinary `model(batch)`;
- graph mean and graph sum readouts are consistent.

## Architecture contracts

The suite checks:

- zero-initialized branch fusion reproduces exact `G + L`;
- learned fusion remains proper/improper O(3)-equivariant;
- router and branch-balance parameters receive gradients;
- minimal config derives heads and local rank deterministically;
- all supported input/output sectors transform correctly;
- forward, feature gradients, and coordinate gradients are finite;
- graph isolation, sparse-edge-order invariance, node permutation, and required
  double backward hold;
- condition and semantic-order PE are neutral at initialization;
- context-free calls bypass trained optional conditioners;
- coordinate refinement is identity initialized, bounded, masked, centered, and
  equivariant;
- migration from compatible historical checkpoints fails closed.

## Data and graph contracts

The suite checks:

- `ptr` is the canonical packed graph membership;
- edges cannot cross graph boundaries;
- mapping samples collate without PyG;
- graph-local edge indices are offset correctly;
- targets and sample IDs survive collation;
- padded tensors pack and restore correctly;
- dense and cell-list radius paths match a dense exact reference;
- radius candidates honor graph isolation, self-edge policy, and
  maximum-neighbor limits.

## Kernel contracts

The PyTorch path is the numerical reference. Focused tests cover:

- CSR sum forward and first/second derivatives;
- packed multi-payload reduction;
- backend fail-closed behavior;
- CUDA FP32 PyTorch/Triton output and gradient agreement;
- CUDA BF16 finiteness;
- native CUDA BF16 full-model equivalence;
- parity-complete forced-Triton O(3), relation, permutation, and refinement;
- full ELA feature, coordinate, every-local-parameter, and force-HVP agreement.

Run backend tests directly:

```bash
uv run pytest -q tests/test_kernel_triton.py
uv run pytest -q tests/test_kernel_triton_cuda.py
uv run pytest -q tests/test_triton_equivariance_cuda.py
```

## Kernel benchmark

```bash
uv run python scripts/benchmark_ela.py \
  --input-irreps "32x0e" \
  --output-irreps "1x0e" \
  --nodes 4096 \
  --degree 32 \
  --width 128 \
  --depth 8 \
  --device cuda \
  --dtype bfloat16 \
  --output artifacts/ela-kernels.json
```

The benchmark uses one prepared `ELABatch`, one model state, and separately
records:

- output error;
- input-feature gradient error;
- coordinate-gradient error;
- local-parameter-gradient error;
- inference latency and peak allocated memory;
- forward/backward latency and peak allocated memory;
- exclusion of neighbor discovery.

Benchmark results are execution evidence, not downstream accuracy evidence.

## Artifact layout

```text
artifacts/canonical-ela/<run-id>/
  kernels.json
  environment.txt
  git.txt
  manifest.json
  focused-tests.log
  benchmark.log
  cuda-focused.log # CUDA
  fast-gate.log    # full
  gpu-gate.log     # full CUDA
```

The manifest records the git SHA, dtype, device, public API contract, backend
profiles, and hashes of required artifacts.
