# Reproducibility hardening packet

## Decision

Establish an executable same-seed noise floor before interpreting another
sub-`0.02 eV` accuracy intervention. This packet changes measurement
infrastructure only. It does not rerun QM9 on CUDA, change an architecture,
open test labels, or promote a default model.

## Claims and falsifiers

- C1: `seeded` preserves the historical nondeterministic execution policy while
  recording its effective state. Falsifier: the default enables strict
  deterministic algorithms or omits the state from result JSON.
- C2: `strict` requests deterministic PyTorch/cuDNN/cuBLAS behavior before model
  construction and CUDA execution. Falsifier: any requested control is false,
  warn-only behavior is enabled, or an invalid existing cuBLAS configuration is
  silently accepted.
- C3: the repeat gate compares only identical source/config/data/split/initial
  state/model-seed runs and reports finite mean, sample standard deviation,
  range, final-state uniqueness, and a machine-readable verdict. Falsifier:
  identity drift is accepted or a nonfinite/missing metric enters the summary.

## Acceptance

- Unit tests cover seeded/strict state, invalid identity, numeric drift, strict
  final-state mismatch, and the CLI artifact.
- One strict CPU synthetic train/eval smoke records a deterministic runtime
  state and completes with finite metrics.
- `scripts/check.sh fast` passes with at least 80% package coverage.
- The future registered seeded CUDA gate uses five fresh processes and accepts
  validation-MAE range at most `0.005 eV`, half the smallest historical
  `0.01 eV` promotion effect.
- The future strict CUDA gate additionally requires identical final-state
  hashes and metric values. An unsupported deterministic operator is a failed
  strict lane.

## Resource boundary

This implementation packet uses CPU tests and one bounded synthetic CPU smoke.
It downloads nothing, changes no dependency, and executes no GPU workload.
The five-run QM9 CUDA measurement requires a separate compute approval packet.
