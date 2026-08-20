# TriELA experiment and evidence contract

## Status

As of the pair-centric architecture migration, there is no registered QM9,
LBA, PSR, promoted runtime, or promoted peak-memory result produced by
canonical TriELA. CPU implementation evidence through E3 is recorded below;
it is not external-task or systems-promotion evidence.

All retained measurements from earlier revisions are **historical pre-TriELA
results**. They used different state, relation, local, and execution contracts
and cannot validate the current model. In particular, they cannot establish:

- accuracy of persistent dense pair memory;
- the contribution of outgoing or incoming triangle multiplication;
- runtime or memory of exact dense triangle blocks;
- equivariance of the pair adapters;
- the scaling boundary of the canonical stage;
- parity with any published biomolecular model.

The tables below are requirements and empty result slots, not claims of
completed experiments.

## 1. Evidence levels

| Level | Required evidence | Current status |
|---|---|---|
| E0: import | canonical public symbols import and construct | verified on CPU |
| E1: algebra | exact triangle/mask/layout references | verified on CPU |
| E2: symmetry | O(3), translation, permutation, directed pair tests | verified on CPU/eval |
| E3: numerics | FP32/BF16 forward/backward and second derivatives | verified on CPU |
| E4: systems | reproducible time and peak-memory benchmark | entry points smoke-tested; not promoted |
| E5: learning | controlled component ablations | mechanism smoke only; not promoted |
| E6: external task | fresh QM9/LBA/PSR or another declared task | not run |

No statement at one level may be promoted to a stronger level. Passing an
equivariance test does not imply useful learning; a one-batch overfit does not
establish validation accuracy; a forward-only benchmark does not establish
training efficiency.

### 1.1 CPU implementation snapshot (2026-08-20)

The source revision is the repository commit containing this document. The
following checks were run from its code tree without a GPU:

```text
uv run bash scripts/check.sh fast
uv run pytest -q --cov=equivariant_linear_attention
uv build
```

Observed implementation evidence:

- 76 tests passed; statement coverage was 89.47%;
- exact FP32/FP64 outgoing and incoming triangle references passed;
- first and second derivatives passed, including zero-motion diagnostics;
- CPU BF16 autocast forward/backward passed with coordinate updates off/on;
- O(3), reflection, translation, graph/group isolation, ordinary permutation,
  and coincident-cutoff permutation regressions passed;
- wheel contents imported with the six-symbol canonical public surface;
- tiny CPU runs of `benchmark_triangle.py`, `benchmark_tri_ela.py`, and
  `ablate_tri_ela.py` completed.

The two-step synthetic intervention smoke produced nonzero `graph_x` changes
for every requested removal: pair FFN `2.43e-4`, outgoing `1.96e-4`, incoming
`2.35e-4`, pair-to-node `1.39e-2`, global `4.34e-2`, and local `9.19e-3`.
These values only prove that the hooks bind to live mechanisms after a tiny
fit; they are not accuracy estimates or component promotion evidence.

Not run: CUDA, GPU peak memory, full scaling grids, multi-seed learning, QM9,
LBA, PSR, or comparison with an external baseline. PyTorch emitted one
environment warning because NumPy is not installed; the package itself does
not require NumPy and all checks completed.

## 2. Source and run provenance

Every recorded run must include:

```text
source revision and dirty-tree status
complete TriELAConfig
dataset name/version and manifest hash
split definition and split hash
feature construction and target normalization
seed and determinism mode
dtype and autocast policy
device model, software stack, and available VRAM/RAM
optimizer, schedule, batch construction, and update count
checkpoint selection rule
wall time, failures, and early-stop reason
raw result artifact path and artifact hash
```

Results without source, split, and configuration identity are diagnostics, not
promotion evidence. Test-set access must be separated from architecture and
hyperparameter selection.

## 3. Correctness matrix

### 3.1 Exact triangle algebra

For small random tensors, compare module outputs with direct references:

```python
out_ref = torch.einsum("bikc,bjkc->bijc", a, b)
in_ref = torch.einsum("bkjc,bkic->bijc", a, b)
```

Required cases:

| Case | Dtypes | Required observation |
|---|---|---|
| outgoing, complete mask | FP64/FP32 | reference agreement |
| incoming, complete mask | FP64/FP32 | reference agreement |
| irregular mask | FP64/FP32 | invalid operands cannot contribute |
| padded batch | FP64/FP32 | padded outputs exactly zero |
| zero-init module | FP64/FP32 | residual begins as no-op |
| first derivative | FP64 | finite, reference agreement |
| second derivative | FP64 | finite |
| gradcheck | FP64, tiny shape | pass |

FP32 target tolerance:

```text
rtol <= 1e-5
atol <= 1e-6
```

Tolerance changes require a recorded numerical reason, not an unexplained test
relaxation.

### 3.2 Pair layout and isolation

| Test | Requirement |
|---|---|
| pack/unpack | exact round trip for variable-size segments |
| transpose | swaps ordered endpoints and layout metadata consistently |
| graph isolation | perturbing one sample cannot change another |
| group isolation | perturbing one group cannot change another |
| chain behavior | differing chain IDs remain interactable |
| padding | values and residuals remain exact zero |
| token guard | allocation above `max_pair_tokens` fails before materialization |
| permutation | both pair axes follow the node permutation |
| directed state | asymmetric `Z_ij` is not silently symmetrized |

### 3.3 Symmetry

Use random proper rotations, reflections, translations, and node permutations.
Test each boundary independently:

```text
PairEmbedding                 invariant
NodeGeometryToPair            invariant
TrianglePairBlock             invariant
PairToNodeSummary             invariant
PairContextInjection          declared-irrep equivariant
GlobalELA                     declared-irrep equivariant
PairConditionedLocalELA       declared-irrep equivariant
coordinate output             polar-vector equivariant
distogram logits              invariant and symmetric only at head
```

Report absolute and relative error per sector and dtype. A near-zero output is
not sufficient evidence; wake zero-initialized projections before testing the
transformation law.

### 3.4 Mixed precision and stability

Required executions:

- FP32 forward and backward;
- CPU BF16 autocast forward and backward;
- GPU BF16 autocast when an unoccupied GPU is explicitly available;
- finite gate, normalization, denominator, pair, node, and coordinate values;
- all-invalid padded rows without NaN or Inf;
- activation checkpointing compatibility if the production model enables it.

Peak gradient norm and clipping fraction should be logged in learning runs.
Clipping must not be treated as a substitute for locating unstable branches.

## 4. Exact-triangle systems benchmark

The standalone exact triangle benchmark must cover:

```text
N  = 64, 128, 256, 384, 512
Cz = 32, 64, 128
```

For both outgoing and incoming modules, record:

| Metric | Forward | Forward + backward |
|---|---:|---:|
| median steady-state time | required | required |
| p10/p90 time or dispersion | required | required |
| peak allocated memory | required | required |
| peak reserved memory | required | required |
| examples or valid pairs per second | required | required |
| compile warm-up | excluded and reported | excluded and reported |

Compare eager and `torch.compile` only when both execute the same exact
operator. Compare activation checkpointing on/off only as a memory/recompute
trade-off; it does not change the mathematical backend.

The report must include log-log slopes and the theoretical references:

```text
pair activation memory: O(N^2 Cz)
exact triangle compute: O(N^3 Ch)
```

Do not label the complete model linear because the Global ELA submodule is
linear in node count.

## 5. End-to-end systems benchmark

The model benchmark must include graph ingestion, pair layout construction,
pair embedding, all stages, output heads, and backward. Excluding dense-layout
or local-support preparation is allowed only as a separately labelled kernel
microbenchmark.

Required configurations:

```text
tokens per sample = 64, 128, 256, 384, 512
pair width        = 32, 64, 128
batching          = uniform and ragged bucketed
coordinate update = off and on
dtype             = FP32 and BF16 where supported
mode              = inference and train step
```

Required reporting:

- parameter count;
- valid and padded pair counts;
- batch padding ratio;
- forward and train-step latency;
- peak allocated/reserved memory;
- throughput;
- graph/pair preparation time;
- local-support rebuild time when coordinates update;
- optimizer-state inclusion or exclusion;
- exact failure boundary under the token guard or available memory.

## 6. Component ablation matrix

Ablations are isolated research harnesses, not alternate public model routes.
They must share data, features, splits, optimizer, update count, parameter
budget as closely as possible, and source revision.

| Arm | Pair state | Out TriMul | In TriMul | Pair->Node | Global ELA | Pair-local ELA |
|---|---:|---:|---:|---:|---:|---:|
| A0 | yes | no | no | no | no | no |
| A1 | yes | yes | no | no | no | no |
| A2 | yes | no | yes | no | no | no |
| A3 | yes | yes | yes | no | no | no |
| A4 | yes | yes | yes | yes | no | no |
| A5 | yes | yes | yes | yes | yes | no |
| A6 canonical | yes | yes | yes | yes | yes | yes |

`A0` still uses pair embedding and Pair FFN, so it isolates triangle reasoning
without resurrecting the removed legacy architecture.

Additional controlled comparisons:

| Question | Arm 1 | Arm 2 |
|---|---|---|
| pair refresh frequency | once per stage | once per block |
| pair-to-node frequency | every block | every second block |
| directed memory | asymmetric latent | forced symmetric diagnostic |
| triangle directions | outgoing only | incoming only |
| coordinate refinement | off | local-only update |
| pair conditioning | pair context only global | global + local |
| distogram supervision | off | fixed declared loss weight |

The forced-symmetric arm is a diagnostic only; the canonical latent remains
ordered.

For each arm, report both accuracy and systems metrics. A component that
improves accuracy but exceeds the declared memory envelope must be described as
that trade-off, not promoted on accuracy alone.

## 7. Learning checks before full benchmarks

### 7.1 Synthetic mechanism tasks

Use small tasks whose target requires the mechanism under test:

- directed relation composition for incoming/outgoing distinction;
- a three-body closure target that pair FFN alone cannot solve;
- cross-chain interaction with different chain IDs but one interaction group;
- O(3)-invariant pair prediction and equivariant vector prediction;
- local coordinate refinement with a known equivariant displacement.

Required sequence:

```text
single batch overfit
held-out synthetic examples
component ablation
permutation/O(3) re-evaluation after training
```

### 7.2 Real-data entry gate

Before a full real-data run, require:

- E1 through E3 complete;
- one successful small-batch overfit;
- memory estimate below the device budget;
- fixed data and split manifest;
- fixed evaluation metric and checkpoint-selection rule;
- no GPU contention with another job.

## 8. Fresh task matrix

The following are candidate validation tasks, not completed evidence:

| Task | Why it is useful | Required comparison | Status |
|---|---|---|---|
| QM9 property | small molecular sanity check | same features/split/training budget | not run on TriELA |
| ATOM3D LBA | protein-ligand pair reasoning | same tokenization and task head | not run on TriELA |
| ATOM3D PSR | global and interface structure ranking | official split and published metrics | not run on TriELA |
| directed synthetic closure | isolates ordered triangle memory | component ablation A0-A6 | not run |

QM9 is too small to establish the large biomolecular value proposition by
itself. LBA or PSR can test pair memory more directly, but only a fresh run from
the canonical source is admissible. Feature identity across compared models is
preferred so the observed difference is attributable to architecture.

## 9. Result record template

Every promoted experiment should add a row like:

| Field | Value |
|---|---|
| source revision | pending |
| dirty tree | pending |
| config hash | pending |
| data manifest | pending |
| split hash | pending |
| seed(s) | pending |
| updates/epochs | pending |
| selected checkpoint | pending |
| validation metric | pending |
| test metric | pending |
| forward/train latency | pending |
| peak memory | pending |
| determinism result | pending |
| artifact hash | pending |
| limitations | pending |

Do not fill missing fields with inferred values.

## 10. Promotion gates

### Correctness promotion

- all exact reference and masking checks pass;
- every declared symmetry check passes on nontrivial outputs;
- first and second derivatives are finite;
- FP32 and CPU BF16 complete without NaN or Inf.

### Systems promotion

- preparation-inclusive and kernel-only numbers are separated;
- forward and backward are both measured;
- warm-up and synchronization policy are recorded;
- memory includes an explicit definition of what is counted;
- observed scaling is consistent with dense pair and exact triangle theory.

### Learning promotion

- architecture selection does not use the final test set;
- at least three preregistered seeds are reported for a promoted comparison;
- confidence interval or seed dispersion is shown;
- accuracy and compute are compared under a shared harness;
- failed and inconclusive arms remain in the ledger.

## 11. Prohibited conclusions

Until fresh canonical runs exist, do not claim that TriELA:

- matches or beats EGNN, Equiformer, SE(3)-Transformer, AF3, Protenix,
  PairMixer, or an ATOM3D published baseline;
- reproduces prior QM9, LBA, or PSR metrics;
- is faster or more memory-efficient than the pre-TriELA model;
- scales linearly as a complete architecture;
- is validated for full protein all-atom dense pair modeling;
- benefits from pair attention, chunking, low-rank factors, or sparse global
  pair memory.

Those are experiment questions. The current architectural claim is narrower:
the canonical model implements exact ordered dense triangle memory coupled to
equivariant global and local node processing under explicit symmetry, masking,
and complexity contracts.
