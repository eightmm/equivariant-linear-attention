# Documentation map

Use this page to distinguish the current ELA contract from retained research
history. The package root and [`PROJECT.md`](../PROJECT.md) remain authoritative
when a historical report describes an older model or command.

## Start here

| Topic | Document |
|---|---|
| Install and validation | [`SETUP.md`](SETUP.md) |
| Model and layer contract | [`CANONICAL_ELA.md`](CANONICAL_ELA.md) |
| Public Python API | [`API_POLICY.md`](API_POLICY.md) |
| `ELABatch` and data ingestion | [`DATA_API.md`](DATA_API.md) |
| Current layer equations | [`CANONICAL_ELA.md`, Sections 3--5](CANONICAL_ELA.md#3-global-and-local-branches) |
| Checkpoint format | [`CHECKPOINTS.md`](CHECKPOINTS.md) |
| Migration from pre-ELA code | [`MIGRATION_TO_ELA.md`](MIGRATION_TO_ELA.md) |

## Execution and evidence

| Topic | Document |
|---|---|
| Kernel policy and measured backend results | [`KERNEL_OPTIMIZATION.md`](KERNEL_OPTIMIZATION.md) |
| Complexity and scaling boundaries | [`SCALING.md`](SCALING.md) |
| Canonical validation matrix | [`CANONICAL_ELA_VALIDATION.md`](CANONICAL_ELA_VALIDATION.md) |
| Runtime logging ownership | [`LOGGING.md`](LOGGING.md) |
| Experiment-record policy | [`EXPERIMENTS.md`](EXPERIMENTS.md) |
| General troubleshooting notes | [`DEBUGGING.md`](DEBUGGING.md) |
| Release history | [`CHANGELOG.md`](CHANGELOG.md) |

## Historical research record

Date-stamped architecture studies and documents explicitly headed
**Historical** preserve the decisions that led to the current model. They are
evidence, not current API documentation. Their imports, scripts, and commands
may require the Git revision recorded with the result.

This category includes the retired [`MODEL.md`](MODEL.md),
[`DATA.md`](DATA.md), [`TRAINING.md`](TRAINING.md),
[`CONFIGS.md`](CONFIGS.md), [`BENCHMARKS.md`](BENCHMARKS.md), and
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) contracts, plus the retired branch
equations retained in [`LAYER_MATH.md`](LAYER_MATH.md).

[`EXPERIMENTS.jsonl`](EXPERIMENTS.jsonl) is the append-only machine ledger. In
particular, the QM9 and LBA reports document historical architecture selection;
they do not make either dataset part of the ELA core.
