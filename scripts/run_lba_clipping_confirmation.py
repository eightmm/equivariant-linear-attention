#!/usr/bin/env python3
"""Paired multi-seed ATOM3D-LBA confirmation of the relaxed clipping policy.

The frozen seed-44 screen found that removing global gradient clipping improved
last-epoch validation RMSE by 0.027843 pK, above its registered 0.020 pK
threshold. That is one-seed evidence and it ran on a candidate list whose
cross-run identity had drifted, so the default remains `grad_clip=1.0`.

This runner executes the registered confirmation: seeds 41--43, clip 1 versus no
clipping, one process, one shared and hashed topology, and thresholds fixed
before any outcome is inspected. Run `scripts/verify_lba_topology.py` first.
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import statistics
import time

import torch

from equivariant_attention.pdbbind import (
    ATOM3D_LBA_REVISION,
    load_atom3d_lba_split_samples,
    topology_sha256,
)
from equivariant_attention.reproducibility import configure_reproducibility
from equivariant_attention.training import fit_target_normalizer


RUN_ID = "lba-clipping-confirmation-20260727"
SEEDS = (41, 42, 43)
BASELINE_POLICY = "global_1"
CANDIDATE_POLICY = "none"
POLICIES: tuple[tuple[str, float | None], ...] = (
    (BASELINE_POLICY, 1.0),
    (CANDIDATE_POLICY, None),
)
MATCHED_EPOCHS = 20
MINIMUM_MEAN_IMPROVEMENT_PK = 0.020
MINIMUM_IMPROVING_SEEDS = 2
MAXIMUM_PAIRED_REGRESSION_PK = 0.050
MAXIMUM_LATENCY_RATIO = 1.05
MAXIMUM_MEMORY_RATIO = 1.05
TRAIN_LBA = runpy.run_path(str(Path(__file__).with_name("train_lba_id30.py")))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/atom3d_lba"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=MATCHED_EPOCHS)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--budget-seconds", type=float, default=5_400.0)
    parser.add_argument("--finalization-grace-seconds", type=float, default=60.0)
    parser.add_argument(
        "--expect-topology",
        default=None,
        help="required topology SHA-256; the run aborts before training on drift",
    )
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.epochs <= 0:
        parser.error("batch size and epochs must be positive")
    if not 0 <= args.warmup_epochs <= args.epochs:
        parser.error("--warmup-epochs must lie in [0, epochs]")
    if args.budget_seconds <= 0.0 or args.finalization_grace_seconds < 0.0:
        parser.error("budget must be positive and grace must be nonnegative")
    for name in ("train_limit", "val_limit"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def _plan(args: argparse.Namespace) -> dict[str, object]:
    limited = args.train_limit is not None or args.val_limit is not None
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "question": (
            "Does removing global gradient clipping improve fixed-budget "
            "ATOM3D-LBA ID30 validation RMSE across independent seeds?"
        ),
        "hypothesis": (
            "Unclipped training improves mean paired last-epoch validation RMSE "
            "by at least 0.020 pK with at least two of three paired wins."
        ),
        "prediction": (
            "The seed-44 screen effect is real but smaller across seeds, so the "
            "mean improvement lands near but possibly below 0.020 pK."
        ),
        "dataset": "vector-institute/atom3d-lba",
        "dataset_revision": ATOM3D_LBA_REVISION,
        "split": "official_ID30_train_validation",
        "model": "current squared-RBF gated-plus-grouped LGL candidate",
        "policies": {name: {"global_grad_clip": value} for name, value in POLICIES},
        "model_seeds": list(SEEDS),
        "order_seeds": list(SEEDS),
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "batch_size": args.batch_size,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 3e-4,
            "weight_decay": 0.01,
            "schedule": "linear warmup then cosine decay",
        },
        "determinism": "strict",
        "dtype": "float32",
        "primary_metric": "last-epoch validation RMSE in pK",
        "topology_contract": "deterministic float64 squared distance, ties retained",
        "acceptance": {
            "minimum_mean_improvement_pK": MINIMUM_MEAN_IMPROVEMENT_PK,
            "minimum_improving_seeds": MINIMUM_IMPROVING_SEEDS,
            "maximum_paired_regression_pK": MAXIMUM_PAIRED_REGRESSION_PK,
            "maximum_step_latency_ratio": MAXIMUM_LATENCY_RATIO,
            "maximum_peak_memory_ratio": MAXIMUM_MEMORY_RATIO,
        },
        "budget_seconds": args.budget_seconds,
        "finalization_grace_seconds": args.finalization_grace_seconds,
        "expected_topology_sha256": args.expect_topology,
        "train_limit": args.train_limit,
        "val_limit": args.val_limit,
        "claim_boundary": "smoke_only" if limited else "paired_three_seed",
        "validation_evaluated": True,
        "test_evaluated": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan = _plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    reproducibility = configure_reproducibility(seed=SEEDS[0], mode="strict")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "summary.json"
    started = time.perf_counter()
    result: dict[str, object] = {
        **plan,
        "status": "loading_data",
        "reproducibility": reproducibility,
        "runtime_environment": TRAIN_LBA["_runtime_environment"](device),
        "git": _git_provenance(),
        "source_sha256": _source_hash(),
        "arm_results": [],
    }
    _write_json(result_path, result)

    train_samples, val_samples = _load_splits(args)
    TRAIN_LBA["_validate_splits"](train_samples, val_samples)
    normalizer = fit_target_normalizer(train_samples)

    result["status"] = "building_topology"
    _write_json(result_path, result)
    train_samples = TRAIN_LBA["_with_matched_sparse_edges"](train_samples, split="train")
    val_samples = TRAIN_LBA["_with_matched_sparse_edges"](val_samples, split="val")
    observed_topology = topology_sha256([*train_samples, *val_samples])
    result["dataset_summary"] = {
        "train_size": len(train_samples),
        "validation_size": len(val_samples),
        "train_identity_sha256": TRAIN_LBA["_sample_identity_hash"](train_samples),
        "validation_identity_sha256": TRAIN_LBA["_sample_identity_hash"](val_samples),
        "topology_sha256": observed_topology,
        "candidate_edge_count": sum(
            int(sample.edge_index.shape[1])
            for sample in [*train_samples, *val_samples]
            if sample.edge_index is not None
        ),
        "target_normalizer": normalizer.as_dict(),
    }
    _write_json(result_path, result)
    if args.expect_topology is not None and observed_topology != args.expect_topology:
        result["status"] = "failed"
        result["failure"] = "topology identity differs from the expected hash"
        _write_json(result_path, result)
        raise RuntimeError(
            f"topology {observed_topology} does not match {args.expect_topology}"
        )

    remaining = (
        args.budget_seconds
        - (time.perf_counter() - started)
        - args.finalization_grace_seconds
    )
    if remaining <= 0.0:
        raise RuntimeError("budget exhausted before policy training")
    per_arm_budget = remaining / (len(SEEDS) * len(POLICIES))

    result["status"] = "training"
    _write_json(result_path, result)
    for seed in SEEDS:
        base_args = _train_args(args, seed=seed, budget_seconds=per_arm_budget)
        for policy, threshold in POLICIES:
            arm_args = copy.copy(base_args)
            arm_args.grad_clip = threshold
            model = TRAIN_LBA["_build_model"]("candidate", None, model_seed=seed)
            record = TRAIN_LBA["_train_arm"](
                arm="candidate",
                model=model,
                train_samples=train_samples,
                val_samples=val_samples,
                normalizer=normalizer,
                device=device,
                amp_dtype=None,
                args=arm_args,
                output_dir=args.output_dir / f"seed{seed}" / policy,
                budget_seconds=per_arm_budget,
            )
            record["policy"] = policy
            record["global_grad_clip"] = threshold
            record["model_seed"] = seed
            record["order_seed"] = seed
            _append(result, "arm_results", record)
            result["elapsed_seconds"] = time.perf_counter() - started
            _write_json(result_path, result)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    result["decision"] = decision(result["arm_results"])
    result["elapsed_seconds"] = time.perf_counter() - started
    result["status"] = (
        "completed"
        if all(
            isinstance(record, Mapping) and record.get("status") == "completed"
            for record in _records(result)
        )
        else "partial"
    )
    _write_json(result_path, result)
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    return 0 if result["status"] == "completed" else 1


def decision(records: object) -> dict[str, object]:
    """Apply the frozen paired promotion rule to the collected arm records."""
    if not isinstance(records, list):
        return {"passed": False, "reason": "arm results are missing"}
    paired: dict[int, dict[str, Mapping[str, object]]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        seed = int(record["model_seed"])
        paired.setdefault(seed, {})[str(record["policy"])] = record
    missing = [
        seed
        for seed in SEEDS
        if set(paired.get(seed, {})) != {BASELINE_POLICY, CANDIDATE_POLICY}
    ]
    if missing:
        return {"passed": False, "reason": f"incomplete seeds: {missing}"}

    improvements: list[float] = []
    latency_ratios: list[float] = []
    memory_ratios: list[float] = []
    per_seed: dict[str, object] = {}
    complete = True
    matched_updates = True
    for seed in SEEDS:
        baseline = paired[seed][BASELINE_POLICY]
        candidate = paired[seed][CANDIDATE_POLICY]
        baseline_rmse = _metric(baseline, "last_epoch_validation", "rmse_pK")
        candidate_rmse = _metric(candidate, "last_epoch_validation", "rmse_pK")
        improvement = baseline_rmse - candidate_rmse
        latency = float(candidate["step_latency_median_seconds"]) / float(
            baseline["step_latency_median_seconds"]
        )
        memory = _ratio(
            candidate.get("peak_cuda_memory_bytes"),
            baseline.get("peak_cuda_memory_bytes"),
        )
        improvements.append(improvement)
        latency_ratios.append(latency)
        if memory is not None:
            memory_ratios.append(memory)
        complete = complete and all(
            record.get("status") == "completed" and record.get("test_evaluated") is False
            for record in (baseline, candidate)
        )
        matched_updates = matched_updates and baseline.get("global_steps") == candidate.get(
            "global_steps"
        )
        per_seed[str(seed)] = {
            "baseline_last_validation_rmse_pK": baseline_rmse,
            "candidate_last_validation_rmse_pK": candidate_rmse,
            "improvement_pK": improvement,
            "step_latency_ratio": latency,
            "peak_memory_ratio": memory,
            "initial_state_matches": (
                baseline.get("initial_state_sha256")
                == candidate.get("initial_state_sha256")
            ),
        }

    mean_improvement = statistics.fmean(improvements)
    worst_regression = -min(improvements)
    criteria = {
        f"mean_improvement_at_least_{MINIMUM_MEAN_IMPROVEMENT_PK}_pK": (
            mean_improvement >= MINIMUM_MEAN_IMPROVEMENT_PK
        ),
        f"improving_seeds_at_least_{MINIMUM_IMPROVING_SEEDS}": (
            sum(1 for value in improvements if value > 0.0) >= MINIMUM_IMPROVING_SEEDS
        ),
        f"worst_regression_at_most_{MAXIMUM_PAIRED_REGRESSION_PK}_pK": (
            worst_regression <= MAXIMUM_PAIRED_REGRESSION_PK
        ),
        f"latency_ratio_at_most_{MAXIMUM_LATENCY_RATIO}": (
            max(latency_ratios) <= MAXIMUM_LATENCY_RATIO
        ),
        f"peak_memory_ratio_at_most_{MAXIMUM_MEMORY_RATIO}": (
            not memory_ratios or max(memory_ratios) <= MAXIMUM_MEMORY_RATIO
        ),
        "all_arms_completed_without_test_access": complete,
        "matched_update_counts": matched_updates,
        "finite_metrics": all(math.isfinite(value) for value in improvements),
    }
    passed = all(criteria.values())
    return {
        "per_seed": per_seed,
        "mean_improvement_pK": mean_improvement,
        "improving_seed_count": sum(1 for value in improvements if value > 0.0),
        "worst_paired_regression_pK": worst_regression,
        "criteria": criteria,
        "passed": passed,
        "default_change_authorized": passed,
        "evidence_grade": "paired_three_seed",
    }


def _load_splits(args: argparse.Namespace) -> tuple[list, list]:
    train_indices = None if args.train_limit is None else tuple(range(args.train_limit))
    val_indices = None if args.val_limit is None else tuple(range(args.val_limit))
    train_samples = load_atom3d_lba_split_samples(
        args.data_root,
        split="train",
        revision=ATOM3D_LBA_REVISION,
        indices=train_indices,
    )
    val_samples = load_atom3d_lba_split_samples(
        args.data_root,
        split="val",
        revision=ATOM3D_LBA_REVISION,
        indices=val_indices,
    )
    return train_samples, val_samples


def _train_args(
    args: argparse.Namespace,
    *,
    seed: int,
    budget_seconds: float,
) -> argparse.Namespace:
    train_args = TRAIN_LBA["parse_args"](
        [
            str(args.output_dir),
            "--arms",
            "candidate",
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
            "--max-epochs",
            str(args.epochs),
            "--min-epochs",
            str(args.epochs),
            "--patience",
            str(args.epochs),
            "--warmup-epochs",
            str(args.warmup_epochs),
            "--learning-rate",
            "0.0003",
            "--weight-decay",
            "0.01",
            "--model-seed",
            str(seed),
            "--order-seed",
            str(seed),
            "--amp-dtype",
            "none",
            "--budget-seconds",
            str(budget_seconds),
        ]
    )
    train_args.train_limit = args.train_limit
    train_args.val_limit = args.val_limit
    return train_args


def _records(result: Mapping[str, object]) -> list[Mapping[str, object]]:
    records = result.get("arm_results")
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, Mapping)]


def _append(result: dict[str, object], key: str, value: object) -> None:
    records = result.get(key)
    if not isinstance(records, list):
        records = []
        result[key] = records
    records.append(value)


def _metric(record: Mapping[str, object], group: str, metric: str) -> float:
    values = record.get(group)
    if not isinstance(values, Mapping):
        raise ValueError(f"missing metric group: {group}")
    return float(values[metric])


def _ratio(numerator: object, denominator: object) -> float | None:
    if numerator is None or denominator is None:
        return None
    denominator_value = float(denominator)
    if denominator_value <= 0.0:
        return None
    return float(numerator) / denominator_value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _git_provenance() -> dict[str, object]:
    return TRAIN_LBA["_git_provenance"]()


def _source_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(
        [
            Path(__file__).resolve(),
            Path(__file__).with_name("train_lba_id30.py").resolve(),
            Path(__file__).resolve().parents[1]
            / "src"
            / "equivariant_attention"
            / "pdbbind.py",
        ]
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
