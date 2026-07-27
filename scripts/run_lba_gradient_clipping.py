#!/usr/bin/env python3
"""Run a matched ATOM3D-LBA gradient-clipping policy screen."""

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
import subprocess
import sys
import time

import torch

from equivariant_attention.pdbbind import (
    ATOM3D_LBA_REVISION,
    load_atom3d_lba_split_samples,
)
from equivariant_attention.reproducibility import configure_reproducibility
from equivariant_attention.training import fit_target_normalizer


RUN_ID = "lba-gradient-clipping-20260727"
MODEL_SEED = 44
ORDER_SEED = 44
POLICIES: tuple[tuple[str, float | None], ...] = (
    ("global_1", 1.0),
    ("global_10", 10.0),
    ("none", None),
)
TRAIN_LBA = runpy.run_path(str(Path(__file__).with_name("train_lba_id30.py")))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/atom3d_lba"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--model-seed", type=int, default=MODEL_SEED)
    parser.add_argument("--order-seed", type=int, default=ORDER_SEED)
    parser.add_argument("--budget-seconds", type=float, default=900.0)
    parser.add_argument("--finalization-grace-seconds", type=float, default=45.0)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.epochs <= 0:
        parser.error("batch size and epochs must be positive")
    if not 0 <= args.warmup_epochs <= args.epochs:
        parser.error("--warmup-epochs must lie in [0, epochs]")
    if args.model_seed < 0 or args.order_seed < 0:
        parser.error("seeds must be nonnegative")
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
            "Does relaxing the candidate's global gradient clipping improve "
            "fixed-budget ATOM3D-LBA ID30 validation RMSE?"
        ),
        "hypothesis": (
            "At least one relaxed policy improves last-epoch validation RMSE "
            "by at least 0.020 pK while remaining finite and resource matched."
        ),
        "prediction": (
            "Relaxation changes effective gradient scale but does not reach "
            "the 0.020 pK accuracy threshold."
        ),
        "dataset": "vector-institute/atom3d-lba",
        "dataset_revision": ATOM3D_LBA_REVISION,
        "split": "official_ID30_train_validation",
        "policies": {
            name: {"global_grad_clip": threshold}
            for name, threshold in POLICIES
        },
        "model": "current squared-RBF gated-plus-grouped LGL candidate",
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 3e-4,
            "weight_decay": 0.01,
            "schedule": "linear warmup then cosine decay",
        },
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "batch_size": args.batch_size,
        "model_seed": args.model_seed,
        "order_seed": args.order_seed,
        "determinism": "strict",
        "dtype": "float32",
        "primary_metric": "last-epoch validation RMSE in pK",
        "acceptance": {
            "last_rmse_improvement_pK": 0.020,
            "maximum_best_rmse_regression_pK": 0.010,
            "minimum_effective_scale_gain": 0.20,
            "maximum_step_latency_ratio": 1.05,
            "maximum_peak_memory_ratio": 1.05,
        },
        "budget_seconds": args.budget_seconds,
        "finalization_grace_seconds": args.finalization_grace_seconds,
        "train_limit": args.train_limit,
        "val_limit": args.val_limit,
        "claim_boundary": "smoke_only" if limited else "exploratory_one_seed",
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
    reproducibility = configure_reproducibility(
        seed=args.model_seed,
        mode="strict",
    )
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
        "policy_results": [],
        "validation_evaluated": True,
        "test_evaluated": False,
    }
    _write_json(result_path, result)

    train_indices = (
        None if args.train_limit is None else tuple(range(args.train_limit))
    )
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
    TRAIN_LBA["_validate_splits"](train_samples, val_samples)
    normalizer = fit_target_normalizer(train_samples)
    result["status"] = "building_topology"
    _write_json(result_path, result)
    train_samples = TRAIN_LBA["_with_matched_sparse_edges"](
        train_samples,
        split="train",
    )
    val_samples = TRAIN_LBA["_with_matched_sparse_edges"](
        val_samples,
        split="val",
    )
    result["dataset_summary"] = {
        "train_size": len(train_samples),
        "validation_size": len(val_samples),
        "train_identity_sha256": TRAIN_LBA["_sample_identity_hash"](
            train_samples
        ),
        "validation_identity_sha256": TRAIN_LBA["_sample_identity_hash"](
            val_samples
        ),
        "topology_sha256": TRAIN_LBA["_topology_hash"](
            [*train_samples, *val_samples]
        ),
        "target_normalizer": normalizer.as_dict(),
        "candidate_edge_count": sum(
            int(sample.edge_index.shape[1])
            for sample in [*train_samples, *val_samples]
            if sample.edge_index is not None
        ),
    }

    remaining = (
        args.budget_seconds
        - (time.perf_counter() - started)
        - args.finalization_grace_seconds
    )
    if remaining <= 0.0:
        raise RuntimeError("budget exhausted before policy training")
    per_policy_budget = remaining / len(POLICIES)
    base_train_args = TRAIN_LBA["parse_args"](
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
            str(args.model_seed),
            "--order-seed",
            str(args.order_seed),
            "--amp-dtype",
            "none",
            "--budget-seconds",
            str(per_policy_budget),
        ]
    )
    base_train_args.train_limit = args.train_limit
    base_train_args.val_limit = args.val_limit
    initial_hashes: set[str] = set()
    result["status"] = "training"
    _write_json(result_path, result)
    for policy, threshold in POLICIES:
        policy_args = copy.copy(base_train_args)
        policy_args.grad_clip = threshold
        model = TRAIN_LBA["_build_model"](
            "candidate",
            None,
            model_seed=args.model_seed,
        )
        record = TRAIN_LBA["_train_arm"](
            arm="candidate",
            model=model,
            train_samples=train_samples,
            val_samples=val_samples,
            normalizer=normalizer,
            device=device,
            amp_dtype=None,
            args=policy_args,
            output_dir=args.output_dir / policy,
            budget_seconds=per_policy_budget,
        )
        record["policy"] = policy
        record["global_grad_clip"] = threshold
        result["policy_results"].append(record)
        initial_hashes.add(str(record["initial_state_sha256"]))
        result["elapsed_seconds"] = time.perf_counter() - started
        _write_json(result_path, result)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result["initial_state_hashes_match"] = len(initial_hashes) == 1
    result["decision"] = _decision(result["policy_results"])
    result["elapsed_seconds"] = time.perf_counter() - started
    result["status"] = (
        "completed"
        if all(
            isinstance(record, Mapping)
            and record.get("status") == "completed"
            for record in result["policy_results"]
        )
        else "partial"
    )
    _write_json(result_path, result)
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    return 0 if result["status"] == "completed" else 1


def _decision(records: object) -> dict[str, object]:
    if not isinstance(records, list):
        return {"passed": False, "reason": "policy results are missing"}
    by_policy = {
        str(record["policy"]): record
        for record in records
        if isinstance(record, Mapping)
    }
    baseline = by_policy.get("global_1")
    if baseline is None:
        return {"passed": False, "reason": "global_1 baseline is missing"}
    baseline_last = _metric(baseline, "last_epoch_validation", "rmse_pK")
    baseline_best = _metric(baseline, "best_validation", "rmse_pK")
    baseline_scale = _monitor_value(
        baseline,
        "effective_grad_scale_mean",
    )
    baseline_latency = float(baseline["step_latency_median_seconds"])
    baseline_memory = baseline.get("peak_cuda_memory_bytes")
    alternatives: dict[str, object] = {}
    admitted: list[tuple[float, str]] = []
    for policy in ("global_10", "none"):
        record = by_policy.get(policy)
        if record is None:
            alternatives[policy] = {"passed": False, "reason": "missing"}
            continue
        last_improvement = baseline_last - _metric(
            record,
            "last_epoch_validation",
            "rmse_pK",
        )
        best_regression = (
            _metric(record, "best_validation", "rmse_pK") - baseline_best
        )
        scale_gain = (
            _monitor_value(record, "effective_grad_scale_mean")
            - baseline_scale
        )
        latency_ratio = (
            float(record["step_latency_median_seconds"]) / baseline_latency
        )
        memory_ratio = _ratio(
            record.get("peak_cuda_memory_bytes"),
            baseline_memory,
        )
        finite_monitor = all(
            math.isfinite(float(value))
            for value in record.get("gradient_monitor", {}).values()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        )
        criteria = {
            "last_rmse_improvement_at_least_0.020_pK": (
                last_improvement >= 0.020
            ),
            "best_rmse_regression_at_most_0.010_pK": (
                best_regression <= 0.010
            ),
            "effective_scale_gain_at_least_0.20": scale_gain >= 0.20,
            "step_latency_ratio_at_most_1.05": latency_ratio <= 1.05,
            "peak_memory_ratio_at_most_1.05": (
                memory_ratio is None or memory_ratio <= 1.05
            ),
            "completed_and_finite": (
                record.get("status") == "completed" and finite_monitor
            ),
            "update_count_matches": (
                record.get("global_steps") == baseline.get("global_steps")
            ),
            "initial_state_matches": (
                record.get("initial_state_sha256")
                == baseline.get("initial_state_sha256")
            ),
            "test_not_evaluated": record.get("test_evaluated") is False,
        }
        passed = all(criteria.values())
        alternatives[policy] = {
            "last_rmse_improvement_pK": last_improvement,
            "best_rmse_regression_pK": best_regression,
            "effective_scale_gain": scale_gain,
            "step_latency_ratio": latency_ratio,
            "peak_memory_ratio": memory_ratio,
            "criteria": criteria,
            "passed": passed,
        }
        if passed:
            admitted.append((last_improvement, policy))
    selected = max(admitted)[1] if admitted else None
    return {
        "baseline_last_validation_rmse_pK": baseline_last,
        "baseline_best_validation_rmse_pK": baseline_best,
        "baseline_effective_grad_scale_mean": baseline_scale,
        "alternatives": alternatives,
        "selected_policy": selected,
        "passed": selected is not None,
        "default_change_authorized": False,
        "evidence_grade": "exploratory_one_seed",
    }


def _metric(
    record: Mapping[str, object],
    group: str,
    metric: str,
) -> float:
    values = record.get(group)
    if not isinstance(values, Mapping):
        raise ValueError(f"missing metric group: {group}")
    return float(values[metric])


def _monitor_value(record: Mapping[str, object], key: str) -> float:
    monitor = record.get("gradient_monitor")
    if not isinstance(monitor, Mapping):
        raise ValueError("gradient monitor is missing")
    return float(monitor[key])


def _ratio(numerator: object, denominator: object) -> float | None:
    if numerator is None or denominator is None:
        return None
    denominator_value = float(denominator)
    if denominator_value <= 0.0:
        raise ValueError("ratio denominator must be positive")
    return float(numerator) / denominator_value


def _git_provenance() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    diff = subprocess.run(
        ["git", "diff", "--no-ext-diff", "--binary"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    return {
        "commit": commit,
        "dirty": bool(status),
        "dirty_path_count": len(status.splitlines()),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _source_hash() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("train_lba_id30.py"),
        root / "src" / "equivariant_attention" / "moment.py",
        root / "src" / "equivariant_attention" / "pdbbind.py",
        root / "src" / "equivariant_attention" / "training.py",
        root / "uv.lock",
    ]
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"LBA clipping screen failed: {error}", file=sys.stderr)
        raise
