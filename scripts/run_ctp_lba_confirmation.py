#!/usr/bin/env python3
"""Run the frozen three-seed CTP-LGL ID30 validation confirmation."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import runpy
import statistics
import time

import torch

from equivariant_attention.pdbbind import (
    ATOM3D_LBA_REVISION,
    load_atom3d_lba_split_samples,
)
from equivariant_attention.reproducibility import configure_reproducibility
from equivariant_attention.training import fit_target_normalizer


TRAIN = runpy.run_path(str(Path(__file__).with_name("train_lba_id30.py")))
SEEDS = (41, 42, 43)
ARMS = ("candidate", "persistent_2e", "ctp")
EPOCHS = 35
BATCH_SIZE = 24


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/atom3d_lba"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--budget-seconds", type=float, default=2_800.0)
    parser.add_argument(
        "--resource-profile",
        type=Path,
        default=Path(
            "artifacts/ctp-lgl-20260727/"
            "lba-resource-profile-isolated-final.json"
        ),
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.budget_seconds) or args.budget_seconds <= 0.0:
        parser.error("--budget-seconds must be positive and finite")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    device = torch.device(args.device)
    configure_reproducibility(seed=SEEDS[0], mode="strict")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    resource_profile = json.loads(args.resource_profile.read_text())
    resource_gate = resource_profile["comparison"]["resource_gate"]
    if not resource_gate.get("passed"):
        raise RuntimeError("the frozen CTP resource gate did not pass")

    train_samples = load_atom3d_lba_split_samples(
        args.data_root,
        split="train",
        revision=ATOM3D_LBA_REVISION,
    )
    val_samples = load_atom3d_lba_split_samples(
        args.data_root,
        split="val",
        revision=ATOM3D_LBA_REVISION,
    )
    TRAIN["_validate_splits"](train_samples, val_samples)
    normalizer = fit_target_normalizer(train_samples)
    train_samples = TRAIN["_with_matched_sparse_edges"](
        train_samples,
        split="train",
    )
    val_samples = TRAIN["_with_matched_sparse_edges"](
        val_samples,
        split="val",
    )
    topology_sha256 = TRAIN["_topology_hash"]([*train_samples, *val_samples])
    edge_count = sum(
        int(sample.edge_index.shape[1])
        for sample in [*train_samples, *val_samples]
        if sample.edge_index is not None
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "summary.json"
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "training",
        "dataset": "vector-institute/atom3d-lba",
        "dataset_revision": ATOM3D_LBA_REVISION,
        "split": "official_ID30_train_validation",
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "topology_sha256": topology_sha256,
        "topology_edge_count": edge_count,
        "topology_materializations": 1,
        "topology_reused_for_every_run": True,
        "train_identity_sha256": TRAIN["_sample_identity_hash"](train_samples),
        "validation_identity_sha256": TRAIN["_sample_identity_hash"](val_samples),
        "resource_profile": str(args.resource_profile),
        "resource_gate": resource_gate,
        "seed_results": [],
        "budget_seconds": args.budget_seconds,
        "validation_evaluated": True,
        "test_evaluated": False,
    }
    _write_json(result_path, result)

    total_runs = len(SEEDS) * len(ARMS)
    completed_runs = 0
    for seed in SEEDS:
        configure_reproducibility(seed=seed, mode="strict")
        seed_dir = args.output_dir / f"seed{seed}"
        train_args = TRAIN["parse_args"](
            [
                str(seed_dir),
                "--data-root",
                str(args.data_root),
                "--device",
                str(device),
                "--arms",
                *ARMS,
                "--batch-size",
                str(BATCH_SIZE),
                "--max-epochs",
                str(EPOCHS),
                "--min-epochs",
                str(EPOCHS),
                "--patience",
                str(EPOCHS),
                "--warmup-epochs",
                "5",
                "--grad-clip",
                "1.0",
                "--model-seed",
                str(seed),
                "--order-seed",
                str(seed),
                "--budget-seconds",
                str(args.budget_seconds),
            ]
        )
        seed_result: dict[str, object] = {
            "model_seed": seed,
            "order_seed": seed,
            "arm_results": [],
            "validation_evaluated": True,
            "test_evaluated": False,
        }
        for arm in ARMS:
            remaining = args.budget_seconds - (time.perf_counter() - started)
            remaining_runs = total_runs - completed_runs
            if remaining <= 0.0:
                seed_result["arm_results"].append(
                    {"arm": arm, "status": "not_run_total_budget_exhausted"}
                )
                completed_runs += 1
                continue
            arm_budget = remaining / remaining_runs
            model = TRAIN["_build_model"](
                arm,
                None,
                model_seed=seed,
            )
            arm_result = TRAIN["_train_arm"](
                arm=arm,
                model=model,
                train_samples=train_samples,
                val_samples=val_samples,
                normalizer=normalizer,
                device=device,
                amp_dtype=None,
                args=train_args,
                output_dir=seed_dir / arm,
                budget_seconds=arm_budget,
            )
            seed_result["arm_results"].append(arm_result)
            completed_runs += 1
            result["elapsed_seconds"] = time.perf_counter() - started
            _write_json(seed_dir / "result.json", seed_result)
            _write_json(result_path, result)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        result["seed_results"].append(seed_result)
        result["decision"] = _promotion_decision(
            result["seed_results"],
            resource_gate_passed=bool(resource_gate["passed"]),
        )
        _write_json(result_path, result)

    decision = _promotion_decision(
        result["seed_results"],
        resource_gate_passed=bool(resource_gate["passed"]),
    )
    result["decision"] = decision
    result["elapsed_seconds"] = time.perf_counter() - started
    result["status"] = (
        "completed"
        if decision["all_registered_runs_completed"]
        else "partial"
    )
    _write_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == "completed" else 1


def _promotion_decision(
    seed_results: object,
    *,
    resource_gate_passed: bool,
) -> dict[str, object]:
    if not isinstance(seed_results, list):
        raise TypeError("seed_results must be a list")
    paired: list[dict[str, float | int]] = []
    all_completed = len(seed_results) == len(SEEDS)
    for seed_result in seed_results:
        if not isinstance(seed_result, Mapping):
            all_completed = False
            continue
        by_arm = {
            str(record["arm"]): record
            for record in seed_result.get("arm_results", [])
            if isinstance(record, Mapping)
        }
        if set(by_arm) != set(ARMS) or any(
            record.get("status") != "completed" for record in by_arm.values()
        ):
            all_completed = False
            continue
        rmse = {
            arm: float(by_arm[arm]["best_validation"]["rmse_pK"])
            for arm in ARMS
        }
        paired.append(
            {
                "seed": int(seed_result["model_seed"]),
                "candidate_rmse_pK": rmse["candidate"],
                "persistent_2e_rmse_pK": rmse["persistent_2e"],
                "ctp_rmse_pK": rmse["ctp"],
                "ctp_improvement_vs_candidate_pK": (
                    rmse["candidate"] - rmse["ctp"]
                ),
                "ctp_improvement_vs_persistent_2e_pK": (
                    rmse["persistent_2e"] - rmse["ctp"]
                ),
            }
        )
    candidate_improvements = [
        float(record["ctp_improvement_vs_candidate_pK"]) for record in paired
    ]
    persistent_improvements = [
        float(record["ctp_improvement_vs_persistent_2e_pK"])
        for record in paired
    ]
    mean_candidate = (
        statistics.fmean(candidate_improvements)
        if candidate_improvements
        else None
    )
    mean_persistent = (
        statistics.fmean(persistent_improvements)
        if persistent_improvements
        else None
    )
    worst = (
        min([*candidate_improvements, *persistent_improvements])
        if paired
        else None
    )
    win_count = sum(value > 0.0 for value in candidate_improvements)
    criteria = {
        "all_registered_runs_completed": all_completed,
        "resource_gate_passed": resource_gate_passed,
        "mean_improvement_vs_candidate_at_least_0.020_pK": (
            mean_candidate is not None
            and math.isfinite(mean_candidate)
            and mean_candidate >= 0.020
        ),
        "mean_improvement_vs_persistent_2e_positive": (
            mean_persistent is not None
            and math.isfinite(mean_persistent)
            and mean_persistent > 0.0
        ),
        "paired_win_count_vs_candidate_at_least_2_of_3": win_count >= 2,
        "worst_paired_improvement_at_least_minus_0.050_pK": (
            worst is not None and math.isfinite(worst) and worst >= -0.050
        ),
    }
    return {
        "paired_results": paired,
        "mean_improvement_vs_candidate_pK": mean_candidate,
        "mean_improvement_vs_persistent_2e_pK": mean_persistent,
        "paired_win_count_vs_candidate": win_count,
        "worst_paired_improvement_pK": worst,
        "all_registered_runs_completed": all_completed,
        "criteria": criteria,
        "passed": all(criteria.values()),
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
