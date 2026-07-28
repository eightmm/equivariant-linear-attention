#!/usr/bin/env python3
"""Run the frozen two-arm QM9 safety smoke for the whitened global read.

This is a bounded numerical/domain safety check, not an accuracy admission
experiment.  Candidate and whitened arms share data, split, initialization,
features, local topology, optimizer, update count, and strict determinism.  The
test split is never evaluated.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time


RUN_ID = "whitened-qm9-safety-20260728"
ARMS = ("candidate", "whitened_ridge_0p1")
MODEL_SEED = 42
STEPS = 500
MAXIMUM_ABSOLUTE_MAE_DELTA_EV = 0.020
MAXIMUM_RESOURCE_RATIO = 1.25


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--budget-seconds", type=float, default=900.0)
    parser.add_argument(
        "--rank-gated-schema-control",
        action="store_true",
        help=(
            "compare the 2F finite-sample gate against a frozen-mix control "
            "with the identical whitening schema and autograd path"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.steps <= STEPS:
        parser.error(f"--steps must lie in [1, {STEPS}]")
    if args.budget_seconds <= 0.0:
        parser.error("--budget-seconds must be positive")
    return args


def build_command(
    arm: str,
    *,
    output: Path,
    steps: int,
    device: str,
    rank_gated_schema_control: bool = False,
) -> list[str]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    command = [
        "uv",
        "run",
        "--locked",
        "python",
        "scripts/train_compare.py",
        "--dataset",
        "qm9",
        "--data-root",
        "data/qm9",
        "--qm9-target-index",
        "4",
        "--num-samples",
        "130000",
        "--train-size",
        "110000",
        "--val-size",
        "10000",
        "--batch-size",
        "64",
        "--steps",
        str(steps),
        "--hidden-dim",
        "64",
        "--num-layers",
        "3",
        "--num-heads",
        "4",
        "--routing",
        "lgl",
        "--global-transport-mode",
        "learned",
        "--no-key-balancing",
        "--gated-local-transport",
        "--grouped-invariant-normalization",
        "--precompute-local-edges",
        "--split-seed",
        "42",
        "--model-seed",
        str(MODEL_SEED),
        "--determinism",
        "strict",
        "--device",
        device,
        "--amp-dtype",
        "none",
        "--skip-test-eval",
        "--metrics-out",
        str(output),
    ]
    if arm == "whitened_ridge_0p1" or rank_gated_schema_control:
        command.extend(
            [
                "--whitened-global-read",
                "--whitened-global-ridge",
                "0.1",
            ]
        )
    if rank_gated_schema_control:
        command.append("--whitened-global-rank-gate")
        if arm == "candidate":
            command.append("--freeze-whitened-global-mix")
    return command


def _plan(args: argparse.Namespace) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "question": (
            "Does the 2F rank-gated whitened global read remain numerically "
            "safe against an identical-schema frozen-mix control on QM9?"
            if args.rank_gated_schema_control
            else (
                "Does the ridge-0.1 whitened global read remain numerically "
                "safe on the small-graph QM9 gap task?"
            )
        ),
        "hypothesis": (
            "The whitened lane remains finite and changes validation MAE by at "
            f"most {MAXIMUM_ABSOLUTE_MAE_DELTA_EV:.3f} eV versus candidate."
        ),
        "prediction": "whitening does not improve QM9 gap",
        "dataset": "QM9 gap",
        "split": "seeded_random_row_warm_start_seed42",
        "prediction_unit": "one equilibrium small molecule",
        "arms": list(ARMS),
        "model_seed": MODEL_SEED,
        "split_seed": 42,
        "steps": args.steps,
        "determinism": "strict",
        "dtype": "float32",
        "primary_metric": "validation MAE in eV after the fixed update budget",
        "maximum_absolute_mae_delta_eV": MAXIMUM_ABSOLUTE_MAE_DELTA_EV,
        "maximum_resource_ratio": MAXIMUM_RESOURCE_RATIO,
        "decision_use": "safety_only",
        "rank_gated_schema_control": args.rank_gated_schema_control,
        "claim_boundary": (
            "registered_safety_smoke"
            if args.steps == STEPS
            else "debug_smoke_only"
        ),
        "budget_seconds": args.budget_seconds,
        "device": args.device,
        "validation_evaluated": True,
        "test_evaluated": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan = _plan(args)
    commands = {
        arm: build_command(
            arm,
            output=args.output_dir / f"{arm}.json",
            steps=args.steps,
            device=args.device,
            rank_gated_schema_control=args.rank_gated_schema_control,
        )
        for arm in ARMS
    }
    plan["commands"] = commands
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    result: dict[str, object] = {
        **plan,
        "status": "running",
        "source_sha256": _source_hash(),
        "records": {},
    }
    _write_json(summary_path, result)
    started = time.monotonic()

    records: dict[str, dict[str, object]] = {}
    for arm, command in commands.items():
        remaining = args.budget_seconds - (time.monotonic() - started)
        if remaining <= 0.0:
            raise TimeoutError("QM9 safety-smoke budget exhausted")
        _run_command(command, timeout=remaining)
        record = _load_json(args.output_dir / f"{arm}.json")
        record["status"] = "completed"
        record["smoke_arm"] = arm
        _validate_record(record, arm=arm, steps=args.steps)
        records[arm] = record
        result["records"] = records
        result["elapsed_seconds"] = time.monotonic() - started
        _write_json(summary_path, result)

    _validate_pair(records)
    result["decision"] = decision(list(records.values()))
    result["elapsed_seconds"] = time.monotonic() - started
    result["status"] = "completed"
    _write_json(summary_path, result)
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    return 0


def decision(records: object) -> dict[str, object]:
    if not isinstance(records, list):
        return {"passed": False, "reason": "arm records are missing"}
    by_arm = {
        str(record.get("smoke_arm")): record
        for record in records
        if isinstance(record, Mapping)
    }
    missing = [arm for arm in ARMS if arm not in by_arm]
    if missing:
        return {"passed": False, "reason": f"missing arms: {missing}"}

    candidate = by_arm["candidate"]
    whitened = by_arm["whitened_ridge_0p1"]
    candidate_mae = float(candidate["val_mae"])
    whitened_mae = float(whitened["val_mae"])
    delta = abs(whitened_mae - candidate_mae)
    improvement = candidate_mae - whitened_mae
    latency_ratio = _ratio(
        whitened.get("step_latency_median_seconds"),
        candidate.get("step_latency_median_seconds"),
    )
    memory_ratio = _ratio(
        whitened.get("peak_cuda_memory_bytes"),
        candidate.get("peak_cuda_memory_bytes"),
    )
    finite = all(
        math.isfinite(value)
        for value in (
            candidate_mae,
            whitened_mae,
            delta,
            improvement,
            latency_ratio,
            memory_ratio,
        )
    )
    criteria = {
        "completed_and_finite": (
            finite
            and all(record.get("status") == "completed" for record in by_arm.values())
        ),
        f"absolute_mae_delta_at_most_{MAXIMUM_ABSOLUTE_MAE_DELTA_EV}_eV": (
            delta <= MAXIMUM_ABSOLUTE_MAE_DELTA_EV
        ),
        "matched_update_count": candidate.get("steps") == whitened.get("steps"),
        "paired_base_initialization": (
            candidate.get("paired_base_initial_state_sha256")
            == whitened.get("paired_base_initial_state_sha256")
        ),
        "matched_split": candidate.get("split_hashes") == whitened.get("split_hashes"),
        "test_not_evaluated": all(
            record.get("test_evaluated") is False for record in by_arm.values()
        ),
    }
    return {
        "candidate_val_mae_eV": candidate_mae,
        "whitened_val_mae_eV": whitened_mae,
        "mae_delta_eV": delta,
        "whitened_improvement_eV": improvement,
        "step_latency_ratio": latency_ratio,
        "peak_memory_ratio": memory_ratio,
        "criteria": criteria,
        "passed": all(criteria.values()),
        "default_change_authorized": False,
        "evidence_grade": "bounded_qm9_safety_smoke",
    }


def _validate_record(
    record: Mapping[str, object],
    *,
    arm: str,
    steps: int,
) -> None:
    if record.get("smoke_arm") != arm:
        raise ValueError(f"wrong arm identity for {arm}")
    if int(record.get("steps", -1)) != steps:
        raise ValueError(f"wrong update count for {arm}")
    if record.get("test_evaluated") is not False:
        raise ValueError(f"test boundary violated for {arm}")
    for key in ("val_mae", "val_rmse", "step_latency_median_seconds"):
        value = float(record[key])
        if not math.isfinite(value):
            raise ValueError(f"nonfinite {key} for {arm}")


def _validate_pair(records: Mapping[str, Mapping[str, object]]) -> None:
    candidate = records["candidate"]
    whitened = records["whitened_ridge_0p1"]
    for key in ("split_hashes", "data_identity"):
        if candidate.get(key) != whitened.get(key):
            raise RuntimeError(f"QM9 arms do not share {key}")
    if (
        candidate.get("paired_base_initial_state_sha256")
        != whitened.get("paired_base_initial_state_sha256")
    ):
        raise RuntimeError("QM9 arms do not share the paired base initialization")


def _ratio(numerator: object, denominator: object) -> float:
    numerator_value = float(numerator)
    denominator_value = float(denominator)
    if denominator_value <= 0.0:
        raise ValueError("ratio denominator must be positive")
    return numerator_value / denominator_value


def _run_command(command: Sequence[str], *, timeout: float) -> None:
    process = subprocess.Popen(command, start_new_session=True)
    try:
        return_code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _terminate_process_group(process)
        raise TimeoutError(f"command timed out after {timeout:.1f}s") from None
    if return_code != 0:
        raise RuntimeError(f"command failed with exit code {return_code}: {command}")


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=5.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _source_hash() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for path in (
        Path(__file__).resolve(),
        Path(__file__).with_name("train_compare.py").resolve(),
        root / "src" / "equivariant_attention" / "training.py",
        root / "src" / "equivariant_attention" / "moment.py",
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
