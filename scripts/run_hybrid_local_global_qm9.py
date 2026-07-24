#!/usr/bin/env python3
"""Run the frozen same-feature QM9 gated x grouped attribution screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence


PACKET_ID = "hybrid-local-global-20260724/soft-normalization-v2"
MAX_STEPS = 500
MAX_GPU_SECONDS = 300.0
MINIMUM_IMPROVEMENT_EV = 0.010
MAXIMUM_REGRESSION_EV = 0.020
MAXIMUM_PARAMETER_RATIO = 1.05
GROUPED_SIMPLICITY_TOLERANCE_EV = 0.005
COMBINED_INTERACTION_MARGIN_EV = 0.010
ARMS = {
    "incumbent": (),
    "grouped": ("--grouped-invariant-normalization",),
    "gated": ("--gated-local-transport",),
    "gated_grouped": (
        "--gated-local-transport",
        "--grouped-invariant-normalization",
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--steps", type=int, default=MAX_STEPS)
    parser.add_argument("--budget-seconds", type=float, default=MAX_GPU_SECONDS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.steps <= MAX_STEPS:
        parser.error(f"--steps must lie in [1, {MAX_STEPS}]")
    if not 0.0 < args.budget_seconds <= MAX_GPU_SECONDS:
        parser.error(f"--budget-seconds must lie in (0, {MAX_GPU_SECONDS}]")
    return args


def build_command(
    arm: str,
    *,
    output: Path,
    steps: int,
    device: str,
) -> list[str]:
    try:
        switches = ARMS[arm]
    except KeyError as error:
        raise ValueError(f"unknown arm: {arm}") from error
    return [
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
        "--precompute-local-edges",
        "--split-seed",
        "42",
        "--model-seed",
        "42",
        "--determinism",
        "strict",
        "--device",
        device,
        "--amp-dtype",
        "none",
        "--skip-test-eval",
        "--metrics-out",
        str(output),
        *switches,
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    commands = {
        arm: build_command(
            arm,
            output=args.output_dir / f"{arm}.json",
            steps=args.steps,
            device=args.device,
        )
        for arm in ARMS
    }
    plan = {
        "schema_version": 1,
        "packet_id": PACKET_ID,
        "dataset": "QM9 gap",
        "raw_feature_contract": "identical node_feats, pos, split, and edge_index",
        "arms": list(ARMS),
        "steps": args.steps,
        "device": args.device,
        "budget_seconds": args.budget_seconds,
        "minimum_improvement_eV": MINIMUM_IMPROVEMENT_EV,
        "maximum_regression_eV": MAXIMUM_REGRESSION_EV,
        "maximum_parameter_ratio": MAXIMUM_PARAMETER_RATIO,
        "grouped_simplicity_tolerance_eV": GROUPED_SIMPLICITY_TOLERANCE_EV,
        "combined_interaction_margin_eV": COMBINED_INTERACTION_MARGIN_EV,
        "test_evaluated": False,
        "commands": commands,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "plan.json", plan)
    records: dict[str, dict[str, object]] = {}
    started = time.monotonic()
    for arm, command in commands.items():
        remaining = args.budget_seconds - (time.monotonic() - started)
        if remaining <= 0.0:
            raise TimeoutError("QM9 packet GPU budget exhausted")
        _run_command(command, timeout=remaining)
        record = _load_json(args.output_dir / f"{arm}.json")
        _validate_record(record, arm=arm, steps=args.steps)
        records[arm] = record

    paired_base_hashes = {
        str(record["paired_base_initial_state_sha256"]) for record in records.values()
    }
    if len(paired_base_hashes) != 1:
        raise RuntimeError("QM9 arms do not share a common base initialization")
    decision = screen_decision(records)
    summary = {
        **plan,
        "status": "completed",
        "elapsed_seconds": time.monotonic() - started,
        "source_sha256": _source_hash(),
        "records": {
            arm: {
                "val_mae_eV": record["val_mae"],
                "parameter_count": record["parameter_count"],
                "elapsed_seconds": record["elapsed_seconds"],
                "peak_cuda_memory_bytes": record["peak_cuda_memory_bytes"],
                "clip_fraction": record["gradient_clipping"]["clip_fraction"],
                "mean_pre_clip_grad_norm": record["gradient_clipping"][
                    "pre_clip_grad_norm_mean"
                ],
                "max_pre_clip_grad_norm": record["gradient_clipping"][
                    "pre_clip_grad_norm_max"
                ],
                "initial_state_sha256": record["initial_state_sha256"],
                "paired_base_initial_state_sha256": record[
                    "paired_base_initial_state_sha256"
                ],
            }
            for arm, record in records.items()
        },
        "decision": decision,
        "validation_evaluated": True,
        "test_evaluated": False,
    }
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def screen_decision(
    records: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    incumbent = records["incumbent"]
    incumbent_mae = float(incumbent["val_mae"])
    incumbent_parameters = int(incumbent["parameter_count"])
    candidates: list[dict[str, object]] = []
    for arm in ("grouped", "gated", "gated_grouped"):
        record = records[arm]
        mae = float(record["val_mae"])
        delta = mae - incumbent_mae
        parameter_ratio = int(record["parameter_count"]) / incumbent_parameters
        candidates.append(
            {
                "arm": arm,
                "val_mae_eV": mae,
                "candidate_minus_incumbent_eV": delta,
                "improvement_eV": -delta,
                "finite": math.isfinite(mae),
                "nonregressing": delta <= MAXIMUM_REGRESSION_EV,
                "parameter_ratio": parameter_ratio,
                "parameter_bounded": parameter_ratio <= MAXIMUM_PARAMETER_RATIO,
                "promotion_screen_passed": (
                    math.isfinite(mae)
                    and -delta >= MINIMUM_IMPROVEMENT_EV
                    and delta <= MAXIMUM_REGRESSION_EV
                    and parameter_ratio <= MAXIMUM_PARAMETER_RATIO
                ),
            }
        )
    admitted = [item for item in candidates if item["promotion_screen_passed"]]
    diagnostic = [
        item
        for item in candidates
        if item["finite"] and item["nonregressing"] and item["parameter_bounded"]
    ]
    selected = min(admitted, key=lambda item: item["val_mae_eV"]) if admitted else None
    diagnostic_selected = (
        min(diagnostic, key=lambda item: item["val_mae_eV"]) if diagnostic else None
    )
    grouped_mae = float(records["grouped"]["val_mae"])
    gated_mae = float(records["gated"]["val_mae"])
    combined_mae = float(records["gated_grouped"]["val_mae"])
    combined_improvement_over_grouped = grouped_mae - combined_mae
    combined_improvement_over_gated = gated_mae - combined_mae
    factorial_interaction_improvement = -(
        combined_mae - gated_mae - grouped_mae + incumbent_mae
    )
    grouped_within_simplicity_tolerance = (
        grouped_mae - combined_mae <= GROUPED_SIMPLICITY_TOLERANCE_EV
    )
    combined_interaction_supported = (
        combined_improvement_over_grouped >= COMBINED_INTERACTION_MARGIN_EV
    )
    if grouped_within_simplicity_tolerance:
        attribution_status = "grouped_only_preferred"
        attribution_selected_arm = "grouped"
    elif combined_interaction_supported:
        attribution_status = "combined_interaction_supported"
        attribution_selected_arm = "gated_grouped"
    else:
        attribution_status = "inconclusive_between_grouped_and_combined"
        attribution_selected_arm = None
    return {
        "incumbent_val_mae_eV": incumbent_mae,
        "candidates": candidates,
        "promotion_selected_arm": None if selected is None else selected["arm"],
        "diagnostic_selected_arm": (
            None if diagnostic_selected is None else diagnostic_selected["arm"]
        ),
        "promotion_screen_passed": selected is not None,
        "grouped_val_mae_eV": grouped_mae,
        "combined_val_mae_eV": combined_mae,
        "combined_improvement_over_grouped_eV": (
            combined_improvement_over_grouped
        ),
        "combined_improvement_over_gated_eV": combined_improvement_over_gated,
        "factorial_interaction_improvement_eV": factorial_interaction_improvement,
        "grouped_within_simplicity_tolerance": (
            grouped_within_simplicity_tolerance
        ),
        "combined_interaction_supported": combined_interaction_supported,
        "attribution_status": attribution_status,
        "attribution_selected_arm": attribution_selected_arm,
    }


def _validate_record(
    record: Mapping[str, object],
    *,
    arm: str,
    steps: int,
) -> None:
    if record.get("dataset") != "qm9" or record.get("steps") != steps:
        raise RuntimeError(f"{arm} emitted an unexpected dataset or update count")
    if record.get("model_seed") != 42 or record.get("split_seed") != 42:
        raise RuntimeError(f"{arm} changed a registered seed")
    if record.get("test_evaluated") is not False:
        raise RuntimeError(f"{arm} evaluated test labels")
    config = record.get("run_config")
    if not isinstance(config, Mapping):
        raise RuntimeError(f"{arm} is missing run_config")
    expected_gated = arm in {"gated", "gated_grouped"}
    expected_grouped = arm in {"grouped", "gated_grouped"}
    expected = {
        "routing": "lgl",
        "precompute_local_edges": True,
        "gated_local_transport": expected_gated,
        "grouped_invariant_normalization": expected_grouped,
        "test_evaluated": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"{arm} emitted unexpected {key}")


def _run_command(command: Sequence[str], *, timeout: float) -> None:
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    print(f"running: {shlex.join(command)}", flush=True)
    process = subprocess.Popen(
        list(command),
        env=environment,
        start_new_session=True,
    )
    try:
        return_code = process.wait(timeout=timeout)
    except BaseException:
        _terminate_process_group(process)
        raise
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=2.0)


def _source_hash() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__).resolve(),
        root / "scripts" / "train_compare.py",
        root / "src" / "equivariant_attention" / "moment.py",
        root / "src" / "equivariant_attention" / "training.py",
        root / "artifacts" / PACKET_ID / "scope.md",
        root / "PROJECT.md",
        root / "uv.lock",
    ]
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"hybrid QM9 screen failed: {error}", file=sys.stderr)
        raise
