#!/usr/bin/env python3
"""Run the preregistered architecture-v3 QM9 ablation screen."""

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


PACKET_ID = "hybrid-local-global-20260724/architecture-v3"
MAX_STEPS = 500
MAX_GPU_SECONDS = 450.0
MINIMUM_IMPROVEMENT_EV = 0.010
MAXIMUM_RESOURCE_RATIO = 1.25

ARMS: dict[str, tuple[str, ...]] = {
    "incumbent": (
        "--gated-local-transport",
        "--grouped-invariant-normalization",
    ),
    "irrep_norm": (
        "--gated-local-transport",
        "--grouped-invariant-normalization",
        "--irrep-rms-normalization",
    ),
    "quartic": (
        "--gated-local-transport",
        "--grouped-invariant-normalization",
        "--quartic-kernel",
    ),
    "rank2": (
        "--gated-local-transport",
        "--grouped-invariant-normalization",
        "--angular-feature-rank",
        "2",
    ),
    "combined": (
        "--gated-local-transport",
        "--grouped-invariant-normalization",
        "--irrep-rms-normalization",
        "--quartic-kernel",
        "--angular-feature-rank",
        "2",
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
        "split": "seeded_random_row_warm_start_seed42",
        "raw_feature_contract": "identical node_feats, pos, split, and edge_index",
        "arms": list(ARMS),
        "steps": args.steps,
        "device": args.device,
        "budget_seconds": args.budget_seconds,
        "minimum_improvement_eV": MINIMUM_IMPROVEMENT_EV,
        "maximum_train_step_latency_ratio": MAXIMUM_RESOURCE_RATIO,
        "maximum_peak_allocation_ratio": MAXIMUM_RESOURCE_RATIO,
        "validation_evaluated": True,
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
            raise TimeoutError("architecture-v3 QM9 GPU budget exhausted")
        _run_command(command, timeout=remaining)
        record = _load_json(args.output_dir / f"{arm}.json")
        _validate_record(record, arm=arm, steps=args.steps)
        records[arm] = record

    base_hashes = {
        str(record["paired_base_initial_state_sha256"])
        for record in records.values()
    }
    if len(base_hashes) != 1:
        raise RuntimeError("v3 arms do not share a common base initialization")
    decision = screen_decision(records)
    summary = {
        **plan,
        "status": "completed",
        "elapsed_seconds": time.monotonic() - started,
        "source_sha256": _source_hash(),
        "records": {
            arm: {
                "val_mae_eV": record["val_mae"],
                "train_probe_mae_eV": record["train_probe"]["mae"],
                "parameter_count": record["parameter_count"],
                "step_latency_median_seconds": record[
                    "step_latency_median_seconds"
                ],
                "peak_cuda_memory_bytes": record["peak_cuda_memory_bytes"],
                "clip_fraction": record["gradient_clipping"]["clip_fraction"],
                "mean_pre_clip_grad_norm": record["gradient_clipping"][
                    "pre_clip_grad_norm_mean"
                ],
                "pathwise_pre_clip_grad_norm": record["gradient_clipping"][
                    "pathwise"
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
    incumbent_latency = float(incumbent["step_latency_median_seconds"])
    incumbent_memory = int(incumbent["peak_cuda_memory_bytes"])
    candidates: list[dict[str, object]] = []
    for arm in ("irrep_norm", "quartic", "rank2", "combined"):
        record = records[arm]
        mae = float(record["val_mae"])
        improvement = incumbent_mae - mae
        latency_ratio = (
            float(record["step_latency_median_seconds"]) / incumbent_latency
        )
        memory_ratio = int(record["peak_cuda_memory_bytes"]) / incumbent_memory
        finite = all(
            math.isfinite(value)
            for value in (mae, improvement, latency_ratio, memory_ratio)
        )
        passed = (
            finite
            and improvement >= MINIMUM_IMPROVEMENT_EV
            and latency_ratio <= MAXIMUM_RESOURCE_RATIO
            and memory_ratio <= MAXIMUM_RESOURCE_RATIO
        )
        candidates.append(
            {
                "arm": arm,
                "val_mae_eV": mae,
                "improvement_eV": improvement,
                "step_latency_ratio": latency_ratio,
                "peak_allocation_ratio": memory_ratio,
                "finite": finite,
                "promotion_screen_passed": passed,
            }
        )
    admitted = [row for row in candidates if row["promotion_screen_passed"]]
    selected = (
        min(admitted, key=lambda row: float(row["val_mae_eV"]))
        if admitted
        else None
    )
    return {
        "incumbent_val_mae_eV": incumbent_mae,
        "candidates": candidates,
        "promotion_screen_passed": selected is not None,
        "promotion_selected_arm": None if selected is None else selected["arm"],
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
    expected = {
        "routing": "lgl",
        "precompute_local_edges": True,
        "gated_local_transport": True,
        "grouped_invariant_normalization": True,
        "irrep_rms_normalization": arm in {"irrep_norm", "combined"},
        "quartic_kernel": arm in {"quartic", "combined"},
        "angular_feature_rank": 2 if arm in {"rank2", "combined"} else 1,
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
        print(f"architecture-v3 QM9 screen failed: {error}", file=sys.stderr)
        raise
