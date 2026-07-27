#!/usr/bin/env python3
"""Run a bounded three-seed ATOM3D-LBA candidate/incumbent confirmation."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import statistics
import subprocess
import sys
import time


PACKET_ID = "lba-multiseed-confirmation-20260727"
SEEDS = (41, 42, 43)
MAX_GPU_SECONDS = 2_250.0
RESERVE_SECONDS = 10.0
MINIMUM_MEAN_IMPROVEMENT_PK = 0.020
MINIMUM_IMPROVING_SEEDS = 2
MAXIMUM_WORST_REGRESSION_PK = 0.050
MAXIMUM_LATENCY_RATIO = 1.25
MAXIMUM_MEMORY_RATIO = 1.50
EXPECTED_DATASET_REVISION = "f93dd2d150a47c270f624620f84e07451a158705"
EXPECTED_TRAIN_IDENTITY = (
    "94d0468cd2c6eb579f5625f9fc74e12c1473c82f44d52186e90bbda17faf3998"
)
EXPECTED_VALIDATION_IDENTITY = (
    "ed4565afc9e87adb926798dd1909a3987fc849a7f0e1f5e3ba92d52c10e7d99c"
)
EXPECTED_TOPOLOGY = (
    "344158d83490a23d25121d84c5ac8d5700281b37c4e699581a6a61385c171080"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--budget-seconds", type=float, default=MAX_GPU_SECONDS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not 0.0 < args.budget_seconds <= MAX_GPU_SECONDS:
        parser.error(
            f"--budget-seconds must lie in (0, {MAX_GPU_SECONDS}]"
        )
    return args


def build_command(
    *,
    seed: int,
    output_dir: Path,
    device: str,
    budget_seconds: float,
) -> list[str]:
    if seed not in SEEDS:
        raise ValueError(f"unregistered seed: {seed}")
    if budget_seconds <= 0.0:
        raise ValueError("per-seed budget must be positive")
    return [
        "uv",
        "run",
        "--locked",
        "python",
        "-u",
        "scripts/train_lba_id30.py",
        str(output_dir),
        "--device",
        device,
        "--arms",
        "candidate",
        "incumbent",
        "--arm-budget-weights",
        "2",
        "1",
        "--batch-size",
        "16",
        "--max-epochs",
        "100",
        "--min-epochs",
        "30",
        "--patience",
        "15",
        "--warmup-epochs",
        "5",
        "--learning-rate",
        "0.0003",
        "--weight-decay",
        "0.01",
        "--grad-clip",
        "1.0",
        "--min-lr-ratio",
        "0.05",
        "--amp-dtype",
        "none",
        "--model-seed",
        str(seed),
        "--order-seed",
        str(seed),
        "--budget-seconds",
        f"{budget_seconds:.6f}",
    ]


def decision(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_seed = {int(record["model_seed"]): record for record in records}
    if set(by_seed) != set(SEEDS):
        raise ValueError("records must contain exactly seeds 41--43")
    improvements: dict[int, float] = {}
    latency_ratios: list[float] = []
    memory_ratios: list[float] = []
    for seed in SEEDS:
        arms = _arms_by_name(by_seed[seed])
        candidate = arms["candidate"]
        incumbent = arms["incumbent"]
        improvements[seed] = _rmse(incumbent) - _rmse(candidate)
        latency_ratios.append(
            _positive_ratio(
                candidate["step_latency_median_seconds"],
                incumbent["step_latency_median_seconds"],
                "step latency",
            )
        )
        memory_ratios.append(
            _positive_ratio(
                candidate["peak_cuda_memory_bytes"],
                incumbent["peak_cuda_memory_bytes"],
                "peak allocation",
            )
        )
    mean_improvement = statistics.fmean(improvements.values())
    improving_seed_count = sum(value > 0.0 for value in improvements.values())
    worst_improvement = min(improvements.values())
    median_latency_ratio = statistics.median(latency_ratios)
    median_memory_ratio = statistics.median(memory_ratios)
    criteria = {
        "mean_improvement_at_least_0.020_pK": (
            mean_improvement >= MINIMUM_MEAN_IMPROVEMENT_PK
        ),
        "at_least_two_of_three_seeds_improve": (
            improving_seed_count >= MINIMUM_IMPROVING_SEEDS
        ),
        "worst_regression_at_most_0.050_pK": (
            worst_improvement >= -MAXIMUM_WORST_REGRESSION_PK
        ),
        "median_step_latency_ratio_at_most_1.25": (
            median_latency_ratio <= MAXIMUM_LATENCY_RATIO
        ),
        "median_peak_memory_ratio_at_most_1.50": (
            median_memory_ratio <= MAXIMUM_MEMORY_RATIO
        ),
    }
    return {
        "passed": all(criteria.values()),
        "criteria": criteria,
        "improvement_by_seed_pK": {
            str(seed): value for seed, value in improvements.items()
        },
        "mean_improvement_pK": mean_improvement,
        "paired_improvement_sample_std_pK": statistics.stdev(
            improvements.values()
        ),
        "improving_seed_count": improving_seed_count,
        "worst_seed_improvement_pK": worst_improvement,
        "median_step_latency_ratio": median_latency_ratio,
        "median_peak_memory_ratio": median_memory_ratio,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan = _plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(args.output_dir / "plan.json", plan)
    started = time.monotonic()
    records: list[dict[str, object]] = []
    executions: list[dict[str, object]] = []
    for index, seed in enumerate(SEEDS):
        elapsed = time.monotonic() - started
        remaining = args.budget_seconds - elapsed - RESERVE_SECONDS
        remaining_seeds = len(SEEDS) - index
        if remaining <= 0.0:
            raise TimeoutError("LBA multi-seed confirmation budget exhausted")
        seed_budget = remaining / remaining_seeds
        output_dir = args.output_dir / f"seed{seed}"
        command = build_command(
            seed=seed,
            output_dir=output_dir,
            device=args.device,
            budget_seconds=seed_budget,
        )
        command_elapsed = _run_command(command, timeout=seed_budget)
        record = _load_json(output_dir / "result.json")
        _validate_record(record, seed=seed)
        records.append(record)
        executions.append(
            {
                "seed": seed,
                "command": command,
                "command_wall_seconds": command_elapsed,
                "result_path": f"seed{seed}/result.json",
                "exit_code": 0,
            }
        )
        arms = _arms_by_name(record)
        print(
            f"completed seed {seed}: candidate={_rmse(arms['candidate']):.6f} "
            f"incumbent={_rmse(arms['incumbent']):.6f} "
            f"packet_wall={time.monotonic() - started:.1f}s",
            flush=True,
        )
    final_decision = decision(records)
    summary = {
        **plan,
        "status": "completed",
        "elapsed_seconds": time.monotonic() - started,
        "source_sha256": records[0]["source_sha256"],
        "executions": executions,
        "arm_summaries": {
            arm: _arm_summary(records, arm)
            for arm in ("candidate", "incumbent")
        },
        "decision": final_decision,
        "validation_evaluated": True,
        "test_evaluated": False,
    }
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _plan(args: argparse.Namespace) -> dict[str, object]:
    return {
        "schema_version": 1,
        "packet_id": PACKET_ID,
        "question": (
            "Does the squared-RBF gated-plus-grouped LGL improve official "
            "ATOM3D-LBA ID30 validation RMSE over the preceding LGL across "
            "model and order seeds 41--43?"
        ),
        "prediction": (
            "Mean paired improvement is at least 0.020 pK, at least two seeds "
            "improve, worst regression is at most 0.050 pK, and median "
            "latency/memory ratios are at most 1.25/1.50."
        ),
        "dataset": "vector-institute/atom3d-lba",
        "dataset_revision": EXPECTED_DATASET_REVISION,
        "split": "official_ID30_train_validation",
        "model_seeds": list(SEEDS),
        "order_seeds": list(SEEDS),
        "arms": ["candidate", "incumbent"],
        "execution_budget_weights": {
            "candidate": 2,
            "incumbent": 1
        },
        "device": args.device,
        "budget_seconds": args.budget_seconds,
        "validation_evaluated": True,
        "test_evaluated": False,
    }


def _validate_record(record: Mapping[str, object], *, seed: int) -> None:
    _assert_finite_json(record)
    if record.get("status") != "completed":
        raise RuntimeError(f"seed {seed} did not complete")
    if record.get("model_seed") != seed or record.get("order_seed") != seed:
        raise RuntimeError(f"seed {seed} changed the registered seeds")
    if record.get("dataset_revision") != EXPECTED_DATASET_REVISION:
        raise RuntimeError(f"seed {seed} changed the dataset revision")
    if record.get("split") != "official_ID30_train_validation":
        raise RuntimeError(f"seed {seed} changed the split")
    if record.get("test_evaluated") is not False:
        raise RuntimeError(f"seed {seed} evaluated test labels")
    dataset = record.get("dataset_summary")
    if not isinstance(dataset, Mapping):
        raise RuntimeError(f"seed {seed} is missing dataset provenance")
    expected = {
        "train_size": 3507,
        "validation_size": 466,
        "train_identity_sha256": EXPECTED_TRAIN_IDENTITY,
        "validation_identity_sha256": EXPECTED_VALIDATION_IDENTITY,
        "topology_sha256": EXPECTED_TOPOLOGY,
    }
    for key, value in expected.items():
        if dataset.get(key) != value:
            raise RuntimeError(f"seed {seed} changed {key}")
    arms = _arms_by_name(record)
    if set(arms) != {"candidate", "incumbent"}:
        raise RuntimeError(f"seed {seed} changed the registered arms")
    if any(arm.get("status") != "completed" for arm in arms.values()):
        raise RuntimeError(f"seed {seed} has an incomplete arm")
    source = record.get("source_sha256")
    if not isinstance(source, str) or len(source) != 64:
        raise RuntimeError(f"seed {seed} is missing source provenance")


def _arms_by_name(
    record: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    arms = record.get("arm_results")
    if not isinstance(arms, list):
        raise ValueError("result is missing arm_results")
    result = {
        str(arm["arm"]): arm
        for arm in arms
        if isinstance(arm, Mapping) and isinstance(arm.get("arm"), str)
    }
    if len(result) != len(arms):
        raise ValueError("arm_results contains an invalid or duplicate arm")
    return result


def _rmse(arm: Mapping[str, object]) -> float:
    validation = arm.get("best_validation")
    if not isinstance(validation, Mapping):
        raise ValueError("arm is missing best validation metrics")
    return float(validation["rmse_pK"])


def _arm_summary(
    records: Sequence[Mapping[str, object]],
    arm: str,
) -> dict[str, object]:
    by_seed = {
        int(record["model_seed"]): _arms_by_name(record)[arm]
        for record in records
    }
    values = [_rmse(by_seed[seed]) for seed in SEEDS]
    return {
        "validation_rmse_by_seed_pK": {
            str(seed): _rmse(by_seed[seed]) for seed in SEEDS
        },
        "mean_validation_rmse_pK": statistics.fmean(values),
        "sample_std_validation_rmse_pK": statistics.stdev(values),
        "median_step_latency_seconds": statistics.median(
            float(by_seed[seed]["step_latency_median_seconds"])
            for seed in SEEDS
        ),
        "median_peak_cuda_memory_bytes": statistics.median(
            int(by_seed[seed]["peak_cuda_memory_bytes"]) for seed in SEEDS
        ),
        "mean_clip_fraction": statistics.fmean(
            float(by_seed[seed]["gradient_monitor"]["clip_fraction"])
            for seed in SEEDS
        ),
    }


def _positive_ratio(numerator: object, denominator: object, name: str) -> float:
    numerator_value = float(numerator)
    denominator_value = float(denominator)
    if not math.isfinite(numerator_value) or numerator_value < 0.0:
        raise ValueError(f"{name} numerator must be finite and nonnegative")
    if not math.isfinite(denominator_value) or denominator_value <= 0.0:
        raise ValueError(f"{name} denominator must be finite and positive")
    return numerator_value / denominator_value


def _assert_finite_json(value: object, *, path: str = "result") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeError(f"nonfinite value at {path}")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite_json(child, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence):
        for index, child in enumerate(value):
            _assert_finite_json(child, path=f"{path}[{index}]")
        return
    raise RuntimeError(f"non-JSON value at {path}: {type(value).__name__}")


def _run_command(command: Sequence[str], *, timeout: float) -> float:
    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "1"
    environment["HF_DATASETS_OFFLINE"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    print(f"running: {shlex.join(command)}", flush=True)
    started = time.monotonic()
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
    elapsed = time.monotonic() - started
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return elapsed


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=3.0)


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def source_hash() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__).resolve(),
        root / "scripts" / "train_lba_id30.py",
        root / "src" / "equivariant_attention" / "moment.py",
        root / "src" / "equivariant_attention" / "pdbbind.py",
        root / "src" / "equivariant_attention" / "training.py",
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


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"LBA multi-seed confirmation failed: {error}", file=sys.stderr)
        raise
