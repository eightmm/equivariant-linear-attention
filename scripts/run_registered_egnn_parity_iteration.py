#!/usr/bin/env python3
"""Run one frozen validation-only iteration against the private static EGNN."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import signal
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path


STUDY_NAME = "private_egnn_parity_20260720"
PACKET_ID = "egnn-parity-20260720"
MAX_PACKET_GPU_SECONDS = 3_600
MAX_ITERATIONS = 3
TERMINATION_RESERVE_SECONDS = 5.0
MODEL_SEEDS = tuple(range(41, 46))
PROMOTION_MAE_EV = 0.398932
SCREEN_MAX_REGRESSION_EV = 0.020
WORST_PAIRED_IMPROVEMENT_EV = -0.020
MAX_PARAMETER_RATIO = 1.05
EXPECTED_DATA_IDENTITY = {
    "processed/data_v3.pt": (
        "9254af077d7bc651631bb56a3a689fb41004731b413bdd0ec8c6efa318229f83"
    ),
    "raw/gdb9.sdf": (
        "98c4e97d50ac549b8c9f0b2114b348a9a944718e17e50d9a724b729f1deaa28e"
    ),
    "raw/gdb9.sdf.csv": (
        "73a67793e3cfa9660f001278bd019c143f57e4785db537a01811cf2ce72aa7eb"
    ),
}
EXPECTED_SPLIT_HASHES = {
    "train": "45c1c82beaa13d91b2ff066b2163bebf701f85186e75a1d90279ac634e917928",
    "validation": "4a98066cf3a91ebc6a1f3b951e65474f3d945baa22e80e9bbfca791c4780fc0b",
    "test": "e100088efe42eddf557ed033486cc8ec7b88b092a118fc83002c3748bf2e0a80",
}
_CANDIDATE_SWITCHES = {
    "radial": ("--learn-local-radial-gate",),
    "pairwise": ("--pairwise-local-content",),
    "pairwise_zero_init": (
        "--pairwise-local-content",
        "--pairwise-residual-scale-init",
        "0.0",
    ),
}


def registered_screen_arms(candidate: str) -> list[dict[str, object]]:
    _candidate_switches(candidate)
    return [
        _attention_arm("screen-attention-lgl-baseline", "attention_baseline", None, 42, 500),
        _attention_arm(f"screen-attention-lgl-{candidate}", "candidate", candidate, 42, 500),
    ]


def registered_confirmation_arms(candidate: str) -> list[dict[str, object]]:
    _candidate_switches(candidate)
    arms: list[dict[str, object]] = []
    for seed in MODEL_SEEDS:
        arms.extend(
            [
                _attention_arm(
                    f"confirmation-attention-{candidate}-seed{seed}",
                    "candidate",
                    candidate,
                    seed,
                    2_000,
                ),
                _egnn_arm(f"confirmation-egnn-static-seed{seed}", seed, 2_000),
            ]
        )
    return arms


def _attention_arm(
    name: str,
    role: str,
    candidate: str | None,
    model_seed: int,
    steps: int,
) -> dict[str, object]:
    return {
        "name": name,
        "role": role,
        "family": "attention",
        "benchmark_model": "factorized_moment",
        "candidate": candidate,
        "model_seed": model_seed,
        "steps": steps,
    }


def _egnn_arm(name: str, model_seed: int, steps: int) -> dict[str, object]:
    return {
        "name": name,
        "role": "egnn",
        "family": "egnn",
        "benchmark_model": "internal_static_egnn_baseline",
        "candidate": None,
        "model_seed": model_seed,
        "steps": steps,
    }


def _candidate_switches(candidate: str) -> tuple[str, ...]:
    try:
        return _CANDIDATE_SWITCHES[candidate]
    except KeyError as error:
        choices = ", ".join(sorted(_CANDIDATE_SWITCHES))
        raise ValueError(f"candidate must be one of: {choices}") from error


def build_train_command(
    arm: Mapping[str, object], metrics_out: Path
) -> list[str]:
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
        str(arm["steps"]),
        "--num-layers",
        "3",
        "--seed",
        "42",
        "--split-seed",
        "42",
        "--model-seed",
        str(arm["model_seed"]),
        "--device",
        "cuda",
        "--amp-dtype",
        "none",
        "--skip-test-eval",
        "--metrics-out",
        str(metrics_out),
    ]
    if arm["family"] == "egnn":
        command.extend(
            [
                "--benchmark-model",
                "internal_static_egnn_baseline",
                "--hidden-dim",
                "91",
            ]
        )
    else:
        command.extend(
            [
                "--benchmark-model",
                "factorized_moment",
                "--hidden-dim",
                "64",
                "--num-heads",
                "4",
                "--routing",
                "lgl",
                "--global-transport-mode",
                "learned",
            ]
        )
        candidate = arm.get("candidate")
        if candidate is not None:
            command.extend(_candidate_switches(str(candidate)))
    return command


def screen_decision(
    candidate: Mapping[str, object], baseline: Mapping[str, object]
) -> dict[str, object]:
    candidate_mae = float(candidate["val_mae"])
    baseline_mae = float(baseline["val_mae"])
    parameter_ratio = _positive_ratio(
        candidate["parameter_count"],
        baseline["parameter_count"],
        name="parameter_count",
    )
    finite = math.isfinite(candidate_mae) and math.isfinite(baseline_mae)
    nonregressing = finite and (
        candidate_mae - baseline_mae <= SCREEN_MAX_REGRESSION_EV
    )
    parameter_bounded = parameter_ratio <= MAX_PARAMETER_RATIO
    criteria = {
        "finite": finite,
        "candidate_minus_baseline_at_most_0.020_eV": nonregressing,
        "parameter_ratio_at_most_1.05": parameter_bounded,
    }
    return {
        "confirmation_admitted": all(criteria.values()),
        "criteria": criteria,
        "candidate_val_mae_eV": candidate_mae,
        "baseline_val_mae_eV": baseline_mae,
        "candidate_minus_baseline_eV": candidate_mae - baseline_mae,
        "parameter_ratio": parameter_ratio,
    }


def promotion_decision(
    candidate: Sequence[Mapping[str, object]],
    egnn: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    candidate_by_seed = _records_by_seed(candidate)
    egnn_by_seed = _records_by_seed(egnn)
    expected = set(MODEL_SEEDS)
    if set(candidate_by_seed) != expected or set(egnn_by_seed) != expected:
        raise ValueError("candidate and EGNN records must contain exactly seeds 41--45")
    candidate_maes = [float(candidate_by_seed[seed]["val_mae"]) for seed in MODEL_SEEDS]
    egnn_maes = [float(egnn_by_seed[seed]["val_mae"]) for seed in MODEL_SEEDS]
    improvements = [
        egnn_mae - candidate_mae
        for candidate_mae, egnn_mae in zip(candidate_maes, egnn_maes, strict=True)
    ]
    candidate_mean = statistics.fmean(candidate_maes)
    egnn_mean = statistics.fmean(egnn_maes)
    improving_seed_count = sum(value > 0.0 for value in improvements)
    worst_improvement = min(improvements)
    criteria = {
        "candidate_mean_mae_at_most_0.398932_eV": candidate_mean <= PROMOTION_MAE_EV,
        "at_least_three_of_five_seed_wins": improving_seed_count >= 3,
        "worst_paired_improvement_at_least_minus_0.020_eV": (
            worst_improvement >= WORST_PAIRED_IMPROVEMENT_EV
        ),
    }
    elapsed_ratios = [
        _positive_ratio(
            candidate_by_seed[seed]["elapsed_seconds"],
            egnn_by_seed[seed]["elapsed_seconds"],
            name="elapsed_seconds",
        )
        for seed in MODEL_SEEDS
    ]
    memory_ratios = [
        _positive_ratio(
            candidate_by_seed[seed]["peak_cuda_memory_bytes"],
            egnn_by_seed[seed]["peak_cuda_memory_bytes"],
            name="peak_cuda_memory_bytes",
        )
        for seed in MODEL_SEEDS
    ]
    return {
        "passed": all(criteria.values()),
        "criteria": criteria,
        "candidate_mean_val_mae_eV": candidate_mean,
        "egnn_mean_val_mae_eV": egnn_mean,
        "mean_improvement_eV": egnn_mean - candidate_mean,
        "improving_seed_count": improving_seed_count,
        "worst_seed_improvement_eV": worst_improvement,
        "seed_improvements_eV": {
            str(seed): value
            for seed, value in zip(MODEL_SEEDS, improvements, strict=True)
        },
        "candidate_val_mae_by_seed_eV": {
            str(seed): value
            for seed, value in zip(MODEL_SEEDS, candidate_maes, strict=True)
        },
        "egnn_val_mae_by_seed_eV": {
            str(seed): value
            for seed, value in zip(MODEL_SEEDS, egnn_maes, strict=True)
        },
        "median_elapsed_ratio_candidate_over_egnn": statistics.median(
            elapsed_ratios
        ),
        "median_peak_memory_ratio_candidate_over_egnn": statistics.median(
            memory_ratios
        ),
    }


def _records_by_seed(
    records: Sequence[Mapping[str, object]],
) -> dict[int, Mapping[str, object]]:
    result: dict[int, Mapping[str, object]] = {}
    for record in records:
        seed = int(record["model_seed"])
        if seed in result:
            raise ValueError(f"duplicate model seed {seed}")
        result[seed] = record
    return result


def _positive_ratio(numerator: object, denominator: object, *, name: str) -> float:
    numerator_value = float(numerator)
    denominator_value = float(denominator)
    if not math.isfinite(numerator_value) or numerator_value < 0.0:
        raise ValueError(f"{name} numerator must be finite and nonnegative")
    if not math.isfinite(denominator_value) or denominator_value <= 0.0:
        raise ValueError(f"{name} denominator must be finite and positive")
    return numerator_value / denominator_value


def _validate_metrics(record: Mapping[str, object], arm: Mapping[str, object]) -> None:
    _assert_finite_json(record)
    expected_pairs = {
        "dataset": "qm9",
        "model": arm["benchmark_model"],
        "model_seed": arm["model_seed"],
        "steps": arm["steps"],
        "split_seed": 42,
        "split_kind": "seeded_random_row_warm_start",
        "test_evaluated": False,
        "target_normalized": True,
        "amp_dtype": "none",
        "train_size": 110_000,
        "val_size": 10_000,
        "test_size": 10_000,
    }
    for key, expected in expected_pairs.items():
        if record.get(key) != expected:
            raise RuntimeError(f"registered run emitted unexpected {key}")
    if record.get("split_hashes") != EXPECTED_SPLIT_HASHES:
        raise RuntimeError("registered run split hashes changed")
    if record.get("data_identity") != EXPECTED_DATA_IDENTITY:
        raise RuntimeError("registered run data hashes changed")
    if record.get("target") != {"index": 4, "name": "gap", "unit": "eV"}:
        raise RuntimeError("registered run emitted an unexpected target")
    train_probe = record.get("train_probe")
    clipping = record.get("gradient_clipping")
    if not isinstance(train_probe, Mapping) or train_probe.get("sample_count") != 256:
        raise RuntimeError("registered fixed train probe is missing")
    if train_probe.get("selection") != "train_split_order_prefix":
        raise RuntimeError("registered fixed train probe selection changed")
    if not isinstance(clipping, Mapping) or clipping.get("measurement_point") != (
        "before_clipping"
    ):
        raise RuntimeError("registered pre-clip gradient diagnostics are missing")
    if clipping.get("step_count") != arm["steps"]:
        raise RuntimeError("gradient clip fraction did not cover every update")
    config = record.get("run_config")
    if not isinstance(config, Mapping) or config.get("device") != "cuda":
        raise RuntimeError("registered run did not execute on CUDA")
    if config.get("coordinate_updates") is not False:
        raise RuntimeError("registered parity study must keep coordinates static")
    if arm["family"] == "egnn":
        if config.get("model") != "internal_static_egnn_baseline":
            raise RuntimeError("registered control was not the private static EGNN")
        return
    if config.get("routing") != "lgl" or config.get("global_transport_mode") != (
        "learned"
    ):
        raise RuntimeError("registered attention route or transport changed")
    expected_candidate = arm.get("candidate")
    expected_radial = expected_candidate == "radial"
    expected_pairwise = str(expected_candidate).startswith("pairwise")
    if config.get("learn_local_radial_gate") is not expected_radial:
        raise RuntimeError("registered radial switch changed")
    if config.get("pairwise_local_content") is not expected_pairwise:
        raise RuntimeError("registered pairwise switch changed")
    expected_pairwise_init = 0.0 if expected_candidate == "pairwise_zero_init" else 0.1
    if float(config.get("pairwise_residual_scale_init", -1.0)) != (
        expected_pairwise_init
    ):
        raise RuntimeError("registered pairwise residual initialization changed")
    if arm["role"] == "candidate":
        diagnostic_key = (
            "local_radial_gradient_parameters"
            if expected_radial
            else "pairwise_local_gradient_parameters"
        )
        diagnostics = record.get(diagnostic_key)
        if not isinstance(diagnostics, Mapping):
            raise RuntimeError(f"missing active diagnostics for {expected_candidate}")
        if int(diagnostics.get("trainable_parameter_count", 0)) <= 0:
            raise RuntimeError(f"{expected_candidate} allocated no trainable parameters")
        if int(diagnostics.get("nonzero_gradient_parameter_count", 0)) <= 0:
            raise RuntimeError(f"{expected_candidate} received no nonzero gradient")


def _assert_finite_json(value: object, *, path: str = "metrics") -> None:
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


def _validate_provenance(
    record: Mapping[str, object],
    arm: Mapping[str, object],
    tracker: dict[str, object],
) -> None:
    source_hash = record.get("source_sha256")
    if tracker.setdefault("source_sha256", source_hash) != source_hash:
        raise RuntimeError("source hash changed within the registered iteration")
    schema_key = str(arm["role"])
    schemas = tracker.setdefault("state_schema_sha256", {})
    if not isinstance(schemas, dict):
        raise RuntimeError("invalid schema provenance tracker")
    schema_hash = record.get("state_schema_sha256")
    if schemas.setdefault(schema_key, schema_hash) != schema_hash:
        raise RuntimeError(f"state schema changed for {schema_key}")
    init_key = f"{arm['role']}.seed{arm['model_seed']}"
    initializations = tracker.setdefault("initial_state_sha256", {})
    if not isinstance(initializations, dict):
        raise RuntimeError("invalid initialization provenance tracker")
    initial_hash = record.get("initial_state_sha256")
    if initializations.setdefault(init_key, initial_hash) != initial_hash:
        raise RuntimeError(f"initialization changed for {init_key}")


def _run_command(command: Sequence[str], *, cwd: Path, timeout: float) -> float:
    if timeout <= 0.0:
        raise TimeoutError("registered GPU budget is exhausted")
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    print(f"running: {shlex.join(command)}", flush=True)
    started = time.monotonic()
    process = subprocess.Popen(
        list(command), cwd=cwd, env=environment, start_new_session=True
    )
    try:
        return_code = process.wait(timeout=timeout)
    except BaseException:
        _terminate_process_group(process)
        raise
    elapsed = time.monotonic() - started
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    return elapsed


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=2.0)


def _load_metrics(path: Path, arm: Mapping[str, object]) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"registered run did not create {path}")
    record = json.loads(path.read_text())
    if not isinstance(record, dict):
        raise RuntimeError("registered metrics root must be an object")
    _validate_metrics(record, arm)
    return record


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _progress_summary(
    *,
    candidate: str,
    iteration: int,
    prior_gpu_seconds: float,
    iteration_started: float,
    provenance: Mapping[str, object],
    run_log: Sequence[Mapping[str, object]],
    status: str,
) -> dict[str, object]:
    iteration_seconds = time.monotonic() - iteration_started
    return {
        "study": STUDY_NAME,
        "packet_id": PACKET_ID,
        "candidate": candidate,
        "iteration": iteration,
        "status": status,
        "budget_seconds": MAX_PACKET_GPU_SECONDS,
        "prior_packet_gpu_seconds": prior_gpu_seconds,
        "iteration_gpu_wall_seconds": iteration_seconds,
        "packet_gpu_wall_seconds": prior_gpu_seconds + iteration_seconds,
        "test_evaluated": False,
        "provenance": dict(provenance),
        "runs": list(run_log),
    }


def _run_arms(
    arms: Sequence[Mapping[str, object]],
    *,
    group: str,
    candidate: str,
    iteration: int,
    repo_root: Path,
    runs_root: Path,
    prior_gpu_seconds: float,
    iteration_started: float,
    provenance: dict[str, object],
    run_log: list[dict[str, object]],
    progress_path: Path,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    group_dir = runs_root / group
    group_dir.mkdir(parents=True, exist_ok=False)
    for arm in arms:
        elapsed_iteration = time.monotonic() - iteration_started
        remaining = (
            MAX_PACKET_GPU_SECONDS
            - prior_gpu_seconds
            - elapsed_iteration
            - TERMINATION_RESERVE_SECONDS
        )
        if remaining <= 0.0:
            raise TimeoutError("registered 3,600 GPU-second packet ceiling reached")
        metrics_path = group_dir / f"{arm['name']}.json"
        command = build_train_command(arm, metrics_path)
        command_seconds = _run_command(command, cwd=repo_root, timeout=remaining)
        record = _load_metrics(metrics_path, arm)
        _validate_provenance(record, arm, provenance)
        records.append(record)
        run_log.append(
            {
                "group": group,
                "name": arm["name"],
                "role": arm["role"],
                "model_seed": arm["model_seed"],
                "metrics_path": str(metrics_path.relative_to(repo_root)),
                "command_wall_seconds": command_seconds,
                "train_elapsed_seconds": record["elapsed_seconds"],
                "val_mae": record["val_mae"],
                "parameter_count": record["parameter_count"],
                "clip_fraction": record["gradient_clipping"]["clip_fraction"],
            }
        )
        _write_json(
            progress_path,
            _progress_summary(
                candidate=candidate,
                iteration=iteration,
                prior_gpu_seconds=prior_gpu_seconds,
                iteration_started=iteration_started,
                provenance=provenance,
                run_log=run_log,
                status=f"running_{group}",
            ),
        )
        print(
            f"completed: {arm['name']} val_mae={float(record['val_mae']):.6f} "
            f"packet_wall={prior_gpu_seconds + time.monotonic() - iteration_started:.1f}s",
            flush=True,
        )
    return records


def _prior_packet_state(
    packet_root: Path, *, current_iteration: int
) -> tuple[float, list[dict[str, object]]]:
    states: dict[int, dict[str, object]] = {}
    for path in sorted(packet_root.glob("iteration-*/study-*.json")):
        record = json.loads(path.read_text())
        if not isinstance(record, dict) or record.get("packet_id") != PACKET_ID:
            continue
        iteration = int(record["iteration"])
        if iteration >= current_iteration:
            continue
        previous = states.get(iteration)
        if previous is None or float(record["packet_gpu_wall_seconds"]) > float(
            previous["packet_gpu_wall_seconds"]
        ):
            states[iteration] = record
    ordered = [states[index] for index in sorted(states)]
    if ordered and bool(ordered[-1].get("promoted", False)):
        raise RuntimeError("a prior iteration already passed the promotion gate")
    expected_prior = list(range(1, current_iteration))
    if sorted(states) != expected_prior:
        raise RuntimeError(
            "iteration order changed or a prior registered iteration is missing"
        )
    prior_seconds = (
        float(ordered[-1]["packet_gpu_wall_seconds"]) if ordered else 0.0
    )
    return prior_seconds, ordered


def _dry_run_plan(output_dir: Path, *, candidate: str, iteration: int) -> dict[str, object]:
    runs = []
    for group, arms in (
        ("screen", registered_screen_arms(candidate)),
        ("confirmation_if_admitted", registered_confirmation_arms(candidate)),
    ):
        for arm in arms:
            metrics_path = output_dir / "registered-runs" / group / f"{arm['name']}.json"
            runs.append(
                {
                    "group": group,
                    "name": arm["name"],
                    "command": build_train_command(arm, metrics_path),
                }
            )
    return {
        "study": STUDY_NAME,
        "packet_id": PACKET_ID,
        "candidate": candidate,
        "iteration": iteration,
        "budget_seconds": MAX_PACKET_GPU_SECONDS,
        "test_evaluated": False,
        "runs": runs,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--candidate", choices=sorted(_CANDIDATE_SWITCHES), required=True)
    parser.add_argument("--iteration", type=int, choices=range(1, MAX_ITERATIONS + 1), required=True)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    expected_candidate = {
        1: "radial",
        2: "pairwise",
        3: "pairwise_zero_init",
    }.get(args.iteration)
    if args.candidate != expected_candidate:
        raise ValueError(
            f"iteration {args.iteration} is frozen to candidate {expected_candidate}"
        )

    repo_root = Path(__file__).resolve().parents[1]
    packet_root = repo_root / "artifacts" / PACKET_ID
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    if output_dir.parent != packet_root:
        raise ValueError(f"output_dir must be a direct child of {packet_root}")
    summary_path = args.summary_out or output_dir / "study-summary.json"
    if not summary_path.is_absolute():
        summary_path = repo_root / summary_path
    if args.dry_run:
        print(
            json.dumps(
                _dry_run_plan(output_dir, candidate=args.candidate, iteration=args.iteration),
                indent=2,
                allow_nan=False,
            )
        )
        return 0

    prior_seconds, prior_iterations = _prior_packet_state(
        packet_root, current_iteration=args.iteration
    )
    if prior_seconds >= MAX_PACKET_GPU_SECONDS:
        raise TimeoutError("registered 3,600 GPU-second packet ceiling already reached")
    runs_root = output_dir / "registered-runs"
    progress_path = output_dir / "study-progress.json"
    if output_dir.exists() or summary_path.exists():
        raise FileExistsError("registered output already exists; choose the frozen fresh path")
    output_dir.mkdir(parents=True)
    runs_root.mkdir()
    iteration_started = time.monotonic()
    provenance: dict[str, object] = {}
    run_log: list[dict[str, object]] = []

    screen_arms = registered_screen_arms(args.candidate)
    screen_records = _run_arms(
        screen_arms,
        group="screen",
        candidate=args.candidate,
        iteration=args.iteration,
        repo_root=repo_root,
        runs_root=runs_root,
        prior_gpu_seconds=prior_seconds,
        iteration_started=iteration_started,
        provenance=provenance,
        run_log=run_log,
        progress_path=progress_path,
    )
    if screen_records[0]["paired_base_initial_state_sha256"] != screen_records[1][
        "paired_base_initial_state_sha256"
    ]:
        raise RuntimeError("screen baseline and candidate common initialization differ")
    screen = screen_decision(screen_records[1], screen_records[0])

    confirmation_records: list[dict[str, object]] = []
    promotion = None
    if screen["confirmation_admitted"]:
        confirmation_arms = registered_confirmation_arms(args.candidate)
        confirmation_records = _run_arms(
            confirmation_arms,
            group="confirmation",
            candidate=args.candidate,
            iteration=args.iteration,
            repo_root=repo_root,
            runs_root=runs_root,
            prior_gpu_seconds=prior_seconds,
            iteration_started=iteration_started,
            provenance=provenance,
            run_log=run_log,
            progress_path=progress_path,
        )
        candidate_records = confirmation_records[::2]
        egnn_records = confirmation_records[1::2]
        for candidate_record, egnn_record in zip(
            candidate_records, egnn_records, strict=True
        ):
            ratio = _positive_ratio(
                candidate_record["parameter_count"],
                egnn_record["parameter_count"],
                name="parameter_count",
            )
            if ratio > MAX_PARAMETER_RATIO:
                raise RuntimeError("candidate exceeds the registered EGNN parameter bound")
        promotion = promotion_decision(candidate_records, egnn_records)

    iteration_seconds = time.monotonic() - iteration_started
    packet_seconds = prior_seconds + iteration_seconds
    if packet_seconds > MAX_PACKET_GPU_SECONDS + 1.0:
        raise TimeoutError("registered 3,600 GPU-second packet ceiling exceeded")
    promoted = bool(promotion and promotion["passed"])
    summary = {
        **_progress_summary(
            candidate=args.candidate,
            iteration=args.iteration,
            prior_gpu_seconds=prior_seconds,
            iteration_started=iteration_started,
            provenance=provenance,
            run_log=run_log,
            status=(
                "promotion_passed"
                if promoted
                else (
                    "confirmation_failed_promotion"
                    if confirmation_records
                    else "screen_not_admitted"
                )
            ),
        ),
        "completed_run_count": len(run_log),
        "screen_run_count": len(screen_records),
        "confirmation_run_count": len(confirmation_records),
        "prior_iterations": prior_iterations,
        "screen": screen,
        "promotion": promotion,
        "promoted": promoted,
        "val_mae": (
            promotion["candidate_mean_val_mae_eV"]
            if promotion is not None
            else screen["candidate_val_mae_eV"]
        ),
    }
    _write_json(summary_path, summary)
    print(f"summary: {summary_path}", flush=True)
    print(
        f"result: status={summary['status']} candidate={args.candidate} "
        f"packet_wall={packet_seconds:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
