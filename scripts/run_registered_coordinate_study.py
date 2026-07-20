#!/usr/bin/env python3
"""Run the frozen validation-only dynamic-coordinate QM9 study."""

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


STUDY_NAME = "dynamic_coordinate_egnn_20260719"
MAX_GPU_SECONDS = 25 * 60
TERMINATION_RESERVE_SECONDS = 5.0
MODEL_SEEDS = tuple(range(41, 46))
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


def registered_screen_arms() -> list[dict[str, object]]:
    return [
        _attention_arm("screen-attention-ggg-static", "ggg", False, 42, 500),
        _attention_arm("screen-attention-ggg-dynamic", "ggg", True, 42, 500),
        _attention_arm("screen-attention-lgl-static", "lgl", False, 42, 500),
        _attention_arm("screen-attention-lgl-dynamic", "lgl", True, 42, 500),
        _egnn_arm("screen-egnn-static", False, 42, 500),
        _egnn_arm("screen-egnn-dynamic", True, 42, 500),
    ]


def registered_confirmation_arms(route: str) -> list[dict[str, object]]:
    if route not in {"ggg", "lgl"}:
        raise ValueError("confirmation route must be ggg or lgl")
    arms = []
    for seed in MODEL_SEEDS:
        arms.extend(
            [
                _attention_arm(
                    f"confirmation-attention-{route}-static-seed{seed}",
                    route,
                    False,
                    seed,
                    2_000,
                ),
                _attention_arm(
                    f"confirmation-attention-{route}-dynamic-seed{seed}",
                    route,
                    True,
                    seed,
                    2_000,
                ),
                _egnn_arm(
                    f"confirmation-egnn-static-seed{seed}",
                    False,
                    seed,
                    2_000,
                ),
                _egnn_arm(
                    f"confirmation-egnn-dynamic-seed{seed}",
                    True,
                    seed,
                    2_000,
                ),
            ]
        )
    return arms


def _attention_arm(
    name: str,
    routing: str,
    coordinate_updates: bool,
    model_seed: int,
    steps: int,
) -> dict[str, object]:
    return {
        "name": name,
        "family": "attention",
        "benchmark_model": "factorized_moment",
        "routing": routing,
        "coordinate_updates": coordinate_updates,
        "model_seed": model_seed,
        "steps": steps,
    }


def _egnn_arm(
    name: str,
    coordinate_updates: bool,
    model_seed: int,
    steps: int,
) -> dict[str, object]:
    return {
        "name": name,
        "family": "egnn",
        "benchmark_model": (
            "internal_dynamic_egnn_baseline"
            if coordinate_updates
            else "internal_static_egnn_baseline"
        ),
        "coordinate_updates": coordinate_updates,
        "model_seed": model_seed,
        "steps": steps,
    }


def build_train_command(arm: Mapping[str, object], metrics_out: Path) -> list[str]:
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
                str(arm["benchmark_model"]),
                "--hidden-dim",
                "91",
            ]
        )
    else:
        command.extend(
            [
                "--hidden-dim",
                "64",
                "--num-heads",
                "4",
                "--routing",
                str(arm["routing"]),
                "--global-transport-mode",
                "learned",
            ]
        )
        if arm["coordinate_updates"]:
            command.append("--coordinate-updates")
    return command


def screen_route_decision(
    arms: Sequence[Mapping[str, object]],
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if len(arms) != len(records):
        raise ValueError("screen arms and records must have equal length")
    by_name = {
        str(arm["name"]): record
        for arm, record in zip(arms, records, strict=True)
    }
    route_eligible: dict[str, bool] = {}
    route_rows: dict[str, dict[str, object]] = {}
    for route in ("ggg", "lgl"):
        static = by_name[f"screen-attention-{route}-static"]
        dynamic = by_name[f"screen-attention-{route}-dynamic"]
        eligible = _screen_pair_eligible(dynamic, static)
        route_eligible[route] = eligible
        route_rows[route] = {
            "static_val_mae": float(static["val_mae"]),
            "dynamic_val_mae": float(dynamic["val_mae"]),
            "dynamic_minus_static_eV": float(dynamic["val_mae"])
            - float(static["val_mae"]),
            "eligible": eligible,
        }
    eligible_routes = [route for route in ("ggg", "lgl") if route_eligible[route]]
    selected_route = (
        min(
            eligible_routes,
            key=lambda route: float(
                by_name[f"screen-attention-{route}-dynamic"]["val_mae"]
            ),
        )
        if eligible_routes
        else None
    )
    egnn_static = by_name["screen-egnn-static"]
    egnn_dynamic = by_name["screen-egnn-dynamic"]
    egnn_eligible = _screen_pair_eligible(egnn_dynamic, egnn_static)
    return {
        "attention_route_eligible": route_eligible,
        "attention_routes": route_rows,
        "selected_attention_route": selected_route,
        "egnn_dynamic_eligible": egnn_eligible,
        "egnn_static_val_mae": float(egnn_static["val_mae"]),
        "egnn_dynamic_val_mae": float(egnn_dynamic["val_mae"]),
        "confirmation_admitted": selected_route is not None and egnn_eligible,
    }


def _screen_pair_eligible(
    dynamic: Mapping[str, object],
    static: Mapping[str, object],
) -> bool:
    diagnostics = dynamic.get("coordinate_diagnostics")
    return bool(
        isinstance(diagnostics, Mapping)
        and diagnostics.get("enabled") is True
        and diagnostics.get("active") is True
        and math.isfinite(float(dynamic["val_mae"]))
        and math.isfinite(float(static["val_mae"]))
        and float(dynamic["val_mae"]) - float(static["val_mae"]) <= 0.020
    )


def paired_coordinate_decision(
    dynamic: Sequence[Mapping[str, object]],
    static: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    dynamic_by_seed = _records_by_seed(dynamic)
    static_by_seed = _records_by_seed(static)
    if set(dynamic_by_seed) != set(MODEL_SEEDS):
        raise ValueError("dynamic records must contain exactly seeds 41--45")
    if set(static_by_seed) != set(MODEL_SEEDS):
        raise ValueError("static records must contain exactly seeds 41--45")
    improvements = [
        float(static_by_seed[seed]["val_mae"])
        - float(dynamic_by_seed[seed]["val_mae"])
        for seed in MODEL_SEEDS
    ]
    elapsed_ratios = [
        _positive_ratio(
            dynamic_by_seed[seed]["elapsed_seconds"],
            static_by_seed[seed]["elapsed_seconds"],
            name="elapsed_seconds",
        )
        for seed in MODEL_SEEDS
    ]
    memory_ratios = [
        _positive_ratio(
            dynamic_by_seed[seed]["peak_cuda_memory_bytes"],
            static_by_seed[seed]["peak_cuda_memory_bytes"],
            name="peak_cuda_memory_bytes",
        )
        for seed in MODEL_SEEDS
    ]
    mean_improvement = statistics.fmean(improvements)
    improving_seed_count = sum(improvement > 0.0 for improvement in improvements)
    worst_improvement = min(improvements)
    median_elapsed_ratio = statistics.median(elapsed_ratios)
    median_memory_ratio = statistics.median(memory_ratios)
    criteria = {
        "mean_improvement_at_least_0.010_eV": mean_improvement >= 0.010,
        "at_least_three_of_five_seeds_improve": improving_seed_count >= 3,
        "worst_regression_at_most_0.020_eV": worst_improvement >= -0.020,
        "median_elapsed_ratio_at_most_1.20": median_elapsed_ratio <= 1.20,
        "median_peak_memory_ratio_at_most_1.20": median_memory_ratio <= 1.20,
    }
    return {
        "passed": all(criteria.values()),
        "criteria": criteria,
        "seed_improvements_eV": {
            str(seed): improvement
            for seed, improvement in zip(MODEL_SEEDS, improvements, strict=True)
        },
        "mean_improvement_eV": mean_improvement,
        "improving_seed_count": improving_seed_count,
        "worst_seed_improvement_eV": worst_improvement,
        "median_elapsed_ratio": median_elapsed_ratio,
        "median_peak_memory_ratio": median_memory_ratio,
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
    if record.get("dataset") != "qm9":
        raise RuntimeError("registered run did not use QM9")
    if record.get("model") != arm["benchmark_model"]:
        raise RuntimeError("registered run emitted an unexpected model identity")
    if record.get("model_seed") != arm["model_seed"]:
        raise RuntimeError("registered run emitted an unexpected model seed")
    if record.get("steps") != arm["steps"]:
        raise RuntimeError("registered run emitted an unexpected update count")
    if record.get("split_seed") != 42 or record.get("split_kind") != (
        "seeded_random_row_warm_start"
    ):
        raise RuntimeError("registered run emitted an unexpected split")
    if record.get("split_hashes") != EXPECTED_SPLIT_HASHES:
        raise RuntimeError("registered run split hashes changed")
    if record.get("data_identity") != EXPECTED_DATA_IDENTITY:
        raise RuntimeError("registered run data hashes changed")
    if record.get("test_evaluated") is not False:
        raise RuntimeError("registered validation study evaluated test labels")
    if record.get("target_normalized") is not True:
        raise RuntimeError("target normalization was not fitted on train")
    if record.get("amp_dtype") != "none":
        raise RuntimeError("registered run must use FP32")
    if record.get("train_size") != 110_000 or record.get("val_size") != 10_000:
        raise RuntimeError("registered run emitted unexpected split sizes")
    if record.get("test_size") != 10_000:
        raise RuntimeError("registered run emitted an unexpected test size")
    if record.get("target") != {"index": 4, "name": "gap", "unit": "eV"}:
        raise RuntimeError("registered run emitted an unexpected target")
    config = record.get("run_config")
    if not isinstance(config, Mapping) or config.get("device") != "cuda":
        raise RuntimeError("registered run did not execute on CUDA")
    if config.get("test_evaluated") is not False:
        raise RuntimeError("registered run config enabled test evaluation")
    if config.get("coordinate_updates") is not arm["coordinate_updates"]:
        raise RuntimeError("registered run changed the coordinate-update switch")
    _validate_coordinate_metrics(record, arm)


def _validate_coordinate_metrics(
    record: Mapping[str, object],
    arm: Mapping[str, object],
) -> None:
    diagnostics = record.get("coordinate_diagnostics")
    gradients = record.get("coordinate_gradient_parameters")
    if not isinstance(diagnostics, Mapping) or not isinstance(gradients, Mapping):
        raise RuntimeError("coordinate diagnostics are missing")
    dynamic = bool(arm["coordinate_updates"])
    if diagnostics.get("enabled") is not dynamic:
        raise RuntimeError("coordinate diagnostics disagree with the arm")
    layers = diagnostics.get("layers")
    if not isinstance(layers, list):
        raise RuntimeError("coordinate layer diagnostics are missing")
    if dynamic:
        if diagnostics.get("active") is not True or len(layers) != 2:
            raise RuntimeError("dynamic coordinate path is inactive")
        if int(gradients.get("nonzero_gradient_parameter_count", 0)) <= 0:
            raise RuntimeError("coordinate parameters received no nonzero gradient")
        if float(diagnostics["centroid_drift_max_angstrom"]) > 1e-6:
            raise RuntimeError("dynamic coordinates changed a graph centroid")
        for layer in layers:
            if not isinstance(layer, Mapping):
                raise RuntimeError("coordinate layer diagnostic must be an object")
            if float(layer["step_max_angstrom"]) > 0.250001:
                raise RuntimeError("coordinate step exceeded the registered bound")
            if float(layer["centroid_drift_max_angstrom"]) > 1e-6:
                raise RuntimeError("coordinate layer changed a graph centroid")
    else:
        if diagnostics.get("active") is not False or layers:
            raise RuntimeError("static arm executed coordinate updates")
        if int(gradients.get("parameter_count", 0)) != 0:
            raise RuntimeError("static arm allocated coordinate parameters")


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


def _validate_common_provenance(
    record: Mapping[str, object],
    arm: Mapping[str, object],
    tracker: dict[str, object],
) -> None:
    source_hash = record.get("source_sha256")
    if tracker.setdefault("source_sha256", source_hash) != source_hash:
        raise RuntimeError("source hash changed within the registered study")
    schema_key = f"{arm['family']}.{'dynamic' if arm['coordinate_updates'] else 'static'}"
    schemas = tracker.setdefault("state_schema_sha256", {})
    if not isinstance(schemas, dict):
        raise RuntimeError("invalid schema provenance tracker")
    schema_hash = record.get("state_schema_sha256")
    if schemas.setdefault(schema_key, schema_hash) != schema_hash:
        raise RuntimeError(f"state schema changed for {schema_key}")
    base_key = f"{arm['family']}.seed{arm['model_seed']}"
    base_hashes = tracker.setdefault("paired_base_initial_state_sha256", {})
    if not isinstance(base_hashes, dict):
        raise RuntimeError("invalid initialization provenance tracker")
    base_hash = record.get("paired_base_initial_state_sha256")
    if base_hashes.setdefault(base_key, base_hash) != base_hash:
        raise RuntimeError(f"static/dynamic base initialization differs for {base_key}")


def _run_command(command: Sequence[str], *, cwd: Path, timeout: float) -> float:
    if timeout <= 0.0:
        raise TimeoutError("registered GPU budget is exhausted")
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    print(f"running: {shlex.join(command)}", flush=True)
    started = time.monotonic()
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=environment,
        start_new_session=True,
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


def _run_arms(
    arms: Sequence[Mapping[str, object]],
    *,
    group: str,
    repo_root: Path,
    runs_root: Path,
    study_started: float,
    provenance: dict[str, object],
    run_log: list[dict[str, object]],
) -> list[dict[str, object]]:
    records = []
    group_dir = runs_root / group
    group_dir.mkdir(parents=True, exist_ok=False)
    for arm in arms:
        elapsed_total = time.monotonic() - study_started
        remaining = MAX_GPU_SECONDS - elapsed_total - TERMINATION_RESERVE_SECONDS
        if remaining <= 0.0:
            raise TimeoutError("registered 25 GPU-minute ceiling reached")
        metrics_path = group_dir / f"{arm['name']}.json"
        command = build_train_command(arm, metrics_path)
        command_seconds = _run_command(command, cwd=repo_root, timeout=remaining)
        record = _load_metrics(metrics_path, arm)
        _validate_common_provenance(record, arm, provenance)
        records.append(record)
        run_log.append(
            {
                "group": group,
                "name": arm["name"],
                "family": arm["family"],
                "coordinate_updates": arm["coordinate_updates"],
                "model_seed": arm["model_seed"],
                "metrics_path": str(metrics_path.relative_to(repo_root)),
                "command_wall_seconds": command_seconds,
                "train_elapsed_seconds": record["elapsed_seconds"],
                "val_mae": record["val_mae"],
            }
        )
        print(
            f"completed: {arm['name']} val_mae={float(record['val_mae']):.6f} "
            f"study_wall={time.monotonic() - study_started:.1f}s",
            flush=True,
        )
    return records


def _group_confirmation_records(
    arms: Sequence[Mapping[str, object]],
    records: Sequence[Mapping[str, object]],
) -> dict[str, list[Mapping[str, object]]]:
    grouped: dict[str, list[Mapping[str, object]]] = {
        "attention_static": [],
        "attention_dynamic": [],
        "egnn_static": [],
        "egnn_dynamic": [],
    }
    for arm, record in zip(arms, records, strict=True):
        key = f"{arm['family']}_{'dynamic' if arm['coordinate_updates'] else 'static'}"
        grouped[key].append(record)
    return grouped


def _mean_val_mae(records: Sequence[Mapping[str, object]]) -> float:
    return statistics.fmean(float(record["val_mae"]) for record in records)


def _write_summary(path: Path, summary: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _dry_run_plan(output_dir: Path) -> dict[str, object]:
    runs = []
    for group, arms in (
        ("screen", registered_screen_arms()),
        ("confirmation_if_ggg", registered_confirmation_arms("ggg")),
        ("confirmation_if_lgl", registered_confirmation_arms("lgl")),
    ):
        for arm in arms:
            path = output_dir / "registered-runs" / group / f"{arm['name']}.json"
            runs.append(
                {
                    "group": group,
                    "mutually_exclusive_confirmation": group.startswith("confirmation"),
                    "name": arm["name"],
                    "command": build_train_command(arm, path),
                }
            )
    return {
        "study": STUDY_NAME,
        "budget_seconds": MAX_GPU_SECONDS,
        "test_evaluated": False,
        "runs": runs,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=Path("artifacts/dynamic-coordinate-egnn-20260719"),
    )
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    summary_path = args.summary_out or output_dir / "study-summary.json"
    if not summary_path.is_absolute():
        summary_path = repo_root / summary_path
    if args.dry_run:
        print(json.dumps(_dry_run_plan(output_dir), indent=2, allow_nan=False))
        return 0

    runs_root = output_dir / "registered-runs"
    if runs_root.exists() or summary_path.exists():
        raise FileExistsError(
            "registered output already exists; choose a fresh output directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir()
    study_started = time.monotonic()
    provenance: dict[str, object] = {}
    run_log: list[dict[str, object]] = []

    screen_arms = registered_screen_arms()
    screen_records = _run_arms(
        screen_arms,
        group="screen",
        repo_root=repo_root,
        runs_root=runs_root,
        study_started=study_started,
        provenance=provenance,
        run_log=run_log,
    )
    screen = screen_route_decision(screen_arms, screen_records)
    confirmation_records: list[dict[str, object]] = []
    attention_decision = None
    egnn_decision = None
    selected_route = screen["selected_attention_route"]
    if screen["confirmation_admitted"]:
        confirmation_arms = registered_confirmation_arms(str(selected_route))
        confirmation_records = _run_arms(
            confirmation_arms,
            group="confirmation",
            repo_root=repo_root,
            runs_root=runs_root,
            study_started=study_started,
            provenance=provenance,
            run_log=run_log,
        )
        grouped = _group_confirmation_records(
            confirmation_arms,
            confirmation_records,
        )
        attention_decision = paired_coordinate_decision(
            grouped["attention_dynamic"],
            grouped["attention_static"],
        )
        egnn_decision = paired_coordinate_decision(
            grouped["egnn_dynamic"],
            grouped["egnn_static"],
        )
        means = {name: _mean_val_mae(records) for name, records in grouped.items()}
    else:
        means = {}

    study_wall_seconds = time.monotonic() - study_started
    summary = {
        "study": STUDY_NAME,
        "status": (
            "confirmation_complete"
            if confirmation_records
            else "screen_not_admitted"
        ),
        "budget_seconds": MAX_GPU_SECONDS,
        "study_wall_seconds": study_wall_seconds,
        "command_wall_seconds": sum(
            float(run["command_wall_seconds"]) for run in run_log
        ),
        "completed_run_count": len(run_log),
        "screen_run_count": len(screen_records),
        "confirmation_run_count": len(confirmation_records),
        "test_evaluated": False,
        "provenance_validated": True,
        "initialization_hash_consistent": True,
        "selected_attention_route": selected_route or "not_selected",
        "screen": screen,
        "attention_coordinate_decision": attention_decision,
        "egnn_coordinate_decision": egnn_decision,
        "mean_validation_mae_eV": means,
        "attention_coordinate_promoted": bool(
            attention_decision and attention_decision["passed"]
        ),
        "egnn_coordinate_promoted": bool(
            egnn_decision and egnn_decision["passed"]
        ),
        "val_mae": (
            means.get("attention_dynamic")
            if means
            else min(float(record["val_mae"]) for record in screen_records)
        ),
        "provenance": provenance,
        "runs": run_log,
    }
    _write_summary(summary_path, summary)
    print(f"summary: {summary_path}", flush=True)
    print(
        f"result: status={summary['status']} route={selected_route} "
        f"wall={study_wall_seconds:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
