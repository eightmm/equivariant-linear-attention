#!/usr/bin/env python3
"""Run the frozen validation-only QM9 transport study within one GPU budget."""

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


STUDY_NAME = "evidence_first_transport_20260719"
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
    "validation": ("4a98066cf3a91ebc6a1f3b951e65474f3d945baa22e80e9bbfca791c4780fc0b"),
    "test": "e100088efe42eddf557ed033486cc8ec7b88b092a118fc83002c3748bf2e0a80",
}


def registered_screen_arms() -> list[dict[str, object]]:
    return [
        _attention_arm("screen-ggg-learned", "ggg", "learned", 42, 500, True),
        _attention_arm("screen-lgg-learned", "lgg", "learned", 42, 500, True),
        _attention_arm("screen-ggl-learned", "ggl", "learned", 42, 500, True),
        _attention_arm("screen-lgl-learned", "lgl", "learned", 42, 500, True),
        _attention_arm("screen-lgl-uniform", "lgl", "uniform", 42, 500, True),
        _attention_arm("screen-lgl-none", "lgl", "none", 42, 500, True),
    ]


def registered_confirmation_arms() -> list[dict[str, object]]:
    return [
        _attention_arm(
            f"confirmation-lgl-{mode}-seed{seed}",
            "lgl",
            mode,
            seed,
            2_000,
            False,
        )
        for seed in MODEL_SEEDS
        for mode in ("learned", "uniform", "none")
    ]


def registered_egnn_arms() -> list[dict[str, object]]:
    return [
        {
            "name": f"egnn-static-seed{seed}",
            "benchmark_model": "internal_static_egnn_baseline",
            "model_seed": seed,
            "steps": 2_000,
            "bounded_diagnostics": False,
        }
        for seed in MODEL_SEEDS
    ]


def _attention_arm(
    name: str,
    routing: str,
    transport_mode: str,
    model_seed: int,
    steps: int,
    bounded_diagnostics: bool,
) -> dict[str, object]:
    return {
        "name": name,
        "benchmark_model": "factorized_moment",
        "routing": routing,
        "transport_mode": transport_mode,
        "model_seed": model_seed,
        "steps": steps,
        "bounded_diagnostics": bounded_diagnostics,
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
        "--skip-test-eval",
        "--metrics-out",
        str(metrics_out),
    ]
    if arm["benchmark_model"] == "internal_static_egnn_baseline":
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
                "--hidden-dim",
                "64",
                "--num-heads",
                "4",
                "--routing",
                str(arm["routing"]),
                "--global-transport-mode",
                str(arm["transport_mode"]),
            ]
        )
        if arm["bounded_diagnostics"]:
            command.extend(
                [
                    "--bounded-diagnostics",
                    "--diagnostic-max-nodes",
                    "32",
                    "--diagnostic-sample-count",
                    "32",
                ]
            )
    return command


def paired_promotion_decision(
    candidate: Sequence[Mapping[str, object]],
    baseline: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    candidate_by_seed = _records_by_seed(candidate)
    baseline_by_seed = _records_by_seed(baseline)
    if set(candidate_by_seed) != set(MODEL_SEEDS):
        raise ValueError("candidate records must contain exactly seeds 41--45")
    if set(baseline_by_seed) != set(MODEL_SEEDS):
        raise ValueError("baseline records must contain exactly seeds 41--45")

    improvements = [
        float(baseline_by_seed[seed]["val_mae"])
        - float(candidate_by_seed[seed]["val_mae"])
        for seed in MODEL_SEEDS
    ]
    elapsed_ratios = [
        _positive_ratio(
            candidate_by_seed[seed]["elapsed_seconds"],
            baseline_by_seed[seed]["elapsed_seconds"],
            name="elapsed_seconds",
        )
        for seed in MODEL_SEEDS
    ]
    memory_ratios = [
        _positive_ratio(
            candidate_by_seed[seed]["peak_cuda_memory_bytes"],
            baseline_by_seed[seed]["peak_cuda_memory_bytes"],
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


def transport_decision(
    records_by_mode: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    if set(records_by_mode) != {"learned", "uniform", "none"}:
        raise ValueError("transport records must contain learned, uniform, and none")
    learned = records_by_mode["learned"]
    uniform = records_by_mode["uniform"]
    none = records_by_mode["none"]
    learned_selectivity = paired_promotion_decision(learned, uniform)
    uniform_selectivity = paired_promotion_decision(uniform, learned)
    learned_global = paired_promotion_decision(learned, none)
    uniform_global = paired_promotion_decision(uniform, none)

    selected_mode: str | None = None
    if learned_selectivity["passed"] and learned_global["passed"]:
        selected_mode = "learned"
    elif uniform_selectivity["passed"] and uniform_global["passed"]:
        selected_mode = "uniform"
    elif uniform_global["passed"]:
        selected_mode = "uniform"
    elif learned_global["passed"]:
        selected_mode = "learned"

    return {
        "learned_selectivity": learned_selectivity,
        "uniform_selectivity": uniform_selectivity,
        "learned_global_transport": learned_global,
        "uniform_global_transport": uniform_global,
        "selected_mode": selected_mode,
        "transport_locked": selected_mode is not None,
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
    expected_model = str(arm["benchmark_model"])
    if record.get("dataset") != "qm9":
        raise RuntimeError("registered run did not use QM9")
    if record.get("model") != expected_model:
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
        raise RuntimeError(
            "registered run split hashes do not match the frozen contract"
        )
    if record.get("data_identity") != EXPECTED_DATA_IDENTITY:
        raise RuntimeError(
            "registered run data hashes do not match the frozen contract"
        )
    if record.get("test_evaluated") is not False:
        raise RuntimeError("registered validation study must not evaluate test labels")
    if record.get("target_normalized") is not True:
        raise RuntimeError(
            "registered run did not use train-fitted target normalization"
        )
    if record.get("amp_dtype") != "none":
        raise RuntimeError("registered run must use FP32")
    if record.get("train_size") != 110_000 or record.get("val_size") != 10_000:
        raise RuntimeError("registered run emitted unexpected train/validation sizes")
    if record.get("test_size") != 10_000:
        raise RuntimeError("registered run emitted an unexpected held-out test size")
    config = record.get("run_config")
    if not isinstance(config, Mapping) or config.get("device") != "cuda":
        raise RuntimeError("registered run did not execute on CUDA")
    if config.get("test_evaluated") is not False:
        raise RuntimeError("registered run config enabled test evaluation")
    target = record.get("target")
    if target != {"index": 4, "name": "gap", "unit": "eV"}:
        raise RuntimeError("registered run emitted an unexpected target contract")


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
    if arm["benchmark_model"] != "factorized_moment":
        return
    schema_hash = record.get("state_schema_sha256")
    if tracker.setdefault("attention_state_schema_sha256", schema_hash) != schema_hash:
        raise RuntimeError("attention state schema changed across registered arms")
    seed = int(arm["model_seed"])
    initial_hashes = tracker.setdefault("attention_initial_state_sha256", {})
    if not isinstance(initial_hashes, dict):
        raise RuntimeError("invalid internal provenance tracker")
    initial_hash = record.get("initial_state_sha256")
    if initial_hashes.setdefault(seed, initial_hash) != initial_hash:
        raise RuntimeError(
            f"attention initialization differs across matched seed-{seed} arms"
        )


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


def _screen_summary(
    arms: Sequence[Mapping[str, object]],
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    rows = [
        {
            "name": arm["name"],
            "routing": arm["routing"],
            "transport_mode": arm["transport_mode"],
            "val_mae": record["val_mae"],
            "val_rmse": record["val_rmse"],
            "elapsed_seconds": record["elapsed_seconds"],
            "peak_cuda_memory_bytes": record["peak_cuda_memory_bytes"],
        }
        for arm, record in zip(arms, records, strict=True)
    ]
    return {
        "claim_boundary": "numerical_screen_only_no_promotion",
        "rows": rows,
        "ranked_by_val_mae": [
            row["name"] for row in sorted(rows, key=lambda row: row["val_mae"])
        ],
    }


def _records_by_transport_mode(
    arms: Sequence[Mapping[str, object]],
    records: Sequence[Mapping[str, object]],
) -> dict[str, list[Mapping[str, object]]]:
    grouped: dict[str, list[Mapping[str, object]]] = {
        "learned": [],
        "uniform": [],
        "none": [],
    }
    for arm, record in zip(arms, records, strict=True):
        grouped[str(arm["transport_mode"])].append(record)
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
        ("confirmation", registered_confirmation_arms()),
        ("egnn", registered_egnn_arms()),
    ):
        for arm in arms:
            path = output_dir / "registered-runs" / group / f"{arm['name']}.json"
            runs.append(
                {
                    "group": group,
                    "conditional": group == "egnn",
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
        default=Path("artifacts/evidence-first-strengthening-20260719"),
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
    confirmation_arms = registered_confirmation_arms()
    confirmation_records = _run_arms(
        confirmation_arms,
        group="confirmation",
        repo_root=repo_root,
        runs_root=runs_root,
        study_started=study_started,
        provenance=provenance,
        run_log=run_log,
    )
    confirmation_by_mode = _records_by_transport_mode(
        confirmation_arms, confirmation_records
    )
    transport = transport_decision(confirmation_by_mode)

    egnn_records: list[dict[str, object]] = []
    egnn_comparison: dict[str, object] | None = None
    selected_mode = transport["selected_mode"]
    if transport["transport_locked"]:
        egnn_arms = registered_egnn_arms()
        egnn_records = _run_arms(
            egnn_arms,
            group="egnn",
            repo_root=repo_root,
            runs_root=runs_root,
            study_started=study_started,
            provenance=provenance,
            run_log=run_log,
        )
        selected_records = confirmation_by_mode[str(selected_mode)]
        egnn_comparison = {
            "claim_boundary": "private_same_harness_not_official_egnn",
            "attention_mode": selected_mode,
            "attention_vs_egnn": paired_promotion_decision(
                selected_records, egnn_records
            ),
            "egnn_vs_attention": paired_promotion_decision(
                egnn_records, selected_records
            ),
            "attention_mean_val_mae": _mean_val_mae(selected_records),
            "egnn_mean_val_mae": _mean_val_mae(egnn_records),
        }

    study_wall_seconds = time.monotonic() - study_started
    selected_records = confirmation_by_mode[str(selected_mode)] if selected_mode else []
    summary = {
        "study": STUDY_NAME,
        "status": (
            "transport_locked_egnn_complete"
            if transport["transport_locked"]
            else "transport_not_locked_egnn_skipped"
        ),
        "budget_seconds": MAX_GPU_SECONDS,
        "study_wall_seconds": study_wall_seconds,
        "command_wall_seconds": sum(
            float(run["command_wall_seconds"]) for run in run_log
        ),
        "completed_run_count": len(run_log),
        "screen_run_count": len(screen_records),
        "confirmation_run_count": len(confirmation_records),
        "egnn_run_count": len(egnn_records),
        "test_evaluated": False,
        "provenance_validated": True,
        "initialization_hash_consistent": True,
        "transport_locked": transport["transport_locked"],
        "selected_transport_mode": selected_mode or "not_selected",
        "val_mae": (
            _mean_val_mae(selected_records)
            if selected_records
            else min(
                _mean_val_mae(confirmation_by_mode["learned"]),
                _mean_val_mae(confirmation_by_mode["uniform"]),
            )
        ),
        "screen": _screen_summary(screen_arms, screen_records),
        "transport": transport,
        "egnn_comparison": egnn_comparison,
        "provenance": provenance,
        "runs": run_log,
    }
    _write_summary(summary_path, summary)
    print(f"summary: {summary_path}", flush=True)
    print(
        f"result: transport_locked={transport['transport_locked']} "
        f"selected_mode={selected_mode} wall={study_wall_seconds:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
