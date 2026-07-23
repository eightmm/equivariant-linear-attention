#!/usr/bin/env python3
"""Run the frozen QM9 screen and conditional confirmation for architecture v2."""

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
from dataclasses import asdict, dataclass
from pathlib import Path


STUDY_NAME = "architecture_v2_positive_tensor_qm9_20260723"
PACKET_ID = "architecture-v2-positive-tensor-20260723"
DEFAULT_OUTPUT_DIR = Path("artifacts") / PACKET_ID / "qm9-study"
DEFAULT_CONFIRMATION_SEEDS = tuple(range(41, 46))
DEFAULT_MAX_WALL_SECONDS = 1_800.0
TERMINATION_RESERVE_SECONDS = 5.0
SCREEN_IMPROVEMENT_EV = 0.010
SCREEN_MAX_REGRESSION_EV = 0.020
CONFIRM_MEAN_IMPROVEMENT_EV = 0.020
CONFIRM_MAX_REGRESSION_EV = 0.020

EXPECTED_QM9_DATA_IDENTITY = {
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
EXPECTED_QM9_SPLIT_HASHES = {
    "train": "45c1c82beaa13d91b2ff066b2163bebf701f85186e75a1d90279ac634e917928",
    "validation": "4a98066cf3a91ebc6a1f3b951e65474f3d945baa22e80e9bbfca791c4780fc0b",
    "test": "e100088efe42eddf557ed033486cc8ec7b88b092a118fc83002c3748bf2e0a80",
}

_VARIANTS = {
    "incumbent": {
        "scalar_content_mode": "unit",
        "hidden_tensor_dim": 0,
        "tensor_product_kernel": False,
    },
    "bounded": {
        "scalar_content_mode": "bounded",
        "hidden_tensor_dim": 0,
        "tensor_product_kernel": False,
    },
    "tensor": {
        "scalar_content_mode": "unit",
        "hidden_tensor_dim": 4,
        "tensor_product_kernel": True,
    },
    "combined": {
        "scalar_content_mode": "bounded",
        "hidden_tensor_dim": 4,
        "tensor_product_kernel": True,
    },
}


@dataclass(frozen=True)
class ExecutionSettings:
    dataset: str = "qm9"
    data_root: Path = Path("data/qm9")
    num_samples: int = 130_000
    train_size: int = 110_000
    val_size: int = 10_000
    batch_size: int = 64
    split_seed: int = 42
    device: str = "cuda"
    determinism: str = "strict"
    max_wall_seconds: float = DEFAULT_MAX_WALL_SECONDS


def screen_arms(*, steps: int, model_seed: int = 42) -> list[dict[str, object]]:
    return [
        _attention_arm(
            name=f"screen-{variant}",
            role="incumbent" if variant == "incumbent" else "candidate",
            variant=variant,
            model_seed=model_seed,
            steps=steps,
        )
        for variant in _VARIANTS
    ]


def confirmation_arms(
    selected_variant: str,
    *,
    seeds: Sequence[int],
    steps: int,
) -> list[dict[str, object]]:
    if selected_variant not in _VARIANTS or selected_variant == "incumbent":
        raise ValueError("selected_variant must be bounded, tensor, or combined")
    arms: list[dict[str, object]] = []
    for seed in seeds:
        arms.extend(
            [
                _attention_arm(
                    name=f"confirmation-incumbent-seed{seed}",
                    role="incumbent",
                    variant="incumbent",
                    model_seed=seed,
                    steps=steps,
                ),
                _attention_arm(
                    name=f"confirmation-{selected_variant}-seed{seed}",
                    role="selected",
                    variant=selected_variant,
                    model_seed=seed,
                    steps=steps,
                ),
                _egnn_arm(
                    name=f"confirmation-egnn-seed{seed}",
                    model_seed=seed,
                    steps=steps,
                ),
            ]
        )
    return arms


def _attention_arm(
    *,
    name: str,
    role: str,
    variant: str,
    model_seed: int,
    steps: int,
) -> dict[str, object]:
    try:
        specification = _VARIANTS[variant]
    except KeyError as error:
        raise ValueError(f"unknown architecture-v2 variant: {variant}") from error
    return {
        "name": name,
        "role": role,
        "family": "attention",
        "benchmark_model": "factorized_moment",
        "variant": variant,
        "model_seed": int(model_seed),
        "steps": int(steps),
        **specification,
    }


def _egnn_arm(*, name: str, model_seed: int, steps: int) -> dict[str, object]:
    return {
        "name": name,
        "role": "egnn",
        "family": "egnn",
        "benchmark_model": "internal_static_egnn_baseline",
        "variant": "egnn",
        "model_seed": int(model_seed),
        "steps": int(steps),
    }


def build_train_command(
    arm: Mapping[str, object],
    metrics_out: Path,
    settings: ExecutionSettings,
) -> list[str]:
    command = [
        "uv",
        "run",
        "--locked",
        "python",
        "scripts/train_compare.py",
        "--dataset",
        settings.dataset,
        "--data-root",
        str(settings.data_root),
        "--num-samples",
        str(settings.num_samples),
        "--train-size",
        str(settings.train_size),
        "--val-size",
        str(settings.val_size),
        "--batch-size",
        str(settings.batch_size),
        "--steps",
        str(arm["steps"]),
        "--num-layers",
        "3",
        "--seed",
        "42",
        "--split-seed",
        str(settings.split_seed),
        "--model-seed",
        str(arm["model_seed"]),
        "--determinism",
        settings.determinism,
        "--device",
        settings.device,
        "--amp-dtype",
        "none",
        "--skip-test-eval",
        "--metrics-out",
        str(metrics_out),
    ]
    if settings.dataset == "qm9":
        command.extend(["--qm9-target-index", "4"])
    if arm["family"] == "egnn":
        command.extend(
            [
                "--benchmark-model",
                "internal_static_egnn_baseline",
                "--hidden-dim",
                "91",
            ]
        )
        return command
    command.extend(
        [
            "--benchmark-model",
            "factorized_moment",
            "--hidden-dim",
            "64",
            "--hidden-tensor-dim",
            str(arm["hidden_tensor_dim"]),
            "--num-heads",
            "4",
            "--routing",
            "lgl",
            "--global-transport-mode",
            "learned",
            "--scalar-content-mode",
            str(arm["scalar_content_mode"]),
        ]
    )
    if arm["tensor_product_kernel"]:
        command.append("--tensor-product-kernel")
    return command


def screen_decision(
    arms: Sequence[Mapping[str, object]],
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if len(arms) != len(records):
        raise ValueError("screen arms and records must have equal length")
    by_variant = {
        str(arm["variant"]): record for arm, record in zip(arms, records, strict=True)
    }
    if set(by_variant) != set(_VARIANTS):
        raise ValueError("screen records must contain exactly the four frozen variants")
    incumbent_mae = float(by_variant["incumbent"]["val_mae"])
    candidate_rows: dict[str, dict[str, object]] = {}
    eligible: list[str] = []
    for variant in ("bounded", "tensor", "combined"):
        candidate_mae = float(by_variant[variant]["val_mae"])
        finite = math.isfinite(incumbent_mae) and math.isfinite(candidate_mae)
        delta = candidate_mae - incumbent_mae
        improvement = incumbent_mae - candidate_mae
        criteria = {
            "finite": finite,
            "candidate_minus_incumbent_at_most_0.020_eV": (
                finite and delta <= SCREEN_MAX_REGRESSION_EV
            ),
            "improvement_at_least_0.010_eV": (
                finite and improvement >= SCREEN_IMPROVEMENT_EV
            ),
        }
        admitted = all(criteria.values())
        candidate_rows[variant] = {
            "val_mae_eV": candidate_mae,
            "candidate_minus_incumbent_eV": delta,
            "improvement_eV": improvement,
            "criteria": criteria,
            "eligible": admitted,
        }
        if admitted:
            eligible.append(variant)
    selected = (
        min(eligible, key=lambda variant: float(by_variant[variant]["val_mae"]))
        if eligible
        else None
    )
    return {
        "claim_id": "C4",
        "incumbent_val_mae_eV": incumbent_mae,
        "candidates": candidate_rows,
        "selected_variant": selected,
        "confirmation_admitted": selected is not None,
        "passed": selected is not None,
    }


def confirmation_decision(
    candidate: Sequence[Mapping[str, object]],
    incumbent: Sequence[Mapping[str, object]],
    *,
    seeds: Sequence[int],
) -> dict[str, object]:
    candidate_by_seed = _records_by_seed(candidate, seeds=seeds, label="candidate")
    incumbent_by_seed = _records_by_seed(incumbent, seeds=seeds, label="incumbent")
    improvements = [
        float(incumbent_by_seed[seed]["val_mae"])
        - float(candidate_by_seed[seed]["val_mae"])
        for seed in seeds
    ]
    candidate_values = [float(candidate_by_seed[seed]["val_mae"]) for seed in seeds]
    incumbent_values = [float(incumbent_by_seed[seed]["val_mae"]) for seed in seeds]
    finite = all(
        math.isfinite(value)
        for value in [*improvements, *candidate_values, *incumbent_values]
    )
    required_wins = max(1, math.ceil(0.8 * len(seeds)))
    mean_improvement = statistics.fmean(improvements)
    win_count = sum(value > 0.0 for value in improvements)
    worst_improvement = min(improvements)
    criteria = {
        "finite": finite,
        "mean_improvement_at_least_0.020_eV": (
            finite and mean_improvement >= CONFIRM_MEAN_IMPROVEMENT_EV
        ),
        "paired_wins_at_least_80_percent": finite and win_count >= required_wins,
        "worst_regression_at_most_0.020_eV": (
            finite and worst_improvement >= -CONFIRM_MAX_REGRESSION_EV
        ),
    }
    return {
        "claim_id": "C5",
        "passed": all(criteria.values()),
        "criteria": criteria,
        "required_paired_win_count": required_wins,
        "candidate_mean_val_mae_eV": statistics.fmean(candidate_values),
        "incumbent_mean_val_mae_eV": statistics.fmean(incumbent_values),
        "mean_improvement_eV": mean_improvement,
        "improving_pair_count": win_count,
        "worst_pair_improvement_eV": worst_improvement,
        "paired_improvements_eV": {
            str(seed): value for seed, value in zip(seeds, improvements, strict=True)
        },
    }


def egnn_competitiveness_decision(
    candidate: Sequence[Mapping[str, object]],
    egnn: Sequence[Mapping[str, object]],
    *,
    seeds: Sequence[int],
) -> dict[str, object]:
    candidate_by_seed = _records_by_seed(candidate, seeds=seeds, label="candidate")
    egnn_by_seed = _records_by_seed(egnn, seeds=seeds, label="egnn")
    candidate_values = [float(candidate_by_seed[seed]["val_mae"]) for seed in seeds]
    egnn_values = [float(egnn_by_seed[seed]["val_mae"]) for seed in seeds]
    improvements = [
        egnn_value - candidate_value
        for candidate_value, egnn_value in zip(
            candidate_values,
            egnn_values,
            strict=True,
        )
    ]
    finite = all(
        math.isfinite(value)
        for value in [*candidate_values, *egnn_values, *improvements]
    )
    candidate_mean = statistics.fmean(candidate_values)
    egnn_mean = statistics.fmean(egnn_values)
    win_count = sum(value > 0.0 for value in improvements)
    required_wins = max(1, math.ceil(0.6 * len(seeds)))
    criteria = {
        "finite": finite,
        "candidate_mean_lower_than_egnn": finite and candidate_mean < egnn_mean,
        "candidate_paired_wins_at_least_60_percent": (
            finite and win_count >= required_wins
        ),
    }
    return {
        "claim": "separate_internal_egnn_competitiveness",
        "passed": all(criteria.values()),
        "criteria": criteria,
        "required_paired_win_count": required_wins,
        "candidate_mean_val_mae_eV": candidate_mean,
        "egnn_mean_val_mae_eV": egnn_mean,
        "mean_improvement_eV": egnn_mean - candidate_mean,
        "candidate_paired_win_count": win_count,
        "paired_improvements_eV": {
            str(seed): value for seed, value in zip(seeds, improvements, strict=True)
        },
    }


def _records_by_seed(
    records: Sequence[Mapping[str, object]],
    *,
    seeds: Sequence[int],
    label: str,
) -> dict[int, Mapping[str, object]]:
    result: dict[int, Mapping[str, object]] = {}
    for record in records:
        seed = int(record["model_seed"])
        if seed in result:
            raise ValueError(f"duplicate {label} model seed {seed}")
        result[seed] = record
    if set(result) != set(seeds):
        raise ValueError(f"{label} records do not match requested seeds")
    return result


def atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_metrics(
    record: Mapping[str, object],
    arm: Mapping[str, object],
    settings: ExecutionSettings,
) -> None:
    _assert_finite_json(record)
    expected = {
        "dataset": settings.dataset,
        "model": arm["benchmark_model"],
        "model_seed": arm["model_seed"],
        "steps": arm["steps"],
        "split_seed": settings.split_seed,
        "split_kind": "seeded_random_row_warm_start",
        "train_size": settings.train_size,
        "val_size": settings.val_size,
        "test_size": settings.num_samples - settings.train_size - settings.val_size,
        "target_normalized": True,
        "test_evaluated": False,
        "amp_dtype": "none",
    }
    for key, expected_value in expected.items():
        if record.get(key) != expected_value:
            raise RuntimeError(f"run {arm['name']} emitted unexpected {key}")
    reproducibility = record.get("reproducibility")
    if not isinstance(reproducibility, Mapping):
        raise RuntimeError("run omitted reproducibility state")
    if reproducibility.get("mode") != settings.determinism:
        raise RuntimeError("run determinism mode changed")
    config = record.get("run_config")
    if not isinstance(config, Mapping):
        raise RuntimeError("run omitted run_config")
    common_config = {
        "device": settings.device,
        "determinism": settings.determinism,
        "test_evaluated": False,
        "model_seed": arm["model_seed"],
        "split_seed": settings.split_seed,
    }
    for key, expected_value in common_config.items():
        if config.get(key) != expected_value:
            raise RuntimeError(f"run_config changed {key}")
    clipping = record.get("gradient_clipping")
    if not isinstance(clipping, Mapping) or clipping.get("step_count") != arm["steps"]:
        raise RuntimeError("gradient clipping diagnostics do not cover every step")
    if arm["family"] == "attention":
        attention_config = {
            "model": "factorized_moment",
            "routing": "lgl",
            "global_transport_mode": "learned",
            "hidden_dim": 64,
            "hidden_tensor_dim": arm["hidden_tensor_dim"],
            "scalar_content_mode": arm["scalar_content_mode"],
            "tensor_product_kernel": arm["tensor_product_kernel"],
        }
        for key, expected_value in attention_config.items():
            if config.get(key) != expected_value:
                raise RuntimeError(f"attention run_config changed {key}")
    else:
        if config.get("model") != "internal_static_egnn_baseline":
            raise RuntimeError("control is not the private static EGNN")
        if config.get("hidden_dim") != 91:
            raise RuntimeError("private static EGNN width changed")
    if settings.dataset == "qm9":
        if record.get("target") != {"index": 4, "name": "gap", "unit": "eV"}:
            raise RuntimeError("QM9 target changed")
        if record.get("data_identity") != EXPECTED_QM9_DATA_IDENTITY:
            raise RuntimeError("QM9 data identity changed")
        if (
            settings.split_seed == 42
            and settings.num_samples == 130_000
            and settings.train_size == 110_000
            and settings.val_size == 10_000
            and record.get("split_hashes") != EXPECTED_QM9_SPLIT_HASHES
        ):
            raise RuntimeError("frozen QM9 split identity changed")


def _assert_finite_json(value: object, *, path: str = "metrics") -> None:
    if value is None or isinstance(value, (bool, str, int)):
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
    for key in ("source_sha256", "split_hashes", "data_identity"):
        if key == "data_identity" and key not in record:
            continue
        value = record.get(key)
        if tracker.setdefault(key, value) != value:
            raise RuntimeError(f"{key} changed within the study")
    schemas = tracker.setdefault("state_schema_sha256", {})
    initializations = tracker.setdefault("initial_state_sha256", {})
    if not isinstance(schemas, dict) or not isinstance(initializations, dict):
        raise RuntimeError("invalid provenance tracker")
    variant = str(arm["variant"])
    schema = record.get("state_schema_sha256")
    if schemas.setdefault(variant, schema) != schema:
        raise RuntimeError(f"state schema changed for {variant}")
    initialization_key = f"{variant}.seed{arm['model_seed']}"
    initialization = record.get("initial_state_sha256")
    if initializations.setdefault(initialization_key, initialization) != initialization:
        raise RuntimeError(f"initial state changed for {initialization_key}")


def _validate_screen_initialization_pairing(
    arms: Sequence[Mapping[str, object]],
    records: Sequence[Mapping[str, object]],
) -> None:
    by_variant = {
        str(arm["variant"]): record for arm, record in zip(arms, records, strict=True)
    }
    for left, right in (("incumbent", "bounded"), ("tensor", "combined")):
        if by_variant[left].get("initial_state_sha256") != by_variant[right].get(
            "initial_state_sha256"
        ):
            raise RuntimeError(f"{left}/{right} paired initialization differs")


def _run_command(command: Sequence[str], *, cwd: Path, timeout: float) -> float:
    if timeout <= 0.0:
        raise TimeoutError("architecture-v2 wall-time budget is exhausted")
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


def _load_metrics(
    path: Path,
    arm: Mapping[str, object],
    settings: ExecutionSettings,
) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"run did not create {path}")
    record = json.loads(path.read_text())
    if not isinstance(record, dict):
        raise RuntimeError("metrics root must be an object")
    _validate_metrics(record, arm, settings)
    return record


def _artifact_reference(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def _run_arms(
    arms: Sequence[Mapping[str, object]],
    *,
    group: str,
    repo_root: Path,
    runs_root: Path,
    settings: ExecutionSettings,
    study_started: float,
    provenance: dict[str, object],
    run_log: list[dict[str, object]],
    progress_path: Path,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    group_dir = runs_root / group
    group_dir.mkdir(parents=True, exist_ok=False)
    for arm in arms:
        elapsed_total = time.monotonic() - study_started
        remaining = (
            settings.max_wall_seconds - elapsed_total - TERMINATION_RESERVE_SECONDS
        )
        if remaining <= 0.0:
            raise TimeoutError("architecture-v2 wall-time ceiling reached")
        metrics_path = group_dir / f"{arm['name']}.json"
        command = build_train_command(arm, metrics_path, settings)
        command_seconds = _run_command(command, cwd=repo_root, timeout=remaining)
        record = _load_metrics(metrics_path, arm, settings)
        _validate_provenance(record, arm, provenance)
        records.append(record)
        run_log.append(
            {
                "group": group,
                "name": arm["name"],
                "role": arm["role"],
                "variant": arm["variant"],
                "model_seed": arm["model_seed"],
                "command": command,
                "metrics_path": _artifact_reference(metrics_path, repo_root),
                "command_wall_seconds": command_seconds,
                "train_elapsed_seconds": record["elapsed_seconds"],
                "val_mae_eV": record["val_mae"],
                "parameter_count": record["parameter_count"],
                "peak_cuda_memory_bytes": record["peak_cuda_memory_bytes"],
            }
        )
        atomic_write_json(
            progress_path,
            {
                "study": STUDY_NAME,
                "status": f"running_{group}",
                "test_evaluated": False,
                "study_wall_seconds": time.monotonic() - study_started,
                "provenance": provenance,
                "runs": run_log,
            },
        )
        print(
            f"completed: {arm['name']} val_mae={float(record['val_mae']):.6f}",
            flush=True,
        )
    return records


def _group_confirmation(
    arms: Sequence[Mapping[str, object]],
    records: Sequence[Mapping[str, object]],
) -> dict[str, list[Mapping[str, object]]]:
    grouped: dict[str, list[Mapping[str, object]]] = {
        "incumbent": [],
        "selected": [],
        "egnn": [],
    }
    for arm, record in zip(arms, records, strict=True):
        grouped[str(arm["role"])].append(record)
    return grouped


def _protocol(
    settings: ExecutionSettings,
    *,
    screen_steps: int,
    confirm_steps: int,
    screen_seed: int,
    confirmation_seeds: Sequence[int],
) -> dict[str, object]:
    frozen = (
        settings.dataset == "qm9"
        and settings.data_root == Path("data/qm9")
        and settings.num_samples == 130_000
        and settings.train_size == 110_000
        and settings.val_size == 10_000
        and settings.batch_size == 64
        and settings.split_seed == 42
        and settings.device == "cuda"
        and settings.determinism == "strict"
        and settings.max_wall_seconds <= DEFAULT_MAX_WALL_SECONDS
        and screen_steps == 500
        and confirm_steps == 2_000
        and screen_seed == 42
        and tuple(confirmation_seeds) == DEFAULT_CONFIRMATION_SEEDS
    )
    return {
        "mode": "frozen_confirmatory" if frozen else "smoke_or_exploratory",
        "claims_admissible": frozen,
        "settings": {
            **asdict(settings),
            "data_root": str(settings.data_root),
        },
        "screen_steps": screen_steps,
        "confirm_steps": confirm_steps,
        "screen_seed": screen_seed,
        "confirmation_seeds": list(confirmation_seeds),
        "screen_thresholds": {
            "maximum_regression_eV": SCREEN_MAX_REGRESSION_EV,
            "minimum_improvement_eV": SCREEN_IMPROVEMENT_EV,
        },
        "confirmation_thresholds": {
            "minimum_mean_improvement_eV": CONFIRM_MEAN_IMPROVEMENT_EV,
            "minimum_paired_win_fraction": 0.8,
            "maximum_worst_regression_eV": CONFIRM_MAX_REGRESSION_EV,
        },
    }


def _c6_not_evaluated() -> dict[str, object]:
    return {
        "claim_id": "C6",
        "status": "not_evaluated",
        "reason": "large-complex PDBBind capacity is handled by the separate runner",
    }


def _plan(
    output_dir: Path,
    *,
    settings: ExecutionSettings,
    screen_steps: int,
    confirm_steps: int,
    screen_seed: int,
    seeds: Sequence[int],
) -> dict[str, object]:
    screen = screen_arms(steps=screen_steps, model_seed=screen_seed)
    confirmation: dict[str, list[dict[str, object]]] = {}
    for variant in ("bounded", "tensor", "combined"):
        variant_runs: list[dict[str, object]] = []
        for arm in confirmation_arms(variant, seeds=seeds, steps=confirm_steps):
            metrics_path = (
                output_dir / "registered-runs" / "confirmation" / f"{arm['name']}.json"
            )
            variant_runs.append(
                {
                    "name": arm["name"],
                    "command": build_train_command(arm, metrics_path, settings),
                }
            )
        confirmation[variant] = variant_runs
    return {
        "study": STUDY_NAME,
        "packet_id": PACKET_ID,
        "protocol": _protocol(
            settings,
            screen_steps=screen_steps,
            confirm_steps=confirm_steps,
            screen_seed=screen_seed,
            confirmation_seeds=seeds,
        ),
        "test_evaluated": False,
        "screen_runs": [
            {
                "name": arm["name"],
                "command": build_train_command(
                    arm,
                    output_dir / "registered-runs" / "screen" / f"{arm['name']}.json",
                    settings,
                ),
            }
            for arm in screen
        ],
        "confirmation_runs_if_admitted": confirmation,
        "c6_large_complex_capacity": _c6_not_evaluated(),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--screen-steps", type=_positive_int, default=500)
    parser.add_argument("--confirm-steps", type=_positive_int, default=2_000)
    parser.add_argument("--screen-seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_CONFIRMATION_SEEDS),
    )
    parser.add_argument("--dataset", choices=["qm9", "synthetic"], default="qm9")
    parser.add_argument("--data-root", type=Path, default=Path("data/qm9"))
    parser.add_argument("--num-samples", type=_positive_int, default=None)
    parser.add_argument("--train-size", type=_positive_int, default=None)
    parser.add_argument("--val-size", type=_positive_int, default=None)
    parser.add_argument("--batch-size", type=_positive_int, default=None)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--determinism",
        choices=["seeded", "strict"],
        default="strict",
    )
    parser.add_argument(
        "--max-wall-seconds",
        type=_positive_float,
        default=DEFAULT_MAX_WALL_SECONDS,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _settings_from_args(args: argparse.Namespace) -> ExecutionSettings:
    if args.dataset == "qm9":
        defaults = (130_000, 110_000, 10_000, 64)
    else:
        defaults = (64, 44, 10, 8)
    num_samples = args.num_samples or defaults[0]
    train_size = args.train_size or defaults[1]
    val_size = args.val_size or defaults[2]
    batch_size = args.batch_size or defaults[3]
    if train_size + val_size >= num_samples:
        raise ValueError("train_size + val_size must be smaller than num_samples")
    return ExecutionSettings(
        dataset=args.dataset,
        data_root=args.data_root,
        num_samples=num_samples,
        train_size=train_size,
        val_size=val_size,
        batch_size=batch_size,
        split_seed=args.split_seed,
        device=args.device,
        determinism=args.determinism,
        max_wall_seconds=args.max_wall_seconds,
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("confirmation seeds must be unique")
    settings = _settings_from_args(args)
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else repo_root / args.output_dir
    )
    summary_path = args.summary_out or output_dir / "summary.json"
    if not summary_path.is_absolute():
        summary_path = repo_root / summary_path
    protocol = _protocol(
        settings,
        screen_steps=args.screen_steps,
        confirm_steps=args.confirm_steps,
        screen_seed=args.screen_seed,
        confirmation_seeds=args.seeds,
    )
    plan = _plan(
        output_dir,
        settings=settings,
        screen_steps=args.screen_steps,
        confirm_steps=args.confirm_steps,
        screen_seed=args.screen_seed,
        seeds=args.seeds,
    )
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False))
        return 0

    runs_root = output_dir / "registered-runs"
    plan_path = output_dir / "plan.json"
    progress_path = output_dir / "progress.json"
    if runs_root.exists() or summary_path.exists() or plan_path.exists():
        raise FileExistsError("study output already exists; choose a fresh output path")
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir()
    atomic_write_json(plan_path, plan)
    study_started = time.monotonic()
    provenance: dict[str, object] = {}
    run_log: list[dict[str, object]] = []
    screen_result: dict[str, object] | None = None
    confirmation_result: dict[str, object] | None = None
    competitiveness: dict[str, object] | None = None
    try:
        registered_screen = screen_arms(
            steps=args.screen_steps,
            model_seed=args.screen_seed,
        )
        screen_records = _run_arms(
            registered_screen,
            group="screen",
            repo_root=repo_root,
            runs_root=runs_root,
            settings=settings,
            study_started=study_started,
            provenance=provenance,
            run_log=run_log,
            progress_path=progress_path,
        )
        _validate_screen_initialization_pairing(
            registered_screen,
            screen_records,
        )
        screen_result = screen_decision(registered_screen, screen_records)
        confirmation_records: list[dict[str, object]] = []
        if screen_result["confirmation_admitted"]:
            selected = str(screen_result["selected_variant"])
            registered_confirmation = confirmation_arms(
                selected,
                seeds=args.seeds,
                steps=args.confirm_steps,
            )
            confirmation_records = _run_arms(
                registered_confirmation,
                group="confirmation",
                repo_root=repo_root,
                runs_root=runs_root,
                settings=settings,
                study_started=study_started,
                provenance=provenance,
                run_log=run_log,
                progress_path=progress_path,
            )
            grouped = _group_confirmation(
                registered_confirmation,
                confirmation_records,
            )
            confirmation_result = confirmation_decision(
                grouped["selected"],
                grouped["incumbent"],
                seeds=args.seeds,
            )
            competitiveness = egnn_competitiveness_decision(
                grouped["selected"],
                grouped["egnn"],
                seeds=args.seeds,
            )
        wall_seconds = time.monotonic() - study_started
        status = (
            "confirmation_complete" if confirmation_records else "screen_not_admitted"
        )
        summary = {
            "study": STUDY_NAME,
            "packet_id": PACKET_ID,
            "status": status,
            "protocol": protocol,
            "test_evaluated": False,
            "study_wall_seconds": wall_seconds,
            "command_wall_seconds": sum(
                float(run["command_wall_seconds"]) for run in run_log
            ),
            "completed_run_count": len(run_log),
            "screen_run_count": len(screen_records),
            "confirmation_run_count": len(confirmation_records),
            "provenance_validated": True,
            "screen_initialization_pairing_validated": True,
            "selected_variant": (screen_result["selected_variant"] or "not_selected"),
            "c4_screen": screen_result,
            "c5_confirmation": confirmation_result,
            "c5_claim_supported": bool(
                protocol["claims_admissible"]
                and confirmation_result
                and confirmation_result["passed"]
            ),
            "egnn_competitiveness": competitiveness,
            "egnn_competitiveness_supported": bool(
                protocol["claims_admissible"]
                and competitiveness
                and competitiveness["passed"]
            ),
            "c6_large_complex_capacity": _c6_not_evaluated(),
            "provenance": provenance,
            "runs": run_log,
        }
        atomic_write_json(summary_path, summary)
    except BaseException as error:
        atomic_write_json(
            summary_path,
            {
                "study": STUDY_NAME,
                "packet_id": PACKET_ID,
                "status": "failed",
                "protocol": protocol,
                "test_evaluated": False,
                "study_wall_seconds": time.monotonic() - study_started,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
                "c4_screen": screen_result,
                "c5_confirmation": confirmation_result,
                "egnn_competitiveness": competitiveness,
                "c6_large_complex_capacity": _c6_not_evaluated(),
                "provenance": provenance,
                "runs": run_log,
            },
        )
        raise
    print(f"summary: {summary_path}", flush=True)
    print(
        f"result: status={status} selected={summary['selected_variant']} "
        f"wall={wall_seconds:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
