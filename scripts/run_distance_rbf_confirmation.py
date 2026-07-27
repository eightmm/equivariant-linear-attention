#!/usr/bin/env python3
"""Confirm the distance-spaced local RBF on QM9 across five model seeds."""

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


PACKET_ID = "distance-rbf-confirmation-20260727"
MODEL_SEEDS = tuple(range(41, 46))
MAX_STEPS = 2_000
MAX_GPU_SECONDS = 1_800.0
TERMINATION_RESERVE_SECONDS = 5.0
MINIMUM_MEAN_IMPROVEMENT_EV = 0.010
MINIMUM_IMPROVING_SEEDS = 3
MAXIMUM_WORST_REGRESSION_EV = 0.020
MAXIMUM_RESOURCE_RATIO = 1.20
ARMS = ("incumbent", "distance_rbf", "egnn_complete", "egnn_matched")

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
    seed: int,
    output: Path,
    steps: int,
    device: str,
) -> list[str]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    common = [
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
        "--num-layers",
        "3",
        "--seed",
        "42",
        "--split-seed",
        "42",
        "--model-seed",
        str(seed),
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
    if arm.startswith("egnn_"):
        common.extend(
            [
                "--benchmark-model",
                "internal_static_egnn_baseline",
                "--hidden-dim",
                "91",
            ]
        )
        if arm == "egnn_matched":
            common.extend(
                ["--local-cutoff", "2.5", "--precompute-local-edges"]
            )
        return common
    spacing = "distance" if arm == "distance_rbf" else "squared"
    common.extend(
        [
            "--routing",
            "lgl",
            "--global-transport-mode",
            "learned",
            "--hidden-dim",
            "64",
            "--num-heads",
            "4",
            "--gated-local-transport",
            "--grouped-invariant-normalization",
            "--local-cutoff",
            "2.5",
            "--local-rbf-spacing",
            spacing,
            "--precompute-local-edges",
        ]
    )
    return common


def paired_promotion_decision(
    candidate: Sequence[Mapping[str, object]],
    baseline: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    candidate_by_seed = _records_by_seed(candidate)
    baseline_by_seed = _records_by_seed(baseline)
    _require_registered_seeds(candidate_by_seed, "candidate")
    _require_registered_seeds(baseline_by_seed, "baseline")
    improvements = {
        seed: float(baseline_by_seed[seed]["val_mae"])
        - float(candidate_by_seed[seed]["val_mae"])
        for seed in MODEL_SEEDS
    }
    latency_ratios = [
        _positive_ratio(
            candidate_by_seed[seed]["step_latency_median_seconds"],
            baseline_by_seed[seed]["step_latency_median_seconds"],
            name="step latency",
        )
        for seed in MODEL_SEEDS
    ]
    memory_ratios = [
        _positive_ratio(
            candidate_by_seed[seed]["peak_cuda_memory_bytes"],
            baseline_by_seed[seed]["peak_cuda_memory_bytes"],
            name="peak CUDA allocation",
        )
        for seed in MODEL_SEEDS
    ]
    mean_improvement = statistics.fmean(improvements.values())
    improving_seed_count = sum(value > 0.0 for value in improvements.values())
    worst_improvement = min(improvements.values())
    criteria = {
        "mean_improvement_at_least_0.010_eV": (
            mean_improvement >= MINIMUM_MEAN_IMPROVEMENT_EV
        ),
        "at_least_three_of_five_seeds_improve": (
            improving_seed_count >= MINIMUM_IMPROVING_SEEDS
        ),
        "worst_regression_at_most_0.020_eV": (
            worst_improvement >= -MAXIMUM_WORST_REGRESSION_EV
        ),
        "median_step_latency_ratio_at_most_1.20": (
            statistics.median(latency_ratios) <= MAXIMUM_RESOURCE_RATIO
        ),
        "median_peak_memory_ratio_at_most_1.20": (
            statistics.median(memory_ratios) <= MAXIMUM_RESOURCE_RATIO
        ),
    }
    return {
        "passed": all(criteria.values()),
        "criteria": criteria,
        "seed_improvements_eV": {
            str(seed): value for seed, value in improvements.items()
        },
        "mean_improvement_eV": mean_improvement,
        "paired_improvement_sample_std_eV": statistics.stdev(improvements.values()),
        "improving_seed_count": improving_seed_count,
        "worst_seed_improvement_eV": worst_improvement,
        "median_step_latency_ratio": statistics.median(latency_ratios),
        "median_peak_memory_ratio": statistics.median(memory_ratios),
    }


def cross_family_decision(
    attention: Sequence[Mapping[str, object]],
    egnn: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    attention_by_seed = _records_by_seed(attention)
    egnn_by_seed = _records_by_seed(egnn)
    _require_registered_seeds(attention_by_seed, "attention")
    _require_registered_seeds(egnn_by_seed, "EGNN")
    improvements = {
        seed: float(egnn_by_seed[seed]["val_mae"])
        - float(attention_by_seed[seed]["val_mae"])
        for seed in MODEL_SEEDS
    }
    mean_improvement = statistics.fmean(improvements.values())
    win_count = sum(value > 0.0 for value in improvements.values())
    worst_improvement = min(improvements.values())
    return {
        "descriptive_competitiveness_passed": (
            mean_improvement > 0.0
            and win_count >= MINIMUM_IMPROVING_SEEDS
            and worst_improvement >= -MAXIMUM_WORST_REGRESSION_EV
        ),
        "attention_advantage_by_seed_eV": {
            str(seed): value for seed, value in improvements.items()
        },
        "mean_attention_advantage_eV": mean_improvement,
        "paired_advantage_sample_std_eV": statistics.stdev(improvements.values()),
        "attention_win_count": win_count,
        "worst_attention_advantage_eV": worst_improvement,
        "mean_attention_val_mae_eV": statistics.fmean(
            float(attention_by_seed[seed]["val_mae"]) for seed in MODEL_SEEDS
        ),
        "mean_egnn_val_mae_eV": statistics.fmean(
            float(egnn_by_seed[seed]["val_mae"]) for seed in MODEL_SEEDS
        ),
        "median_attention_to_egnn_step_latency_ratio": statistics.median(
            _positive_ratio(
                attention_by_seed[seed]["step_latency_median_seconds"],
                egnn_by_seed[seed]["step_latency_median_seconds"],
                name="step latency",
            )
            for seed in MODEL_SEEDS
        ),
        "median_attention_to_egnn_peak_memory_ratio": statistics.median(
            _positive_ratio(
                attention_by_seed[seed]["peak_cuda_memory_bytes"],
                egnn_by_seed[seed]["peak_cuda_memory_bytes"],
                name="peak CUDA allocation",
            )
            for seed in MODEL_SEEDS
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    commands = {
        f"{arm}-seed{seed}": build_command(
            arm,
            seed=seed,
            output=args.output_dir / f"{arm}-seed{seed}.json",
            steps=args.steps,
            device=args.device,
        )
        for seed in MODEL_SEEDS
        for arm in ARMS
    }
    plan = _plan(args, commands)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(args.output_dir / "plan.json", plan)
    records: dict[str, list[dict[str, object]]] = {arm: [] for arm in ARMS}
    executions: list[dict[str, object]] = []
    provenance: dict[str, object] = {}
    started = time.monotonic()
    for seed in MODEL_SEEDS:
        for arm in ARMS:
            name = f"{arm}-seed{seed}"
            remaining = (
                args.budget_seconds
                - (time.monotonic() - started)
                - TERMINATION_RESERVE_SECONDS
            )
            if remaining <= 0.0:
                raise TimeoutError("distance-RBF confirmation GPU budget exhausted")
            command = commands[name]
            command_seconds = _run_command(command, timeout=remaining)
            record = _load_json(args.output_dir / f"{name}.json")
            _validate_record(record, arm=arm, seed=seed, steps=args.steps)
            _validate_provenance(record, arm=arm, seed=seed, tracker=provenance)
            records[arm].append(record)
            executions.append(
                {
                    "name": name,
                    "command": command,
                    "command_wall_seconds": command_seconds,
                    "metrics_path": f"{name}.json",
                    "val_mae_eV": record["val_mae"],
                    "exit_code": 0,
                }
            )
            print(
                f"completed: {name} val_mae={float(record['val_mae']):.6f} "
                f"packet_wall={time.monotonic() - started:.1f}s",
                flush=True,
            )

    primary = paired_promotion_decision(
        records["distance_rbf"],
        records["incumbent"],
    )
    decision = {
        "distance_rbf_vs_incumbent": primary,
        "distance_rbf_vs_complete_egnn": cross_family_decision(
            records["distance_rbf"],
            records["egnn_complete"],
        ),
        "distance_rbf_vs_matched_egnn": cross_family_decision(
            records["distance_rbf"],
            records["egnn_matched"],
        ),
        "promotion_selected_arm": "distance_rbf" if primary["passed"] else None,
        "conditional_lba_validation_authorized": bool(primary["passed"]),
    }
    summary = {
        **plan,
        "status": "completed",
        "elapsed_seconds": time.monotonic() - started,
        "packet_source_sha256": source_hash(),
        "source_sha256": provenance["source_sha256"],
        "provenance": provenance,
        "executions": executions,
        "arm_summaries": {
            arm: _arm_summary(arm_records)
            for arm, arm_records in records.items()
        },
        "decision": decision,
        "validation_evaluated": True,
        "test_evaluated": False,
    }
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _plan(
    args: argparse.Namespace,
    commands: Mapping[str, Sequence[str]],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "packet_id": PACKET_ID,
        "question": (
            "Does distance-spaced local RBF improve the squared-spacing LGL "
            "incumbent across five 2,000-update QM9 seeds, and how does it "
            "compare descriptively with complete and topology-matched EGNN?"
        ),
        "prediction": (
            "Mean paired improvement is at least 0.010 eV, at least three seeds "
            "improve, worst regression is at most 0.020 eV, and resource ratios "
            "stay at most 1.20."
        ),
        "dataset": "QM9 gap",
        "split": "seeded_random_row_warm_start_seed42",
        "model_seeds": list(MODEL_SEEDS),
        "arms": list(ARMS),
        "steps": args.steps,
        "device": args.device,
        "budget_seconds": args.budget_seconds,
        "test_evaluated": False,
        "commands": {name: list(command) for name, command in commands.items()},
    }


def _validate_record(
    record: Mapping[str, object],
    *,
    arm: str,
    seed: int,
    steps: int,
) -> None:
    _assert_finite_json(record)
    if record.get("dataset") != "qm9" or record.get("steps") != steps:
        raise RuntimeError(f"{arm} seed {seed} changed dataset or update count")
    if record.get("model_seed") != seed or record.get("split_seed") != 42:
        raise RuntimeError(f"{arm} seed {seed} changed a registered seed")
    if record.get("split_hashes") != EXPECTED_SPLIT_HASHES:
        raise RuntimeError(f"{arm} seed {seed} changed the frozen split")
    if record.get("data_identity") != EXPECTED_DATA_IDENTITY:
        raise RuntimeError(f"{arm} seed {seed} changed the QM9 snapshot")
    if record.get("test_evaluated") is not False:
        raise RuntimeError(f"{arm} seed {seed} evaluated test labels")
    if record.get("target") != {"index": 4, "name": "gap", "unit": "eV"}:
        raise RuntimeError(f"{arm} seed {seed} changed the target")
    config = record.get("run_config")
    if not isinstance(config, Mapping):
        raise RuntimeError(f"{arm} seed {seed} is missing run_config")
    expected_model = (
        "internal_static_egnn_baseline"
        if arm.startswith("egnn_")
        else "factorized_moment"
    )
    if record.get("model") != expected_model:
        raise RuntimeError(f"{arm} seed {seed} changed the model family")
    if config.get("determinism") != "strict":
        raise RuntimeError(f"{arm} seed {seed} did not use strict determinism")
    if arm in {"incumbent", "distance_rbf"}:
        expected_spacing = "distance" if arm == "distance_rbf" else "squared"
        expected = {
            "routing": "lgl",
            "gated_local_transport": True,
            "grouped_invariant_normalization": True,
            "local_cutoff": 2.5,
            "local_rbf_spacing": expected_spacing,
            "precompute_local_edges": True,
        }
    elif arm == "egnn_matched":
        expected = {
            "local_cutoff": 2.5,
            "precompute_local_edges": True,
            "edge_topology": "precomputed_radius_candidates_without_self",
        }
    else:
        expected = {
            "local_cutoff": "not_applicable",
            "precompute_local_edges": False,
            "edge_topology": "same_graph_directed_complete_without_self",
        }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"{arm} seed {seed} emitted unexpected {key}")


def _validate_provenance(
    record: Mapping[str, object],
    *,
    arm: str,
    seed: int,
    tracker: dict[str, object],
) -> None:
    source_hash = record.get("source_sha256")
    if tracker.setdefault("source_sha256", source_hash) != source_hash:
        raise RuntimeError("source hash changed within the packet")
    family = "egnn" if arm.startswith("egnn_") else "attention"
    schema_key = f"{family}_state_schema_sha256"
    schema_hash = record.get("state_schema_sha256")
    if tracker.setdefault(schema_key, schema_hash) != schema_hash:
        raise RuntimeError(f"{family} state schema changed within the packet")
    initial_key = f"{family}_initial_state_sha256"
    initial_by_seed = tracker.setdefault(initial_key, {})
    if not isinstance(initial_by_seed, dict):
        raise RuntimeError("invalid provenance tracker")
    initial_hash = record.get("initial_state_sha256")
    if initial_by_seed.setdefault(str(seed), initial_hash) != initial_hash:
        raise RuntimeError(
            f"{family} initialization changed within matched seed {seed}"
        )


def _arm_summary(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    values = [float(record["val_mae"]) for record in records]
    return {
        "val_mae_by_seed_eV": {
            str(record["model_seed"]): record["val_mae"] for record in records
        },
        "mean_val_mae_eV": statistics.fmean(values),
        "sample_std_val_mae_eV": statistics.stdev(values),
        "median_step_latency_seconds": statistics.median(
            float(record["step_latency_median_seconds"]) for record in records
        ),
        "median_peak_cuda_memory_bytes": statistics.median(
            int(record["peak_cuda_memory_bytes"]) for record in records
        ),
        "mean_clip_fraction": statistics.fmean(
            float(record["gradient_clipping"]["clip_fraction"])
            for record in records
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


def _require_registered_seeds(
    records: Mapping[int, Mapping[str, object]],
    name: str,
) -> None:
    if set(records) != set(MODEL_SEEDS):
        raise ValueError(f"{name} records must contain exactly seeds 41--45")


def _positive_ratio(numerator: object, denominator: object, *, name: str) -> float:
    numerator_value = float(numerator)
    denominator_value = float(denominator)
    if not math.isfinite(numerator_value) or numerator_value < 0.0:
        raise ValueError(f"{name} numerator must be finite and nonnegative")
    if not math.isfinite(denominator_value) or denominator_value <= 0.0:
        raise ValueError(f"{name} denominator must be finite and positive")
    return numerator_value / denominator_value


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


def _run_command(command: Sequence[str], *, timeout: float) -> float:
    environment = os.environ.copy()
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
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=2.0)


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def source_hash() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__).resolve(),
        root / "scripts" / "train_compare.py",
        root / "src" / "equivariant_attention" / "moment.py",
        root / "src" / "equivariant_attention" / "training.py",
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


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"distance-RBF confirmation failed: {error}", file=sys.stderr)
        raise
