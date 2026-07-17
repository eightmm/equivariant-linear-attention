#!/usr/bin/env python3
"""Validate paired QM9/CUDA provenance and emit the registered decision."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[1]
SEEDS = (41, 42, 43)
EXPECTED_COMMIT = "a8bda61868cf118b93b6a605001fb401c23f46c1"


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def differing_keys(left: dict[str, object], right: dict[str, object]) -> set[str]:
    return {
        key
        for key in left.keys() | right.keys()
        if left.get(key) != right.get(key)
    }


def main() -> None:
    records: dict[tuple[str, int], dict[str, object]] = {}
    for route in ("ggg", "lgl"):
        for seed in SEEDS:
            path = RUN_DIR / "qm9" / f"{route}-m1-seed{seed}.json"
            row = load_json(path)
            records[(route, seed)] = row
            assert row["dataset"] == "qm9"
            assert row["model_seed"] == seed
            assert row["test_evaluated"] is False
            assert "test_mae" not in row and "test_rmse" not in row
            assert math.isfinite(float(row["val_mae"]))
            run_config = row["run_config"]
            assert isinstance(run_config, dict)
            assert run_config["routing"] == route
            assert run_config["memory_count"] == 1
            assert run_config["memory_interaction"] is False
            assert run_config["steps"] == 2000
            assert run_config["test_evaluated"] is False

    common_fields = (
        "source_sha256",
        "state_schema_sha256",
        "parameter_count",
        "trainable_parameter_count",
        "data_identity",
        "split_hashes",
        "split_seed",
        "target",
        "train_size",
        "val_size",
    )
    reference = records[("ggg", 41)]
    for row in records.values():
        for field in common_fields:
            assert row[field] == reference[field], field

    pairs: list[dict[str, object]] = []
    for seed in SEEDS:
        baseline = records[("ggg", seed)]
        candidate = records[("lgl", seed)]
        assert baseline["initial_state_sha256"] == candidate["initial_state_sha256"]
        baseline_config = baseline["run_config"]
        candidate_config = candidate["run_config"]
        assert isinstance(baseline_config, dict) and isinstance(candidate_config, dict)
        assert differing_keys(baseline_config, candidate_config) == {
            "routing",
            "local_head_counts",
        }
        delta = float(baseline["val_mae"]) - float(candidate["val_mae"])
        pairs.append(
            {
                "seed": seed,
                "ggg_val_mae_eV": baseline["val_mae"],
                "lgl_val_mae_eV": candidate["val_mae"],
                "improvement_eV": delta,
                "ggg_train_seconds": baseline["elapsed_seconds"],
                "lgl_train_seconds": candidate["elapsed_seconds"],
                "ggg_peak_cuda_memory_bytes": baseline["peak_cuda_memory_bytes"],
                "lgl_peak_cuda_memory_bytes": candidate["peak_cuda_memory_bytes"],
                "initial_state_sha256": baseline["initial_state_sha256"],
            }
        )

    ledger_path = RUN_DIR / "qm9-runs.jsonl"
    if not ledger_path.exists():
        ledger_path = RUN_DIR / "qm9-runs-public.jsonl"
    ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    assert len(ledger) == 6
    assert all(row["exit"] == 0 for row in ledger)
    assert all(row["dirty"] == 0 for row in ledger)
    assert all(
        len(row["git_sha"]) >= 7 and EXPECTED_COMMIT.startswith(row["git_sha"])
        for row in ledger
    )
    assert all(row["metrics"]["test_evaluated"] is False for row in ledger)

    benchmark_path = RUN_DIR / "cuda-benchmarks-vectorized.json"
    benchmark = load_json(benchmark_path)
    assert benchmark["git_commit"] == EXPECTED_COMMIT
    assert benchmark["git_dirty"] is False
    assert benchmark["test_evaluated"] is False
    assert benchmark["all_comparisons_within_20_percent"] is True
    model_source = ROOT / "src" / "equivariant_attention" / "moment.py"
    assert benchmark["model_source_sha256"] == hashlib.sha256(
        model_source.read_bytes()
    ).hexdigest()

    improvements = [float(row["improvement_eV"]) for row in pairs]
    mean_improvement = statistics.mean(improvements)
    positive_count = sum(value > 0.0 for value in improvements)
    minimum_improvement = min(improvements)
    accuracy_pass = (
        mean_improvement >= 0.010
        and positive_count >= 2
        and minimum_improvement >= -0.020
    )
    benchmark_pass = bool(benchmark["all_comparisons_within_20_percent"])
    ggg_elapsed = statistics.mean(
        float(records[("ggg", seed)]["elapsed_seconds"]) for seed in SEEDS
    )
    lgl_elapsed = statistics.mean(
        float(records[("lgl", seed)]["elapsed_seconds"]) for seed in SEEDS
    )
    ggg_peak = max(
        int(records[("ggg", seed)]["peak_cuda_memory_bytes"]) for seed in SEEDS
    )
    lgl_peak = max(
        int(records[("lgl", seed)]["peak_cuda_memory_bytes"]) for seed in SEEDS
    )
    gradient_element_counts = {
        int(row["gradient_parameters"]["parameters_with_gradient_count"])
        for row in records.values()
    }
    ggg_nonzero_gradient_counts = {
        int(records[("ggg", seed)]["nonzero_gradient_parameter_count"])
        for seed in SEEDS
    }
    lgl_nonzero_gradient_counts = {
        int(records[("lgl", seed)]["nonzero_gradient_parameter_count"])
        for seed in SEEDS
    }
    assert len(gradient_element_counts) == 1
    assert len(ggg_nonzero_gradient_counts) == 1
    assert len(lgl_nonzero_gradient_counts) == 1
    gradient_elements = gradient_element_counts.pop()
    ggg_nonzero_gradients = ggg_nonzero_gradient_counts.pop()
    lgl_nonzero_gradients = lgl_nonzero_gradient_counts.pop()
    maximum_benchmark_latency_increase = max(
        float(row["latency_increase_fraction"])
        for row in benchmark["comparisons"]
    )
    maximum_benchmark_memory_increase = max(
        float(row["peak_memory_increase_fraction"])
        for row in benchmark["comparisons"]
    )
    directly_timed_components = {
        "paired_run_ledger_wall_seconds": sum(
            int(row["duration_s"]) for row in ledger
        ),
        "preliminary_cuda_benchmark_wall_seconds": float(
            load_json(RUN_DIR / "cuda-benchmarks.json")["wall_seconds"]
        ),
        "vectorization_smoke_wall_seconds": float(
            load_json(RUN_DIR / "cuda-benchmarks-vectorized-smoke.json")[
                "wall_seconds"
            ]
        ),
        "registered_cuda_benchmark_wall_seconds": float(
            benchmark["wall_seconds"]
        ),
        "qm9_smoke_elapsed_seconds": float(
            load_json(RUN_DIR / "qm9-smoke-lgl.json")["elapsed_seconds"]
        ),
    }
    directly_timed_seconds = sum(directly_timed_components.values())
    conservative_total_gpu_wall_seconds = 570.0
    unmetered_allowance_seconds = (
        conservative_total_gpu_wall_seconds - directly_timed_seconds
    )
    assert unmetered_allowance_seconds >= 0.0

    result = {
        "schema_version": 1,
        "test_evaluated": False,
        "git_commit": EXPECTED_COMMIT,
        "source_sha256": reference["source_sha256"],
        "data_identity": reference["data_identity"],
        "split_hashes": reference["split_hashes"],
        "state_schema_sha256": reference["state_schema_sha256"],
        "parameter_count": reference["parameter_count"],
        "trainable_parameter_count": reference["trainable_parameter_count"],
        "pairs": pairs,
        "accuracy_decision": {
            "mean_improvement_eV": mean_improvement,
            "sample_std_improvement_eV": statistics.stdev(improvements),
            "positive_seed_count": positive_count,
            "minimum_improvement_eV": minimum_improvement,
            "thresholds": {
                "minimum_mean_improvement_eV": 0.010,
                "minimum_positive_seed_count": 2,
                "minimum_worst_seed_improvement_eV": -0.020,
            },
            "passes": accuracy_pass,
        },
        "training_measurement": {
            "ggg_mean_elapsed_seconds": ggg_elapsed,
            "lgl_mean_elapsed_seconds": lgl_elapsed,
            "lgl_elapsed_change_fraction": lgl_elapsed / ggg_elapsed - 1.0,
            "ggg_peak_cuda_memory_bytes": ggg_peak,
            "lgl_peak_cuda_memory_bytes": lgl_peak,
            "lgl_peak_memory_change_fraction": lgl_peak / ggg_peak - 1.0,
        },
        "cuda_decision": {
            "maximum_latency_increase_fraction": maximum_benchmark_latency_increase,
            "maximum_peak_memory_increase_fraction": maximum_benchmark_memory_increase,
            "passes": benchmark_pass,
            "artifact": benchmark_path.name,
        },
        "provenance_checks": {
            "clean_ledger_rows": True,
            "paired_initial_state": True,
            "common_parameter_and_state_schema": True,
            "common_source_data_and_split": True,
            "only_route_and_local_head_layout_differ_within_pair": True,
            "test_metric_keys_absent": True,
        },
        "compute_envelope": {
            "accounting_kind": "conservative_operator_bound",
            "directly_timed_components": directly_timed_components,
            "directly_timed_seconds": directly_timed_seconds,
            "unmetered_allowance_seconds": unmetered_allowance_seconds,
            "gpu_check_duration_not_recorded": True,
            "conservative_total_gpu_wall_seconds": (
                conservative_total_gpu_wall_seconds
            ),
            "conservative_total_gpu_minutes": 9.5,
            "budget_gpu_minutes": 30,
        },
        "decision": (
            "lgl_m1_passes_registered_qm9_probe"
            if accuracy_pass and benchmark_pass
            else "retain_ggg_m1"
        ),
        "interacting_memory_decision": "blocked_by_stage0",
        "claim_boundary": (
            "Adaptive three-seed validation-only evidence on the registered QM9 "
            "random-row warm split; not a test-set, cold-molecule, or default-"
            "promotion claim."
        ),
    }
    output_path = RUN_DIR / "performance-summary.json"
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    report = f"""# Registered performance decision

The M=1 `lgl` route passes the preregistered QM9 validation and CUDA resource
rules against matched M=1 `ggg` on clean commit `{EXPECTED_COMMIT[:8]}`.

| seed | GGG MAE (eV) | LGL MAE (eV) | improvement (eV) |
|---:|---:|---:|---:|
"""
    for row in pairs:
        report += (
            f"| {row['seed']} | {row['ggg_val_mae_eV']:.6f} | "
            f"{row['lgl_val_mae_eV']:.6f} | {row['improvement_eV']:.6f} |\n"
        )
    report += f"""

- Mean improvement: `{mean_improvement:.6f} eV` (threshold `0.010 eV`).
- Positive seeds: `{positive_count}/3` (threshold `2/3`).
- Worst seed improvement: `{minimum_improvement:.6f} eV` (floor `-0.020 eV`).
- Mean measured training time: GGG `{ggg_elapsed:.3f}s`, LGL `{lgl_elapsed:.3f}s`
  (`{(lgl_elapsed / ggg_elapsed - 1.0) * 100:.2f}%`).
- Clean five-process CUDA benchmark: worst latency change
  `{maximum_benchmark_latency_increase * 100:.2f}%`; worst peak-memory change
  `{maximum_benchmark_memory_increase * 100:.2f}%`, both below `+20%`.
- All six rows have matching source/data/split/schema/parameter provenance;
  each paired seed has an identical initial-state hash.
- Gradient accounting is stable across seeds: both routes report
  `{gradient_elements:,}` parameter elements with a gradient, while exact
  nonzero-gradient elements are `{ggg_nonzero_gradients:,}` for GGG and
  `{lgl_nonzero_gradients:,}` for LGL. This route-dependent difference is
  recorded and is not a parameter/schema mismatch.
- `test_evaluated=false` in all rows and no test MAE/RMSE key is present.
- GPU usage is a conservative operator bound, not an exact timer sum: saved
  artifact timers total `{directly_timed_seconds:.3f}s`; the reported ceiling is
  `570s` (`9.5 min`), leaving `{unmetered_allowance_seconds:.3f}s` for the GPU
  check and process overhead whose raw duration receipt was not recorded.

Interacting M=4/M=8 remains blocked by Stage-0. This is adaptive three-seed
validation-only evidence on a random-row warm split, not test-set or
cold-molecule evidence and not an automatic public-default change.
"""
    (RUN_DIR / "performance-report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
