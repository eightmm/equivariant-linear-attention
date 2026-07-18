from __future__ import annotations

import json
import runpy
from pathlib import Path


def _symbols() -> dict[str, object]:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_registered_transport_study.py"
    )
    return runpy.run_path(script)


def _record(
    seed: int,
    val_mae: float,
    *,
    elapsed_seconds: float = 10.0,
    peak_cuda_memory_bytes: int = 100,
) -> dict[str, object]:
    return {
        "model_seed": seed,
        "val_mae": val_mae,
        "elapsed_seconds": elapsed_seconds,
        "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
    }


def test_registered_study_builds_only_the_frozen_validation_runs() -> None:
    symbols = _symbols()
    screen = symbols["registered_screen_arms"]()
    confirmation = symbols["registered_confirmation_arms"]()
    egnn = symbols["registered_egnn_arms"]()

    assert [(arm["routing"], arm["transport_mode"]) for arm in screen] == [
        ("ggg", "learned"),
        ("lgg", "learned"),
        ("ggl", "learned"),
        ("lgl", "learned"),
        ("lgl", "uniform"),
        ("lgl", "none"),
    ]
    assert {arm["model_seed"] for arm in screen} == {42}
    assert {arm["steps"] for arm in screen} == {500}

    assert len(confirmation) == 15
    assert {arm["routing"] for arm in confirmation} == {"lgl"}
    assert {arm["transport_mode"] for arm in confirmation} == {
        "learned",
        "uniform",
        "none",
    }
    assert {arm["model_seed"] for arm in confirmation} == set(range(41, 46))
    assert {arm["steps"] for arm in confirmation} == {2_000}

    assert len(egnn) == 5
    assert {arm["benchmark_model"] for arm in egnn} == {"internal_static_egnn_baseline"}
    assert {arm["model_seed"] for arm in egnn} == set(range(41, 46))
    assert {arm["steps"] for arm in egnn} == {2_000}

    for arm in [*screen, *confirmation, *egnn]:
        command = symbols["build_train_command"](
            arm,
            Path("artifacts/study") / f"{arm['name']}.json",
        )
        assert command[:5] == [
            "uv",
            "run",
            "--locked",
            "python",
            "scripts/train_compare.py",
        ]
        assert "--dataset" in command and "qm9" in command
        assert "--num-samples" in command and "130000" in command
        assert "--train-size" in command and "110000" in command
        assert "--val-size" in command and "10000" in command
        assert "--split-seed" in command and "42" in command
        assert "--device" in command and "cuda" in command
        assert "--skip-test-eval" in command
        assert "--evaluate-test" not in command
        assert "--amp-dtype" not in command


def test_five_seed_promotion_rule_enforces_every_registered_threshold() -> None:
    symbols = _symbols()
    baseline = [_record(seed, 0.60) for seed in range(41, 46)]
    candidate = [
        _record(41, 0.58, elapsed_seconds=11.0, peak_cuda_memory_bytes=110),
        _record(42, 0.59, elapsed_seconds=11.0, peak_cuda_memory_bytes=110),
        _record(43, 0.58, elapsed_seconds=11.0, peak_cuda_memory_bytes=110),
        _record(44, 0.61, elapsed_seconds=11.0, peak_cuda_memory_bytes=110),
        _record(45, 0.57, elapsed_seconds=11.0, peak_cuda_memory_bytes=110),
    ]

    decision = symbols["paired_promotion_decision"](candidate, baseline)

    assert decision["passed"]
    assert decision["mean_improvement_eV"] >= 0.010
    assert decision["improving_seed_count"] == 4
    assert decision["worst_seed_improvement_eV"] >= -0.020
    assert decision["median_elapsed_ratio"] <= 1.20
    assert decision["median_peak_memory_ratio"] <= 1.20

    candidate[3] = _record(44, 0.621)
    assert not symbols["paired_promotion_decision"](candidate, baseline)["passed"]


def test_uniform_is_selected_when_selectivity_is_unproven() -> None:
    symbols = _symbols()
    records = {
        "learned": [_record(seed, 0.55) for seed in range(41, 46)],
        "uniform": [_record(seed, 0.555) for seed in range(41, 46)],
        "none": [_record(seed, 0.58) for seed in range(41, 46)],
    }

    decision = symbols["transport_decision"](records)

    assert not decision["learned_selectivity"]["passed"]
    assert decision["uniform_global_transport"]["passed"]
    assert decision["selected_mode"] == "uniform"
    assert decision["transport_locked"]


def test_transport_does_not_lock_when_neither_global_arm_beats_none() -> None:
    symbols = _symbols()
    records = {
        "learned": [_record(seed, 0.575) for seed in range(41, 46)],
        "uniform": [_record(seed, 0.576) for seed in range(41, 46)],
        "none": [_record(seed, 0.58) for seed in range(41, 46)],
    }

    decision = symbols["transport_decision"](records)

    assert not decision["transport_locked"]
    assert decision["selected_mode"] is None


def test_completed_study_is_recorded_without_test_or_egnn_evaluation() -> None:
    root = Path(__file__).resolve().parents[1]
    expected_command = [
        "uv",
        "run",
        "--locked",
        "python",
        "scripts/run_registered_transport_study.py",
        "artifacts/evidence-first-strengthening-20260719",
        "--summary-out",
        "artifacts/evidence-first-strengthening-20260719/study-summary.json",
    ]
    records = [
        json.loads(line)
        for line in (root / "docs" / "EXPERIMENTS.jsonl").read_text().splitlines()
    ]
    matches = [record for record in records if record["cmd"] == expected_command]

    assert matches
    metrics = matches[-1]["metrics"]
    assert matches[-1]["exit"] == 0
    assert metrics["completed_run_count"] == 21
    assert metrics["transport_locked"] is False
    assert metrics["egnn_run_count"] == 0
    assert metrics["test_evaluated"] is False
