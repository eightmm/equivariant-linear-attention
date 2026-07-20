from __future__ import annotations

import json
import runpy
from pathlib import Path


def _symbols() -> dict[str, object]:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_registered_coordinate_study.py"
    )
    return runpy.run_path(script)


def _record(
    seed: int,
    val_mae: float,
    *,
    coordinate_updates: bool,
    active: bool | None = None,
    elapsed_seconds: float = 10.0,
    peak_cuda_memory_bytes: int = 100,
) -> dict[str, object]:
    if active is None:
        active = coordinate_updates
    return {
        "model_seed": seed,
        "val_mae": val_mae,
        "elapsed_seconds": elapsed_seconds,
        "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
        "run_config": {"coordinate_updates": coordinate_updates},
        "coordinate_diagnostics": {
            "enabled": coordinate_updates,
            "active": active,
            "centroid_drift_max_angstrom": 0.0,
            "layers": (
                [
                    {
                        "step_max_angstrom": 0.1,
                        "centroid_drift_max_angstrom": 0.0,
                    },
                    {
                        "step_max_angstrom": 0.1,
                        "centroid_drift_max_angstrom": 0.0,
                    },
                ]
                if coordinate_updates
                else []
            ),
        },
    }


def test_registered_coordinate_study_builds_only_the_frozen_runs() -> None:
    symbols = _symbols()
    screen = symbols["registered_screen_arms"]()
    confirmation = symbols["registered_confirmation_arms"]("lgl")

    assert [arm["name"] for arm in screen] == [
        "screen-attention-ggg-static",
        "screen-attention-ggg-dynamic",
        "screen-attention-lgl-static",
        "screen-attention-lgl-dynamic",
        "screen-egnn-static",
        "screen-egnn-dynamic",
    ]
    assert {arm["model_seed"] for arm in screen} == {42}
    assert {arm["steps"] for arm in screen} == {500}
    assert len(confirmation) == 20
    assert {arm["model_seed"] for arm in confirmation} == set(range(41, 46))
    assert {arm["steps"] for arm in confirmation} == {2_000}
    assert {arm["routing"] for arm in confirmation if "routing" in arm} == {"lgl"}

    for arm in [*screen, *confirmation]:
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
        assert ("--coordinate-updates" in command) is bool(
            arm["coordinate_updates"]
            and arm["benchmark_model"] == "factorized_moment"
        )


def test_screen_selects_lowest_eligible_dynamic_attention_route() -> None:
    symbols = _symbols()
    arms = symbols["registered_screen_arms"]()
    records = [
        _record(42, 0.60, coordinate_updates=False),
        _record(42, 0.59, coordinate_updates=True),
        _record(42, 0.58, coordinate_updates=False),
        _record(42, 0.57, coordinate_updates=True),
        _record(42, 0.62, coordinate_updates=False),
        _record(42, 0.60, coordinate_updates=True),
    ]

    decision = symbols["screen_route_decision"](arms, records)

    assert decision["selected_attention_route"] == "lgl"
    assert decision["attention_route_eligible"]["ggg"] is True
    assert decision["attention_route_eligible"]["lgl"] is True
    assert decision["egnn_dynamic_eligible"] is True

    records[3] = _record(
        42, 0.57, coordinate_updates=True, active=False
    )
    decision = symbols["screen_route_decision"](arms, records)
    assert decision["selected_attention_route"] == "ggg"
    assert decision["attention_route_eligible"]["lgl"] is False


def test_screen_rejects_dynamic_arm_more_than_point_zero_two_ev_worse() -> None:
    symbols = _symbols()
    arms = symbols["registered_screen_arms"]()
    records = [
        _record(42, 0.60, coordinate_updates=False),
        _record(42, 0.621, coordinate_updates=True),
        _record(42, 0.58, coordinate_updates=False),
        _record(42, 0.601, coordinate_updates=True),
        _record(42, 0.62, coordinate_updates=False),
        _record(42, 0.641, coordinate_updates=True),
    ]

    decision = symbols["screen_route_decision"](arms, records)

    assert decision["selected_attention_route"] is None
    assert not any(decision["attention_route_eligible"].values())
    assert decision["egnn_dynamic_eligible"] is False


def test_coordinate_promotion_rule_enforces_every_registered_threshold() -> None:
    symbols = _symbols()
    static = [
        _record(seed, 0.60, coordinate_updates=False)
        for seed in range(41, 46)
    ]
    dynamic = [
        _record(41, 0.58, coordinate_updates=True, elapsed_seconds=11.0, peak_cuda_memory_bytes=110),
        _record(42, 0.59, coordinate_updates=True, elapsed_seconds=11.0, peak_cuda_memory_bytes=110),
        _record(43, 0.58, coordinate_updates=True, elapsed_seconds=11.0, peak_cuda_memory_bytes=110),
        _record(44, 0.61, coordinate_updates=True, elapsed_seconds=11.0, peak_cuda_memory_bytes=110),
        _record(45, 0.57, coordinate_updates=True, elapsed_seconds=11.0, peak_cuda_memory_bytes=110),
    ]

    decision = symbols["paired_coordinate_decision"](dynamic, static)

    assert decision["passed"]
    assert decision["mean_improvement_eV"] >= 0.010
    assert decision["improving_seed_count"] == 4
    assert decision["worst_seed_improvement_eV"] >= -0.020
    assert decision["median_elapsed_ratio"] <= 1.20
    assert decision["median_peak_memory_ratio"] <= 1.20

    dynamic[3] = _record(44, 0.621, coordinate_updates=True)
    assert not symbols["paired_coordinate_decision"](dynamic, static)["passed"]


def test_completed_coordinate_study_is_recorded_without_promotion_or_test() -> None:
    root = Path(__file__).resolve().parents[1]
    expected_command = [
        "uv",
        "run",
        "--locked",
        "python",
        "scripts/run_registered_coordinate_study.py",
        "artifacts/dynamic-coordinate-egnn-20260719",
        "--summary-out",
        "artifacts/dynamic-coordinate-egnn-20260719/study-summary.json",
    ]
    records = [
        json.loads(line)
        for line in (root / "docs" / "EXPERIMENTS.jsonl").read_text().splitlines()
    ]
    matches = [record for record in records if record["cmd"] == expected_command]

    assert matches
    metrics = matches[-1]["metrics"]
    assert matches[-1]["exit"] == 0
    assert metrics["completed_run_count"] == 26
    assert metrics["confirmation_run_count"] == 20
    assert metrics["attention_coordinate_promoted"] is False
    assert metrics["egnn_coordinate_promoted"] is False
    assert metrics["test_evaluated"] is False
