from __future__ import annotations

from pathlib import Path
import runpy

import pytest


def _symbols() -> dict[str, object]:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_whitened_ridge_screen.py"
    )
    return runpy.run_path(script)


def _record(
    arm: str,
    *,
    last_rmse: float,
    ridge: float | None = None,
    latency: float = 1.0,
    memory: int = 100,
    status: str = "completed",
    steps: int = 4_400,
    mix: float = 0.05,
) -> dict[str, object]:
    return {
        "screen_arm": arm,
        "arm": "candidate" if ridge is None else "whitened",
        "whitened_global_ridge": ridge,
        "status": status,
        "last_epoch_validation": {"rmse_pK": last_rmse},
        "best_validation": {"rmse_pK": last_rmse - 0.01},
        "step_latency_median_seconds": latency,
        "peak_cuda_memory_bytes": memory,
        "global_steps": steps,
        "test_evaluated": False,
        "whitened_mix_magnitude": {
            "scalar_absmax": 0.0 if ridge is None else mix,
            "vector_absmax": 0.0 if ridge is None else mix,
            "per_layer_scalar_absmax": [] if ridge is None else [mix],
            "per_layer_vector_absmax": [] if ridge is None else [mix],
        },
    }


def test_plan_freezes_the_ridge_grid_thresholds_and_no_test_boundary() -> None:
    symbols = _symbols()
    args = symbols["parse_args"](["artifacts/ridge", "--dry-run"])

    plan = symbols["_plan"](args)

    assert plan["arms"] == [
        "candidate",
        "whitened_ridge_0p5",
        "whitened_ridge_0p1",
        "whitened_ridge_0p01",
    ]
    assert plan["ridge_grid"] == [0.5, 0.1, 0.01]
    assert plan["model_seed"] == 44
    assert plan["epochs"] == 20
    assert plan["acceptance"] == {
        "must_improve_last_rmse_over_candidate": True,
        "maximum_regression_pK": 0.050,
        "maximum_step_latency_ratio": 1.25,
        "maximum_peak_memory_ratio": 1.25,
    }
    assert plan["expected_topology_sha256"] == (
        "57f40fb157e6416558db5507d95c3a5e4f828881e0bc92e142e1b85de802dc6c"
    )
    assert plan["test_evaluated"] is False
    assert plan["shared_topology"].startswith("one in-memory")


def test_decision_selects_the_best_improving_ridge() -> None:
    symbols = _symbols()

    result = symbols["decision"](
        [
            _record("candidate", last_rmse=1.600),
            _record("whitened_ridge_0p5", last_rmse=1.590, ridge=0.5),
            _record("whitened_ridge_0p1", last_rmse=1.570, ridge=0.1),
            _record("whitened_ridge_0p01", last_rmse=1.640, ridge=0.01),
        ]
    )

    assert result["passed"] is True
    assert result["selected_arm"] == "whitened_ridge_0p1"
    assert result["selected_ridge"] == 0.1
    assert result["advances_to_multiseed_confirmation"] is True
    assert result["default_change_authorized"] is False
    assert result["ridges"]["whitened_ridge_0p01"]["passed"] is False
    assert result["ridges"]["whitened_ridge_0p1"][
        "last_rmse_improvement_pK"
    ] == pytest.approx(0.030)


def test_decision_rejects_every_regressing_ridge() -> None:
    symbols = _symbols()

    result = symbols["decision"](
        [
            _record("candidate", last_rmse=1.600),
            _record("whitened_ridge_0p5", last_rmse=1.601, ridge=0.5),
            _record("whitened_ridge_0p1", last_rmse=1.700, ridge=0.1),
        ]
    )

    assert result["passed"] is False
    assert result["selected_arm"] is None
    assert result["advances_to_multiseed_confirmation"] is False
    assert result["ridges"]["whitened_ridge_0p5"]["criteria"][
        "improves_last_rmse"
    ] is False
    assert result["ridges"]["whitened_ridge_0p1"]["criteria"][
        "regression_at_most_0.05_pK"
    ] is False


def test_decision_rejects_a_resource_or_budget_failure() -> None:
    symbols = _symbols()

    slow = symbols["decision"](
        [
            _record("candidate", last_rmse=1.600),
            _record("whitened_ridge_0p1", last_rmse=1.560, ridge=0.1, latency=1.4),
        ]
    )
    assert slow["passed"] is False
    assert slow["ridges"]["whitened_ridge_0p1"]["criteria"][
        "latency_ratio_at_most_1.25"
    ] is False

    hungry = symbols["decision"](
        [
            _record("candidate", last_rmse=1.600),
            _record("whitened_ridge_0p1", last_rmse=1.560, ridge=0.1, memory=200),
        ]
    )
    assert hungry["passed"] is False
    assert hungry["ridges"]["whitened_ridge_0p1"]["criteria"][
        "peak_memory_ratio_at_most_1.25"
    ] is False

    unfinished = symbols["decision"](
        [
            _record("candidate", last_rmse=1.600),
            _record(
                "whitened_ridge_0p1",
                last_rmse=1.560,
                ridge=0.1,
                status="budget_exhausted",
                steps=900,
            ),
        ]
    )
    assert unfinished["passed"] is False
    assert unfinished["ridges"]["whitened_ridge_0p1"]["criteria"][
        "matched_update_count"
    ] is False


def test_decision_reports_whether_the_lane_was_used() -> None:
    symbols = _symbols()

    result = symbols["decision"](
        [
            _record("candidate", last_rmse=1.600),
            _record("whitened_ridge_0p1", last_rmse=1.590, ridge=0.1, mix=0.0),
            _record("whitened_ridge_0p5", last_rmse=1.595, ridge=0.5, mix=0.2),
        ]
    )

    assert result["ridges"]["whitened_ridge_0p1"]["lane_active"] is False
    assert result["ridges"]["whitened_ridge_0p5"]["lane_active"] is True


def test_decision_requires_the_candidate_baseline() -> None:
    symbols = _symbols()

    result = symbols["decision"](
        [_record("whitened_ridge_0p1", last_rmse=1.590, ridge=0.1)]
    )

    assert result["passed"] is False
    assert "baseline" in str(result["reason"])


def test_arm_names_are_filesystem_safe_and_unique() -> None:
    symbols = _symbols()

    names = [symbols["_arm_name"](ridge) for ridge in (0.5, 0.1, 0.01, 1.0)]

    assert names == [
        "whitened_ridge_0p5",
        "whitened_ridge_0p1",
        "whitened_ridge_0p01",
        "whitened_ridge_1",
    ]
    assert len(set(names)) == len(names)
    assert all(name.replace("_", "").replace("p", "").isalnum() for name in names)


def test_parse_args_rejects_an_invalid_grid_or_budget() -> None:
    symbols = _symbols()

    with pytest.raises(SystemExit):
        symbols["parse_args"](["artifacts/ridge", "--ridge-grid", "0.1", "0.1"])
    with pytest.raises(SystemExit):
        symbols["parse_args"](["artifacts/ridge", "--ridge-grid", "0"])
    with pytest.raises(SystemExit):
        symbols["parse_args"](["artifacts/ridge", "--budget-seconds", "0"])
    with pytest.raises(SystemExit):
        symbols["parse_args"](
            ["artifacts/ridge", "--warmup-epochs", "30", "--epochs", "20"]
        )
