from __future__ import annotations

from pathlib import Path
import runpy

import pytest


def _symbols() -> dict[str, object]:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_whitened_qm9_smoke.py"
    )
    return runpy.run_path(script)


def _record(
    arm: str,
    *,
    mae: float,
    status: str = "completed",
    steps: int = 500,
) -> dict[str, object]:
    return {
        "smoke_arm": arm,
        "status": status,
        "steps": steps,
        "val_mae": mae,
        "val_rmse": mae + 0.1,
        "step_latency_median_seconds": 1.0,
        "peak_cuda_memory_bytes": 100,
        "test_evaluated": False,
        "paired_base_initial_state_sha256": "shared",
        "split_hashes": {
            "train": "train",
            "validation": "validation",
            "test": "test",
        },
    }


def test_plan_freezes_the_qm9_safety_contract() -> None:
    symbols = _symbols()
    args = symbols["parse_args"](["artifacts/qm9-smoke", "--dry-run"])

    plan = symbols["_plan"](args)

    assert plan["arms"] == ["candidate", "whitened_ridge_0p1"]
    assert plan["model_seed"] == 42
    assert plan["steps"] == 500
    assert plan["maximum_absolute_mae_delta_eV"] == pytest.approx(0.020)
    assert plan["prediction"] == "whitening does not improve QM9 gap"
    assert plan["test_evaluated"] is False


def test_decision_passes_a_finite_bounded_change_in_either_direction() -> None:
    symbols = _symbols()

    result = symbols["decision"](
        [
            _record("candidate", mae=0.50),
            _record("whitened_ridge_0p1", mae=0.515),
        ]
    )

    assert result["passed"] is True
    assert result["mae_delta_eV"] == pytest.approx(0.015)
    assert result["whitened_improvement_eV"] == pytest.approx(-0.015)
    assert result["default_change_authorized"] is False


def test_decision_rejects_a_large_change_or_incomplete_pair() -> None:
    symbols = _symbols()

    large = symbols["decision"](
        [
            _record("candidate", mae=0.50),
            _record("whitened_ridge_0p1", mae=0.53),
        ]
    )
    missing = symbols["decision"]([_record("candidate", mae=0.50)])

    assert large["passed"] is False
    assert large["criteria"]["absolute_mae_delta_at_most_0.02_eV"] is False
    assert missing["passed"] is False
    assert "missing" in str(missing["reason"])


def test_build_command_keeps_the_feature_and_test_boundary_matched() -> None:
    symbols = _symbols()
    candidate = symbols["build_command"](
        "candidate",
        output=Path("candidate.json"),
        steps=500,
        device="cuda",
    )
    whitened = symbols["build_command"](
        "whitened_ridge_0p1",
        output=Path("whitened.json"),
        steps=500,
        device="cuda",
    )

    shared = {
        "--gated-local-transport",
        "--grouped-invariant-normalization",
        "--no-key-balancing",
        "--precompute-local-edges",
        "--skip-test-eval",
    }
    assert shared.issubset(candidate)
    assert shared.issubset(whitened)
    assert "--whitened-global-read" not in candidate
    assert "--whitened-global-read" in whitened
    ridge_index = whitened.index("--whitened-global-ridge")
    assert whitened[ridge_index + 1] == "0.1"


def test_rank_gated_commands_use_an_identical_schema_control() -> None:
    symbols = _symbols()
    control = symbols["build_command"](
        "candidate",
        output=Path("control.json"),
        steps=500,
        device="cuda",
        rank_gated_schema_control=True,
    )
    active = symbols["build_command"](
        "whitened_ridge_0p1",
        output=Path("active.json"),
        steps=500,
        device="cuda",
        rank_gated_schema_control=True,
    )

    for command in (control, active):
        assert "--whitened-global-read" in command
        assert "--whitened-global-rank-gate" in command
    assert "--freeze-whitened-global-mix" in control
    assert "--freeze-whitened-global-mix" not in active
