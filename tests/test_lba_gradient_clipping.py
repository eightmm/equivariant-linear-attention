from __future__ import annotations

from pathlib import Path
import runpy

import pytest


def _symbols() -> dict[str, object]:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_lba_gradient_clipping.py"
    )
    return runpy.run_path(script)


def _record(
    policy: str,
    *,
    last_rmse: float,
    best_rmse: float,
    scale: float,
    latency: float = 1.0,
    memory: int = 100,
) -> dict[str, object]:
    return {
        "policy": policy,
        "status": "completed",
        "last_epoch_validation": {"rmse_pK": last_rmse},
        "best_validation": {"rmse_pK": best_rmse},
        "step_latency_median_seconds": latency,
        "peak_cuda_memory_bytes": memory,
        "gradient_monitor": {
            "effective_grad_scale_mean": scale,
            "pre_clip_grad_norm_mean": 2.0,
        },
        "global_steps": 100,
        "initial_state_sha256": "same",
        "test_evaluated": False,
    }


def test_plan_freezes_three_policies_and_no_test_boundary() -> None:
    symbols = _symbols()
    args = symbols["parse_args"](["artifacts/lba-clip", "--dry-run"])

    plan = symbols["_plan"](args)

    assert plan["model_seed"] == 44
    assert plan["epochs"] == 20
    assert plan["policies"] == {
        "global_1": {"global_grad_clip": 1.0},
        "global_10": {"global_grad_clip": 10.0},
        "none": {"global_grad_clip": None},
    }
    assert plan["test_evaluated"] is False


def test_decision_admits_only_a_matched_accuracy_improvement() -> None:
    decision = _symbols()["_decision"](
        [
            _record(
                "global_1",
                last_rmse=1.60,
                best_rmse=1.58,
                scale=0.10,
            ),
            _record(
                "global_10",
                last_rmse=1.57,
                best_rmse=1.57,
                scale=0.50,
            ),
            _record(
                "none",
                last_rmse=1.61,
                best_rmse=1.59,
                scale=1.00,
            ),
        ]
    )

    assert decision["passed"] is True
    assert decision["selected_policy"] == "global_10"
    assert decision["default_change_authorized"] is False
    assert decision["alternatives"]["global_10"]["passed"] is True
    assert decision["alternatives"]["none"]["passed"] is False


def test_decision_rejects_scale_change_without_accuracy_gain() -> None:
    decision = _symbols()["_decision"](
        [
            _record(
                "global_1",
                last_rmse=1.60,
                best_rmse=1.58,
                scale=0.10,
            ),
            _record(
                "global_10",
                last_rmse=1.59,
                best_rmse=1.57,
                scale=0.50,
            ),
            _record(
                "none",
                last_rmse=1.60,
                best_rmse=1.58,
                scale=1.00,
            ),
        ]
    )

    assert decision["passed"] is False
    assert decision["selected_policy"] is None
    assert decision["alternatives"]["global_10"][
        "last_rmse_improvement_pK"
    ] == pytest.approx(0.01)
