from __future__ import annotations

from pathlib import Path
import runpy

import pytest


def _symbols() -> dict[str, object]:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_lba_clipping_confirmation.py"
    )
    return runpy.run_path(script)


def _record(
    policy: str,
    seed: int,
    *,
    last_rmse: float,
    latency: float = 1.0,
    memory: int = 100,
    status: str = "completed",
    steps: int = 100,
) -> dict[str, object]:
    return {
        "policy": policy,
        "model_seed": seed,
        "order_seed": seed,
        "status": status,
        "last_epoch_validation": {"rmse_pK": last_rmse},
        "best_validation": {"rmse_pK": last_rmse},
        "step_latency_median_seconds": latency,
        "peak_cuda_memory_bytes": memory,
        "global_steps": steps,
        "initial_state_sha256": f"seed-{seed}",
        "test_evaluated": False,
    }


def _paired(improvements: dict[int, float], **overrides: object) -> list[dict]:
    records: list[dict] = []
    for seed, improvement in improvements.items():
        records.append(_record("global_1", seed, last_rmse=1.60))
        records.append(
            _record("none", seed, last_rmse=1.60 - improvement, **overrides)
        )
    return records


def test_plan_freezes_paired_seeds_thresholds_and_no_test_boundary() -> None:
    symbols = _symbols()
    args = symbols["parse_args"](["artifacts/lba-clip-confirm", "--dry-run"])

    plan = symbols["_plan"](args)

    assert plan["model_seeds"] == [41, 42, 43]
    assert plan["order_seeds"] == [41, 42, 43]
    assert plan["policies"] == {
        "global_1": {"global_grad_clip": 1.0},
        "none": {"global_grad_clip": None},
    }
    assert plan["acceptance"]["minimum_mean_improvement_pK"] == 0.020
    assert plan["acceptance"]["minimum_improving_seeds"] == 2
    assert plan["acceptance"]["maximum_paired_regression_pK"] == 0.050
    assert plan["primary_metric"] == "last-epoch validation RMSE in pK"
    assert plan["claim_boundary"] == "paired_three_seed"
    assert plan["test_evaluated"] is False


def test_decision_promotes_a_consistent_paired_improvement() -> None:
    symbols = _symbols()

    result = symbols["decision"](
        _paired({41: 0.030, 42: 0.025, 43: 0.020})
    )

    assert result["passed"] is True
    assert result["default_change_authorized"] is True
    assert result["improving_seed_count"] == 3
    assert result["mean_improvement_pK"] == pytest.approx(0.025)


def test_decision_rejects_a_mean_below_the_frozen_threshold() -> None:
    symbols = _symbols()

    result = symbols["decision"](_paired({41: 0.030, 42: 0.010, 43: 0.005}))

    assert result["passed"] is False
    assert result["default_change_authorized"] is False
    assert result["criteria"]["mean_improvement_at_least_0.02_pK"] is False
    assert result["improving_seed_count"] == 3


def test_decision_rejects_a_single_seed_carrying_the_mean() -> None:
    symbols = _symbols()

    result = symbols["decision"](_paired({41: 0.090, 42: -0.010, 43: -0.005}))

    assert result["passed"] is False
    assert result["criteria"]["improving_seeds_at_least_2"] is False
    assert result["mean_improvement_pK"] == pytest.approx(0.025)


def test_decision_rejects_an_unbounded_worst_seed() -> None:
    symbols = _symbols()

    result = symbols["decision"](_paired({41: 0.200, 42: 0.100, 43: -0.060}))

    assert result["passed"] is False
    assert result["criteria"]["worst_regression_at_most_0.05_pK"] is False
    assert result["worst_paired_regression_pK"] == pytest.approx(0.060)


def test_decision_rejects_a_resource_regression() -> None:
    symbols = _symbols()

    result = symbols["decision"](
        _paired({41: 0.030, 42: 0.025, 43: 0.020}, latency=1.30)
    )

    assert result["passed"] is False
    assert result["criteria"]["latency_ratio_at_most_1.05"] is False


def test_decision_rejects_an_incomplete_or_mismatched_arm() -> None:
    symbols = _symbols()

    unfinished = symbols["decision"](
        _paired({41: 0.030, 42: 0.025, 43: 0.020}, status="budget_exhausted")
    )
    assert unfinished["passed"] is False
    assert unfinished["criteria"]["all_arms_completed_without_test_access"] is False

    unmatched = symbols["decision"](
        _paired({41: 0.030, 42: 0.025, 43: 0.020}, steps=90)
    )
    assert unmatched["passed"] is False
    assert unmatched["criteria"]["matched_update_counts"] is False


def test_decision_requires_every_registered_seed() -> None:
    symbols = _symbols()

    result = symbols["decision"](_paired({41: 0.030, 42: 0.025}))

    assert result["passed"] is False
    assert "43" in str(result["reason"])


def test_parse_args_rejects_an_invalid_budget_or_warmup() -> None:
    symbols = _symbols()

    with pytest.raises(SystemExit):
        symbols["parse_args"](
            ["artifacts/lba-clip-confirm", "--warmup-epochs", "40", "--epochs", "20"]
        )
    with pytest.raises(SystemExit):
        symbols["parse_args"](["artifacts/lba-clip-confirm", "--budget-seconds", "0"])
