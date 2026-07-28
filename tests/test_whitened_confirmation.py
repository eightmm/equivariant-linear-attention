from __future__ import annotations

from pathlib import Path
import runpy

import pytest


def _symbols() -> dict[str, object]:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_whitened_confirmation.py"
    )
    return runpy.run_path(script)


def _record(
    arm: str,
    seed: int,
    *,
    last_rmse: float,
    ridge: float | None = None,
    latency: float = 1.0,
    memory: int = 100,
    status: str = "completed",
    steps: int = 4_400,
) -> dict[str, object]:
    return {
        "confirmation_arm": arm,
        "arm": "candidate" if ridge is None else "whitened",
        "whitened_global_ridge": ridge,
        "model_seed": seed,
        "order_seed": seed,
        "status": status,
        "last_epoch_validation": {
            "rmse_pK": last_rmse,
            "mae_pK": last_rmse - 0.2,
            "pearson": 0.5,
            "spearman": 0.5,
        },
        "best_validation": {
            "rmse_pK": last_rmse - 0.01,
            "mae_pK": last_rmse - 0.21,
            "pearson": 0.51,
            "spearman": 0.51,
        },
        "step_latency_median_seconds": latency,
        "peak_cuda_memory_bytes": memory,
        "global_steps": steps,
        "test_evaluated": False,
        "paired_base_initial_state_sha256": f"seed-{seed}",
        "gradient_monitor": {
            "clip_fraction": 0.98,
            "pre_clip_grad_norm_mean": 10.0,
            "effective_grad_scale_mean": 0.18,
        },
    }


def _paired(
    improvements: dict[int, float],
    *,
    ridge: float,
    **overrides: object,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    arm = f"whitened_ridge_{str(ridge).replace('.', 'p')}"
    for seed, improvement in improvements.items():
        records.append(_record("candidate", seed, last_rmse=1.60))
        records.append(
            _record(
                arm,
                seed,
                last_rmse=1.60 - improvement,
                ridge=ridge,
                **overrides,
            )
        )
    return records


def test_plan_freezes_primary_secondary_ridges_and_topology() -> None:
    symbols = _symbols()
    args = symbols["parse_args"](["artifacts/whitened-confirm", "--dry-run"])

    plan = symbols["_plan"](args)

    assert plan["model_seeds"] == [41, 42, 43]
    assert plan["order_seeds"] == [41, 42, 43]
    assert plan["primary_ridge"] == pytest.approx(0.1)
    assert plan["secondary_ridge"] == pytest.approx(0.01)
    assert plan["epochs"] == 20
    assert plan["primary_metric"] == "last-epoch validation RMSE in pK"
    assert plan["acceptance"]["minimum_mean_improvement_pK"] == pytest.approx(
        0.020
    )
    assert plan["test_evaluated"] is False
    assert len(plan["expected_topology_sha256"]) == 64


def test_decision_promotes_only_the_preregistered_primary_ridge() -> None:
    symbols = _symbols()
    records = _paired(
        {41: 0.030, 42: 0.025, 43: 0.020},
        ridge=0.1,
    ) + [
        record
        for record in _paired(
            {41: 0.040, 42: 0.035, 43: 0.030},
            ridge=0.01,
        )
        if record["confirmation_arm"] != "candidate"
    ]

    result = symbols["decision"](records, qm9_safety_passed=True)

    assert result["passed"] is True
    assert result["default_change_authorized"] is True
    assert result["selected_ridge"] == pytest.approx(0.1)
    assert result["primary"]["mean_improvement_pK"] == pytest.approx(0.025)
    assert result["secondary"]["mean_improvement_pK"] == pytest.approx(0.035)


def test_secondary_success_cannot_rescue_a_failed_primary() -> None:
    symbols = _symbols()
    records = _paired(
        {41: 0.010, 42: 0.005, 43: -0.005},
        ridge=0.1,
    ) + [
        record
        for record in _paired(
            {41: 0.040, 42: 0.035, 43: 0.030},
            ridge=0.01,
        )
        if record["confirmation_arm"] != "candidate"
    ]

    result = symbols["decision"](records, qm9_safety_passed=True)

    assert result["passed"] is False
    assert result["default_change_authorized"] is False
    assert result["primary"]["passed"] is False
    assert result["secondary"]["passed"] is True
    assert result["exact_ridge_resolved"] is False


def test_rank_gated_confirmation_uses_the_schema_matched_control() -> None:
    symbols = _symbols()
    records: list[dict[str, object]] = []
    for seed, improvement in {41: 0.030, 42: 0.025, 43: 0.020}.items():
        control = _record(
            symbols["RANK_CONTROL_ARM"],
            seed,
            last_rmse=1.60,
            ridge=0.1,
        )
        active = _record(
            symbols["RANK_ACTIVE_ARM"],
            seed,
            last_rmse=1.60 - improvement,
            ridge=0.1,
        )
        control["initial_state_sha256"] = f"full-{seed}"
        active["initial_state_sha256"] = f"full-{seed}"
        records.extend([control, active])

    result = symbols["decision"](
        records,
        qm9_safety_passed=True,
        rank_gated_schema_control=True,
    )

    assert result["passed"] is True
    assert result["schema_matched_control"] is True
    assert result["secondary"] is None
    assert result["primary"]["criteria"]["full_initial_state_matches"] is True


def test_decision_rejects_resource_or_paired_seed_failures() -> None:
    symbols = _symbols()
    records = _paired(
        {41: 0.030, 42: 0.025, 43: 0.020},
        ridge=0.1,
        latency=1.30,
    ) + [
        record
        for record in _paired(
            {41: 0.030, 42: 0.025, 43: 0.020},
            ridge=0.01,
        )
        if record["confirmation_arm"] != "candidate"
    ]

    result = symbols["decision"](records, qm9_safety_passed=True)

    assert result["passed"] is False
    assert (
        result["primary"]["criteria"]["latency_ratio_at_most_1.25"] is False
    )


def test_parse_args_rejects_contract_changes() -> None:
    symbols = _symbols()

    with pytest.raises(SystemExit):
        symbols["parse_args"](
            ["artifacts/whitened-confirm", "--epochs", "21"]
        )
    with pytest.raises(SystemExit):
        symbols["parse_args"](
            ["artifacts/whitened-confirm", "--budget-seconds", "0"]
        )
