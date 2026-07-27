"""ATOM3D-LBA multi-seed confirmation contracts."""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest


def _symbols() -> dict[str, object]:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_lba_multiseed_confirmation.py"
    )
    return runpy.run_path(script)


def _arm(name: str, rmse: float, latency: float, memory: int) -> dict[str, object]:
    return {
        "arm": name,
        "status": "completed",
        "best_validation": {"rmse_pK": rmse},
        "step_latency_median_seconds": latency,
        "peak_cuda_memory_bytes": memory,
        "gradient_monitor": {"clip_fraction": 0.99},
    }


def _record(
    seed: int,
    candidate_rmse: float,
    incumbent_rmse: float,
) -> dict[str, object]:
    return {
        "model_seed": seed,
        "arm_results": [
            _arm("candidate", candidate_rmse, 0.02, 1_400),
            _arm("incumbent", incumbent_rmse, 0.02, 1_000),
        ],
    }


def test_command_freezes_seeds_arms_and_test_boundary() -> None:
    symbols = _symbols()
    command = symbols["build_command"](
        seed=42,
        output_dir=Path("out"),
        device="cuda",
        budget_seconds=600.0,
    )

    assert command[command.index("--model-seed") + 1] == "42"
    assert command[command.index("--order-seed") + 1] == "42"
    arms = command[command.index("--arms") + 1 : command.index("--batch-size")]
    assert arms == [
        "candidate",
        "incumbent",
        "--arm-budget-weights",
        "1",
        "1",
    ]
    assert command[command.index("--max-epochs") + 1] == "35"
    assert command[command.index("--min-epochs") + 1] == "35"
    assert "--evaluate-test" not in command


def test_decision_requires_consistent_gain_and_resource_bounds() -> None:
    symbols = _symbols()
    records = [
        _record(41, 1.50, 1.54),
        _record(42, 1.51, 1.54),
        _record(43, 1.52, 1.52),
    ]

    result = symbols["decision"](records)

    assert result["passed"] is True
    assert result["mean_improvement_pK"] == pytest.approx(0.0233333333333333)
    assert result["improving_seed_count"] == 2
    assert result["median_peak_memory_ratio"] == pytest.approx(1.4)


def test_decision_rejects_a_one_seed_only_gain() -> None:
    symbols = _symbols()
    records = [
        _record(41, 1.40, 1.55),
        _record(42, 1.56, 1.54),
        _record(43, 1.56, 1.54),
    ]

    result = symbols["decision"](records)

    assert result["passed"] is False
    assert result["improving_seed_count"] == 1
