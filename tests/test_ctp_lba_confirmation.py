from __future__ import annotations

from pathlib import Path
import runpy


CONFIRM = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "run_ctp_lba_confirmation.py")
)


def _seed(seed: int, current: float, persistent: float, ctp: float) -> dict:
    def arm(name: str, rmse: float) -> dict:
        return {
            "arm": name,
            "status": "completed",
            "best_validation": {"rmse_pK": rmse},
        }

    return {
        "model_seed": seed,
        "arm_results": [
            arm("candidate", current),
            arm("persistent_2e", persistent),
            arm("ctp", ctp),
        ],
    }


def test_promotion_requires_mean_control_gains_wins_and_worst_seed_guard() -> None:
    decision = CONFIRM["_promotion_decision"](
        [
            _seed(41, 1.60, 1.61, 1.56),
            _seed(42, 1.62, 1.63, 1.58),
            _seed(43, 1.61, 1.60, 1.62),
        ],
        resource_gate_passed=True,
    )

    assert decision["mean_improvement_vs_candidate_pK"] >= 0.020
    assert decision["mean_improvement_vs_persistent_2e_pK"] > 0.0
    assert decision["paired_win_count_vs_candidate"] == 2
    assert decision["worst_paired_improvement_pK"] >= -0.050
    assert decision["passed"] is True

    decision["criteria"]["resource_gate_passed"] = False
    assert not all(decision["criteria"].values())
