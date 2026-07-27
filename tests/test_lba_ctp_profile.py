from __future__ import annotations

from pathlib import Path
import runpy


PROFILE = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "profile_lba_train_step.py")
)


def test_ctp_resource_gate_is_relative_to_current_candidate() -> None:
    arms = {
        "candidate": {
            "parameter_count": 100,
            "median_synchronized_step_seconds": 1.0,
            "timed_peak_cuda_memory_bytes": 1_000,
        },
        "persistent_2e": {
            "parameter_count": 104,
            "median_synchronized_step_seconds": 1.08,
            "timed_peak_cuda_memory_bytes": 1_050,
        },
        "ctp": {
            "parameter_count": 109,
            "median_synchronized_step_seconds": 1.20,
            "timed_peak_cuda_memory_bytes": 1_200,
        },
    }

    comparison = PROFILE["_ctp_resource_comparison"](arms)

    assert comparison["ctp_to_candidate_parameter_ratio"] == 1.09
    assert comparison["ctp_to_candidate_median_step_ratio"] == 1.2
    assert comparison["ctp_to_candidate_peak_allocation_ratio"] == 1.2
    assert comparison["resource_gate"]["passed"] is True
