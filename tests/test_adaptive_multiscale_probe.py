from __future__ import annotations

import json
from pathlib import Path
import runpy


PROBE = runpy.run_path(
    str(
        Path(__file__).parents[1]
        / "scripts"
        / "probe_adaptive_multiscale_spatial.py"
    )
)


def test_tiny_adaptive_multiscale_probe_records_witness_and_matched_models() -> None:
    result = PROBE["run_probe"](
        nodes=16,
        edge_multiplier=2,
        hidden_dim=16,
        heads=2,
        warmup=0,
        repeats=1,
        threads=1,
        seed=2807,
    )

    assert result["schema_version"] == 1
    assert result["status"] == "completed"
    assert result["device"] == "cpu"
    assert result["candidate"]["adaptive_middle_layer"] == 1
    assert result["candidate"]["common_state_equal"] is True
    assert result["candidate"]["finite_output"] is True
    assert result["witness"]["centroid_max_error"] < 1e-12
    assert result["witness"]["second_moment_max_error"] < 1e-12
    assert result["witness"]["spatial_summary_distance"] > 1e-6
    assert result["resource"]["baseline_median_ms"] >= 0.0
    assert result["resource"]["candidate_median_ms"] >= 0.0
    assert len(result["resource"]["baseline_samples_ms"]) == 1
    assert len(result["resource"]["candidate_samples_ms"]) == 1
    assert result["resource"]["timing_order"] == [["baseline", "candidate"]]
    assert result["resource"]["state_byte_ratio"] >= 1.0
    assert "cpu_affinity" in result["host"]
    json.dumps(result, allow_nan=False)
