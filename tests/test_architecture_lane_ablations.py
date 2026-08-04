from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path


def test_architecture_lane_ablations_are_isolated_and_paired(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "architecture-lanes.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/ablate_architecture_lanes.py",
            "--lane",
            "all",
            "--steps",
            "2",
            "--threads",
            "1",
            "--output",
            str(output),
        ],
        check=True,
        cwd=repository,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["device"] == "cpu"
    assert payload["deterministic_algorithms"] is True
    assert set(payload["lanes"]) == {"cg12", "multiscale"}

    for lane in payload["lanes"].values():
        identity = lane["initial_identity"]
        assert identity["full_state_equal"] is True
        assert identity["parameter_schema_equal"] is True
        assert identity["task_tensors_shared"] is True
        assert identity["initial_output_max_abs"] == 0.0
        ablation = lane["ablation"]
        assert ablation["candidate_lane_enabled"] is True
        assert ablation["control_lane_enabled"] is False
        assert ablation["candidate_only_target_lane_trainable"] is True
        assert ablation["shared_parameters_frozen"] is True
        assert ablation["control_trainable_parameters"] == []
        assert lane["task"]["teacher_signal_rms"] > 0.0
        candidate = lane["target_parameter_max_abs_after_training"]["candidate"]
        control = lane["target_parameter_max_abs_after_training"]["control"]
        assert any(value > 0.0 for value in candidate.values())
        assert all(value == 0.0 for value in control.values())
        assert math.isfinite(lane["arms"]["candidate"]["final_loss"])
        assert lane["arms"]["candidate"]["steps"] == 2
        assert (
            lane["arms"]["control"]["initial_loss"]
            == lane["arms"]["control"]["final_loss"]
        )

    cg12 = payload["lanes"]["cg12"]
    sector_rms = cg12["task"]["input_sector_rms"]
    assert sector_rms["scalar_l0"] == 0.0
    assert sector_rms["vector_l1"] > 0.0
    assert sector_rms["tensor_l2"] > 0.0
    multiscale = payload["lanes"]["multiscale"]
    assert multiscale["task"]["same_receiver_sender_csr_pattern"] is True
    assert multiscale["task"]["teacher_radial_response_span"] > 0.0
