from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path


def test_relation_transport_ablation_is_paired_and_frozen(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "relation-ablation.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/ablate_relation_transport.py",
            "--graphs",
            "4",
            "--steps",
            "2",
            "--width",
            "16",
            "--depth",
            "1",
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
    assert payload["task"]["only_varying_input"] == "edge_type"
    assert payload["ablation"]["shared_relation_cutoffs"] is True
    assert payload["ablation"]["shared_topology"] is True
    assert payload["initial_identity"]["full_state_equal"] is True
    assert payload["initial_identity"]["parameter_schema_equal"] is True
    assert payload["initial_identity"]["task_tensors_equal"] is True
    assert payload["initial_identity"]["initial_output_max_abs"] == 0.0

    frozen = payload["ablation"]["control_frozen_parameters"]
    assert len(frozen) == 3
    candidate_parameters = payload["relation_parameter_max_abs_after_training"][
        "candidate"
    ]
    control_parameters = payload["relation_parameter_max_abs_after_training"]["control"]
    assert set(candidate_parameters) == set(frozen)
    assert set(control_parameters) == set(frozen)
    assert any(value > 0.0 for value in candidate_parameters.values())
    assert all(value == 0.0 for value in control_parameters.values())
    for arm in payload["arms"].values():
        assert math.isfinite(arm["initial_loss"])
        assert math.isfinite(arm["final_loss"])
        assert arm["steps"] == 2
