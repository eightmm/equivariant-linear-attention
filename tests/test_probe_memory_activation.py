import json
from pathlib import Path
import runpy

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probe_memory_activation.py"


def _symbols() -> dict[str, object]:
    return runpy.run_path(SCRIPT)


def _active_head(symbols: dict[str, object]) -> dict[str, object]:
    pair_gate = {
        "min": 0.4,
        "p01": 0.4,
        "median": 0.6,
        "p99": 0.8,
        "max": 0.8,
        "mean": 0.6,
        "cv": 0.1,
        "centered_frobenius_ratio": 0.09950371902099893,
        "nonconstant_fraction": 0.5,
        "nonconstant_relative_tolerance": 1e-3,
        "symmetry_relative_max_error": 0.0,
    }
    return {
        "head_index": 0,
        "assignment": {
            "memory_count": 4,
            "occupancy.min": 1.0,
            "occupancy.mean": 4.0,
            "occupancy.max": 7.0,
            "occupancy_fraction.min": 0.0625,
            "occupancy_fraction.mean": 0.25,
            "occupancy_fraction.max": 0.4375,
            "assignment_entropy_over_log_m": 0.7,
        },
        "coupling": {
            "coupling.q00": 0.5,
            "coupling.q50": 0.8,
            "coupling.q100": 1.0,
            "off_diagonal_nonunit_fraction": 0.5,
        },
        "pair_gate": pair_gate,
    }


def test_stage0_decision_requires_every_head_and_output_change() -> None:
    symbols = _symbols()
    head = _active_head(symbols)
    activation = {
        "scope": "single_graph_per_head",
        "node_count": 16,
        "head_count": 2,
        "memory_count": 4,
        "heads": [head, {**head, "head_index": 1}],
    }

    passed = symbols["stage0_decision"](activation, relative_output_rms=1e-3)
    failed_head = {
        **head,
        "head_index": 1,
        "pair_gate": {
            **head["pair_gate"],
            "cv": 0.0,
            "centered_frobenius_ratio": 0.0,
            "nonconstant_fraction": 0.0,
        },
    }
    failed = symbols["stage0_decision"](
        {**activation, "heads": [head, failed_head]},
        relative_output_rms=1e-3,
    )
    no_output_change = symbols["stage0_decision"](
        activation,
        relative_output_rms=0.0,
    )

    assert passed["passed"] is True
    assert all(check["passed"] for check in passed["checks"])
    assert failed["passed"] is False
    assert no_output_change["passed"] is False
    json.dumps(passed, allow_nan=False)


def test_probe_graph_is_fixed_heterogeneous_sixteen_node_input() -> None:
    symbols = _symbols()

    node_feats, pos, batch = symbols["probe_graph"](
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    assert node_feats.shape == (16, 8)
    assert pos.shape == (16, 3)
    assert torch.equal(batch, torch.zeros(16, dtype=torch.long))
    assert torch.unique(node_feats[:, :4], dim=0).shape[0] == 4
    assert node_feats.dtype == torch.float64
    assert pos.dtype == torch.float64


def test_actual_probe_uses_common_state_and_emits_strict_json() -> None:
    symbols = _symbols()

    result = symbols["run_probe"](
        memory_counts=(4,),
        device=torch.device("cpu"),
        dtype=torch.float64,
        seed=401,
        hidden_dim=8,
        num_heads=2,
    )

    assert result["schema_version"] == 1
    assert result["test_evaluated"] is False
    assert len(result["source_sha256"]) == 64
    assert result["baseline"]["memory_count"] == 1
    assert len(result["baseline"]["state_sha256"]) == 64
    arm = result["arms"][0]
    assert arm["memory_count"] == 4
    assert arm["state_sha256"] == result["baseline"]["state_sha256"]
    assert arm["state_schema_sha256"] == result["baseline"]["state_schema_sha256"]
    assert arm["activation"]["scope"] == "single_graph_per_head"
    assert arm["activation"]["head_count"] == 2
    assert len(arm["activation"]["heads"]) == 2
    assert arm["relative_output_rms"] >= 0.0
    assert isinstance(arm["decision"]["passed"], bool)
    assert result["decision"] in {
        "admit_interacting_memory_arms",
        "block_interacting_memory_arms",
    }
    json.dumps(result, allow_nan=False)


def test_relative_rms_rejects_zero_baseline() -> None:
    symbols = _symbols()
    with pytest.raises(ValueError, match="baseline RMS"):
        symbols["relative_output_rms"](
            {"x": torch.zeros(2)},
            {"x": torch.ones(2)},
        )
