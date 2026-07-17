import json
from pathlib import Path
import runpy

import pytest
import torch
import equivariant_attention.moment as moment


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
            "conditional_entropy_over_log_m": 0.7,
            "marginal_entropy_over_log_m": 0.9,
            "mutual_information_over_log_m": 0.2,
        },
        "coupling": {
            "coupling.q00": 0.5,
            "coupling.q50": 0.8,
            "coupling.q100": 1.0,
            "off_diagonal_nonunit_fraction": 0.5,
            "centered_frobenius_ratio": 0.2,
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

    mechanism = {
        "messages": {"aggregate": 1e-3},
        "post_middle": {"aggregate": 1e-3},
        "gradients": {"scalars": 1e-3, "vectors": 1e-3, "positions": 1e-3},
    }
    passed = symbols["stage0_decision"](
        activation, mechanism=mechanism, relative_output_rms=1e-3
    )
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
        mechanism=mechanism,
        relative_output_rms=1e-3,
    )
    no_output_change = symbols["stage0_decision"](
        activation,
        mechanism=mechanism,
        relative_output_rms=0.0,
    )
    no_gradient_change = symbols["stage0_decision"](
        activation,
        mechanism={
            **mechanism,
            "gradients": {**mechanism["gradients"], "positions": 0.0},
        },
        relative_output_rms=1e-3,
    )

    assert passed["passed"] is True
    assert all(check["passed"] for check in passed["checks"])
    assert failed["passed"] is False
    assert no_output_change["passed"] is False
    assert no_gradient_change["passed"] is False
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


def test_probe_graph_suite_has_distinct_registered_roles() -> None:
    symbols = _symbols()
    graphs = {
        role: symbols["probe_graph"](
            dtype=torch.float64,
            device=torch.device("cpu"),
            scenario=role,
        )
        for role in ("aligned", "crossed", "spatial_only", "semantic_only")
    }

    aligned_features, aligned_pos, _ = graphs["aligned"]
    crossed_features, crossed_pos, _ = graphs["crossed"]
    spatial_features, spatial_pos, _ = graphs["spatial_only"]
    semantic_features, semantic_pos, _ = graphs["semantic_only"]
    assert torch.equal(crossed_pos, aligned_pos)
    assert not torch.equal(crossed_features, aligned_features)
    assert torch.unique(spatial_features, dim=0).shape[0] == 1
    assert torch.equal(spatial_pos, aligned_pos)
    assert torch.unique(semantic_features[:, :4], dim=0).shape[0] == 4
    assert semantic_pos.square().sum(dim=-1).sqrt().max() < 1.0


def test_symmetric_relative_rms_is_scale_free_symmetric_and_zero_safe() -> None:
    symbols = _symbols()
    left = torch.tensor([1.0, 2.0], dtype=torch.float64)
    right = torch.tensor([2.0, 4.0], dtype=torch.float64)

    value = symbols["symmetric_relative_rms"](left, right)

    assert value == pytest.approx(symbols["symmetric_relative_rms"](right, left))
    assert value == pytest.approx(
        symbols["symmetric_relative_rms"](10.0 * left, 10.0 * right)
    )
    assert symbols["symmetric_relative_rms"](
        torch.zeros(2), torch.zeros(2)
    ) == pytest.approx(0.0)


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

    assert result["schema_version"] == 2
    assert result["test_evaluated"] is False
    assert result["hidden_dim"] == 8
    assert result["num_heads"] == 2
    assert len(result["source_sha256"]) == 64
    assert result["baseline"]["memory_count"] == 1
    assert len(result["baseline"]["state_sha256"]) == 64
    arm = result["arms"][0]
    assert arm["memory_count"] == 4
    assert arm["state_sha256"] == result["baseline"]["state_sha256"]
    assert arm["state_schema_sha256"] == result["baseline"]["state_schema_sha256"]
    assert set(arm["counterfactuals"]) == {
        "ones",
        "radial",
        "identity",
        "lambda_0.10",
        "lambda_0.25",
        "lambda_0.50",
    }
    for counterfactual in arm["counterfactuals"].values():
        assert counterfactual["activation"]["scope"] == "single_graph_per_head"
        assert counterfactual["activation"]["head_count"] == 2
        assert len(counterfactual["activation"]["heads"]) == 2
        assert counterfactual["relative_output_rms"] >= 0.0
        assert "messages" in counterfactual["mechanism"]
        assert "post_middle" in counterfactual["mechanism"]
        assert "gradients" in counterfactual["mechanism"]
    assert arm["diagnosis"] in {
        "router_functionally_inactive",
        "radial_coupling_collapsed",
        "identity_activates",
    }
    assert isinstance(arm["counterfactuals"]["radial"]["decision"]["passed"], bool)
    assert result["decision"] in {
        "admit_interacting_memory_arms",
        "block_interacting_memory_arms",
    }
    json.dumps(result, allow_nan=False)


def test_probe_restores_private_helpers_after_counterfactual_execution() -> None:
    symbols = _symbols()
    assignment_helper = moment._memory_assignments_and_coupling
    message_helper = moment._global_moment_messages

    symbols["run_probe"](
        memory_counts=(4,),
        device=torch.device("cpu"),
        dtype=torch.float64,
        seed=401,
        hidden_dim=8,
        num_heads=2,
    )

    assert moment._memory_assignments_and_coupling is assignment_helper
    assert moment._global_moment_messages is message_helper


def test_registered_suite_aggregates_lanes_and_selects_only_aligned_repairs() -> None:
    symbols = _symbols()

    result = symbols["run_suite"](
        memory_counts=(4,),
        hidden_dims=(8,),
        seeds=(401,),
        scenarios=("aligned", "semantic_only"),
        device=torch.device("cpu"),
        dtype=torch.float64,
        num_heads=2,
    )

    assert result["schema_version"] == 1
    assert result["probe"] == "hemm_stage0_registered_suite"
    assert len(result["lanes"]) == 2
    assert {lane["scenario"] for lane in result["lanes"]} == {
        "aligned",
        "semantic_only",
    }
    assert result["selected_identity_mix"] in {None, 0.1, 0.25, 0.5}
    assert result["scientific_decision"] in {
        "admit_interacting_memory_arms",
        "block_interacting_memory_arms",
    }
    assert result["test_evaluated"] is False
    json.dumps(result, allow_nan=False)


def test_relative_rms_rejects_zero_baseline() -> None:
    symbols = _symbols()
    with pytest.raises(ValueError, match="baseline RMS"):
        symbols["relative_output_rms"](
            {"x": torch.zeros(2)},
            {"x": torch.ones(2)},
        )
