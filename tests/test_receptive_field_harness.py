"""Receptive-field harness contracts.

The QM9 EGNN control has been receiving the complete same-graph edge list while
the attention arms' local heads only see candidates inside ``local_cutoff``.
Comparing them is only a same-topology architecture comparison when both arms
consume the same candidates, so the EGNN arm must be able to take the frozen
radius candidates at the screened cutoff, and the cutoff must be rejected when
it would be inert.
"""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest

from equivariant_attention.training import build_regression_model


def _train_compare() -> dict[str, object]:
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_compare.py"
    return runpy.run_path(script)


def _screen() -> dict[str, object]:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_receptive_field_qm9.py"
    )
    return runpy.run_path(script)


def _record(
    val_mae: float,
    *,
    latency: float = 0.01,
    memory: int = 1_000,
) -> dict[str, object]:
    return {
        "val_mae": val_mae,
        "step_latency_median_seconds": latency,
        "peak_cuda_memory_bytes": memory,
    }


def test_build_regression_model_threads_the_radial_spacing() -> None:
    model = build_regression_model(
        node_dim=7,
        hidden_dim=32,
        num_heads=2,
        local_head_counts=(2, 0, 2),
        local_rbf_spacing="distance",
        local_cutoff=5.0,
    )

    assert model.config.local_rbf_spacing == "distance"
    assert model.config.local_cutoff == 5.0


def test_default_regression_model_keeps_the_incumbent_spacing() -> None:
    model = build_regression_model(node_dim=7, hidden_dim=32, num_heads=2)

    assert model.config.local_rbf_spacing == "squared"


def test_run_config_records_the_radial_spacing_and_cutoff() -> None:
    symbols = _train_compare()
    args = symbols["parse_args"](
        [
            "--routing",
            "lgl",
            "--gated-local-transport",
            "--local-cutoff",
            "5.0",
            "--local-rbf-spacing",
            "distance",
            "--precompute-local-edges",
        ]
    )

    config = symbols["_run_config"](args, split_seed=42, model_seed=42)

    assert config["local_cutoff"] == 5.0
    assert config["local_rbf_spacing"] == "distance"
    assert config["local_rbf_center_variable"] == "normalized_radius"
    assert config["precompute_local_edges"] is True
    assert config["edge_topology"] == "precomputed_radius_candidates_with_self"


def test_egnn_accepts_matched_radius_candidates_at_the_screened_cutoff() -> None:
    symbols = _train_compare()
    args = symbols["parse_args"](
        [
            "--benchmark-model",
            "internal_static_egnn_baseline",
            "--local-cutoff",
            "5.0",
            "--precompute-local-edges",
        ]
    )

    model = symbols["_build_benchmark_model"](args, node_dim=11)

    assert type(model).__name__ == "_StaticEGNNBaseline"


def test_egnn_rejects_an_inert_cutoff_without_precomputed_candidates() -> None:
    symbols = _train_compare()
    args = symbols["parse_args"](
        [
            "--benchmark-model",
            "internal_static_egnn_baseline",
            "--local-cutoff",
            "5.0",
        ]
    )

    with pytest.raises(ValueError, match="--precompute-local-edges"):
        symbols["_build_benchmark_model"](args, node_dim=11)


def test_egnn_rejects_attention_only_radial_controls() -> None:
    symbols = _train_compare()
    args = symbols["parse_args"](
        [
            "--benchmark-model",
            "internal_static_egnn_baseline",
            "--local-rbf-spacing",
            "distance",
            "--precompute-local-edges",
        ]
    )

    with pytest.raises(ValueError, match="local_rbf_spacing"):
        symbols["_build_benchmark_model"](args, node_dim=11)


def test_egnn_records_the_matched_edge_topology() -> None:
    symbols = _train_compare()
    args = symbols["parse_args"](
        [
            "--benchmark-model",
            "internal_static_egnn_baseline",
            "--local-cutoff",
            "5.0",
            "--precompute-local-edges",
        ]
    )

    config = symbols["_run_config"](args, split_seed=42, model_seed=42)

    assert config["local_cutoff"] == 5.0
    assert config["precompute_local_edges"] is True
    assert config["edge_topology"] == "precomputed_radius_candidates_without_self"


def test_screen_arms_cover_the_factorial_and_both_egnn_topologies() -> None:
    symbols = _screen()
    arms = symbols["ARMS"]

    assert list(arms) == [
        "incumbent",
        "wide_cutoff",
        "distance_rbf",
        "wide_distance",
        "egnn_complete",
        "egnn_matched",
    ]
    for arm, expected_cutoff, expected_spacing in (
        ("incumbent", "2.5", "squared"),
        ("wide_cutoff", "5.0", "squared"),
        ("distance_rbf", "2.5", "distance"),
        ("wide_distance", "5.0", "distance"),
    ):
        command = symbols["build_command"](
            arm,
            output=Path("out.json"),
            steps=500,
            device="cuda",
        )
        assert command[command.index("--local-cutoff") + 1] == expected_cutoff
        assert command[command.index("--local-rbf-spacing") + 1] == expected_spacing
        assert "--gated-local-transport" in command
        assert "--grouped-invariant-normalization" in command
        assert "--skip-test-eval" in command
        assert "--precompute-local-edges" in command


def test_egnn_arms_use_matched_and_complete_topologies() -> None:
    symbols = _screen()

    complete = symbols["build_command"](
        "egnn_complete",
        output=Path("out.json"),
        steps=500,
        device="cuda",
    )
    matched = symbols["build_command"](
        "egnn_matched",
        output=Path("out.json"),
        steps=500,
        device="cuda",
    )

    assert "--precompute-local-edges" not in complete
    assert "--local-rbf-spacing" not in complete
    assert "--precompute-local-edges" in matched
    assert matched[matched.index("--local-cutoff") + 1] == "5.0"
    for command in (complete, matched):
        assert command[command.index("--benchmark-model") + 1] == (
            "internal_static_egnn_baseline"
        )
        assert "--gated-local-transport" not in command
        assert "--skip-test-eval" in command


def test_screen_admits_only_a_material_improvement_within_the_resource_ceiling() -> None:
    symbols = _screen()
    minimum = symbols["MINIMUM_IMPROVEMENT_EV"]
    ceiling = symbols["MAXIMUM_RESOURCE_RATIO"]
    records = {
        "incumbent": _record(0.646),
        "wide_cutoff": _record(0.646 - minimum, latency=0.01 * ceiling),
        "distance_rbf": _record(0.646 - minimum / 2.0),
        "wide_distance": _record(0.500, latency=0.01 * ceiling * 2.0),
        "egnn_complete": _record(0.409),
        "egnn_matched": _record(0.450),
    }

    decision = symbols["screen_decision"](records)

    rows = {row["arm"]: row for row in decision["candidates"]}
    assert rows["wide_cutoff"]["promotion_screen_passed"] is True
    assert rows["distance_rbf"]["promotion_screen_passed"] is False
    assert rows["wide_distance"]["promotion_screen_passed"] is False
    assert decision["promotion_selected_arm"] == "wide_cutoff"
    assert decision["egnn_topology_confound_eV"] == pytest.approx(0.450 - 0.409)


def test_screen_reports_no_selection_when_every_arm_regresses() -> None:
    symbols = _screen()
    records = {
        "incumbent": _record(0.646),
        "wide_cutoff": _record(0.700),
        "distance_rbf": _record(0.660),
        "wide_distance": _record(0.650),
        "egnn_complete": _record(0.409),
        "egnn_matched": _record(0.409),
    }

    decision = symbols["screen_decision"](records)

    assert decision["promotion_screen_passed"] is False
    assert decision["promotion_selected_arm"] is None
    assert decision["egnn_topology_confound_eV"] == pytest.approx(0.0)


def test_screen_plan_is_dry_runnable_and_never_opens_test_labels(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    symbols = _screen()

    assert symbols["main"]([str(tmp_path / "out"), "--dry-run"]) == 0

    plan = json.loads(capsys.readouterr().out)
    assert plan["test_evaluated"] is False
    assert plan["validation_evaluated"] is True
    assert plan["steps"] == symbols["MAX_STEPS"]
    assert not (tmp_path / "out").exists()
    for command in plan["commands"].values():
        assert "--evaluate-test" not in command
