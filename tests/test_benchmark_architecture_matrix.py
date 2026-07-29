from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
import runpy
from typing import Any

import torch

from equivariant_attention.config import ArchitectureConfig
from equivariant_attention.training import build_regression_model


MATRIX = runpy.run_path(
    str(
        Path(__file__).parents[1]
        / "scripts"
        / "benchmark_architecture_matrix.py"
    )
)


def _assert_finite_json(value: Any) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _assert_finite_json(child)
    elif isinstance(value, list):
        for child in value:
            _assert_finite_json(child)
    elif isinstance(value, float):
        assert math.isfinite(value)


def test_required_architecture_arms_are_executable_structured_configs() -> None:
    required = MATRIX["REQUIRED_ARMS"]
    configs = MATRIX["build_architecture_arms"](
        required,
        node_dim=8,
        width=8,
        num_heads=2,
        global_backend="auto",
        local_backend="materialized",
    )

    assert tuple(configs) == (
        "legacy_lgl",
        "deep_global",
        "h3_r4",
        "h4_r2",
        "h6_r2",
        "workspace_l3",
    )
    assert configs["legacy_lgl"].local.local_head_counts == (2, 0, 2)
    assert configs["legacy_lgl"].local.use_gated_local_transport
    assert configs["deep_global"].num_layers == 6
    assert not configs["deep_global"].local.use_sparse_low_rank_local_residual
    assert configs["h3_r4"].local.local_residual_rank == 4
    assert configs["h4_r2"].num_layers == 4
    assert configs["h6_r2"].num_layers == 6
    assert configs["workspace_l3"].profile == "high_order"
    assert configs[
        "workspace_l3"
    ].representation.transient_workspace_layers == (0, 1, 2)

    for config in configs.values():
        restored = ArchitectureConfig.from_json(config.to_json())
        assert restored == config
        model = build_regression_model(
            config.node_dim,
            architecture_config=config,
        )
        assert sum(parameter.numel() for parameter in model.parameters()) > 0


def test_n8_k2_variants_are_unique_selfed_exact_and_never_cross_graphs() -> None:
    make_case = MATRIX["make_synthetic_case"]
    cases = {
        variant: make_case(
            8,
            degree=2,
            variant=variant,
            node_dim=7,
            seed=19,
        )
        for variant in MATRIX["GRAPH_VARIANTS"]
    }

    for variant, case in cases.items():
        batch = case.batch
        assert batch.edge_index is not None
        assert batch.edge_index_is_validated
        assert batch.edge_index.shape == (2, 8 * 2)
        assert case.metadata["exact_e_equals_k_n"] is True
        assert case.metadata["supplied_edges"] == 16
        assert case.metadata["unique_directed_edges"] is True
        assert case.metadata["self_edge_per_node"] is True
        receiver, sender = batch.edge_index
        pair_id = receiver * 8 + sender
        assert torch.unique(pair_id).numel() == 16
        assert torch.equal(
            torch.sort(receiver[receiver == sender]).values,
            torch.arange(8),
        )
        assert torch.equal(
            batch.batch[receiver],
            batch.batch[sender],
        )
        repeated = make_case(
            8,
            degree=2,
            variant=variant,
            node_dim=7,
            seed=19,
        )
        assert repeated.metadata["edge_index_sha256"] == case.metadata[
            "edge_index_sha256"
        ]
        assert torch.equal(repeated.batch.node_feats, batch.node_feats)

    assert cases["uniform"].metadata["receiver_max_degree"] == 2
    assert cases["skew"].metadata["receiver_max_degree"] > 2
    assert cases["ragged"].metadata["graph_count"] == 3
    assert len(set(cases["ragged"].metadata["graph_sizes"])) > 1


def test_high_degree_and_ragged_capacity_adjustments_never_fake_multigraphs() -> None:
    make_case = MATRIX["make_synthetic_case"]
    complete = make_case(
        8,
        degree=8,
        variant="skew",
        node_dim=5,
        seed=3,
    )
    assert complete.batch.edge_index is not None
    assert complete.batch.edge_index.shape == (2, 64)
    assert torch.unique(
        complete.batch.edge_index[0] * 8 + complete.batch.edge_index[1]
    ).numel() == 64
    assert complete.metadata["effective_topology_variant"] == "uniform"
    assert complete.metadata["topology_limitations"]

    adjusted = make_case(
        8,
        degree=6,
        variant="ragged",
        node_dim=5,
        seed=4,
    )
    assert adjusted.metadata["graph_sizes"] == [1, 7]
    assert adjusted.metadata["ragged_partition_adjusted"] is True
    assert adjusted.metadata["ragged_partition_collapsed"] is False
    assert adjusted.batch.edge_index is not None
    assert torch.unique(
        adjusted.batch.edge_index[0] * 8 + adjusted.batch.edge_index[1]
    ).numel() == 48

    collapsed = make_case(
        8,
        degree=7,
        variant="ragged",
        node_dim=5,
        seed=5,
    )
    assert collapsed.metadata["graph_sizes"] == [8]
    assert collapsed.metadata["ragged_partition_collapsed"] is True
    assert collapsed.metadata["effective_topology_variant"] == (
        "single_graph_capacity_fallback"
    )
    assert collapsed.batch.edge_index is not None
    assert torch.unique(
        collapsed.batch.edge_index[0] * 8 + collapsed.batch.edge_index[1]
    ).numel() == 56
    for case in (complete, adjusted, collapsed):
        assert case.metadata["self_edge_count"] == 8
        assert "candidate_multigraph" not in case.metadata
        assert "repeated_senders_possible" not in case.metadata


def test_validated_unique_ragged_topology_executes_actual_model_forward() -> None:
    case = MATRIX["make_synthetic_case"](
        8,
        degree=6,
        variant="ragged",
        node_dim=8,
        seed=29,
    )
    config = MATRIX["build_arm_config"](
        "legacy_lgl",
        node_dim=8,
        width=8,
        num_heads=2,
        global_backend="auto",
        local_backend="materialized",
    )
    model = build_regression_model(8, architecture_config=config)
    prediction = MATRIX["predict_graph_scalar"](model, case.batch)

    assert case.batch.edge_index_is_validated
    assert prediction.shape == (2, 1)
    assert torch.isfinite(prediction).all()


def test_tiny_matrix_records_forward_train_state_backends_and_claim_boundary() -> None:
    result = MATRIX["run_architecture_matrix"](
        nodes=(8,),
        degrees=(2,),
        variants=("uniform",),
        arms=(
            "legacy_lgl",
            "deep_global",
            "h3_r4",
            "workspace_l3",
            "standard",
        ),
        node_dim=8,
        width=8,
        num_heads=2,
        parameter_match=False,
        local_backend="auto",
        device="cpu",
        warmup=0,
        repeats=1,
        max_wall_seconds=60.0,
        threads=1,
        seed=23,
    )

    assert result["schema"] == (
        "equivariant_attention.architecture_resource_matrix"
    )
    assert result["schema_version"] == 1
    assert [row["status"] for row in result["rows"]] == [
        "completed",
        "completed",
        "completed",
        "completed",
        "completed",
    ]
    rows = {row["arm"]: row for row in result["rows"]}
    assert rows["legacy_lgl"]["neighbor_representation"] == "validated_coo"
    assert rows["deep_global"]["neighbor_representation"] == "not_consumed"
    assert rows["h3_r4"]["neighbor_representation"] == (
        "packed_receiver_csr"
    )
    assert rows["workspace_l3"]["neighbor_representation"] == (
        "validated_coo"
    )
    for row in rows.values():
        assert row["edges"] == row["nodes"] * row["degree"]
        assert row["parameter_count"] > 0
        assert row["parameter_bytes"] > 0
        assert row["model_state_bytes"] >= row["parameter_bytes"]
        assert row["optimizer_state_bytes"] > 0
        assert row["forward"]["repeat_count"] == 1
        assert row["forward"]["median_ms"] > 0.0
        assert row["train_step"]["optimizer_inclusive"] is True
        assert row["train_step"]["backward_included"] is True
        assert row["train_step"]["optimizer_step_included"] is True
        assert row["train_step"]["median_ms"] > 0.0
        assert row["forward"]["cuda_memory_available"] is False
        assert row["forward"]["cuda_peak_allocated_bytes"] is None
        assert row["graph_construction"][
            "included_in_forward_timing"
        ] is False
        assert row["graph_construction"][
            "included_in_train_step_timing"
        ] is False
        assert row["execution"]["requested_global_lane"] == "auto"
        assert row["execution"]["effective_global_lane"] in {
            "direct",
            "outer_scatter",
            "padded_bmm",
            "bucket_bmm",
            "ragged_gemm",
        }
        assert row["execution"]["requested_cache_mode"] == "auto"

    assert rows["deep_global"]["execution"]["effective_local_backend"] == (
        "not_executed"
    )
    assert rows["deep_global"]["execution"]["requested_local_backend"] == (
        "not_executed"
    )
    assert rows["deep_global"]["execution"]["configured_local_backend"] == (
        "materialized"
    )
    assert rows["deep_global"]["execution"]["local_operation"] == (
        "not_executed"
    )
    assert rows["deep_global"]["execution"]["local_backend_status"] == (
        "not_executed"
    )
    assert rows["standard"]["execution"]["effective_local_backend"] == (
        "not_executed"
    )
    assert rows["standard"]["execution"]["local_operation"] == "not_executed"
    assert rows["h3_r4"]["execution"]["local_backend_status"] == "executed"
    assert rows["workspace_l3"]["execution"]["local_backend_status"] == (
        "not_executed"
    )
    assert rows["workspace_l3"]["execution"][
        "transient_workspace_status"
    ] == "executed"
    assert rows["workspace_l3"]["execution"][
        "transient_workspace_backend"
    ] == "filtered_coo_reference"
    assert rows["legacy_lgl"]["execution"]["global_feature_width"] == 14
    assert rows["legacy_lgl"]["execution"]["global_value_width"] == 20
    assert rows["workspace_l3"]["execution"]["global_feature_width"] == 14
    assert rows["workspace_l3"]["execution"]["global_value_width"] == 20

    comparisons = {
        comparison["arm"]: comparison
        for comparison in result["relative_to_legacy_lgl"]
    }
    assert comparisons["legacy_lgl"]["parameter_ratio"] == 1.0
    assert comparisons["h3_r4"]["status"] == "measured"
    assert (
        comparisons["h3_r4"]["resource_match_is_observed_not_enforced"]
        is True
    )
    assert result["claim_boundary"]["accuracy_evaluated"] is False
    assert (
        result["claim_boundary"]["architecture_superiority_claimed"] is False
    )
    _assert_finite_json(result)
    json.dumps(result, allow_nan=False, sort_keys=True)


def test_global_transport_width_receipt_matches_spatial_feature_families() -> None:
    base = MATRIX["build_arm_config"](
        "deep_global",
        node_dim=8,
        width=8,
        num_heads=2,
    )
    static = replace(
        base,
        global_transport=replace(
            base.global_transport,
            use_multiscale_spatial_kernel=True,
        ),
    )
    adaptive_base = MATRIX["build_arm_config"](
        "legacy_lgl",
        node_dim=8,
        width=8,
        num_heads=2,
    )
    adaptive = replace(
        adaptive_base,
        global_transport=replace(
            adaptive_base.global_transport,
            use_adaptive_multiscale_spatial_kernel=True,
        ),
    )

    assert MATRIX["_global_transport_widths"](base) == (14, 20)
    assert MATRIX["_global_transport_widths"](static) == (24, 20)
    assert MATRIX["_global_transport_widths"](adaptive) == (54, 20)


def test_auto_sparse_preparation_falls_back_from_excessive_ell_padding() -> None:
    case = MATRIX["make_synthetic_case"](
        512,
        degree=4,
        variant="skew",
        node_dim=8,
        seed=29,
    )

    prepared, representation = MATRIX["_prepare_batch"](
        case.batch,
        edge_consumption="sparse_residual",
        local_backend="auto",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert representation == "packed_receiver_csr"
    assert prepared.packed_neighbors is not None
    assert prepared.packed_neighbors.ell_sender is None


def test_parameter_width_search_is_bounded_and_reports_failed_matches() -> None:
    widths, receipt = MATRIX["resolve_parameter_matched_widths"](
        ("legacy_lgl", "deep_global", "h3_r4"),
        node_dim=8,
        reference_width=8,
        num_heads=2,
        maximum_width=12,
        global_backend="auto",
        local_backend="materialized",
        geometry_cache_mode="auto",
        workspace_channels=1,
    )

    assert widths["legacy_lgl"] == 8
    assert all(width in {2, 4, 6, 8, 10, 12} for width in widths.values())
    assert receipt["candidate_widths"] == [2, 4, 6, 8, 10, 12]
    assert receipt["guaranteed_match"] is False
    for selection in receipt["selection"].values():
        assert selection["parameter_count"] > 0
        assert selection["ratio_to_legacy_target"] > 0.0
        assert isinstance(selection["within_one_percent"], bool)


def test_wall_budget_skips_rows_with_machine_readable_status() -> None:
    result = MATRIX["run_architecture_matrix"](
        nodes=(8,),
        degrees=(2,),
        variants=("uniform",),
        arms=("legacy_lgl", "h3_r4"),
        node_dim=8,
        width=8,
        num_heads=2,
        parameter_match=False,
        local_backend="materialized",
        device="cpu",
        warmup=0,
        repeats=1,
        max_wall_seconds=0.0,
        threads=1,
    )

    assert [row["status"] for row in result["rows"]] == [
        "skipped_time_budget",
        "skipped_time_budget",
    ]
    assert all(
        "wall budget" in row["reason"] for row in result["rows"]
    )
    json.dumps(result, allow_nan=False)


def test_mathematically_impossible_degree_is_a_machine_readable_skip() -> None:
    result = MATRIX["run_architecture_matrix"](
        nodes=(8,),
        degrees=(9,),
        variants=("ragged",),
        arms=("legacy_lgl", "h3_r4"),
        node_dim=8,
        width=8,
        num_heads=2,
        parameter_match=False,
        local_backend="materialized",
        device="cpu",
        warmup=0,
        repeats=1,
        max_wall_seconds=60.0,
        threads=1,
    )

    assert [row["status"] for row in result["rows"]] == [
        "skipped_infeasible_topology",
        "skipped_infeasible_topology",
    ]
    assert all(row["graph"]["topology_feasible"] is False for row in result["rows"])
    json.dumps(result, allow_nan=False)
