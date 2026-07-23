from __future__ import annotations

import json
import math
from pathlib import Path
import runpy
from typing import Any

import pytest
import torch


SCALING = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "benchmark_sparse_scaling.py")
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


def test_bounded_ring_edges_have_exact_degree_self_edges_and_no_duplicates() -> None:
    edge_index = SCALING["bounded_ring_edge_index"](
        11, degree=4, device=torch.device("cpu")
    )

    assert edge_index.shape == (2, 44)
    assert torch.equal(torch.bincount(edge_index[0]), torch.full((11,), 4))
    assert len(set(map(tuple, edge_index.T.tolist()))) == edge_index.shape[1]
    self_nodes = edge_index[0, edge_index[0] == edge_index[1]]
    assert torch.equal(self_nodes, torch.arange(11))


def test_same_kernel_dense_and_factorized_outputs_match() -> None:
    inputs = SCALING["same_kernel_inputs"](
        17,
        device=torch.device("cpu"),
        dtype=torch.float64,
        scalar_dim=7,
        value_dim=5,
    )

    dense = SCALING["dense_same_kernel"](*inputs)
    factorized = SCALING["factorized_same_kernel"](*inputs)

    assert torch.allclose(dense, factorized, atol=1e-11, rtol=1e-10)


def test_tiny_scaling_run_records_resource_boundary_and_strict_json() -> None:
    result = SCALING["run_scaling_benchmark"](
        sizes=(8, 16),
        device="cpu",
        dtype="float64",
        degree=4,
        same_kernel_dense_max=16,
        dense_egnn_max=8,
        density_size=8,
        warmup=0,
        repeats=1,
        include_model_benchmarks=False,
    )

    assert result["schema_version"] == 1
    assert result["sizes"] == [8, 16]
    assert result["same_kernel"]["max_abs_error"] < 1e-10
    assert {row["method"] for row in result["same_kernel"]["rows"]} == {
        "materialized_dense",
        "factorized",
    }
    for row in result["same_kernel"]["rows"]:
        if row["method"] == "materialized_dense":
            assert row["materialized_pair_elements"] == row["nodes"] ** 2
        else:
            assert row["materialized_pair_elements"] == 0
    assert result["model_scaling"]["status"] == "skipped"
    assert result["inference_boundary"]["neighbor_builder_included"] is False
    _assert_finite_json(result)
    json.dumps(result, allow_nan=False)


def test_density_degrees_match_registered_controls() -> None:
    assert SCALING["density_degrees"](64) == [1, 4, 16, 64]
    assert SCALING["density_degrees"](512) == [1, 4, 16, 128, 512]


def test_seeded_exact_edges_have_requested_count_and_are_reproducible() -> None:
    build = SCALING["seeded_exact_edge_index"]

    first = build(
        17,
        edge_multiplier=5,
        seed=20260723,
        device=torch.device("cpu"),
    )
    repeated = build(
        17,
        edge_multiplier=5,
        seed=20260723,
        device=torch.device("cpu"),
    )
    changed = build(
        17,
        edge_multiplier=5,
        seed=20260724,
        device=torch.device("cpu"),
    )

    assert first.shape == (2, 17 * 5)
    assert torch.equal(first, repeated)
    assert not torch.equal(first, changed)
    assert int(first.min()) == 0
    assert int(first.max()) == 16
    assert torch.unique(first[0] * 17 + first[1]).numel() == first.shape[1]
    self_nodes = first[0, first[0] == first[1]]
    assert torch.equal(torch.sort(self_nodes).values, torch.arange(17))
    assert torch.equal(torch.bincount(first[0]), torch.full((17,), 5))


def test_seeded_exact_edges_saturate_at_complete_graph() -> None:
    edge_index = SCALING["seeded_exact_edge_index"](
        7,
        edge_multiplier=100,
        seed=3,
        device=torch.device("cpu"),
    )

    assert edge_index.shape == (2, 49)
    assert torch.unique(edge_index[0] * 7 + edge_index[1]).numel() == 49


@pytest.mark.parametrize(
    ("num_nodes", "edge_multiplier", "seed"),
    [(0, 2, 1), (8, 0, 1), (8, 2, -1), (True, 2, 1), (8, True, 1)],
)
def test_seeded_exact_edges_reject_invalid_controls(
    num_nodes: int,
    edge_multiplier: int,
    seed: int,
) -> None:
    with pytest.raises(ValueError):
        SCALING["seeded_exact_edge_index"](
            num_nodes,
            edge_multiplier=edge_multiplier,
            seed=seed,
            device=torch.device("cpu"),
        )


def test_tiny_edge_multiplier_grid_uses_identical_exact_edges() -> None:
    result = SCALING["run_edge_multiplier_benchmark"](
        sizes=(12, 16),
        edge_multipliers=(2, 4),
        device="cpu",
        seed=20260723,
        warmup=0,
        repeats=1,
        max_wall_seconds=30.0,
    )

    assert result["schema_version"] == 1
    assert result["benchmark"] == "same_edge_exact_multiplier_grid"
    assert result["graph_generator"] == "seeded_exact_receiver_regular_directed"
    assert result["graph_seed"] == 20260723
    assert result["model_seed"] == 20260723
    assert all(len(value) == 64 for value in result["model_state_sha256"].values())
    assert result["sizes"] == [12, 16]
    assert result["edge_multipliers"] == [2, 4]
    assert len(result["cells"]) == 4
    assert result["inference_boundary"]["edge_construction_timed"] is False
    assert result["inference_boundary"]["same_edge_tensor_for_models"] is True
    for cell in result["cells"]:
        assert cell["status"] == "completed"
        assert cell["candidate_edges_including_self"] == (
            cell["nodes"] * cell["edge_multiplier"]
        )
        assert cell["effective_nonself_edges"] == (
            cell["nodes"] * (cell["edge_multiplier"] - 1)
        )
        assert cell["receiver_candidate_degree"] == {
            "minimum": cell["edge_multiplier"],
            "maximum": cell["edge_multiplier"],
            "mean": float(cell["edge_multiplier"]),
            "population_std": 0.0,
        }
        assert len(cell["edge_index_sha256"]) == 64
        assert set(cell["models"]) == {"ec_lgl", "static_egnn"}
        assert cell["ec_lgl_to_egnn_latency_ratio"] > 0.0
    assert set(result["fits_by_nodes"]) == {"12", "16"}
    _assert_finite_json(result)
    json.dumps(result, allow_nan=False)


def test_graph_seed_varies_topology_without_varying_model_state() -> None:
    common = {
        "sizes": (12,),
        "edge_multipliers": (4,),
        "device": "cpu",
        "model_seed": 20260723,
        "warmup": 0,
        "repeats": 1,
        "max_wall_seconds": 30.0,
    }

    first = SCALING["run_edge_multiplier_benchmark"](seed=11, **common)
    changed_graph = SCALING["run_edge_multiplier_benchmark"](seed=12, **common)
    changed_model = SCALING["run_edge_multiplier_benchmark"](
        seed=11,
        **{**common, "model_seed": 20260724},
    )

    assert first["model_seed"] == changed_graph["model_seed"] == 20260723
    assert first["graph_seed"] == 11
    assert changed_graph["graph_seed"] == 12
    assert (
        first["cells"][0]["edge_index_sha256"]
        != (changed_graph["cells"][0]["edge_index_sha256"])
    )
    assert first["model_state_sha256"] == changed_graph["model_state_sha256"]
    assert first["model_state_sha256"] != changed_model["model_state_sha256"]


@pytest.mark.parametrize("model_seed", [-1, True])
def test_edge_multiplier_grid_rejects_invalid_model_seed(model_seed: int) -> None:
    with pytest.raises(ValueError, match="model_seed"):
        SCALING["run_edge_multiplier_benchmark"](
            sizes=(12,),
            edge_multipliers=(4,),
            device="cpu",
            seed=11,
            model_seed=model_seed,
            warmup=0,
            repeats=1,
            max_wall_seconds=30.0,
        )


def test_tiny_edge_free_grid_records_exact_boundaries_and_model_identity() -> None:
    result = SCALING["run_edge_free_spatial_benchmark"](
        sizes=(8,),
        edge_multipliers=(2, 4),
        device="cpu",
        seed=20260723,
        model_seed=20260723,
        warmup=0,
        repeats=1,
        max_wall_seconds=30.0,
    )

    assert result["schema_version"] == 1
    assert result["benchmark"] == "edge_free_spatial_vs_edge_scaled_egnn"
    assert result["sizes"] == [8]
    assert result["edge_multipliers"] == [2, 4]
    assert result["graph_seed"] == result["model_seed"] == 20260723
    assert result["inference_boundary"] == {
        "edge_construction_timed": False,
        "coordinate_updates_timed": True,
        "candidate_edge_index": None,
        "egnn_receives_prebuilt_edges": True,
        "topology_matched": False,
    }
    assert result["model_state_sha256"]["ggg"] == (
        result["model_state_sha256"]["spatial_static"]
    )
    assert result["model_parameters_semantics"] == "trainable_parameters"
    assert result["model_parameters"]["ggg"] < result["model_total_parameters"]["ggg"]
    assert result["model_parameters"]["spatial_static"] == (
        result["model_parameters"]["ggg"]
    )
    assert result["model_total_parameters"]["spatial_static"] == (
        result["model_total_parameters"]["ggg"]
    )
    assert result["model_parameters"]["static_egnn"] == (
        result["model_total_parameters"]["static_egnn"]
    )
    assert result["model_parameter_bytes"]["ggg"] == (
        4 * result["model_total_parameters"]["ggg"]
    )
    assert set(result["rows"][0]["edge_free_models"]) == {
        "ggg",
        "spatial_static",
        "spatial_dynamic",
    }
    assert all(
        metrics["status"] == "completed"
        for metrics in result["rows"][0]["edge_free_models"].values()
    )
    assert len(result["rows"][0]["egnn"]) == 2
    for cell in result["rows"][0]["egnn"]:
        assert cell["status"] == "completed"
        assert cell["candidate_edges_including_self"] == (
            cell["nodes"] * cell["edge_multiplier"]
        )
        assert len(cell["edge_index_sha256"]) == 64
        assert cell["receiver_candidate_degree"]["minimum"] == (
            cell["edge_multiplier"]
        )
        assert cell["receiver_candidate_degree"]["maximum"] == (
            cell["edge_multiplier"]
        )
    _assert_finite_json(result)
    json.dumps(result, allow_nan=False)


def test_edge_free_timer_rejects_nonfinite_output_before_timing() -> None:
    class NonfiniteModel(torch.nn.Module):
        def forward(
            self,
            node_feats: torch.Tensor,
            pos: torch.Tensor,
            *,
            batch: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            del pos, batch
            return {
                "node_scalars": node_feats.new_full(
                    (node_feats.shape[0], 1),
                    math.inf,
                )
            }

    node_feats = torch.zeros(4, 3)
    metrics = SCALING["_timed_edge_free_metrics"](
        model=NonfiniteModel(),
        node_feats=node_feats,
        pos=torch.zeros(4, 3),
        batch=torch.zeros(4, dtype=torch.long),
        warmup=0,
        repeats=1,
    )

    assert metrics == {
        "status": "failed",
        "failure_class": "nonfinite_output",
    }


def test_tiny_train_step_grid_records_full_step_and_memory_boundaries() -> None:
    result = SCALING["run_edge_free_train_step_benchmark"](
        sizes=(8,),
        edge_multipliers=(2,),
        device="cpu",
        seed=20260723,
        model_seed=20260723,
        warmup=1,
        repeats=1,
        max_wall_seconds=30.0,
    )

    assert result["schema_version"] == 1
    assert result["benchmark"] == "edge_free_train_step_vs_edge_scaled_egnn"
    assert result["timed_step"] == [
        "optimizer.zero_grad(set_to_none=True)",
        "forward",
        "mse_loss",
        "backward",
        "adamw.step",
    ]
    assert result["sizes"] == [8]
    assert result["edge_multipliers"] == [2]
    assert result["inference_boundary"]["graph_construction_timed_separately"] is True
    assert result["inference_boundary"]["model_step_excludes_graph_construction"] is True
    assert result["inference_boundary"]["task_accuracy_inferred"] is False

    cell = result["cells"][0]
    assert cell["status"] == "completed"
    assert cell["candidate_edges_including_self"] == 16
    assert cell["graph_construction"]["cpu_build_ms"] >= 0.0
    assert cell["graph_construction"]["device_transfer_ms"] == 0.0
    assert len(cell["edge_index_sha256"]) == 64
    assert set(cell["models"]) == {
        "spatial_static",
        "spatial_dynamic",
        "static_egnn",
    }
    for name, metrics in cell["models"].items():
        assert metrics["status"] == "completed"
        assert metrics["median_train_step_ms"] > 0.0
        assert metrics["final_loss"] >= 0.0
        assert metrics["gradient_validation"]["all_finite"] is True
        assert metrics["gradient_validation"]["nonzero_elements"] > 0
        assert len(metrics["initial_state_sha256"]) == 64
        assert metrics["optimizer"] == "AdamW"
        assert metrics["peak_cuda_bytes"] is None
        assert metrics["peak_cuda_bytes_delta"] is None
        assert metrics["edge_index_supplied"] is (name == "static_egnn")
    assert cell["spatial_static_to_egnn_train_step_ratio"] > 0.0
    assert cell["spatial_dynamic_to_egnn_train_step_ratio"] > 0.0
    _assert_finite_json(result)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    ("warmup", "repeats", "max_wall_seconds"),
    [(-1, 1, 10.0), (0, 0, 10.0), (0, 1, 0.0)],
)
def test_train_step_grid_rejects_invalid_execution_controls(
    warmup: int,
    repeats: int,
    max_wall_seconds: float,
) -> None:
    with pytest.raises(ValueError):
        SCALING["run_edge_free_train_step_benchmark"](
            sizes=(8,),
            edge_multipliers=(2,),
            device="cpu",
            warmup=warmup,
            repeats=repeats,
            max_wall_seconds=max_wall_seconds,
        )


def test_train_step_cli_mode_is_exclusive() -> None:
    args = SCALING["parse_args"](
        [
            "--edge-free-train-step-grid",
            "--metrics-out",
            "unused.json",
        ]
    )

    assert args.edge_free_train_step_grid is True
