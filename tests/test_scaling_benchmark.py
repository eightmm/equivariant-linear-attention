from __future__ import annotations

import json
import math
from pathlib import Path
import runpy
from typing import Any

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
