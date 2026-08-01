from __future__ import annotations

import torch

from equivariant_attention import (
    ELA,
    ELABatch,
    collate_graphs,
    prepare_3d_graph,
)


def test_collator_returns_ela_batch_with_counts() -> None:
    samples = [
        {
            "x": torch.randn(2, 3),
            "pos": torch.randn(2, 3),
            "target": torch.tensor([1.0]),
            "sample_id": "a",
        },
        {
            "x": torch.randn(4, 3),
            "pos": torch.randn(4, 3),
            "target": torch.tensor([2.0]),
            "sample_id": "b",
        },
    ]
    batch = collate_graphs(samples)
    assert isinstance(batch, ELABatch)
    assert batch.num_nodes == 6
    assert batch.num_graphs == 2
    assert batch["sample_ids"] == ("a", "b")
    assert batch["target"].shape == (2, 1)


def test_ela_collate_returns_ela_batch() -> None:
    samples = [
        {"x": torch.randn(2, 3), "pos": torch.randn(2, 3)},
        {"x": torch.randn(3, 3), "pos": torch.randn(3, 3)},
    ]
    assert isinstance(ELA.collate(samples), ELABatch)


def test_device_only_move_preserves_floating_dtypes() -> None:
    batch = ELABatch(
        {
            "node_irreps": torch.randn(5, 4, dtype=torch.float64),
            "pos": torch.randn(5, 3, dtype=torch.float64),
            "batch": torch.tensor([0, 0, 1, 1, 1]),
            "edge_index": torch.tensor([[0, 1], [1, 0]]),
        }
    )
    moved = batch.to("cpu")
    assert moved["node_irreps"].dtype == torch.float64
    assert moved["pos"].dtype == torch.float64
    assert moved["batch"].dtype == torch.long
    assert moved["edge_index"].dtype == torch.long


def test_model_dtype_move_keeps_geometry_in_fp32() -> None:
    batch = ELABatch(
        {
            "node_irreps": torch.randn(5, 4, dtype=torch.float32),
            "pos": torch.randn(5, 3, dtype=torch.float64),
            "target": torch.randn(2, 1, dtype=torch.float32),
        }
    )
    moved = batch.to("cpu", dtype=torch.bfloat16)
    assert moved["node_irreps"].dtype == torch.bfloat16
    assert moved["target"].dtype == torch.bfloat16
    assert moved["pos"].dtype == torch.float32


def test_explicit_geometry_dtype_override() -> None:
    batch = ELABatch(
        {
            "node_irreps": torch.randn(5, 4),
            "pos": torch.randn(5, 3),
        }
    )
    moved = batch.to(
        "cpu",
        dtype=torch.float32,
        geometry_dtype=torch.float64,
    )
    assert moved["node_irreps"].dtype == torch.float32
    assert moved["pos"].dtype == torch.float64


def test_prepared_graph_moves_with_batch() -> None:
    graph_batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    edges = torch.tensor(
        [[0, 1, 2, 3], [1, 0, 3, 2]],
        dtype=torch.long,
    )
    graph = prepare_3d_graph(graph_batch, edges)
    batch = ELABatch(
        {
            "node_irreps": torch.randn(4, 3),
            "pos": torch.randn(4, 3),
            "graph": graph,
        }
    )
    moved = batch.to("cpu")
    assert moved["graph"].device.type == "cpu"
    assert moved["graph"].num_edges == graph.num_edges
