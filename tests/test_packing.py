from __future__ import annotations

import torch

from equivariant_linear_attention.packing import collate_graphs, pack_edges, pack_node_input


def _padded_input() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.arange(18, dtype=torch.float64).reshape(2, 3, 3)
    positions = torch.arange(18, dtype=torch.float64).reshape(2, 3, 3) / 10.0
    mask = torch.tensor([[True, True, False], [True, False, True]])
    return features, positions, mask


def test_padded_nodes_pack_restore_and_accept_all_edge_layouts() -> None:
    features, positions, mask = _padded_input()
    packed = pack_node_input(features, positions, mask=mask)

    assert packed.layout.kind == "padded"
    assert packed.layout.batch_size == 2
    assert torch.equal(packed.batch, torch.tensor([0, 0, 1, 1]))
    torch.testing.assert_close(packed.node_irreps, features[mask])
    torch.testing.assert_close(
        packed.layout.restore_node_tensor(packed.node_irreps),
        torch.where(mask.unsqueeze(-1), features, torch.zeros_like(features)),
    )

    padded_edges = torch.tensor(
        [
            [[0, 1, -1], [1, 0, -1]],
            [[0, 2, -1], [2, 0, -1]],
        ]
    )
    relation = torch.tensor([[0, 1, 9], [2, 3, 9]])
    expected_edges = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    expected_relation = torch.tensor([0, 1, 2, 3])

    tensor_edges, tensor_relation = pack_edges(
        packed,
        edge_index=padded_edges,
        edge_relation_id=relation,
    )
    assert torch.equal(tensor_edges, expected_edges)
    assert torch.equal(tensor_relation, expected_relation)

    transposed_edges, _ = pack_edges(
        packed,
        edge_index=padded_edges.transpose(1, 2),
        edge_mask=torch.tensor([[True, True, False], [True, True, False]]),
    )
    assert torch.equal(transposed_edges, expected_edges)

    ragged_edges, ragged_relation = pack_edges(
        packed,
        edge_index=(padded_edges[0, :, :2], padded_edges[1, :, :2]),
        edge_relation_id=(relation[0, :2], relation[1, :2]),
    )
    assert torch.equal(ragged_edges, expected_edges)
    assert torch.equal(ragged_relation, expected_relation)

    adjacency = torch.zeros(2, 3, 3, dtype=torch.bool)
    adjacency[0, 0, 1] = True
    adjacency[0, 2, 0] = True  # masked receiver: filtered
    adjacency[1, 2, 0] = True
    adjacency_edges, adjacency_relation = pack_edges(
        packed,
        adjacency=adjacency,
    )
    assert torch.equal(adjacency_edges, torch.tensor([[0, 3], [1, 2]]))
    assert adjacency_relation is None


def test_flat_nodes_and_edges_preserve_explicit_metadata() -> None:
    features = torch.randn(4, 3)
    positions = torch.randn(4, 3)
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    packed = pack_node_input(
        features,
        positions,
        batch=batch,
        mask=torch.ones(4, dtype=torch.bool),
    )
    assert packed.layout.kind == "flat"
    assert packed.layout.restore_node_tensor(features) is features
    assert torch.equal(packed.batch, batch.to(dtype=torch.long))

    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    relation = torch.tensor([3, 2, 1, 0], dtype=torch.int32)
    actual_edges, actual_relation = pack_edges(
        packed,
        edge_index=edge_index,
        edge_relation_id=relation,
    )
    assert torch.equal(actual_edges, edge_index)
    assert torch.equal(actual_relation, relation.to(dtype=torch.long))

    adjacency = torch.eye(4, dtype=torch.bool)
    adjacency_edges, adjacency_relation = pack_edges(packed, adjacency=adjacency)
    assert torch.equal(adjacency_edges, torch.arange(4).repeat(2, 1))
    assert adjacency_relation is None


def test_mapping_collation_carries_every_supported_training_field() -> None:
    edges_a = torch.tensor([[0, 1], [1, 0]])
    edges_b = torch.tensor([[0, 2, 1], [2, 0, 1]])
    samples = [
        {
            "x": torch.randn(2, 3),
            "positions": torch.randn(2, 3),
            "edge_index": edges_a,
            "edge_relation_id": torch.tensor([0, 1]),
            "target": torch.tensor([1.0]),
            "sample_id": "a",
            "condition": torch.tensor([1.0, 2.0]),
            "order": torch.tensor([[0.0], [1.0]]),
            "order_group": torch.tensor([0, 0]),
            "order_mask": torch.tensor([True, False]),
        },
        {
            "node_features": torch.randn(3, 3),
            "pos": torch.randn(3, 3),
            "edge_index": edges_b,
            "edge_relation_id": torch.tensor([2, 3, 4]),
            "target": torch.tensor([2.0]),
            "sample_id": "b",
            "condition": torch.tensor([3.0, 4.0]),
            "order": torch.tensor([[0.0], [1.0], [2.0]]),
            "order_group": torch.tensor([0, 0, 1]),
            "order_mask": torch.tensor([True, True, True]),
        },
    ]

    batch = collate_graphs(samples)
    assert batch["node_irreps"].shape == (5, 3)
    assert torch.equal(batch["batch"], torch.tensor([0, 0, 1, 1, 1]))
    assert torch.equal(
        batch["edge_index"],
        torch.cat([edges_a, edges_b + 2], dim=1),
    )
    assert torch.equal(batch["edge_relation_id"], torch.arange(5))
    assert batch["target"].shape == (2, 1)
    assert batch["condition"].shape == (2, 2)
    assert batch["order"].shape == (5, 1)
    assert batch["order_group"].shape == (5,)
    assert batch["order_mask"].dtype == torch.bool
    assert batch["sample_ids"] == ("a", "b")

    node_conditioned = [
        {**sample, "condition": torch.randn(count, 2)}
        for sample, count in zip(samples, (2, 3), strict=True)
    ]
    node_batch = collate_graphs(node_conditioned)
    assert node_batch["condition"].shape == (5, 2)
