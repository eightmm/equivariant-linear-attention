from __future__ import annotations

import torch

from equivariant_attention import (
    ELA,
    ELAConfig,
    ELAContext,
    ELAFeatures,
    SparseGeometry,
)


def _padded_fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(13)
    features = torch.randn(2, 5, 4, dtype=torch.float64)
    positions = 0.1 * torch.randn(2, 5, 3, dtype=torch.float64)
    mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, True],
        ]
    )
    positions[~mask] = 99.0
    return features, positions, mask


def _batched_complete_edges(
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    parts = []
    max_edges = 0
    for count in mask.sum(dim=1).tolist():
        count = int(count)
        receiver = torch.arange(count).repeat_interleave(count)
        sender = torch.arange(count).repeat(count)
        edge = torch.stack([receiver, sender])
        parts.append(edge)
        max_edges = max(max_edges, edge.shape[1])
    padded = torch.full((len(parts), 2, max_edges), -1, dtype=torch.long)
    edge_mask = torch.zeros((len(parts), max_edges), dtype=torch.bool)
    for index, edge in enumerate(parts):
        padded[index, :, : edge.shape[1]] = edge
        edge_mask[index, : edge.shape[1]] = True
    return padded, edge_mask


def test_padded_automatic_batch_matches_flat_packed_batch() -> None:
    features, positions, mask = _padded_fixture()
    model = ELA.scalar(
        4,
        output_dim=2,
        width=32,
        depth=1,
        cutoff=10.0,
        num_rbf=8,
    ).double().eval()

    flat_features = features[mask]
    flat_positions = positions[mask]
    flat_batch = torch.repeat_interleave(
        torch.arange(2),
        mask.sum(dim=1),
    )
    with torch.inference_mode():
        padded = model(features, positions, mask=mask)
        flat = model(
            flat_features,
            flat_positions,
            batch=flat_batch,
        )

    assert padded["node_irreps"].shape == (2, 5, 2)
    assert padded["positions"].shape == (2, 5, 3)
    assert padded["node_mask"].shape == (2, 5)
    torch.testing.assert_close(
        padded["node_irreps"][mask],
        flat["node_irreps"],
        atol=2e-10,
        rtol=2e-10,
    )
    torch.testing.assert_close(
        padded["graph_irreps"],
        flat["graph_irreps"],
        atol=2e-10,
        rtol=2e-10,
    )
    torch.testing.assert_close(
        padded["node_irreps"][~mask],
        torch.zeros_like(padded["node_irreps"][~mask]),
    )
    torch.testing.assert_close(padded["positions"][~mask], positions[~mask])


def test_padded_edge_index_adjacency_and_ragged_edges_match() -> None:
    features, positions, mask = _padded_fixture()
    edge_index, edge_mask = _batched_complete_edges(mask)
    adjacency = torch.zeros((2, 5, 5), dtype=torch.bool)
    adjacency[0, :3, :3] = True
    adjacency[1, :5, :5] = True
    ragged = [
        edge_index[0, :, edge_mask[0]],
        edge_index[1, :, edge_mask[1]],
    ]
    model = ELA.scalar(
        4,
        width=32,
        depth=1,
        cutoff=10.0,
        num_rbf=8,
    ).double().eval()

    with torch.inference_mode():
        tensor_edges = model(
            features,
            positions,
            mask=mask,
            edge_index=edge_index,
            edge_mask=edge_mask,
        )["node_irreps"]
        adjacency_edges = model(
            features,
            positions,
            mask=mask,
            adjacency=adjacency,
        )["node_irreps"]
        ragged_edges = model(
            features,
            positions,
            mask=mask,
            edge_index=ragged,
        )["node_irreps"]

    torch.testing.assert_close(tensor_edges, adjacency_edges, atol=0.0, rtol=0.0)
    torch.testing.assert_close(tensor_edges, ragged_edges, atol=0.0, rtol=0.0)


def test_padded_condition_order_and_refinement_shortcuts() -> None:
    features, positions, mask = _padded_fixture()
    config = ELAConfig(
        input_irreps="4x0e",
        output_irreps="1x0e",
        width=32,
        depth=1,
        geometry=SparseGeometry(cutoff=10.0, num_rbf=8),
        features=ELAFeatures(
            condition_dim=1,
            order_dim=1,
            coordinate_refinement=True,
        ),
    )
    model = ELA(config).double().eval()
    condition = torch.randn(2, 5, dtype=torch.float64)
    order = torch.arange(5, dtype=torch.float64).expand(2, 5)
    order_group = torch.zeros(2, 5, dtype=torch.long)
    update_mask = mask.clone()
    update_mask[:, 0] = False

    with torch.inference_mode():
        output = model(
            features,
            positions,
            mask=mask,
            condition=condition,
            order=order,
            order_group=order_group,
            order_mask=mask,
            refine_steps=1,
            max_coordinate_step=0.1,
            update_mask=update_mask,
        )
    assert output["node_irreps"].shape == (2, 5, 1)
    assert output["positions"].shape == (2, 5, 3)
    torch.testing.assert_close(
        output["coordinate_delta"][~update_mask],
        torch.zeros_like(output["coordinate_delta"][~update_mask]),
    )


def test_explicit_context_is_normalized_for_padded_nodes() -> None:
    features, positions, mask = _padded_fixture()
    config = ELAConfig(
        input_irreps="4x0e",
        width=32,
        depth=1,
        geometry=SparseGeometry(cutoff=10.0, num_rbf=8),
        features=ELAFeatures(condition_dim=2),
    )
    model = ELA(config).double().eval()
    context = ELAContext(
        condition=torch.randn(2, 5, 2, dtype=torch.float64)
    )
    with torch.inference_mode():
        output = model(features, positions, mask=mask, context=context)
    assert output["node_irreps"].shape[:2] == (2, 5)


def test_graph_condition_is_not_confused_with_padded_node_axis() -> None:
    features, positions, mask = _padded_fixture()
    config = ELAConfig(
        input_irreps="4x0e",
        width=32,
        depth=1,
        geometry=SparseGeometry(cutoff=10.0, num_rbf=8),
        features=ELAFeatures(condition_dim=5),
    )
    model = ELA(config).double().eval()
    graph_condition = torch.randn(2, 5, dtype=torch.float64)

    with torch.inference_mode():
        shortcut = model(
            features,
            positions,
            mask=mask,
            condition=graph_condition,
        )["graph_irreps"]
        explicit = model(
            features,
            positions,
            mask=mask,
            context=ELAContext(condition=graph_condition),
        )["graph_irreps"]
    torch.testing.assert_close(shortcut, explicit, atol=0.0, rtol=0.0)
