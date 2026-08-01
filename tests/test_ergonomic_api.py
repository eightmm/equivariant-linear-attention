from __future__ import annotations

import torch

from equivariant_attention import ELA, ELAConfig, SparseGeometry, collate_graphs


def _complete_edges(nodes: int) -> torch.Tensor:
    receiver = torch.arange(nodes).repeat_interleave(nodes)
    sender = torch.arange(nodes).repeat(nodes)
    return torch.stack([receiver, sender])


def test_direct_constructor_and_config_constructor_match() -> None:
    torch.manual_seed(3)
    config = ELAConfig(
        input_irreps="4x0e",
        output_irreps="1x0e",
        width=32,
        depth=1,
        geometry=SparseGeometry(cutoff=10.0, num_rbf=8),
    )
    configured = ELA(config).double().eval()
    direct = ELA(
        input_irreps="4x0e",
        output_irreps="1x0e",
        width=32,
        depth=1,
        cutoff=10.0,
        num_rbf=8,
    ).double().eval()
    direct.load_state_dict(configured.state_dict(), strict=True)

    features = torch.randn(5, 4, dtype=torch.float64)
    positions = 0.1 * torch.randn(5, 3, dtype=torch.float64)
    edges = _complete_edges(5)
    with torch.inference_mode():
        expected = configured(
            features,
            positions,
            edge_index=edges,
        )["node_irreps"]
        actual = direct(
            features,
            positions,
            edge_index=edges,
        )["node_irreps"]
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_scalar_factory_and_automatic_radius_graph() -> None:
    torch.manual_seed(5)
    model = ELA.scalar(
        4,
        output_dim=2,
        width=32,
        depth=1,
        cutoff=10.0,
        num_rbf=8,
    ).double().eval()
    features = torch.randn(6, 4, dtype=torch.float64)
    positions = 0.1 * torch.randn(6, 3, dtype=torch.float64)
    prepared = model.prepare_graph(
        positions,
        edge_index=_complete_edges(6),
    )

    with torch.inference_mode():
        automatic = model(features, positions)
        cached = model(features, positions, prepared)
    assert automatic["node_irreps"].shape == (6, 2)
    assert automatic["graph_irreps"].shape == (1, 2)
    torch.testing.assert_close(
        automatic["node_irreps"],
        cached["node_irreps"],
        atol=2e-10,
        rtol=2e-10,
    )


def test_collated_mapping_runs_without_pyg() -> None:
    torch.manual_seed(7)
    samples = [
        {
            "x": torch.randn(3, 4, dtype=torch.float64),
            "positions": 0.1 * torch.randn(3, 3, dtype=torch.float64),
            "edge_index": _complete_edges(3),
            "target": torch.tensor([1.0], dtype=torch.float64),
            "sample_id": "a",
        },
        {
            "node_irreps": torch.randn(5, 4, dtype=torch.float64),
            "pos": 0.1 * torch.randn(5, 3, dtype=torch.float64),
            "edge_index": _complete_edges(5),
            "target": torch.tensor([2.0], dtype=torch.float64),
            "sample_id": "b",
        },
    ]
    batch = collate_graphs(samples)
    model = ELA.scalar(
        4,
        width=32,
        depth=1,
        cutoff=10.0,
        num_rbf=8,
    ).double().eval()

    with torch.inference_mode():
        output = model(batch)
    assert output["node_irreps"].shape == (8, 1)
    assert output["graph_irreps"].shape == (2, 1)
    assert batch["target"].shape == (2, 1)
    assert batch["sample_ids"] == ("a", "b")


def test_automatic_graph_keeps_coordinate_gradients() -> None:
    torch.manual_seed(11)
    model = ELA.scalar(
        4,
        width=32,
        depth=1,
        cutoff=10.0,
        num_rbf=8,
    ).double()
    features = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    positions = 0.1 * torch.randn(
        5,
        3,
        dtype=torch.float64,
        requires_grad=True,
    )
    output = model(features, positions)
    output["node_irreps"].square().mean().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert positions.grad is not None and torch.isfinite(positions.grad).all()
