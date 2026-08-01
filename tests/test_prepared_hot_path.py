from __future__ import annotations

import torch

from equivariant_attention import ELA, ELAContext


def _complete_edges(nodes: int) -> torch.Tensor:
    receiver = torch.arange(nodes).repeat_interleave(nodes)
    sender = torch.arange(nodes).repeat(nodes)
    return torch.stack([receiver, sender])


def test_forward_prepared_matches_validated_forward() -> None:
    torch.manual_seed(47)
    model = ELA.scalar(
        4,
        output_dim=2,
        width=32,
        depth=2,
        cutoff=10.0,
        num_rbf=8,
    ).double().eval()
    features = torch.randn(7, 4, dtype=torch.float64)
    positions = 0.1 * torch.randn(7, 3, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    edges = torch.cat(
        [
            _complete_edges(3),
            _complete_edges(4) + torch.tensor([[3], [3]]),
        ],
        dim=1,
    )
    graph = model.prepare_graph(
        positions,
        batch=batch,
        edge_index=edges,
    )

    with torch.inference_mode():
        validated = model(features, positions, graph)
        hot = model.forward_prepared(features, positions, graph)
    assert validated.keys() == hot.keys()
    for name in validated:
        torch.testing.assert_close(hot[name], validated[name], atol=0.0, rtol=0.0)


def test_forward_prepared_supports_reusable_context() -> None:
    torch.manual_seed(53)
    model = ELA(
        input_irreps="4x0e",
        output_irreps="1x0e",
        width=32,
        depth=1,
        cutoff=10.0,
        condition_dim=3,
    ).double().eval()
    features = torch.randn(5, 4, dtype=torch.float64)
    positions = 0.1 * torch.randn(5, 3, dtype=torch.float64)
    graph = model.prepare_graph(
        positions,
        edge_index=_complete_edges(5),
    )
    context = ELAContext(condition=torch.randn(1, 3, dtype=torch.float64))

    with torch.inference_mode():
        validated = model(features, positions, graph, context=context)
        hot = model.forward_prepared(
            features,
            positions,
            graph,
            context=context,
        )
    for name in validated:
        torch.testing.assert_close(hot[name], validated[name], atol=0.0, rtol=0.0)


def test_forward_prepared_keeps_coordinate_gradients() -> None:
    torch.manual_seed(59)
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
    graph = model.prepare_graph(
        positions.detach(),
        edge_index=_complete_edges(5),
    )
    output = model.forward_prepared(features, positions, graph)
    output["node_irreps"].square().mean().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert positions.grad is not None and torch.isfinite(positions.grad).all()
