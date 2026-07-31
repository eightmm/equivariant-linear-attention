from __future__ import annotations

import torch

from equivariant_attention.canonical_regression import ELARegressionModel


def _batched_complete_edges(nodes_per_graph: int, num_graphs: int) -> torch.Tensor:
    blocks = []
    for graph in range(num_graphs):
        offset = graph * nodes_per_graph
        receiver = (
            torch.arange(nodes_per_graph).repeat_interleave(nodes_per_graph)
            + offset
        )
        sender = torch.arange(nodes_per_graph).repeat(nodes_per_graph) + offset
        blocks.append(torch.stack([receiver, sender]))
    return torch.cat(blocks, dim=1)


def test_regression_adapter_forward_backward_and_masked_readout() -> None:
    torch.manual_seed(31)
    model = ELARegressionModel(
        node_dim=4,
        width=32,
        depth=1,
        cutoff=10.0,
        num_rbf=8,
    ).double()
    nodes = 8
    features = torch.randn(nodes, 4, dtype=torch.float64, requires_grad=True)
    positions = torch.randn(nodes, 3, dtype=torch.float64, requires_grad=True)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    readout_mask = torch.tensor(
        [True, True, False, False, True, False, True, False]
    )
    output = model(
        features,
        positions,
        batch=batch,
        edge_index=_batched_complete_edges(4, 2),
        readout_mask=readout_mask,
    )
    output["graph_scalars"].square().mean().backward()

    assert output["graph_scalars"].shape == (2, 1)
    assert torch.isfinite(output["graph_scalars"]).all()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert positions.grad is not None and torch.isfinite(positions.grad).all()
