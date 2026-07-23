from __future__ import annotations

import torch

from equivariant_attention.moment import _graph_summary_for_nodes


def test_single_graph_summary_uses_broadcast_view_and_accumulates_gradients() -> None:
    summary = torch.arange(12, dtype=torch.float64).reshape(1, 3, 4)
    summary.requires_grad_()
    batch = torch.zeros(17, dtype=torch.long)

    broadcast = _graph_summary_for_nodes(summary, batch, num_graphs=1)

    assert broadcast.shape == (17, 3, 4)
    assert broadcast.stride(0) == 0
    assert torch.equal(broadcast[0], summary[0])
    broadcast.sum().backward()
    assert torch.equal(summary.grad, torch.full_like(summary, 17.0))


def test_multi_graph_summary_preserves_indexed_broadcast() -> None:
    summary = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    batch = torch.tensor([0, 1, 1, 0], dtype=torch.long)

    broadcast = _graph_summary_for_nodes(summary, batch, num_graphs=2)

    assert torch.equal(broadcast, summary[batch])
