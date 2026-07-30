from __future__ import annotations

import torch

from equivariant_attention import (
    Unified3DConfig,
    UnifiedEquivariantAttention,
    prepare_3d_graph,
)


def _complete_edge_index(num_nodes: int) -> torch.Tensor:
    receiver = torch.arange(num_nodes).repeat_interleave(num_nodes)
    sender = torch.arange(num_nodes).repeat(num_nodes)
    return torch.stack([receiver, sender])


def test_chiral_bridge_is_deterministic_cyclic_selector() -> None:
    config = Unified3DConfig(
        input_irreps="4x0e",
        output_irreps="1x0e",
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        local_rank=3,
    )
    model = UnifiedEquivariantAttention(config)

    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    for block in model.core.blocks:
        torch.testing.assert_close(
            block.local_chiral_scalar_out.weight,
            expected,
        )


def test_even_only_objective_reaches_chiral_bridge_on_first_backward() -> None:
    torch.manual_seed(23)
    config = Unified3DConfig(
        input_irreps="4x0e",
        output_irreps="1x0e",
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        local_rank=3,
        local_cutoff=10.0,
    )
    model = UnifiedEquivariantAttention(config).double()
    num_nodes = 6
    batch = torch.zeros(num_nodes, dtype=torch.long)
    graph = prepare_3d_graph(batch, _complete_edge_index(num_nodes))
    node_features = torch.randn(num_nodes, 4, dtype=torch.float64)
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.2, -0.1],
            [-0.3, 1.2, 0.4],
            [0.4, -0.6, 1.4],
            [-1.1, -0.2, 0.7],
            [0.8, 1.1, -0.9],
        ],
        dtype=torch.float64,
    )

    output = model(node_features, positions, graph)["node_irreps"]
    output.square().mean().backward()

    for block in model.core.blocks:
        gradient = block.local_chiral_scalar_out.weight.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0
