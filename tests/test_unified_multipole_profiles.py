from __future__ import annotations

import torch

from equivariant_attention.unified import (
    Unified3DConfig,
    UnifiedEquivariantAttention,
    prepare_3d_graph,
)


def _complete_edge_index(num_nodes: int) -> torch.Tensor:
    receiver = torch.arange(num_nodes).repeat_interleave(num_nodes)
    sender = torch.arange(num_nodes).repeat(num_nodes)
    return torch.stack([receiver, sender])


def _positions() -> torch.Tensor:
    return torch.tensor(
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


def test_positive_radial_profiles_do_not_collapse_inside_large_cutoff() -> None:
    config = Unified3DConfig(
        input_irreps="3x0e",
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        local_rank=3,
        local_cutoff=10.0,
        num_rbf=8,
    )
    model = UnifiedEquivariantAttention(config).double().eval()
    positions = _positions()
    graph = prepare_3d_graph(
        torch.zeros(positions.shape[0], dtype=torch.long),
        _complete_edge_index(positions.shape[0]),
    )

    geometry = model.core._build_geometry(positions, graph.neighbors)
    multipoles = model.core.node_multipoles(geometry)

    assert torch.count_nonzero(multipoles.axial.abs() > 1e-10)
    assert torch.count_nonzero(multipoles.odd_scalar.abs() > 1e-12)
    assert torch.count_nonzero(multipoles.odd_tensor.abs() > 1e-12)


def test_radial_profiles_are_positive_and_distinct() -> None:
    config = Unified3DConfig(
        input_irreps="2x0e",
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        local_rank=4,
        num_rbf=3,
    )
    bank = UnifiedEquivariantAttention(config).core.node_multipoles

    scales = bank._radial_scales
    assert scales.shape == (4,)
    assert torch.all(scales[1:] > scales[:-1])

    coordinate = torch.linspace(0.0, 1.0, 17).unsqueeze(-1)
    profiles = torch.exp((coordinate - 0.5) * scales.unsqueeze(0))
    assert torch.all(profiles > 0)
    assert torch.linalg.matrix_rank(profiles) == scales.numel()
