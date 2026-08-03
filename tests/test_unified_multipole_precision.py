from __future__ import annotations

import pytest
import torch

from equivariant_linear_attention.geometry import prepare_3d_graph
from equivariant_linear_attention.model.runtime import (
    _BaseStackConfig,
    _ELARuntime,
)


def _complete_edge_index(num_nodes: int, *, device: torch.device) -> torch.Tensor:
    receiver = torch.arange(num_nodes, device=device).repeat_interleave(num_nodes)
    sender = torch.arange(num_nodes, device=device).repeat(num_nodes)
    return torch.stack([receiver, sender])


def test_multipole_core_supports_coordinate_double_backward() -> None:
    torch.manual_seed(211)
    config = _BaseStackConfig(
        input_irreps="3x0e",
        output_irreps="1x0e",
        hidden_dim=12,
        num_layers=1,
        num_heads=3,
        local_rank=3,
        local_cutoff=5.0,
        num_rbf=6,
    )
    model = _ELARuntime(config).double()
    num_nodes = 5
    node_features = torch.randn(
        num_nodes,
        3,
        dtype=torch.float64,
        requires_grad=True,
    )
    positions = torch.randn(
        num_nodes,
        3,
        dtype=torch.float64,
        requires_grad=True,
    )
    graph = prepare_3d_graph(
        torch.zeros(num_nodes, dtype=torch.long),
        _complete_edge_index(num_nodes, device=torch.device("cpu")),
    )

    energy = model(node_features, positions, graph)["graph_irreps"].sum()
    force = -torch.autograd.grad(
        energy,
        positions,
        create_graph=True,
    )[0]
    curvature_loss = force.square().sum()
    coordinate_hessian_vector = torch.autograd.grad(
        curvature_loss,
        positions,
    )[0]

    assert torch.isfinite(energy)
    assert torch.isfinite(force).all()
    assert torch.isfinite(coordinate_hessian_vector).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_multipole_core_cuda_bfloat16_forward_backward() -> None:
    torch.manual_seed(223)
    device = torch.device("cuda")
    config = _BaseStackConfig(
        input_irreps="4x0e",
        output_irreps="1x0e + 1x0o + 1x1o + 1x2e",
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        local_rank=3,
        local_cutoff=6.0,
        num_rbf=8,
    )
    model = _ELARuntime(config).to(
        device=device,
        dtype=torch.bfloat16,
    )
    num_nodes = 8
    node_features = torch.randn(
        num_nodes,
        4,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    positions = torch.randn(
        num_nodes,
        3,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    graph = prepare_3d_graph(
        torch.zeros(num_nodes, dtype=torch.long, device=device),
        _complete_edge_index(num_nodes, device=device),
    )

    output = model(node_features, positions, graph)["node_irreps"]
    loss = output.float().square().mean()
    loss.backward()

    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output.float()).all()
    assert node_features.grad is not None
    assert positions.grad is not None
    assert torch.isfinite(node_features.grad.float()).all()
    assert torch.isfinite(positions.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad.float()).all()
        for parameter in model.parameters()
    )
