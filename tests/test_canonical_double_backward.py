from __future__ import annotations

import torch

from equivariant_linear_attention import ELA, ELAGraph


def _complete_edges(nodes: int) -> torch.Tensor:
    receiver = torch.arange(nodes).repeat_interleave(nodes)
    sender = torch.arange(nodes).repeat(nodes)
    return torch.stack([receiver, sender])


def test_canonical_fixed_fusion_supports_input_and_coordinate_double_backward() -> None:
    torch.manual_seed(53)
    model = ELA(
        input_irreps="4x0e",
        output_irreps="1x0e",
        width=16,
        depth=1,
        cutoff=10.0,
    ).double()
    nodes = 5
    features = torch.randn(
        nodes,
        4,
        dtype=torch.float64,
        requires_grad=True,
    )
    positions = torch.randn(
        nodes,
        3,
        dtype=torch.float64,
        requires_grad=True,
    )
    graph = ELAGraph(
        x=features,
        pos=positions,
        edge_index=_complete_edges(nodes),
    )
    output = model(graph).x
    first_features, first_positions = torch.autograd.grad(
        output.square().sum(),
        (features, positions),
        create_graph=True,
    )
    second_features, second_positions = torch.autograd.grad(
        first_features.square().sum() + first_positions.square().sum(),
        (features, positions),
    )

    for gradient in (
        first_features,
        first_positions,
        second_features,
        second_positions,
    ):
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0
