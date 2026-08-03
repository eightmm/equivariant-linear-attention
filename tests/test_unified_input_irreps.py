from __future__ import annotations

import pytest
import torch

from equivariant_linear_attention import (
    IrrepLayout,
    matrix_to_st5,
    pack_irreps,
    split_irreps,
    st5_to_matrix,
)
from equivariant_linear_attention.geometry import prepare_3d_graph
from equivariant_linear_attention.model.runtime import (
    _BaseStackConfig,
    _ELARuntime,
)


INPUT_IRREPS = "2x0e + 1x0o + 1x1e + 2x1o + 1x2e + 1x2o"


def _complete_edge_index(num_nodes: int) -> torch.Tensor:
    receiver = torch.arange(num_nodes).repeat_interleave(num_nodes)
    sender = torch.arange(num_nodes).repeat(num_nodes)
    return torch.stack([receiver, sender])


def _transform_blocks(
    layout: IrrepLayout,
    value: torch.Tensor,
    orthogonal: torch.Tensor,
) -> torch.Tensor:
    determinant = torch.linalg.det(orthogonal)
    transformed: dict[str, torch.Tensor] = {}
    for block in layout.blocks:
        irrep = block.irrep
        block_value = split_irreps(layout, value)[str(irrep)]
        sign = determinant ** (
            irrep.degree + (1 if irrep.parity == "o" else 0)
        )
        if irrep.degree == 0:
            transformed[str(irrep)] = sign * block_value
        elif irrep.degree == 1:
            transformed[str(irrep)] = sign * torch.einsum(
                "...c,dc->...d", block_value, orthogonal
            )
        else:
            matrix = st5_to_matrix(block_value)
            rotated = torch.einsum(
                "ab,...bc,dc->...ad", orthogonal, matrix, orthogonal
            )
            transformed[str(irrep)] = sign * matrix_to_st5(rotated)
    return pack_irreps(layout, transformed)


def test_flat_irrep_pack_split_and_st5_round_trips() -> None:
    layout = IrrepLayout.parse(INPUT_IRREPS)
    blocks = {
        str(block.irrep): torch.randn(4, block.multiplicity, block.irrep.dim)
        for block in layout.blocks
    }
    packed = pack_irreps(layout, blocks)

    assert packed.shape == (4, layout.dim)
    unpacked = split_irreps(layout, packed)
    for name, value in blocks.items():
        torch.testing.assert_close(unpacked[name], value)

    matrix = torch.randn(4, 2, 3, 3)
    matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
    matrix = matrix - (
        torch.diagonal(matrix, dim1=-2, dim2=-1).sum(-1)[..., None, None]
        * torch.eye(3)
        / 3.0
    )
    torch.testing.assert_close(st5_to_matrix(matrix_to_st5(matrix)), matrix)


def test_unified_accepts_all_lte2_input_sectors_and_backpropagates() -> None:
    torch.manual_seed(730)
    config = _BaseStackConfig(
        input_irreps=INPUT_IRREPS,
        output_irreps="1x0e + 1x0o + 1x1e + 1x1o + 1x2e + 1x2o",
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        local_rank=2,
        local_cutoff=10.0,
    )
    model = _ELARuntime(config).double()
    node_irreps = torch.randn(
        6, config.input_layout.dim, dtype=torch.float64, requires_grad=True
    )
    pos = torch.randn(6, 3, dtype=torch.float64, requires_grad=True)
    graph = prepare_3d_graph(
        torch.zeros(6, dtype=torch.long), _complete_edge_index(6)
    )

    output = model(node_irreps, pos, graph)["node_irreps"]
    output.square().mean().backward()

    assert node_irreps.grad is not None
    assert torch.isfinite(node_irreps.grad).all()
    for block in config.input_layout.blocks:
        sector_grad = node_irreps.grad[..., config.input_layout.slice_for(block.irrep)]
        assert torch.count_nonzero(sector_grad) > 0
    assert pos.grad is not None
    assert torch.isfinite(pos.grad).all()


def test_flat_input_obeys_reflection_and_translation_contract() -> None:
    torch.manual_seed(731)
    config = _BaseStackConfig(
        input_irreps=INPUT_IRREPS,
        output_irreps="1x0e + 1x0o + 1x1e + 1x1o + 1x2e + 1x2o",
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        local_rank=2,
        local_cutoff=10.0,
    )
    model = _ELARuntime(config).double().eval()
    node_irreps = torch.randn(6, config.input_layout.dim, dtype=torch.float64)
    pos = torch.randn(6, 3, dtype=torch.float64)
    graph = prepare_3d_graph(
        torch.zeros(6, dtype=torch.long), _complete_edge_index(6)
    )
    reflection = torch.diag(
        torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)
    )

    reference = model(node_irreps, pos, graph)["node_irreps"]
    transformed = model(
        _transform_blocks(config.input_layout, node_irreps, reflection),
        pos @ reflection.T + torch.tensor([3.0, -2.0, 1.0]),
        graph,
    )["node_irreps"]
    expected = _transform_blocks(config.output_layout, reference, reflection)

    torch.testing.assert_close(transformed, expected, atol=3e-8, rtol=3e-8)


def test_input_projector_bias_and_geometry_only_contract() -> None:
    config = _BaseStackConfig(
        input_irreps=INPUT_IRREPS,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
    )
    model = _ELARuntime(config)
    projectors = model.core.input_projection.projectors

    assert projectors["0e"].bias is not None
    assert all(
        getattr(layer, "bias", None) is None
        for name, layer in projectors.items()
        if name != "0e"
    )

    geometry_only = _ELARuntime(
        _BaseStackConfig(
            input_irreps="0",
            hidden_dim=16,
            num_layers=1,
            num_heads=4,
        )
    )
    graph = prepare_3d_graph(
        torch.zeros(4, dtype=torch.long), _complete_edge_index(4)
    )
    output = geometry_only(
        torch.empty(4, 0), torch.randn(4, 3), graph
    )["graph_irreps"]
    assert torch.isfinite(output).all()


def test_unified_rejects_high_degree_input() -> None:
    with pytest.raises(ValueError, match="l<=2"):
        _BaseStackConfig(input_irreps="1x3o")
