from __future__ import annotations

import pytest
import torch

from equivariant_linear_attention.geometry import Prepared3DGraph, prepare_3d_graph
from equivariant_linear_attention.model.runtime import (
    _BaseStackConfig,
    _ELARuntime,
)


def _complete_edge_index(num_nodes: int) -> torch.Tensor:
    receiver = torch.arange(num_nodes).repeat_interleave(num_nodes)
    sender = torch.arange(num_nodes).repeat(num_nodes)
    return torch.stack([receiver, sender])


def _st_to_matrix(value: torch.Tensor) -> torch.Tensor:
    xx, yy, xy, xz, yz = value.unbind(dim=-1)
    zz = -xx - yy
    return torch.stack(
        [
            torch.stack([xx, xy, xz], dim=-1),
            torch.stack([xy, yy, yz], dim=-1),
            torch.stack([xz, yz, zz], dim=-1),
        ],
        dim=-2,
    )


def _matrix_to_st(value: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            value[..., 0, 0],
            value[..., 1, 1],
            value[..., 0, 1],
            value[..., 0, 2],
            value[..., 1, 2],
        ],
        dim=-1,
    )


def _transform_output(
    model: _ELARuntime,
    value: torch.Tensor,
    orthogonal: torch.Tensor,
) -> torch.Tensor:
    determinant = torch.linalg.det(orthogonal)
    split = model.split_output(value)
    outputs = []
    for block in model.output_irreps.blocks:
        irrep = block.irrep
        block_value = split[str(irrep)]
        # Parity labels specify inversion parity. Relative to the ordinary
        # Cartesian degree-l action, an improper transform contributes
        # det(R) ** (l + 1[parity=o]).
        sign = determinant ** (
            irrep.degree + (1 if irrep.parity == "o" else 0)
        )
        if irrep.degree == 0:
            transformed = sign * block_value
        elif irrep.degree == 1:
            transformed = sign * torch.einsum(
                "...c,dc->...d",
                block_value,
                orthogonal,
            )
        elif irrep.degree == 2:
            matrix = _st_to_matrix(block_value)
            transformed_matrix = torch.einsum(
                "ab,...bc,dc->...ad",
                orthogonal,
                matrix,
                orthogonal,
            )
            transformed = sign * _matrix_to_st(transformed_matrix)
        else:
            raise AssertionError("test helper supports l<=2")
        outputs.append(transformed.flatten(start_dim=-2))
    return torch.cat(outputs, dim=-1)


def test_unified_config_exposes_only_output_representation() -> None:
    config = _BaseStackConfig(
        input_irreps="5x0e",
        output_irreps="2x0e + 1x0o + 2x1o + 1x1e + 1x2e + 1x2o",
        hidden_dim=16,
        num_heads=4,
    )

    assert str(config.internal_irreps) == (
        "16x0e + 4x0o + 4x1e + 4x1o + 4x2e + 4x2o"
    )
    assert config.output_layout.dim == 22
    contract = config.canonical_contract()
    assert contract["public_symmetry"] == "SE3"
    assert contract["internal_symmetry"] == "O3_parity_complete"
    assert contract["user_representation_control"] == "input_and_output_irreps"
    assert contract["fallbacks"] == ()


def test_unified_config_rejects_unsupported_output_degree() -> None:
    with pytest.raises(ValueError, match="l<=2"):
        _BaseStackConfig(input_irreps="3x0e", output_irreps="1x3o")


def test_relation_cutoff_may_only_narrow_shared_domain() -> None:
    with pytest.raises(ValueError, match="only narrow"):
        _BaseStackConfig(
            input_irreps="3x0e",
            local_cutoff=4.0,
            relation_cutoffs=(2.0, 5.0),
        )


def test_prepare_3d_graph_rejects_cross_graph_edges() -> None:
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 0], [0, 1, 2, 3, 2]],
        dtype=torch.long,
    )
    with pytest.raises(ValueError, match="different graphs"):
        prepare_3d_graph(batch, edge_index)


def test_prepared_graph_same_device_move_is_identity() -> None:
    batch = torch.zeros(4, dtype=torch.long)
    graph = prepare_3d_graph(batch, _complete_edge_index(4))
    assert isinstance(graph, Prepared3DGraph)
    assert graph.to("cpu") is graph


def test_output_irreps_shape_and_split_contract() -> None:
    torch.manual_seed(4)
    config = _BaseStackConfig(
        input_irreps="4x0e",
        output_irreps="2x0e + 1x0o + 2x1o + 1x1e + 1x2e + 1x2o",
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        local_rank=2,
        local_cutoff=10.0,
    )
    model = _ELARuntime(config)
    graph = prepare_3d_graph(
        torch.zeros(6, dtype=torch.long),
        _complete_edge_index(6),
    )
    output = model(torch.randn(6, 4), torch.randn(6, 3), graph)

    assert output["node_irreps"].shape == (6, config.output_layout.dim)
    assert output["graph_irreps"].shape == (1, config.output_layout.dim)
    split = model.split_output(output["node_irreps"])
    assert split["0e"].shape == (6, 2, 1)
    assert split["0o"].shape == (6, 1, 1)
    assert split["1o"].shape == (6, 2, 3)
    assert split["1e"].shape == (6, 1, 3)
    assert split["2e"].shape == (6, 1, 5)
    assert split["2o"].shape == (6, 1, 5)


def test_parity_complete_output_obeys_improper_transform() -> None:
    torch.manual_seed(9)
    config = _BaseStackConfig(
        input_irreps="3x0e",
        output_irreps="1x0e + 1x0o + 1x1o + 1x1e + 1x2e + 1x2o",
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        local_rank=3,
        local_cutoff=20.0,
    )
    model = _ELARuntime(config).double().eval()
    with torch.no_grad():
        for block in model.core.blocks:
            block.local_chiral_scalar_out.weight.fill_(0.2)
            block.local_chiral_axial_out.weight.fill_(0.2)
            block.local_chiral_tensor_out.weight.fill_(0.2)

    node_feats = torch.randn(7, 3, dtype=torch.float64)
    pos = torch.randn(7, 3, dtype=torch.float64)
    batch = torch.zeros(7, dtype=torch.long)
    graph = prepare_3d_graph(batch, _complete_edge_index(7))
    reflection = torch.diag(
        torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)
    )
    translation = torch.tensor([2.0, -1.0, 0.5], dtype=torch.float64)

    reference = model(node_feats, pos, graph)["node_irreps"]
    transformed = model(
        node_feats,
        pos @ reflection.T + translation,
        graph,
    )["node_irreps"]
    expected = _transform_output(model, reference, reflection)

    torch.testing.assert_close(
        transformed,
        expected,
        atol=2e-8,
        rtol=2e-8,
    )
    odd = model.split_output(reference)["0o"]
    assert torch.count_nonzero(odd.abs() > 1e-12)


def test_unified_forward_and_coordinate_gradients_are_finite() -> None:
    torch.manual_seed(12)
    config = _BaseStackConfig(
        input_irreps="4x0e",
        output_irreps="1x0e + 1x0o + 1x1o",
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        local_rank=2,
        local_cutoff=10.0,
    )
    model = _ELARuntime(config).double()
    node_feats = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
    pos = torch.randn(6, 3, dtype=torch.float64, requires_grad=True)
    graph = prepare_3d_graph(
        torch.zeros(6, dtype=torch.long),
        _complete_edge_index(6),
    )

    output = model(node_feats, pos, graph)["node_irreps"]
    loss = output.square().mean()
    loss.backward()

    assert torch.isfinite(output).all()
    assert node_feats.grad is not None
    assert pos.grad is not None
    assert torch.isfinite(node_feats.grad).all()
    assert torch.isfinite(pos.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_relation_metadata_is_packed_once_and_range_checked() -> None:
    config = _BaseStackConfig(
        input_irreps="3x0e",
        output_irreps="1x0e",
        hidden_dim=16,
        num_heads=4,
        relation_cutoffs=(2.0, 4.0),
        local_cutoff=4.0,
    )
    model = _ELARuntime(config)
    batch = torch.zeros(4, dtype=torch.long)
    edge_index = _complete_edge_index(4)
    relation_id = torch.arange(edge_index.shape[1], dtype=torch.long) % 2
    graph = prepare_3d_graph(
        batch,
        edge_index,
        edge_relation_id=relation_id,
    )

    assert graph.neighbors.relation_id is not None
    output = model(torch.randn(4, 3), torch.randn(4, 3), graph)
    assert torch.isfinite(output["graph_irreps"]).all()

    invalid_graph = prepare_3d_graph(
        batch,
        edge_index,
        edge_relation_id=torch.full(
            (edge_index.shape[1],),
            2,
            dtype=torch.long,
        ),
    )
    with pytest.raises(ValueError, match="outside"):
        model(torch.randn(4, 3), torch.randn(4, 3), invalid_graph)
