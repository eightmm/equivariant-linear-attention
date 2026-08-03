from __future__ import annotations

import pytest
import torch

from equivariant_linear_attention import (
    ELA,
    ELABatch,
    IrrepLayout,
    matrix_to_st5,
    pack_irreps,
    split_irreps,
    st5_to_matrix,
)

FULL_IRREPS = "1x0e + 1x0o + 1x1e + 1x1o + 1x2e + 1x2o"


def _complete_edges(nodes: int) -> torch.Tensor:
    receiver = torch.arange(nodes).repeat_interleave(nodes)
    sender = torch.arange(nodes).repeat(nodes)
    return torch.stack([receiver, sender])


def _batched_complete_edges(nodes_per_graph: int, graphs: int) -> torch.Tensor:
    return torch.cat(
        [
            _complete_edges(nodes_per_graph) + graph * nodes_per_graph
            for graph in range(graphs)
        ],
        dim=1,
    )


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


def _transform_vector(
    value: torch.Tensor,
    orthogonal: torch.Tensor,
    *,
    axial: bool,
) -> torch.Tensor:
    result = torch.einsum("...c,dc->...d", value, orthogonal)
    return torch.linalg.det(orthogonal) * result if axial else result


def _transform_tensor(
    value: torch.Tensor,
    orthogonal: torch.Tensor,
    *,
    odd: bool,
) -> torch.Tensor:
    matrix = _st_to_matrix(value)
    result = torch.einsum(
        "ab,...bc,dc->...ad",
        orthogonal,
        matrix,
        orthogonal,
    )
    compact = _matrix_to_st(result)
    return torch.linalg.det(orthogonal) * compact if odd else compact


def _transform_irreps(
    layout: str | IrrepLayout,
    value: torch.Tensor,
    orthogonal: torch.Tensor,
) -> torch.Tensor:
    parsed = IrrepLayout.parse(layout)
    determinant = torch.linalg.det(orthogonal)
    source = split_irreps(parsed, value)
    transformed: dict[str, torch.Tensor] = {}
    for block in parsed.blocks:
        irrep = block.irrep
        block_value = source[str(irrep)]
        parity_factor = determinant ** (
            irrep.degree + (1 if irrep.parity == "o" else 0)
        )
        if irrep.degree == 0:
            result = block_value
        elif irrep.degree == 1:
            result = torch.einsum(
                "...c,dc->...d",
                block_value,
                orthogonal,
            )
        else:
            matrix = st5_to_matrix(block_value)
            result = matrix_to_st5(
                torch.einsum(
                    "ab,...bc,dc->...ad",
                    orthogonal,
                    matrix,
                    orthogonal,
                )
            )
        transformed[str(irrep)] = parity_factor * result
    return pack_irreps(parsed, transformed)


def _learned_model(
    *,
    input_irreps: str = "4x0e",
    output_irreps: str = FULL_IRREPS,
    width: int = 16,
    depth: int = 1,
) -> ELA:
    model = ELA(
        input_irreps=input_irreps,
        output_irreps=output_irreps,
        width=width,
        depth=depth,
        cutoff=10.0,
        num_rbf=8,
    ).double()
    with torch.no_grad():
        for layer in model.layers:
            layer.branch_fusion.router[-1].weight.normal_(mean=0.0, std=0.1)
            layer.branch_fusion.router[-1].bias.normal_(mean=0.0, std=0.1)
            layer.branch_fusion.balance_strength.normal_(mean=0.0, std=0.2)
    return model.eval()


def test_learned_ela_obeys_full_o3_and_translation() -> None:
    torch.manual_seed(41)
    model = _learned_model(width=32, depth=2)
    nodes = 7
    features = torch.randn(nodes, 4, dtype=torch.float64)
    positions = torch.randn(nodes, 3, dtype=torch.float64)
    edge_index = _complete_edges(nodes)
    orthogonal, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.linalg.det(orthogonal) > 0:
        orthogonal[:, 0] = -orthogonal[:, 0]
    translation = torch.tensor([0.7, -1.1, 0.3], dtype=torch.float64)

    with torch.inference_mode():
        reference = model.split_output(
            model(ELABatch(features, positions, edge_index=edge_index))[
                "node_irreps"
            ]
        )
        actual = model.split_output(
            model(
                ELABatch(
                    features,
                    positions @ orthogonal.T + translation,
                    edge_index=edge_index,
                )
            )["node_irreps"]
        )

    determinant = torch.linalg.det(orthogonal)
    torch.testing.assert_close(actual["0e"], reference["0e"], atol=5e-8, rtol=5e-8)
    torch.testing.assert_close(
        actual["0o"], determinant * reference["0o"], atol=5e-8, rtol=5e-8
    )
    torch.testing.assert_close(
        actual["1o"],
        _transform_vector(reference["1o"], orthogonal, axial=False),
        atol=6e-8,
        rtol=6e-8,
    )
    torch.testing.assert_close(
        actual["1e"],
        _transform_vector(reference["1e"], orthogonal, axial=True),
        atol=6e-8,
        rtol=6e-8,
    )
    torch.testing.assert_close(
        actual["2e"],
        _transform_tensor(reference["2e"], orthogonal, odd=False),
        atol=8e-8,
        rtol=8e-8,
    )
    torch.testing.assert_close(
        actual["2o"],
        _transform_tensor(reference["2o"], orthogonal, odd=True),
        atol=8e-8,
        rtol=8e-8,
    )


def test_ela_is_node_permutation_equivariant() -> None:
    torch.manual_seed(43)
    model = _learned_model(width=32, depth=2)
    nodes = 7
    features = torch.randn(nodes, 4, dtype=torch.float64)
    positions = torch.randn(nodes, 3, dtype=torch.float64)
    edge_index = _complete_edges(nodes)
    permutation = torch.randperm(nodes)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(nodes)

    with torch.inference_mode():
        reference = model(
            ELABatch(features, positions, edge_index=edge_index)
        )["node_irreps"]
        actual = model(
            ELABatch(
                features[permutation],
                positions[permutation],
                edge_index=inverse[edge_index],
            )
        )["node_irreps"]
    torch.testing.assert_close(
        actual,
        reference[permutation],
        atol=6e-8,
        rtol=6e-8,
    )


@pytest.mark.parametrize("determinant_sign", [1, -1], ids=["proper", "improper"])
def test_ela_obeys_generic_o3_with_non_scalar_inputs(
    determinant_sign: int,
) -> None:
    torch.manual_seed(45 + determinant_sign)
    model = _learned_model(input_irreps=FULL_IRREPS)
    nodes = 5
    features = torch.randn(
        nodes,
        model.config.input_layout.dim,
        dtype=torch.float64,
    )
    positions = torch.randn(nodes, 3, dtype=torch.float64)
    edge_index = _complete_edges(nodes)
    orthogonal, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if int(torch.linalg.det(orthogonal).sign().item()) != determinant_sign:
        orthogonal[:, 0].neg_()
    translation = torch.tensor([0.4, -0.7, 1.2], dtype=torch.float64)

    with torch.inference_mode():
        reference = model(
            ELABatch(features, positions, edge_index=edge_index)
        )["node_irreps"]
        actual = model(
            ELABatch(
                _transform_irreps(model.config.input_layout, features, orthogonal),
                positions @ orthogonal.T + translation,
                edge_index=edge_index,
            )
        )["node_irreps"]
    expected = _transform_irreps(
        model.config.output_layout,
        reference,
        orthogonal,
    )
    torch.testing.assert_close(actual, expected, atol=8e-8, rtol=8e-8)


def test_ela_is_invariant_to_sparse_edge_order() -> None:
    torch.manual_seed(49)
    model = _learned_model(input_irreps="3x0e + 1x1o")
    nodes = 6
    features = torch.randn(
        nodes,
        model.config.input_layout.dim,
        dtype=torch.float64,
    )
    positions = torch.randn(nodes, 3, dtype=torch.float64)
    edge_index = _complete_edges(nodes)
    permuted_edges = edge_index[:, torch.randperm(edge_index.shape[1])]

    with torch.inference_mode():
        reference = model(
            ELABatch(features, positions, edge_index=edge_index)
        )
        actual = model(
            ELABatch(features, positions, edge_index=permuted_edges)
        )
    torch.testing.assert_close(
        actual["node_irreps"], reference["node_irreps"], atol=2e-10, rtol=2e-10
    )
    torch.testing.assert_close(
        actual["graph_irreps"], reference["graph_irreps"], atol=2e-10, rtol=2e-10
    )


def test_ela_keeps_batched_graphs_isolated() -> None:
    torch.manual_seed(51)
    model = _learned_model(
        input_irreps="3x0e + 1x1o",
        output_irreps="1x0e + 1x1o",
    )
    nodes_per_graph = 4
    features = torch.randn(
        2 * nodes_per_graph,
        model.config.input_layout.dim,
        dtype=torch.float64,
    )
    positions = torch.randn(2 * nodes_per_graph, 3, dtype=torch.float64)
    edge_index = _batched_complete_edges(nodes_per_graph, 2)
    ptr = torch.tensor([0, nodes_per_graph, 2 * nodes_per_graph])

    with torch.inference_mode():
        reference = model(
            ELABatch(features, positions, ptr=ptr, edge_index=edge_index)
        )
        isolated = model(
            ELABatch(
                features[:nodes_per_graph],
                positions[:nodes_per_graph],
                edge_index=_complete_edges(nodes_per_graph),
            )
        )
        changed_features = features.clone()
        changed_positions = positions.clone()
        changed_features[nodes_per_graph:] = (
            7.0 * changed_features[nodes_per_graph:] + 3.0
        )
        changed_positions[nodes_per_graph:] = (
            2.0 * changed_positions[nodes_per_graph:]
            + torch.tensor([11.0, -5.0, 2.0], dtype=torch.float64)
        )
        changed = model(
            ELABatch(
                changed_features,
                changed_positions,
                ptr=ptr,
                edge_index=edge_index,
            )
        )

    torch.testing.assert_close(
        reference["node_irreps"][:nodes_per_graph],
        isolated["node_irreps"],
        atol=2e-10,
        rtol=2e-10,
    )
    torch.testing.assert_close(
        reference["graph_irreps"][0],
        isolated["graph_irreps"][0],
        atol=2e-10,
        rtol=2e-10,
    )
    torch.testing.assert_close(
        changed["node_irreps"][:nodes_per_graph],
        reference["node_irreps"][:nodes_per_graph],
        atol=2e-10,
        rtol=2e-10,
    )
