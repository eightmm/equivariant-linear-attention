from __future__ import annotations

import pytest
import torch

from equivariant_linear_attention import ELA, ELAGraph
from equivariant_linear_attention.irreps import pack_irreps, split_irreps
from equivariant_linear_attention.irreps import (
    IrrepLayout,
    matrix_to_st5,
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
    ).double()
    with torch.no_grad():
        for layer in model.layers:
            for name in (
                "local_scalar_out",
                "local_odd_out",
                "local_polar_out",
                "local_axial_out",
                "local_even_tensor_out",
                "local_odd_tensor_out",
                "local_chiral_axial_out",
                "local_chiral_tensor_out",
                "local_mass_out",
            ):
                getattr(layer, name).weight.normal_(mean=0.0, std=0.03)
            layer.local_radial_value.weight.normal_(mean=0.0, std=0.02)
            layer.raw_odd_alignment.normal_(mean=-1.0, std=0.1)
            layer.raw_global_radial_alignment.normal_(mean=-1.0, std=0.1)
            layer.second_moment_chiral_mix.fill_(0.1)
    _wake_zero_initialized(model)
    return model.eval()


def _wake_zero_initialized(model: ELA, *, std: float = 0.05) -> int:
    """Perturb every still-zero parameter so no lane sits out the test.

    Several lanes ship zero-initialized so an untrained model starts at the
    identity -- the tensor closure and the l=2 local score among them. Left
    alone they emit exactly zero, so an equivariance test built on a fresh
    model never transports anything through them and would pass even if the
    lane were deleted. Waking them makes the test cover the whole operator.
    """

    woken = 0
    with torch.no_grad():
        for parameter in model.parameters():
            if not torch.count_nonzero(parameter):
                parameter.normal_(mean=0.0, std=std)
                woken += 1
    return woken


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
            model(ELAGraph(features, positions, edge_index=edge_index)).x
        )
        actual = model.split_output(
            model(
                ELAGraph(
                    features,
                    positions @ orthogonal.T + translation,
                    edge_index=edge_index,
                )
            ).x
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
            ELAGraph(features, positions, edge_index=edge_index)
        ).x
        actual = model(
            ELAGraph(
                features[permutation],
                positions[permutation],
                edge_index=inverse[edge_index],
            )
        ).x
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
            ELAGraph(features, positions, edge_index=edge_index)
        ).x
        actual = model(
            ELAGraph(
                _transform_irreps(model.config.input_layout, features, orthogonal),
                positions @ orthogonal.T + translation,
                edge_index=edge_index,
            )
        ).x
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
            ELAGraph(features, positions, edge_index=edge_index)
        )
        actual = model(
            ELAGraph(features, positions, edge_index=permuted_edges)
        )
    torch.testing.assert_close(actual.x, reference.x, atol=2e-10, rtol=2e-10)
    assert actual.graph_x is not None and reference.graph_x is not None
    torch.testing.assert_close(
        actual.graph_x, reference.graph_x, atol=2e-10, rtol=2e-10
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
    batch = torch.repeat_interleave(torch.arange(2), nodes_per_graph)

    with torch.inference_mode():
        reference = model(
            ELAGraph(features, positions, batch=batch, edge_index=edge_index)
        )
        isolated = model(
            ELAGraph(
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
            ELAGraph(
                changed_features,
                changed_positions,
                batch=batch,
                edge_index=edge_index,
            )
        )

    torch.testing.assert_close(
        reference.x[:nodes_per_graph],
        isolated.x,
        atol=2e-10,
        rtol=2e-10,
    )
    assert reference.graph_x is not None and isolated.graph_x is not None
    torch.testing.assert_close(
        reference.graph_x[0],
        isolated.graph_x[0],
        atol=2e-10,
        rtol=2e-10,
    )
    torch.testing.assert_close(
        changed.x[:nodes_per_graph],
        reference.x[:nodes_per_graph],
        atol=2e-10,
        rtol=2e-10,
    )


def test_learned_model_leaves_no_lane_inert() -> None:
    """Guard the guard: the learned fixture must exercise every lane.

    The tensor closure and the l=2 local score are zero-initialized, so a
    fixture that only randomizes the local output projections transports
    nothing through them and the O(3) tests below would still pass with those
    modules removed.
    """

    bare = ELA(
        input_irreps="4x0e",
        output_irreps=FULL_IRREPS,
        width=32,
        depth=2,
        cutoff=10.0,
    ).double()
    assert _wake_zero_initialized(bare) > 0, "expected zero-initialized lanes to exist"

    model = _learned_model(width=32, depth=2)
    inert = [
        name
        for name, parameter in model.named_parameters()
        if not torch.count_nonzero(parameter)
    ]
    assert not inert, f"lanes still inert in the learned fixture: {inert}"
