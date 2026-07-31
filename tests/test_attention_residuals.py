from __future__ import annotations

import pytest
import torch

from equivariant_attention import (
    EquivariantAttentionResidualConfig,
    EquivariantAttentionResidualCore,
    EquivariantAttentionResiduals,
    EquivariantBlockAttentionResidual,
    prepare_3d_graph,
)
from equivariant_attention.parity_se3 import _ParityState


def _state(
    *,
    nodes: int = 5,
    scalar_width: int = 12,
    heads: int = 3,
) -> _ParityState:
    return _ParityState(
        even_scalar=torch.randn(nodes, scalar_width, dtype=torch.float64),
        odd_scalar=torch.randn(nodes, heads, dtype=torch.float64),
        polar_vector=torch.randn(nodes, heads, 3, dtype=torch.float64),
        axial_vector=torch.randn(nodes, heads, 3, dtype=torch.float64),
        even_tensor=torch.randn(nodes, heads, 5, dtype=torch.float64),
        odd_tensor=torch.randn(nodes, heads, 5, dtype=torch.float64),
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


def _transform(state: _ParityState, orthogonal: torch.Tensor) -> _ParityState:
    determinant = torch.linalg.det(orthogonal)

    def vector(value: torch.Tensor, *, axial: bool) -> torch.Tensor:
        result = torch.einsum("...c,dc->...d", value, orthogonal)
        return determinant * result if axial else result

    def tensor(value: torch.Tensor, *, odd: bool) -> torch.Tensor:
        matrix = _st_to_matrix(value)
        result = torch.einsum(
            "ab,...bc,dc->...ad",
            orthogonal,
            matrix,
            orthogonal,
        )
        compact = _matrix_to_st(result)
        return determinant * compact if odd else compact

    return _ParityState(
        even_scalar=state.even_scalar,
        odd_scalar=determinant * state.odd_scalar,
        polar_vector=vector(state.polar_vector, axial=False),
        axial_vector=vector(state.axial_vector, axial=True),
        even_tensor=tensor(state.even_tensor, odd=False),
        odd_tensor=tensor(state.odd_tensor, odd=True),
    )


def _assert_state_close(left: _ParityState, right: _ParityState) -> None:
    for name in (
        "even_scalar",
        "odd_scalar",
        "polar_vector",
        "axial_vector",
        "even_tensor",
        "odd_tensor",
    ):
        torch.testing.assert_close(
            getattr(left, name),
            getattr(right, name),
            atol=2e-9,
            rtol=2e-9,
        )


def _complete_edges(nodes: int) -> torch.Tensor:
    receiver = torch.arange(nodes).repeat_interleave(nodes)
    sender = torch.arange(nodes).repeat(nodes)
    return torch.stack([receiver, sender])


def test_zero_initialized_depth_query_is_uniform() -> None:
    torch.manual_seed(3)
    router = EquivariantBlockAttentionResidual(
        scalar_width=12,
        num_heads=3,
        eps=1e-8,
    ).double()
    sources = (_state(), _state(), _state())
    weights = router.routing_weights(sources)
    torch.testing.assert_close(
        weights,
        torch.full_like(weights, 1.0 / 3.0),
        atol=0.0,
        rtol=0.0,
    )


def test_depth_router_commutes_with_reflection() -> None:
    torch.manual_seed(5)
    router = EquivariantBlockAttentionResidual(
        scalar_width=12,
        num_heads=3,
        eps=1e-8,
    ).double()
    with torch.no_grad():
        router.pseudo_query.normal_(mean=0.0, std=0.2)
    sources = (_state(), _state(), _state())
    reflection = torch.diag(
        torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)
    )
    actual = router(tuple(_transform(source, reflection) for source in sources))
    expected = _transform(router(sources), reflection)
    _assert_state_close(actual, expected)


def test_depth_weights_are_shared_by_every_irrep_sector() -> None:
    router = EquivariantBlockAttentionResidual(
        scalar_width=2,
        num_heads=1,
        eps=1e-8,
    ).double()
    first = _ParityState(
        even_scalar=torch.ones(1, 2, dtype=torch.float64),
        odd_scalar=torch.ones(1, 1, dtype=torch.float64),
        polar_vector=torch.ones(1, 1, 3, dtype=torch.float64),
        axial_vector=torch.ones(1, 1, 3, dtype=torch.float64),
        even_tensor=torch.ones(1, 1, 5, dtype=torch.float64),
        odd_tensor=torch.ones(1, 1, 5, dtype=torch.float64),
    )
    second = _ParityState(
        even_scalar=3.0 * first.even_scalar,
        odd_scalar=3.0 * first.odd_scalar,
        polar_vector=3.0 * first.polar_vector,
        axial_vector=3.0 * first.axial_vector,
        even_tensor=3.0 * first.even_tensor,
        odd_tensor=3.0 * first.odd_tensor,
    )
    mixed = router((first, second))
    for name in (
        "even_scalar",
        "odd_scalar",
        "polar_vector",
        "axial_vector",
        "even_tensor",
        "odd_tensor",
    ):
        torch.testing.assert_close(
            getattr(mixed, name),
            2.0 * getattr(first, name),
        )


def test_attention_residual_config_rejects_too_many_blocks() -> None:
    with pytest.raises(ValueError, match="must not exceed num_layers"):
        EquivariantAttentionResidualConfig(
            input_irreps="4x0e",
            num_layers=2,
            attention_residual_blocks=3,
        )


def test_nondivisible_depth_uses_exact_requested_block_count() -> None:
    starts = EquivariantAttentionResidualCore._block_start_indices(6, 4)
    assert starts == (2, 4, 5)
    assert len(starts) + 1 == 4


def test_final_state_retains_routed_embedding_when_residuals_are_zero() -> None:
    torch.manual_seed(19)
    config = EquivariantAttentionResidualConfig(
        input_irreps="4x0e",
        output_irreps="1x0e",
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        local_rank=3,
        local_cutoff=10.0,
        num_rbf=8,
        attention_residual_blocks=1,
    )
    model = EquivariantAttentionResiduals(config).double().eval()
    layer = model.layers[0]
    with torch.no_grad():
        for name, parameter in layer.named_parameters():
            if name in {
                "scalar_scale",
                "odd_scale",
                "polar_scale",
                "axial_scale",
                "even_tensor_scale",
                "odd_tensor_scale",
                "closure_scalar_scale",
                "closure_odd_scale",
                "closure_polar_scale",
                "closure_axial_scale",
                "closure_even_tensor_scale",
                "closure_odd_tensor_scale",
                "ffn_scalar_scale",
                "ffn_odd_scale",
                "ffn_polar_scale",
                "ffn_axial_scale",
                "ffn_even_tensor_scale",
                "ffn_odd_tensor_scale",
            }:
                parameter.zero_()
    nodes = 5
    features = torch.randn(nodes, 4, dtype=torch.float64)
    positions = torch.randn(nodes, 3, dtype=torch.float64)
    graph = prepare_3d_graph(
        torch.zeros(nodes, dtype=torch.long),
        _complete_edges(nodes),
    )
    embedding, _ = model.embed_input(features, positions, graph)
    final, _, _ = model.forward_features(features, positions, graph)
    for name in (
        "even_scalar",
        "odd_scalar",
        "polar_vector",
        "axial_vector",
        "even_tensor",
        "odd_tensor",
    ):
        torch.testing.assert_close(
            getattr(final, name),
            0.5 * getattr(embedding, name),
            atol=2e-10,
            rtol=2e-10,
        )


def test_attention_residual_stack_forward_backward() -> None:
    torch.manual_seed(7)
    config = EquivariantAttentionResidualConfig(
        input_irreps="4x0e",
        output_irreps="1x0e + 1x1o",
        hidden_dim=16,
        num_layers=4,
        num_heads=4,
        local_rank=3,
        local_cutoff=10.0,
        num_rbf=8,
        attention_residual_blocks=2,
        condition_dim=5,
    )
    model = EquivariantAttentionResiduals(config).double()
    nodes = 6
    features = torch.randn(nodes, 4, dtype=torch.float64, requires_grad=True)
    positions = torch.randn(nodes, 3, dtype=torch.float64, requires_grad=True)
    graph = prepare_3d_graph(
        torch.zeros(nodes, dtype=torch.long),
        _complete_edges(nodes),
    )
    condition = torch.randn(1, 5, dtype=torch.float64)
    output = model(features, positions, graph, condition=condition)
    loss = output["node_irreps"].square().mean()
    loss.backward()

    assert torch.isfinite(output["node_irreps"]).all()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert positions.grad is not None and torch.isfinite(positions.grad).all()
    query = model.layers[1].attention_depth_router.pseudo_query
    assert query.grad is not None and torch.isfinite(query.grad).all()


def test_coordinate_refinement_with_attnres_is_se3_equivariant() -> None:
    torch.manual_seed(11)
    config = EquivariantAttentionResidualConfig(
        input_irreps="4x0e",
        output_irreps="1x0e",
        hidden_dim=16,
        num_layers=3,
        num_heads=4,
        local_rank=3,
        local_cutoff=10.0,
        num_rbf=8,
        attention_residual_blocks=2,
        coordinate_updates=True,
        max_coordinate_step=0.1,
    )
    model = EquivariantAttentionResiduals(config).double().eval()
    with torch.no_grad():
        for layer in model.layers:
            layer.coordinate_vector.weight.normal_(mean=0.0, std=0.2)
    nodes = 6
    features = torch.randn(nodes, 4, dtype=torch.float64)
    positions = torch.randn(nodes, 3, dtype=torch.float64)
    graph = prepare_3d_graph(
        torch.zeros(nodes, dtype=torch.long),
        _complete_edges(nodes),
    )
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.linalg.det(rotation) < 0:
        rotation[:, 0] = -rotation[:, 0]
    translation = torch.tensor([0.7, -1.1, 0.2], dtype=torch.float64)

    reference = model(features, positions, graph)
    transformed = model(
        features,
        positions @ rotation.T + translation,
        graph,
    )
    torch.testing.assert_close(
        transformed["positions"],
        reference["positions"] @ rotation.T + translation,
        atol=4e-8,
        rtol=4e-8,
    )
