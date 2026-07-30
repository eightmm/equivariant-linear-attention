from __future__ import annotations

import torch
from torch import nn

from equivariant_attention import prepare_3d_graph
from equivariant_attention.equivariant_linear_attention import (
    EquivariantLinearAttention,
    EquivariantLinearAttentionConfig,
    _EquivariantDropout,
    _EquivariantRMSNorm,
    _GraphDropPath,
    _GroupedRMSLinear,
    _NormGatedIrrepActivation,
)
from equivariant_attention.parity_se3 import _ParityState


def _state(nodes: int = 5, scalar_width: int = 12, heads: int = 3) -> _ParityState:
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
        transformed = torch.einsum("...c,dc->...d", value, orthogonal)
        return determinant * transformed if axial else transformed

    def tensor(value: torch.Tensor, *, odd: bool) -> torch.Tensor:
        matrix = _st_to_matrix(value)
        transformed = torch.einsum(
            "ab,...bc,dc->...ad",
            orthogonal,
            matrix,
            orthogonal,
        )
        compact = _matrix_to_st(transformed)
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


def test_config_identifies_equivariant_linear_attention() -> None:
    config = EquivariantLinearAttentionConfig(
        input_irreps="4x0e + 1x1o",
        output_irreps="1x0e",
        hidden_dim=16,
        num_heads=4,
        local_rank=3,
        residual_dropout=0.1,
        drop_path_rate=0.2,
    )
    contract = config.canonical_contract()
    assert contract["architecture"] == "equivariant_linear_attention"
    assert contract["attention_qk_normalization"] == (
        "headwise_rms_before_positive_feature"
    )
    assert contract["branch_normalization"] == (
        "all_sector_equivariant_rms_pre_norm"
    )


def test_equivariant_rms_norm_commutes_with_reflection() -> None:
    torch.manual_seed(3)
    state = _state()
    norm = _EquivariantRMSNorm(scalar_width=12, num_heads=3, eps=1e-8).double()
    reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))
    _assert_state_close(
        norm(_transform(state, reflection)),
        _transform(norm(state), reflection),
    )


def test_norm_gated_activation_is_identity_at_initialization() -> None:
    torch.manual_seed(5)
    delta = _state()
    reference = _state()
    activation = _NormGatedIrrepActivation(
        scalar_width=12,
        num_heads=3,
        eps=1e-8,
    ).double()
    _assert_state_close(activation(delta, reference), delta)


def test_learned_norm_gate_remains_equivariant() -> None:
    torch.manual_seed(6)
    delta = _state()
    reference = _state()
    activation = _NormGatedIrrepActivation(
        scalar_width=12,
        num_heads=3,
        eps=1e-8,
    ).double()
    with torch.no_grad():
        activation.projection[-1].weight.normal_(mean=0.0, std=0.1)
        activation.projection[-1].bias.normal_(mean=0.0, std=0.1)
    reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))
    actual = activation(
        _transform(delta, reflection),
        _transform(reference, reflection),
    )
    expected = _transform(activation(delta, reference), reflection)
    _assert_state_close(actual, expected)


def test_grouped_qk_projection_has_unit_rms_per_group() -> None:
    linear = nn.Linear(4, 4, bias=False).double()
    with torch.no_grad():
        linear.weight.copy_(torch.eye(4, dtype=torch.float64))
    projection = _GroupedRMSLinear(
        linear,
        groups=2,
        width=2,
        eps=1e-12,
    ).double()
    value = torch.tensor([[3.0, 4.0, 5.0, 12.0]], dtype=torch.float64)
    output = projection(value).reshape(1, 2, 2)
    rms = output.square().mean(dim=-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-10, rtol=1e-10)


def test_equivariant_dropout_never_splits_vector_components() -> None:
    state = _state(nodes=8)
    dropout = _EquivariantDropout(0.5).train()
    torch.manual_seed(7)
    output = dropout(state)
    original = state.polar_vector
    result = output.polar_vector
    original_norm = torch.linalg.vector_norm(original, dim=-1)
    result_norm = torch.linalg.vector_norm(result, dim=-1)
    active = result_norm > 0
    scale = result_norm[active] / original_norm[active]
    torch.testing.assert_close(scale, torch.full_like(scale, 2.0))
    direction_error = torch.linalg.vector_norm(
        result[active] / result_norm[active, None]
        - original[active] / original_norm[active, None],
        dim=-1,
    )
    torch.testing.assert_close(direction_error, torch.zeros_like(direction_error))


def test_drop_path_uses_one_mask_per_graph() -> None:
    state = _state(nodes=6)
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    drop_path = _GraphDropPath(0.5).train()
    torch.manual_seed(11)
    output = drop_path(state, batch=batch, num_graphs=2)
    ratio = output.even_scalar / state.even_scalar
    for graph in (0, 1):
        selected = ratio[batch == graph]
        torch.testing.assert_close(selected, selected[:1].expand_as(selected))


def test_refined_stack_forward_backward_and_condition() -> None:
    torch.manual_seed(13)
    config = EquivariantLinearAttentionConfig(
        input_irreps="4x0e + 1x1o",
        output_irreps="1x0e + 1x1o",
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        local_rank=3,
        local_cutoff=10.0,
        num_rbf=8,
        condition_dim=5,
        residual_dropout=0.0,
        drop_path_rate=0.0,
    )
    model = EquivariantLinearAttention(config).double()
    nodes = 6
    scalar = torch.randn(nodes, 4, 1, dtype=torch.float64)
    vector = torch.randn(nodes, 1, 3, dtype=torch.float64)
    node_irreps = torch.cat([scalar.flatten(1), vector.flatten(1)], dim=-1)
    node_irreps.requires_grad_(True)
    positions = torch.randn(nodes, 3, dtype=torch.float64, requires_grad=True)
    graph = prepare_3d_graph(
        torch.zeros(nodes, dtype=torch.long),
        _complete_edges(nodes),
    )
    condition = torch.randn(1, 5, dtype=torch.float64)
    output = model(node_irreps, positions, graph, condition=condition)
    loss = output["node_irreps"].square().mean()
    loss.backward()
    assert torch.isfinite(output["node_irreps"]).all()
    assert node_irreps.grad is not None and torch.isfinite(node_irreps.grad).all()
    assert positions.grad is not None and torch.isfinite(positions.grad).all()


def test_refined_stack_obeys_rotation_and_translation() -> None:
    torch.manual_seed(17)
    config = EquivariantLinearAttentionConfig(
        input_irreps="4x0e",
        output_irreps="1x0e + 1x1o",
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        local_rank=3,
        local_cutoff=10.0,
        num_rbf=8,
    )
    model = EquivariantLinearAttention(config).double().eval()
    nodes = 6
    features = torch.randn(nodes, 4, dtype=torch.float64)
    positions = torch.randn(nodes, 3, dtype=torch.float64)
    graph = prepare_3d_graph(
        torch.zeros(nodes, dtype=torch.long),
        _complete_edges(nodes),
    )
    orthogonal, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.linalg.det(orthogonal) < 0:
        orthogonal[:, 0] = -orthogonal[:, 0]
    translation = torch.tensor([1.2, -0.3, 0.7], dtype=torch.float64)

    reference = model(features, positions, graph)["node_irreps"]
    transformed = model(
        features,
        positions @ orthogonal.T + translation,
        graph,
    )["node_irreps"]
    reference_blocks = model.split_output(reference)
    transformed_blocks = model.split_output(transformed)
    torch.testing.assert_close(
        transformed_blocks["0e"],
        reference_blocks["0e"],
        atol=3e-8,
        rtol=3e-8,
    )
    expected_vector = torch.einsum(
        "...c,dc->...d",
        reference_blocks["1o"],
        orthogonal,
    )
    torch.testing.assert_close(
        transformed_blocks["1o"],
        expected_vector,
        atol=3e-8,
        rtol=3e-8,
    )
