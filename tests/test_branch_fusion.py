from __future__ import annotations

import torch

from equivariant_attention.branch_fusion import RMSAwareBranchFusion
from equivariant_attention.parity_se3 import _ParityState


def _state(nodes: int = 5, width: int = 12, heads: int = 3) -> _ParityState:
    return _ParityState(
        even_scalar=torch.randn(nodes, width, dtype=torch.float64),
        odd_scalar=torch.randn(nodes, heads, dtype=torch.float64),
        polar_vector=torch.randn(nodes, heads, 3, dtype=torch.float64),
        axial_vector=torch.randn(nodes, heads, 3, dtype=torch.float64),
        even_tensor=torch.randn(nodes, heads, 5, dtype=torch.float64),
        odd_tensor=torch.randn(nodes, heads, 5, dtype=torch.float64),
    )


def _messages(nodes: int = 5, heads: int = 3, head_dim: int = 4):
    global_message = (
        torch.randn(nodes, heads, head_dim, dtype=torch.float64),
        torch.randn(nodes, heads, dtype=torch.float64),
        torch.randn(nodes, heads, 3, dtype=torch.float64),
        torch.randn(nodes, heads, 3, dtype=torch.float64),
        torch.randn(nodes, heads, 5, dtype=torch.float64),
        torch.randn(nodes, heads, 5, dtype=torch.float64),
    )
    local_message = (
        torch.randn(nodes, heads, head_dim, dtype=torch.float64),
        torch.randn(nodes, heads, dtype=torch.float64),
        torch.randn(nodes, heads, 3, dtype=torch.float64),
        torch.randn(nodes, heads, 3, dtype=torch.float64),
        torch.randn(nodes, heads, 5, dtype=torch.float64),
        torch.randn(nodes, heads, 5, dtype=torch.float64),
        torch.randn(nodes, heads, dtype=torch.float64),
    )
    return global_message, local_message


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


def _transform_value(
    value: torch.Tensor,
    index: int,
    orthogonal: torch.Tensor,
) -> torch.Tensor:
    determinant = torch.linalg.det(orthogonal)
    if index == 0:
        return value
    if index == 1:
        return determinant * value
    if index in {2, 3}:
        transformed = torch.einsum("...c,dc->...d", value, orthogonal)
        return determinant * transformed if index == 3 else transformed
    matrix = _st_to_matrix(value)
    transformed = torch.einsum(
        "ab,...bc,dc->...ad",
        orthogonal,
        matrix,
        orthogonal,
    )
    compact = _matrix_to_st(transformed)
    return determinant * compact if index == 5 else compact


def _transform_state(state: _ParityState, orthogonal: torch.Tensor) -> _ParityState:
    values = [
        state.even_scalar,
        state.odd_scalar,
        state.polar_vector,
        state.axial_vector,
        state.even_tensor,
        state.odd_tensor,
    ]
    transformed = [
        _transform_value(value, index, orthogonal)
        for index, value in enumerate(values)
    ]
    return _ParityState(*transformed)


def test_branch_fusion_is_exact_old_sum_at_initialization() -> None:
    torch.manual_seed(3)
    state = _state()
    global_message, local_message = _messages()
    fusion = RMSAwareBranchFusion(scalar_width=12, eps=1e-8).double()

    routed_global, routed_local = fusion(state, global_message, local_message)
    for index in range(6):
        torch.testing.assert_close(
            routed_global[index],
            global_message[index] + local_message[index],
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            routed_local[index],
            torch.zeros_like(local_message[index]),
            atol=0.0,
            rtol=0.0,
        )
    torch.testing.assert_close(
        routed_local[6],
        local_message[6],
        atol=0.0,
        rtol=0.0,
    )


def test_branch_weights_are_positive_and_sum_to_two() -> None:
    torch.manual_seed(5)
    fusion = RMSAwareBranchFusion(scalar_width=12).double()
    state = _state()
    global_message, local_message = _messages()
    weights, _, _ = fusion.routing_weights(state, global_message, local_message)

    assert torch.all(weights > 0)
    torch.testing.assert_close(
        weights.sum(dim=-1),
        torch.full_like(weights[..., 0], 2.0),
    )


def test_learned_branch_fusion_commutes_with_reflection() -> None:
    torch.manual_seed(7)
    fusion = RMSAwareBranchFusion(scalar_width=12, eps=1e-8).double()
    with torch.no_grad():
        fusion.router[-1].weight.normal_(mean=0.0, std=0.1)
        fusion.router[-1].bias.normal_(mean=0.0, std=0.1)
        fusion.balance_strength.copy_(
            torch.tensor([0.2, -0.1, 0.3, 0.4, -0.2, 0.15], dtype=torch.float64)
        )
    state = _state()
    global_message, local_message = _messages()
    reflection = torch.diag(
        torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)
    )

    reference_global, reference_local = fusion(
        state,
        global_message,
        local_message,
    )
    transformed_global = tuple(
        _transform_value(value, index, reflection)
        for index, value in enumerate(global_message)
    )
    transformed_local = tuple(
        [
            *(
                _transform_value(value, index, reflection)
                for index, value in enumerate(local_message[:6])
            ),
            torch.linalg.det(reflection) * local_message[6],
        ]
    )
    actual_global, actual_local = fusion(
        _transform_state(state, reflection),
        transformed_global,
        transformed_local,
    )

    for index in range(6):
        torch.testing.assert_close(
            actual_global[index],
            _transform_value(reference_global[index], index, reflection),
            atol=3e-9,
            rtol=3e-9,
        )
    torch.testing.assert_close(
        actual_local[6],
        torch.linalg.det(reflection) * reference_local[6],
        atol=3e-9,
        rtol=3e-9,
    )


def test_learned_branch_fusion_commutes_with_generic_o3_actions() -> None:
    torch.manual_seed(9)
    fusion = RMSAwareBranchFusion(scalar_width=12, eps=1e-8).double()
    with torch.no_grad():
        fusion.router[-1].weight.normal_(mean=0.0, std=0.1)
        fusion.router[-1].bias.normal_(mean=0.0, std=0.1)
        fusion.balance_strength.copy_(
            torch.tensor([0.2, -0.1, 0.3, 0.4, -0.2, 0.15], dtype=torch.float64)
        )
    state = _state()
    global_message, local_message = _messages()
    proper, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.linalg.det(proper) < 0:
        proper[:, 0].neg_()
    improper = proper @ torch.diag(
        torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)
    )

    reference_global, reference_local = fusion(
        state,
        global_message,
        local_message,
    )
    for orthogonal in (proper, improper):
        transformed_global = tuple(
            _transform_value(value, index, orthogonal)
            for index, value in enumerate(global_message)
        )
        transformed_local = (
            *(
                _transform_value(value, index, orthogonal)
                for index, value in enumerate(local_message[:6])
            ),
            torch.linalg.det(orthogonal) * local_message[6],
        )
        actual_global, actual_local = fusion(
            _transform_state(state, orthogonal),
            transformed_global,
            transformed_local,
        )

        for index in range(6):
            torch.testing.assert_close(
                actual_global[index],
                _transform_value(reference_global[index], index, orthogonal),
                atol=3e-9,
                rtol=3e-9,
            )
        torch.testing.assert_close(
            actual_local[6],
            torch.linalg.det(orthogonal) * reference_local[6],
            atol=3e-9,
            rtol=3e-9,
        )


def test_router_and_balance_parameters_receive_gradients() -> None:
    torch.manual_seed(11)
    fusion = RMSAwareBranchFusion(scalar_width=12).double()
    with torch.no_grad():
        fusion.balance_strength.fill_(0.2)
    state = _state()
    global_message, local_message = _messages()
    routed_global, routed_local = fusion(state, global_message, local_message)
    loss = sum(value.square().mean() for value in routed_global)
    loss = loss + routed_local[6].square().mean()
    loss.backward()

    assert fusion.balance_strength.grad is not None
    assert torch.isfinite(fusion.balance_strength.grad).all()
    output = fusion.router[-1]
    assert output.weight.grad is not None
    assert torch.isfinite(output.weight.grad).all()


def test_coefficient_fusion_matches_materialized_balance_formula() -> None:
    torch.manual_seed(12)
    fusion = RMSAwareBranchFusion(scalar_width=12, eps=1e-8).double()
    with torch.no_grad():
        for parameter in fusion.router.parameters():
            parameter.normal_(mean=0.0, std=0.1)
        fusion.balance_strength.copy_(
            torch.tensor([0.2, -0.1, 0.3, 0.4, -0.2, 0.15], dtype=torch.float64)
        )
    state = _state()
    global_message, local_message = _messages()

    actual, _ = fusion(state, global_message, local_message)
    weights, global_rms, local_rms = fusion.routing_weights(
        state,
        global_message,
        local_message,
    )
    for index, (global_value, local_value) in enumerate(
        zip(global_message, local_message[:6], strict=True)
    ):
        global_weight = fusion._broadcast(weights[:, index, 0], global_value)
        local_weight = fusion._broadcast(weights[:, index, 1], local_value)
        weighted = global_weight * global_value + local_weight * local_value
        reference_scale = torch.sqrt(
            0.5
            * (
                global_rms[index].square()
                + local_rms[index].square()
            )
        )
        weight_norm = torch.sqrt(
            0.5 * (global_weight.square() + local_weight.square())
            + fusion.eps
        )
        balanced = (
            reference_scale
            * (
                global_weight * global_value / global_rms[index]
                + local_weight * local_value / local_rms[index]
            )
            / weight_norm
        )
        strength = torch.tanh(fusion.balance_strength[index])
        expected = weighted + strength * (balanced - weighted)
        torch.testing.assert_close(actual[index], expected, atol=1e-12, rtol=1e-12)


def test_zero_router_matches_native_promotion_for_mixed_precision_branches() -> None:
    torch.manual_seed(13)
    fusion = RMSAwareBranchFusion(scalar_width=12).double()
    state = _state()
    global_message, local_message = _messages()
    mixed_global = tuple(value.to(torch.bfloat16) for value in global_message)
    mixed_local = (
        *(value.to(torch.float32) for value in local_message[:6]),
        local_message[6].to(torch.float32),
    )

    routed_global, routed_local = fusion(
        state,
        mixed_global,
        mixed_local,
    )

    for index in range(6):
        expected = mixed_global[index] + mixed_local[index]
        assert routed_global[index].dtype == expected.dtype
        torch.testing.assert_close(
            routed_global[index],
            expected,
            atol=0.0,
            rtol=0.0,
        )
    torch.testing.assert_close(
        routed_local[6],
        mixed_local[6],
        atol=0.0,
        rtol=0.0,
    )
