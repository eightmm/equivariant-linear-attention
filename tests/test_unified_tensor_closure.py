from __future__ import annotations

import torch

from equivariant_attention.multipole_ops import (
    NodeMultipoles,
    _LowOrderTensorClosure,
    _ParitySectorNorm,
    _st_commutator_vector,
    _st_jordan_product,
)
from equivariant_attention.parity_se3 import _ParityState


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
    sign = torch.linalg.det(orthogonal) if axial else value.new_tensor(1.0)
    return sign * torch.einsum("...c,dc->...d", value, orthogonal)


def _transform_tensor(
    value: torch.Tensor,
    orthogonal: torch.Tensor,
    *,
    odd: bool,
) -> torch.Tensor:
    matrix = _st_to_matrix(value)
    transformed = torch.einsum(
        "ab,...bc,dc->...ad",
        orthogonal,
        matrix,
        orthogonal,
    )
    if odd:
        transformed = torch.linalg.det(orthogonal) * transformed
    return _matrix_to_st(transformed)


def _transform_state(
    state: _ParityState,
    orthogonal: torch.Tensor,
) -> _ParityState:
    determinant = torch.linalg.det(orthogonal)
    return _ParityState(
        even_scalar=state.even_scalar,
        odd_scalar=determinant * state.odd_scalar,
        polar_vector=_transform_vector(
            state.polar_vector,
            orthogonal,
            axial=False,
        ),
        axial_vector=_transform_vector(
            state.axial_vector,
            orthogonal,
            axial=True,
        ),
        even_tensor=_transform_tensor(
            state.even_tensor,
            orthogonal,
            odd=False,
        ),
        odd_tensor=_transform_tensor(
            state.odd_tensor,
            orthogonal,
            odd=True,
        ),
    )


def _transform_multipoles(
    value: NodeMultipoles,
    orthogonal: torch.Tensor,
) -> NodeMultipoles:
    determinant = torch.linalg.det(orthogonal)
    return NodeMultipoles(
        mass=value.mass,
        mass_square=value.mass_square,
        polar=_transform_vector(value.polar, orthogonal, axial=False),
        even_tensor=_transform_tensor(value.even_tensor, orthogonal, odd=False),
        axial=_transform_vector(value.axial, orthogonal, axial=True),
        odd_scalar=determinant * value.odd_scalar,
        odd_tensor=_transform_tensor(value.odd_tensor, orthogonal, odd=True),
    )


def _random_state(
    *,
    nodes: int,
    scalar_width: int,
    heads: int,
) -> _ParityState:
    return _ParityState(
        even_scalar=torch.randn(nodes, scalar_width, dtype=torch.float64),
        odd_scalar=torch.randn(nodes, heads, dtype=torch.float64),
        polar_vector=torch.randn(nodes, heads, 3, dtype=torch.float64),
        axial_vector=torch.randn(nodes, heads, 3, dtype=torch.float64),
        even_tensor=torch.randn(nodes, heads, 5, dtype=torch.float64),
        odd_tensor=torch.randn(nodes, heads, 5, dtype=torch.float64),
    )


def _random_multipoles(*, nodes: int, rank: int) -> NodeMultipoles:
    return NodeMultipoles(
        mass=torch.rand(nodes, rank, dtype=torch.float64),
        mass_square=torch.rand(nodes, rank, dtype=torch.float64),
        polar=torch.randn(nodes, rank, 3, dtype=torch.float64),
        even_tensor=torch.randn(nodes, rank, 5, dtype=torch.float64),
        axial=torch.randn(nodes, rank, 3, dtype=torch.float64),
        odd_scalar=torch.randn(nodes, rank, dtype=torch.float64),
        odd_tensor=torch.randn(nodes, rank, 5, dtype=torch.float64),
    )


def test_rank_two_products_obey_expected_parities() -> None:
    torch.manual_seed(101)
    even_left = torch.randn(4, 3, 5, dtype=torch.float64)
    even_right = torch.randn(4, 3, 5, dtype=torch.float64)
    odd_right = torch.randn(4, 3, 5, dtype=torch.float64)
    orthogonal, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.linalg.det(orthogonal) > 0:
        orthogonal[:, 0] = -orthogonal[:, 0]

    even_even_vector = _st_commutator_vector(even_left, even_right)
    even_odd_vector = _st_commutator_vector(even_left, odd_right)
    even_even_tensor = _st_jordan_product(even_left, even_right)
    even_odd_tensor = _st_jordan_product(even_left, odd_right)

    transformed_even_left = _transform_tensor(
        even_left,
        orthogonal,
        odd=False,
    )
    transformed_even_right = _transform_tensor(
        even_right,
        orthogonal,
        odd=False,
    )
    transformed_odd_right = _transform_tensor(
        odd_right,
        orthogonal,
        odd=True,
    )

    torch.testing.assert_close(
        _st_commutator_vector(
            transformed_even_left,
            transformed_even_right,
        ),
        _transform_vector(even_even_vector, orthogonal, axial=True),
        atol=1e-10,
        rtol=1e-10,
    )
    torch.testing.assert_close(
        _st_commutator_vector(
            transformed_even_left,
            transformed_odd_right,
        ),
        _transform_vector(even_odd_vector, orthogonal, axial=False),
        atol=1e-10,
        rtol=1e-10,
    )
    torch.testing.assert_close(
        _st_jordan_product(
            transformed_even_left,
            transformed_even_right,
        ),
        _transform_tensor(even_even_tensor, orthogonal, odd=False),
        atol=1e-10,
        rtol=1e-10,
    )
    torch.testing.assert_close(
        _st_jordan_product(
            transformed_even_left,
            transformed_odd_right,
        ),
        _transform_tensor(even_odd_tensor, orthogonal, odd=True),
        atol=1e-10,
        rtol=1e-10,
    )


def test_low_order_closure_is_parity_complete() -> None:
    torch.manual_seed(103)
    nodes = 5
    heads = 4
    rank = 3
    scalar_width = 16
    closure = _LowOrderTensorClosure(
        scalar_width=scalar_width,
        num_heads=heads,
        rank=rank,
        multipole_rank=rank,
        eps=1e-10,
    ).double()
    with torch.no_grad():
        for parameter in closure.parameters():
            parameter.normal_(mean=0.0, std=0.2)

    state = _random_state(
        nodes=nodes,
        scalar_width=scalar_width,
        heads=heads,
    )
    multipoles = _random_multipoles(nodes=nodes, rank=rank)
    reflection = torch.diag(
        torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)
    )

    reference = closure(state, multipoles)
    transformed = closure(
        _transform_state(state, reflection),
        _transform_multipoles(multipoles, reflection),
    )
    expected = _transform_state(reference, reflection)

    for actual, target in zip(
        (
            transformed.even_scalar,
            transformed.odd_scalar,
            transformed.polar_vector,
            transformed.axial_vector,
            transformed.even_tensor,
            transformed.odd_tensor,
        ),
        (
            expected.even_scalar,
            expected.odd_scalar,
            expected.polar_vector,
            expected.axial_vector,
            expected.even_tensor,
            expected.odd_tensor,
        ),
        strict=True,
    ):
        torch.testing.assert_close(actual, target, atol=1e-9, rtol=1e-9)


def test_sector_norm_is_finite_at_zero_state() -> None:
    nodes = 3
    heads = 4
    scalar_width = 16
    zero = _ParityState(
        even_scalar=torch.zeros(nodes, scalar_width),
        odd_scalar=torch.zeros(nodes, heads),
        polar_vector=torch.zeros(nodes, heads, 3),
        axial_vector=torch.zeros(nodes, heads, 3),
        even_tensor=torch.zeros(nodes, heads, 5),
        odd_tensor=torch.zeros(nodes, heads, 5),
    )
    norm = _ParitySectorNorm(
        scalar_width=scalar_width,
        num_heads=heads,
        eps=1e-8,
    )
    normalized = norm(zero)

    assert all(
        torch.isfinite(value).all()
        for value in (
            normalized.odd_scalar,
            normalized.polar_vector,
            normalized.axial_vector,
            normalized.even_tensor,
            normalized.odd_tensor,
        )
    )
