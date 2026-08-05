from __future__ import annotations

import pytest
import torch

from equivariant_linear_attention.irreps import (
    matrix_to_st5,
    project_symmetric_traceless,
    st5_inner,
    st5_mse,
    st5_norm,
    st5_to_matrix,
)


def _orthogonal(*, improper: bool = False) -> torch.Tensor:
    matrix, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    desired = -1 if improper else 1
    if int(torch.linalg.det(matrix).sign().item()) != desired:
        matrix[:, 0].neg_()
    return matrix


def test_matrix_conversion_projects_to_symmetric_traceless() -> None:
    value = torch.randn(5, 3, 3, dtype=torch.float64)
    projected = project_symmetric_traceless(value)
    recovered = st5_to_matrix(matrix_to_st5(value))

    torch.testing.assert_close(recovered, projected)
    torch.testing.assert_close(recovered, recovered.transpose(-1, -2))
    torch.testing.assert_close(
        torch.diagonal(recovered, dim1=-2, dim2=-1).sum(-1),
        torch.zeros(5, dtype=torch.float64),
        atol=2e-15,
        rtol=0.0,
    )


@pytest.mark.parametrize("improper", [False, True])
def test_st5_metric_matches_frobenius_and_is_o3_invariant(
    improper: bool,
) -> None:
    left = matrix_to_st5(torch.randn(7, 3, 3, dtype=torch.float64))
    right = matrix_to_st5(torch.randn(7, 3, 3, dtype=torch.float64))
    rotation = _orthogonal(improper=improper)

    left_matrix = st5_to_matrix(left)
    right_matrix = st5_to_matrix(right)
    rotated_left = matrix_to_st5(rotation @ left_matrix @ rotation.T)
    rotated_right = matrix_to_st5(rotation @ right_matrix @ rotation.T)

    expected_inner = (left_matrix * right_matrix).sum(dim=(-2, -1))
    torch.testing.assert_close(st5_inner(left, right), expected_inner)
    torch.testing.assert_close(
        st5_inner(rotated_left, rotated_right),
        expected_inner,
        atol=2e-14,
        rtol=2e-14,
    )
    torch.testing.assert_close(st5_norm(rotated_left), st5_norm(left))
    torch.testing.assert_close(
        st5_mse(rotated_left, rotated_right),
        st5_mse(left, right),
    )


def test_st5_mse_reduction_and_validation() -> None:
    prediction = torch.randn(2, 3, 5, dtype=torch.float64)
    target = torch.randn(2, 3, 5, dtype=torch.float64)
    elementwise = st5_mse(prediction, target, reduction="none")

    assert elementwise.shape == (2, 3)
    torch.testing.assert_close(st5_mse(prediction, target), elementwise.mean())
    torch.testing.assert_close(
        st5_mse(prediction, target, reduction="sum"),
        elementwise.sum(),
    )
    with pytest.raises(ValueError, match="reduction"):
        st5_mse(prediction, target, reduction="median")


def test_st5_norm_has_a_finite_gradient_at_the_zero_tensor() -> None:
    """A zero tensor target is legal, so the norm must stay differentiable.

    ``sqrt`` has an infinite derivative at the origin, which previously turned
    any zero row into ``nan`` gradients for a public helper.
    """

    value = torch.zeros(4, 5, dtype=torch.float64, requires_grad=True)

    gradient, = torch.autograd.grad(st5_norm(value).sum(), value)

    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) == 0


def test_st5_norm_values_are_unchanged_by_the_zero_guard() -> None:
    torch.manual_seed(5)
    value = torch.randn(6, 5, dtype=torch.float64)
    mixed = torch.cat([value, torch.zeros(2, 5, dtype=torch.float64)])

    expected = torch.sqrt(st5_inner(mixed, mixed).clamp_min(0.0))

    torch.testing.assert_close(st5_norm(mixed), expected, atol=0.0, rtol=0.0)


def test_st5_norm_keeps_finite_gradients_on_mixed_zero_and_nonzero_rows() -> None:
    torch.manual_seed(6)
    raw = torch.cat(
        [torch.randn(3, 5, dtype=torch.float64), torch.zeros(3, 5, dtype=torch.float64)]
    ).requires_grad_(True)

    gradient, = torch.autograd.grad(st5_norm(raw).sum(), raw)

    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient[3:]) == 0
    assert torch.count_nonzero(gradient[:3]) > 0
