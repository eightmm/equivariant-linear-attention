from __future__ import annotations

from math import pi, sqrt

import pytest
import torch

from equivariant_attention.spherical import (
    cartesian_to_real_l1,
    matrix_to_real_l2,
    real_clebsch_gordan,
    real_l1_to_cartesian,
    real_l2_to_matrix,
    real_spherical_harmonics,
)


def _random_orthogonal(*, proper: bool, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(
        torch.randn(3, 3, dtype=torch.float64, generator=generator)
    )
    if (torch.linalg.det(q) > 0) != proper:
        q[:, 0] = -q[:, 0]
    return q


@pytest.mark.parametrize(
    ("args", "error", "message"),
    [
        ((-1, 0, 0), ValueError, "nonnegative"),
        ((True, 0, 0), TypeError, "integer"),
        ((1.0, 0, 0), TypeError, "integer"),
    ],
)
def test_clebsch_gordan_rejects_invalid_degrees(
    args: tuple[object, object, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        real_clebsch_gordan(
            *args,  # type: ignore[arg-type]
            dtype=torch.float64,
            device="cpu",
        )


def test_clebsch_gordan_selection_rule_returns_a_typed_zero_tensor() -> None:
    coefficients = real_clebsch_gordan(
        1,
        1,
        3,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert coefficients.shape == (7, 3, 3)
    assert coefficients.dtype == torch.float32
    assert coefficients.device.type == "cpu"
    assert torch.count_nonzero(coefficients) == 0


def test_clebsch_gordan_rejects_nonreal_dtype_and_guarded_allocation() -> None:
    with pytest.raises(TypeError, match="real floating"):
        real_clebsch_gordan(1, 1, 1, dtype=torch.complex128, device="cpu")
    with pytest.raises(ValueError, match="element guard"):
        real_clebsch_gordan(100, 100, 100, dtype=torch.float64, device="cpu")


def test_canonical_clebsch_gordan_cache_is_private_and_cast_on_request() -> None:
    first = real_clebsch_gordan(3, 2, 4, dtype=torch.float64, device="cpu")
    second = real_clebsch_gordan(3, 2, 4, dtype=torch.float64, device="cpu")
    lower_precision = real_clebsch_gordan(
        3, 2, 4, dtype=torch.float32, device="cpu"
    )

    assert first.data_ptr() != second.data_ptr()
    assert first.dtype == torch.float64
    assert first.device.type == "cpu"
    assert lower_precision.dtype == torch.float32
    torch.testing.assert_close(lower_precision.double(), first, atol=6e-8, rtol=6e-8)

    expected = second.clone()
    first.zero_()
    third = real_clebsch_gordan(
        3, 2, 4, dtype=torch.float64, device="cpu"
    )
    torch.testing.assert_close(third, expected, rtol=0, atol=0)


def test_low_degree_clebsch_gordan_matches_dot_and_cross_conventions() -> None:
    scalar = real_clebsch_gordan(1, 1, 0, dtype=torch.float64, device="cpu")
    vector = real_clebsch_gordan(1, 1, 1, dtype=torch.float64, device="cpu")

    torch.testing.assert_close(
        scalar[0],
        torch.eye(3, dtype=torch.float64) / sqrt(3.0),
        atol=2e-15,
        rtol=2e-15,
    )

    left_cartesian = torch.tensor([0.3, -0.5, 0.8], dtype=torch.float64)
    right_cartesian = torch.tensor([-0.7, 0.2, 0.4], dtype=torch.float64)
    left = cartesian_to_real_l1(left_cartesian)
    right = cartesian_to_real_l1(right_cartesian)
    coupled = torch.einsum("mab,a,b->m", vector, left, right)
    expected = cartesian_to_real_l1(
        torch.linalg.cross(left_cartesian, right_cartesian)
    ) / sqrt(2.0)
    torch.testing.assert_close(coupled, expected, atol=2e-15, rtol=2e-15)


def test_clebsch_gordan_is_orthogonal_and_complete_through_degree_four() -> None:
    for left_degree in range(5):
        for right_degree in range(5):
            blocks = []
            for output_degree in range(
                abs(left_degree - right_degree), left_degree + right_degree + 1
            ):
                coefficients = real_clebsch_gordan(
                    left_degree,
                    right_degree,
                    output_degree,
                    dtype=torch.float64,
                    device="cpu",
                )
                flattened = coefficients.reshape(2 * output_degree + 1, -1)
                torch.testing.assert_close(
                    flattened @ flattened.mT,
                    torch.eye(2 * output_degree + 1, dtype=torch.float64),
                    atol=3e-14,
                    rtol=3e-14,
                )
                blocks.append(flattened)

            complete = torch.cat(blocks, dim=0)
            torch.testing.assert_close(
                complete @ complete.mT,
                torch.eye(complete.shape[0], dtype=torch.float64),
                atol=5e-14,
                rtol=5e-14,
            )


def test_clebsch_gordan_swap_symmetry_through_degree_four() -> None:
    for left_degree in range(5):
        for right_degree in range(5):
            for output_degree in range(
                abs(left_degree - right_degree), left_degree + right_degree + 1
            ):
                left_right = real_clebsch_gordan(
                    left_degree,
                    right_degree,
                    output_degree,
                    dtype=torch.float64,
                    device="cpu",
                )
                right_left = real_clebsch_gordan(
                    right_degree,
                    left_degree,
                    output_degree,
                    dtype=torch.float64,
                    device="cpu",
                )
                sign = (-1) ** (left_degree + right_degree - output_degree)
                torch.testing.assert_close(
                    left_right,
                    sign * right_left.transpose(-1, -2),
                    atol=3e-14,
                    rtol=3e-14,
                )


def test_real_spherical_harmonics_low_degree_golden_values() -> None:
    points = torch.tensor(
        [[0.2, -0.4, 0.7], [-0.8, 0.1, -0.3]],
        dtype=torch.float64,
    )
    unit = points / torch.linalg.vector_norm(points, dim=-1, keepdim=True)
    x, y, z = unit.unbind(dim=-1)

    degree_zero = real_spherical_harmonics(0, points)
    degree_one = real_spherical_harmonics(1, points)
    degree_two = real_spherical_harmonics(2, points)

    torch.testing.assert_close(
        degree_zero,
        torch.full((2, 1), 1.0 / sqrt(4.0 * pi), dtype=torch.float64),
        atol=2e-15,
        rtol=2e-15,
    )
    torch.testing.assert_close(
        degree_one,
        sqrt(3.0 / (4.0 * pi)) * torch.stack((y, z, x), dim=-1),
        atol=2e-15,
        rtol=2e-15,
    )
    expected_two = torch.stack(
        (
            sqrt(15.0 / (4.0 * pi)) * x * y,
            sqrt(15.0 / (4.0 * pi)) * y * z,
            sqrt(5.0 / (16.0 * pi)) * (2.0 * z.square() - x.square() - y.square()),
            sqrt(15.0 / (4.0 * pi)) * x * z,
            sqrt(15.0 / (16.0 * pi)) * (x.square() - y.square()),
        ),
        dim=-1,
    )
    torch.testing.assert_close(
        degree_two,
        expected_two,
        atol=3e-15,
        rtol=3e-15,
    )


def test_solid_harmonics_are_finite_at_coincident_points() -> None:
    coincident = torch.zeros(4, 3, dtype=torch.float64, requires_grad=True)
    for degree in range(7):
        solid = real_spherical_harmonics(degree, coincident, normalize=False)
        normalized = real_spherical_harmonics(degree, coincident, normalize=True)
        assert torch.isfinite(solid).all()
        assert torch.isfinite(normalized).all()
        if degree == 0:
            torch.testing.assert_close(
                solid,
                torch.full_like(solid, 1.0 / sqrt(4.0 * pi)),
            )
        else:
            assert torch.count_nonzero(solid) == 0
            assert torch.count_nonzero(normalized) == 0


def test_spherical_harmonic_inputs_are_validated() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        real_spherical_harmonics(-1, torch.ones(2, 3))
    with pytest.raises(TypeError, match="integer"):
        real_spherical_harmonics(True, torch.ones(2, 3))
    with pytest.raises(ValueError, match="final dimension 3"):
        real_spherical_harmonics(2, torch.ones(2, 4))
    with pytest.raises(TypeError, match="real floating"):
        real_spherical_harmonics(2, torch.ones(2, 3, dtype=torch.int64))
    with pytest.raises(TypeError, match="normalize"):
        real_spherical_harmonics(2, torch.ones(2, 3), normalize=1)  # type: ignore[arg-type]


def test_spherical_harmonic_addition_theorem_through_degree_four() -> None:
    generator = torch.Generator().manual_seed(382)
    left = torch.randn(19, 3, dtype=torch.float64, generator=generator)
    right = torch.randn(19, 3, dtype=torch.float64, generator=generator)
    left = left / torch.linalg.vector_norm(left, dim=-1, keepdim=True)
    right = right / torch.linalg.vector_norm(right, dim=-1, keepdim=True)
    cosine = (left * right).sum(dim=-1)

    legendre_previous = torch.ones_like(cosine)
    legendre_current = cosine
    for degree in range(5):
        if degree == 0:
            legendre = legendre_previous
        elif degree == 1:
            legendre = legendre_current
        else:
            legendre_next = (
                (2 * degree - 1) * cosine * legendre_current
                - (degree - 1) * legendre_previous
            ) / degree
            legendre_previous, legendre_current = (
                legendre_current,
                legendre_next,
            )
            legendre = legendre_current
        observed = (
            real_spherical_harmonics(degree, left)
            * real_spherical_harmonics(degree, right)
        ).sum(dim=-1)
        expected = (2 * degree + 1) / (4.0 * pi) * legendre
        torch.testing.assert_close(observed, expected, atol=2e-13, rtol=2e-13)


def test_spherical_harmonic_products_decompose_with_real_cg_through_l4() -> None:
    generator = torch.Generator().manual_seed(914)
    points = torch.randn(11, 3, dtype=torch.float64, generator=generator)
    points = points / torch.linalg.vector_norm(points, dim=-1, keepdim=True)

    for left_degree in range(5):
        for right_degree in range(5 - left_degree):
            left = real_spherical_harmonics(left_degree, points)
            right = real_spherical_harmonics(right_degree, points)
            product = torch.einsum("na,nb->nab", left, right)
            reconstructed = torch.zeros_like(product)
            for output_degree in range(
                abs(left_degree - right_degree), left_degree + right_degree + 1
            ):
                coefficients = real_clebsch_gordan(
                    left_degree,
                    right_degree,
                    output_degree,
                    dtype=torch.float64,
                    device="cpu",
                )
                zero_coupling = coefficients[
                    output_degree, left_degree, right_degree
                ]
                prefactor = sqrt(
                    (2 * left_degree + 1)
                    * (2 * right_degree + 1)
                    / (4.0 * pi * (2 * output_degree + 1))
                )
                output = real_spherical_harmonics(output_degree, points)
                reconstructed = reconstructed + (
                    prefactor
                    * zero_coupling
                    * torch.einsum("mab,nm->nab", coefficients, output)
                )
            torch.testing.assert_close(
                reconstructed,
                product,
                atol=3e-13,
                rtol=3e-13,
            )


@pytest.mark.parametrize("proper", [False, True])
def test_spherical_harmonics_are_o3_covariant_through_degree_four(
    proper: bool,
) -> None:
    rotation = _random_orthogonal(proper=proper, seed=175 + int(proper))
    generator = torch.Generator().manual_seed(507)
    calibration = torch.randn(48, 3, dtype=torch.float64, generator=generator)
    probe = torch.randn(23, 3, dtype=torch.float64, generator=generator)

    for degree in range(5):
        calibration_values = real_spherical_harmonics(degree, calibration)
        rotated_calibration = real_spherical_harmonics(
            degree, calibration @ rotation.mT
        )
        representation_transpose = torch.linalg.lstsq(
            calibration_values, rotated_calibration
        ).solution
        representation = representation_transpose.mT
        torch.testing.assert_close(
            representation @ representation.mT,
            torch.eye(2 * degree + 1, dtype=torch.float64),
            atol=2e-12,
            rtol=2e-12,
        )

        observed = real_spherical_harmonics(degree, probe @ rotation.mT)
        expected = torch.einsum(
            "mn,kn->km",
            representation,
            real_spherical_harmonics(degree, probe),
        )
        torch.testing.assert_close(observed, expected, atol=2e-12, rtol=2e-12)

        inverted = real_spherical_harmonics(degree, -probe)
        torch.testing.assert_close(
            inverted,
            (-1) ** degree * real_spherical_harmonics(degree, probe),
            atol=2e-13,
            rtol=2e-13,
        )


@pytest.mark.parametrize("normalize", [False, True])
def test_spherical_harmonics_have_fp64_first_and_second_derivatives(
    normalize: bool,
) -> None:
    points = torch.tensor(
        [[0.4, -0.7, 1.1], [-0.9, 0.3, 0.6]],
        dtype=torch.float64,
        requires_grad=True,
    )

    def evaluate(value: torch.Tensor) -> torch.Tensor:
        return real_spherical_harmonics(4, value, normalize=normalize)

    assert torch.autograd.gradcheck(
        evaluate,
        (points,),
        eps=1e-6,
        atol=2e-5,
        rtol=2e-4,
        fast_mode=True,
    )
    assert torch.autograd.gradgradcheck(
        evaluate,
        (points,),
        eps=1e-6,
        atol=3e-5,
        rtol=3e-4,
        fast_mode=True,
    )


def test_low_degree_cartesian_bridges_are_isometric_and_invertible() -> None:
    generator = torch.Generator().manual_seed(73)
    vector = torch.randn(7, 3, dtype=torch.float64, generator=generator)
    tesseral_vector = cartesian_to_real_l1(vector)
    torch.testing.assert_close(
        tesseral_vector,
        vector[..., [1, 2, 0]],
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(real_l1_to_cartesian(tesseral_vector), vector)

    tesseral_tensor = torch.randn(7, 5, dtype=torch.float64, generator=generator)
    matrix = real_l2_to_matrix(tesseral_tensor)
    torch.testing.assert_close(matrix, matrix.mT, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1),
        torch.zeros(7, dtype=torch.float64),
        atol=2e-15,
        rtol=2e-15,
    )
    torch.testing.assert_close(
        matrix.square().sum(dim=(-2, -1)),
        tesseral_tensor.square().sum(dim=-1),
        atol=3e-15,
        rtol=3e-15,
    )
    torch.testing.assert_close(
        matrix_to_real_l2(matrix),
        tesseral_tensor,
        atol=3e-15,
        rtol=3e-15,
    )
