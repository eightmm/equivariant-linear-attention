r"""Dependency-free real spherical-harmonic and Clebsch--Gordan reference.

The spherical-harmonic convention is the orthonormal Condon--Shortley
convention, transformed to real tesseral components ordered by
``m = -l, ..., l``:

.. math::

    Y^R_{l0} &= Y^0_l, \\
    Y^R_{l,+m} &= \sqrt{2}(-1)^m \Re Y^m_l, \\
    Y^R_{l,-m} &= \sqrt{2}(-1)^m \Im Y^m_l \quad (m > 0).

Clebsch--Gordan tensors have shape ``(2 L + 1, 2 l1 + 1, 2 l2 + 1)`` and
couple with ``einsum("Mab,...a,...b->...M", C, left, right)``.  The
complex-to-real transform includes the conventional path phase
``(-i) ** (l1 + l2 - L)``.  It makes every coefficient real, gives
``1 x 1 -> 0`` as the normalized dot product, and gives ``1 x 1 -> 1`` as
the normalized cross product.

This module is intentionally a correctness reference.  It uses only Python
integer arithmetic and PyTorch, builds coefficients once on CPU in float64,
and does not depend on e3nn, SciPy, or a runtime symbolic package.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from math import copysign, exp, factorial, lgamma, log, pi, sqrt
from typing import TypeAlias

import torch


DeviceLike: TypeAlias = torch.device | str | int

_REAL_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}
_MAX_CG_ELEMENTS = 2_000_000
_REALNESS_RTOL = 256.0 * torch.finfo(torch.float64).eps

__all__ = [
    "cartesian_to_real_l1",
    "matrix_to_real_l2",
    "real_clebsch_gordan",
    "real_l1_to_cartesian",
    "real_l2_to_matrix",
    "real_spherical_harmonics",
]


def _validate_degree(name: str, degree: int) -> int:
    if isinstance(degree, bool) or not isinstance(degree, int):
        raise TypeError(f"{name} must be an integer")
    if degree < 0:
        raise ValueError(f"{name} must be nonnegative")
    return degree


def _validate_real_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype not in _REAL_DTYPES:
        raise TypeError("dtype must be a real floating PyTorch dtype")
    return dtype


def _coefficient_shape(
    left_degree: int,
    right_degree: int,
    output_degree: int,
) -> tuple[int, int, int]:
    shape = (
        2 * output_degree + 1,
        2 * left_degree + 1,
        2 * right_degree + 1,
    )
    elements = shape[0] * shape[1] * shape[2]
    if elements > _MAX_CG_ELEMENTS:
        raise ValueError(
            "Clebsch--Gordan coefficient element guard exceeded: "
            f"requested {elements:,}, maximum {_MAX_CG_ELEMENTS:,}"
        )
    return shape


def _log_positive_fraction(value: Fraction) -> float:
    if value <= 0:
        raise ValueError("the logarithm requires a positive fraction")
    return log(value.numerator) - log(value.denominator)


@lru_cache(maxsize=None)
def _complex_clebsch_gordan_value(
    left_degree: int,
    left_m: int,
    right_degree: int,
    right_m: int,
    output_degree: int,
    output_m: int,
) -> float:
    """Evaluate one Condon--Shortley CG coefficient by the Racah formula.

    Every alternating summand is accumulated as :class:`fractions.Fraction`.
    Only the final square root and conversion use floating point.  Computing
    the final magnitude in log space avoids factorial overflow.
    """

    if (
        output_m != left_m + right_m
        or abs(left_m) > left_degree
        or abs(right_m) > right_degree
        or abs(output_m) > output_degree
        or output_degree < abs(left_degree - right_degree)
        or output_degree > left_degree + right_degree
    ):
        return 0.0

    triangle = Fraction(
        (2 * output_degree + 1)
        * factorial(output_degree + left_degree - right_degree)
        * factorial(output_degree - left_degree + right_degree)
        * factorial(left_degree + right_degree - output_degree),
        factorial(left_degree + right_degree + output_degree + 1),
    )
    magnetic_factorials = (
        factorial(output_degree + output_m)
        * factorial(output_degree - output_m)
        * factorial(left_degree - left_m)
        * factorial(left_degree + left_m)
        * factorial(right_degree - right_m)
        * factorial(right_degree + right_m)
    )

    lower = max(
        0,
        right_degree - output_degree - left_m,
        left_degree - output_degree + right_m,
    )
    upper = min(
        left_degree + right_degree - output_degree,
        left_degree - left_m,
        right_degree + right_m,
    )
    alternating_sum = Fraction(0)
    for index in range(lower, upper + 1):
        denominator = (
            factorial(index)
            * factorial(left_degree + right_degree - output_degree - index)
            * factorial(left_degree - left_m - index)
            * factorial(right_degree + right_m - index)
            * factorial(output_degree - right_degree + left_m + index)
            * factorial(output_degree - left_degree - right_m + index)
        )
        alternating_sum += Fraction(-1 if index % 2 else 1, denominator)

    if not alternating_sum:
        return 0.0
    magnitude_log = (
        _log_positive_fraction(abs(alternating_sum))
        + 0.5 * _log_positive_fraction(triangle)
        + 0.5 * log(magnetic_factorials)
    )
    return copysign(exp(magnitude_log), alternating_sum)


@lru_cache(maxsize=None)
def _complex_to_real_matrix(degree: int) -> torch.Tensor:
    """Return ``Y_real = U @ Y_complex`` in the documented tesseral order."""

    size = 2 * degree + 1
    transform = torch.zeros(size, size, dtype=torch.complex128)
    transform[degree, degree] = 1.0
    inverse_root_two = 1.0 / sqrt(2.0)
    for order in range(1, degree + 1):
        positive = degree + order
        negative = degree - order
        sign = -1.0 if order % 2 else 1.0
        transform[positive, positive] = sign * inverse_root_two
        transform[positive, negative] = inverse_root_two
        transform[negative, positive] = -1j * sign * inverse_root_two
        transform[negative, negative] = 1j * inverse_root_two
    return transform


@lru_cache(maxsize=128)
def _canonical_real_clebsch_gordan(
    left_degree: int,
    right_degree: int,
    output_degree: int,
) -> torch.Tensor:
    shape = _coefficient_shape(left_degree, right_degree, output_degree)
    if not (
        abs(left_degree - right_degree)
        <= output_degree
        <= left_degree + right_degree
    ):
        return torch.zeros(shape, dtype=torch.float64)

    coefficients = torch.zeros(shape, dtype=torch.complex128)
    for left_m in range(-left_degree, left_degree + 1):
        for right_m in range(-right_degree, right_degree + 1):
            output_m = left_m + right_m
            if abs(output_m) <= output_degree:
                coefficients[
                    output_m + output_degree,
                    left_m + left_degree,
                    right_m + right_degree,
                ] = _complex_clebsch_gordan_value(
                    left_degree,
                    left_m,
                    right_degree,
                    right_m,
                    output_degree,
                    output_m,
                )

    output_transform = _complex_to_real_matrix(output_degree)
    left_transform = _complex_to_real_matrix(left_degree)
    right_transform = _complex_to_real_matrix(right_degree)
    transformed = torch.einsum(
        "aM,Mij,bi,cj->abc",
        output_transform,
        coefficients,
        left_transform.conj(),
        right_transform.conj(),
    )
    transformed = (-1j) ** (
        left_degree + right_degree - output_degree
    ) * transformed

    real_scale = max(1.0, transformed.real.abs().max().item())
    imaginary_residual = transformed.imag.abs().max().item()
    if imaginary_residual > _REALNESS_RTOL * real_scale:
        raise RuntimeError(
            "complex-to-real Clebsch--Gordan transform was not real: "
            f"maximum imaginary residual {imaginary_residual:.3e}"
        )
    result = transformed.real.contiguous()
    result.masked_fill_(
        result.abs() < 32.0 * torch.finfo(torch.float64).eps,
        0.0,
    )
    return result


def real_clebsch_gordan(
    left_degree: int,
    right_degree: int,
    output_degree: int,
    *,
    dtype: torch.dtype,
    device: DeviceLike,
) -> torch.Tensor:
    """Return a real tesseral Clebsch--Gordan coefficient tensor.

    A triangle-rule violation is a mathematically valid zero coupling and
    therefore returns a correctly shaped zero tensor.  Invalid degrees and
    requests exceeding the bounded reference allocation are rejected.

    A canonical CPU/float64 tensor is memoized internally.  Every public call
    returns its own tensor so caller mutation cannot corrupt the process-wide
    coefficient cache.
    """

    left_degree = _validate_degree("left_degree", left_degree)
    right_degree = _validate_degree("right_degree", right_degree)
    output_degree = _validate_degree("output_degree", output_degree)
    dtype = _validate_real_dtype(dtype)
    target_device = torch.device(device)
    _coefficient_shape(left_degree, right_degree, output_degree)
    canonical = _canonical_real_clebsch_gordan(
        left_degree,
        right_degree,
        output_degree,
    )
    return canonical.to(device=target_device, dtype=dtype, copy=True)


@lru_cache(maxsize=None)
def _solid_harmonic_terms(
    degree: int,
    order: int,
) -> tuple[tuple[float, int, int], ...]:
    """Coefficients of ``r^(l-m) d^m P_l(z/r)``.

    Each returned term is ``coefficient * z**z_power * r2**r2_power``.
    The Condon--Shortley phase cancels the phase in the documented real
    tesseral conversion for nonzero ``m``.
    """

    normalization_log = 0.5 * (
        log(2 * degree + 1)
        - log(4.0 * pi)
        + lgamma(degree - order + 1)
        - lgamma(degree + order + 1)
    )
    tesseral_scale = sqrt(2.0) if order else 1.0
    terms = []
    for radial_power in range((degree - order) // 2 + 1):
        z_power = degree - 2 * radial_power - order
        coefficient = Fraction(
            (-1 if radial_power % 2 else 1)
            * factorial(2 * degree - 2 * radial_power),
            (2**degree)
            * factorial(radial_power)
            * factorial(degree - radial_power)
            * factorial(z_power),
        )
        terms.append(
            (
                tesseral_scale
                * exp(normalization_log)
                * float(coefficient),
                z_power,
                radial_power,
            )
        )
    return tuple(terms)


def real_spherical_harmonics(
    degree: int,
    x: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """Evaluate real tesseral spherical or regular solid harmonics.

    Args:
        degree: Nonnegative angular degree ``l``.
        x: Real floating tensor with final dimension three.
        normalize: If true, evaluate on ``x / ||x||``.  If false, return the
            regular solid harmonic ``||x||**l Y_l(x / ||x||)``, evaluated as
            a polynomial without divisions.

    At a coincident point, normalized harmonics are defined as zero for
    ``l > 0`` and retain the constant ``Y_00`` for ``l = 0``.  This explicit
    extension is finite; no directional value at the origin is implied.
    """

    degree = _validate_degree("degree", degree)
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a PyTorch tensor")
    if x.ndim < 1 or x.shape[-1] != 3:
        raise ValueError("x must have final dimension 3")
    if x.dtype not in _REAL_DTYPES:
        raise TypeError("x must have a real floating PyTorch dtype")
    if not isinstance(normalize, bool):
        raise TypeError("normalize must be a bool")

    coordinates = x
    if normalize:
        squared_radius = coordinates.square().sum(dim=-1, keepdim=True)
        safe_squared_radius = torch.where(
            squared_radius > 0,
            squared_radius,
            torch.ones_like(squared_radius),
        )
        coordinates = coordinates * torch.rsqrt(safe_squared_radius)

    x_component, y_component, z_component = coordinates.unbind(dim=-1)
    squared_radius = coordinates.square().sum(dim=-1)
    real_power = torch.ones_like(x_component)
    imaginary_power = torch.zeros_like(x_component)
    components: list[torch.Tensor | None] = [None] * (2 * degree + 1)

    for order in range(degree + 1):
        if order:
            previous_real = real_power
            real_power = previous_real * x_component - imaginary_power * y_component
            imaginary_power = (
                previous_real * y_component + imaginary_power * x_component
            )

        radial_polynomial = torch.zeros_like(x_component)
        for coefficient, z_power, radial_power in _solid_harmonic_terms(
            degree, order
        ):
            radial_polynomial = radial_polynomial + (
                coefficient
                * z_component.pow(z_power)
                * squared_radius.pow(radial_power)
            )

        if order == 0:
            components[degree] = radial_polynomial
        else:
            components[degree - order] = imaginary_power * radial_polynomial
            components[degree + order] = real_power * radial_polynomial

    # All entries are assigned by the loop; the assertion protects maintenance
    # changes while leaving the output stack statically ordered by m.
    if any(component is None for component in components):
        raise RuntimeError("internal spherical-harmonic component was not assigned")
    return torch.stack(
        [component for component in components if component is not None],
        dim=-1,
    )


def cartesian_to_real_l1(value: torch.Tensor) -> torch.Tensor:
    """Reorder Cartesian ``(x, y, z)`` to tesseral ``m=(-1, 0, 1)``."""

    _validate_bridge_tensor(value, final_shape=(3,), name="value")
    return value[..., [1, 2, 0]]


def real_l1_to_cartesian(value: torch.Tensor) -> torch.Tensor:
    """Reorder tesseral ``m=(-1, 0, 1)`` to Cartesian ``(x, y, z)``."""

    _validate_bridge_tensor(value, final_shape=(3,), name="value")
    return value[..., [2, 0, 1]]


def real_l2_to_matrix(value: torch.Tensor) -> torch.Tensor:
    """Map tesseral ``l=2`` coordinates to an isometric traceless matrix."""

    _validate_bridge_tensor(value, final_shape=(5,), name="value")
    minus_two, minus_one, zero, plus_one, plus_two = value.unbind(dim=-1)
    inverse_root_two = 1.0 / sqrt(2.0)
    inverse_root_six = 1.0 / sqrt(6.0)
    xx = -inverse_root_six * zero + inverse_root_two * plus_two
    yy = -inverse_root_six * zero - inverse_root_two * plus_two
    zz = 2.0 * inverse_root_six * zero
    xy = inverse_root_two * minus_two
    xz = inverse_root_two * plus_one
    yz = inverse_root_two * minus_one
    return torch.stack(
        (
            torch.stack((xx, xy, xz), dim=-1),
            torch.stack((xy, yy, yz), dim=-1),
            torch.stack((xz, yz, zz), dim=-1),
        ),
        dim=-2,
    )


def matrix_to_real_l2(value: torch.Tensor) -> torch.Tensor:
    """Project a Cartesian matrix onto the isometric real ``l=2`` basis."""

    _validate_bridge_tensor(value, final_shape=(3, 3), name="value")
    root_two = sqrt(2.0)
    root_six = sqrt(6.0)
    return torch.stack(
        (
            (value[..., 0, 1] + value[..., 1, 0]) / root_two,
            (value[..., 1, 2] + value[..., 2, 1]) / root_two,
            (
                -value[..., 0, 0]
                - value[..., 1, 1]
                + 2.0 * value[..., 2, 2]
            )
            / root_six,
            (value[..., 0, 2] + value[..., 2, 0]) / root_two,
            (value[..., 0, 0] - value[..., 1, 1]) / root_two,
        ),
        dim=-1,
    )


def _validate_bridge_tensor(
    value: torch.Tensor,
    *,
    final_shape: tuple[int, ...],
    name: str,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a PyTorch tensor")
    if value.shape[-len(final_shape) :] != final_shape:
        raise ValueError(f"{name} must have final shape {final_shape}")
    if value.dtype not in _REAL_DTYPES:
        raise TypeError(f"{name} must have a real floating PyTorch dtype")
