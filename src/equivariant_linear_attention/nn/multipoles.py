from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .parity import (
    _ChannelMix,
    _ParityState,
    _StaticGeometry,
    _compute_dtype,
    _csr_sum,
    _st_cross,
    _st_from_vector,
    _st_inner,
    _st_square,
)


def _c2_cutoff(squared_ratio: torch.Tensor) -> torch.Tensor:
    """Compact C2 envelope in the squared-distance coordinate."""
    inside = squared_ratio < 1.0
    u = squared_ratio.clamp(min=0.0, max=1.0)
    value = 1.0 - 10.0 * u.pow(3) + 15.0 * u.pow(4) - 6.0 * u.pow(5)
    return torch.where(inside, value, torch.zeros_like(value))


def _safe_unit_direction(
    direction: torch.Tensor,
    squared_distance: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    inverse = torch.where(
        squared_distance > eps,
        squared_distance.clamp_min(eps).rsqrt(),
        torch.zeros_like(squared_distance),
    )
    return direction * inverse.unsqueeze(-1)


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


def _st_commutator_vector(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Degree-one component of two degree-two tensors."""
    left_matrix = _st_to_matrix(left)
    right_matrix = _st_to_matrix(right)
    commutator = left_matrix @ right_matrix - right_matrix @ left_matrix
    return torch.stack(
        [
            commutator[..., 2, 1],
            commutator[..., 0, 2],
            commutator[..., 1, 0],
        ],
        dim=-1,
    )


def _st_jordan_product(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Degree-two component of the symmetric tensor product."""
    left_matrix = _st_to_matrix(left)
    right_matrix = _st_to_matrix(right)
    symmetric = 0.5 * (
        left_matrix @ right_matrix + right_matrix @ left_matrix
    )
    trace_third = symmetric.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / 3.0
    identity = torch.eye(
        3,
        dtype=symmetric.dtype,
        device=symmetric.device,
    )
    traceless = symmetric - trace_third[..., None, None] * identity
    return _matrix_to_st(traceless)


def _st_orthonormal(value: torch.Tensor) -> torch.Tensor:
    """Map the compact ST5 coordinates to an orthonormal Frobenius basis."""
    xx, yy, xy, xz, yz = value.unbind(dim=-1)
    zz = -xx - yy
    inv_sqrt_two = value.new_tensor(0.5).sqrt()
    inv_sqrt_six = value.new_tensor(1.0 / 6.0).sqrt()
    sqrt_two = value.new_tensor(2.0).sqrt()
    return torch.stack(
        [
            (xx - yy) * inv_sqrt_two,
            (xx + yy - 2.0 * zz) * inv_sqrt_six,
            sqrt_two * xy,
            sqrt_two * xz,
            sqrt_two * yz,
        ],
        dim=-1,
    )


def _normalize_st(value: torch.Tensor, eps: float) -> torch.Tensor:
    square = _st_square(value).unsqueeze(-1)
    return value / torch.sqrt(1.0 + square + eps)


@dataclass(frozen=True)
class NodeMultipoles:
    """Static receiver-centered l<=2 multipoles and parity-complete products."""

    mass: torch.Tensor
    mass_square: torch.Tensor
    polar: torch.Tensor
    even_tensor: torch.Tensor
    axial: torch.Tensor
    odd_scalar: torch.Tensor
    odd_tensor: torch.Tensor


class NodeMultipoleBank(nn.Module):
    """Deterministic positive radial profiles with no edge hidden state."""

    def __init__(
        self,
        *,
        num_rbf: int,
        rank: int,
        eps: float,
    ) -> None:
        super().__init__()
        if num_rbf < 1:
            raise ValueError("num_rbf must be positive")
        if rank < 3:
            raise ValueError("node multipoles require rank>=3 for chirality")
        self.num_rbf = num_rbf
        self.rank = rank
        self.eps = eps
        scales = torch.linspace(-4.0, 4.0, rank)
        self.register_buffer("_radial_scales", scales, persistent=False)

    def forward(self, geometry: _StaticGeometry) -> NodeMultipoles:
        dtype = _compute_dtype(
            geometry.direction,
            geometry.squared_distance,
            geometry.cutoff,
        )
        squared_distance = geometry.squared_distance.to(dtype=dtype)
        radial_coordinate = squared_distance.clamp(min=0.0, max=1.0)
        radial = torch.exp(
            (radial_coordinate.unsqueeze(-1) - 0.5)
            * self._radial_scales.to(dtype=dtype)
        )
        cutoff = geometry.cutoff.to(dtype=dtype)
        weight = cutoff.unsqueeze(-1) * radial
        mass = _csr_sum(weight, geometry.row_ptr)
        mass_square = _csr_sum(weight.square(), geometry.row_ptr)
        denominator = 1.0 + mass

        unit = _safe_unit_direction(
            geometry.direction.to(dtype=dtype),
            squared_distance,
            self.eps,
        )
        polar = _csr_sum(
            weight.unsqueeze(-1) * unit.unsqueeze(1),
            geometry.row_ptr,
        ) / denominator.unsqueeze(-1)
        even_tensor = _csr_sum(
            weight.unsqueeze(-1) * _st_from_vector(unit).unsqueeze(1),
            geometry.row_ptr,
        ) / denominator.unsqueeze(-1)

        first = polar
        second = polar.roll(shifts=1, dims=1)
        third = polar.roll(shifts=2, dims=1)
        axial = torch.cross(first, second, dim=-1)
        odd_scalar = (axial * third).sum(dim=-1)
        odd_tensor = _st_cross(third, axial)
        return NodeMultipoles(
            mass=mass,
            mass_square=mass_square,
            polar=polar,
            even_tensor=even_tensor,
            axial=axial,
            odd_scalar=odd_scalar,
            odd_tensor=odd_tensor,
        )


class _ParitySectorNorm(nn.Module):
    """Invariant sector RMS normalization with copy-wise learned gains."""

    def __init__(
        self,
        *,
        scalar_width: int,
        num_heads: int,
        eps: float,
    ) -> None:
        super().__init__()
        if scalar_width <= 0:
            raise ValueError("scalar_width must be positive")
        self.eps = max(float(eps), 1e-8)
        self.odd_gain = nn.Parameter(torch.ones(num_heads))
        self.polar_gain = nn.Parameter(torch.ones(num_heads))
        self.axial_gain = nn.Parameter(torch.ones(num_heads))
        self.even_tensor_gain = nn.Parameter(torch.ones(num_heads))
        self.odd_tensor_gain = nn.Parameter(torch.ones(num_heads))

    def _scalar_sector(
        self,
        value: torch.Tensor,
        gain: torch.Tensor,
    ) -> torch.Tensor:
        inverse = (value.square().mean(dim=-1, keepdim=True) + self.eps).rsqrt()
        return value * inverse * gain.to(dtype=value.dtype).unsqueeze(0)

    def _vector_sector(
        self,
        value: torch.Tensor,
        gain: torch.Tensor,
    ) -> torch.Tensor:
        inverse = (
            value.square().mean(dim=(-2, -1), keepdim=True) + self.eps
        ).rsqrt()
        return value * inverse * gain.to(dtype=value.dtype).reshape(1, -1, 1)

    def _tensor_sector(
        self,
        value: torch.Tensor,
        gain: torch.Tensor,
    ) -> torch.Tensor:
        square = _st_square(value).mean(dim=-1, keepdim=True) / 5.0
        inverse = (square + self.eps).rsqrt().unsqueeze(-1)
        return value * inverse * gain.to(dtype=value.dtype).reshape(1, -1, 1)

    def forward(self, state: _ParityState) -> _ParityState:
        return _ParityState(
            even_scalar=state.even_scalar,
            odd_scalar=self._scalar_sector(state.odd_scalar, self.odd_gain),
            polar_vector=self._vector_sector(state.polar_vector, self.polar_gain),
            axial_vector=self._vector_sector(state.axial_vector, self.axial_gain),
            even_tensor=self._tensor_sector(
                state.even_tensor,
                self.even_tensor_gain,
            ),
            odd_tensor=self._tensor_sector(
                state.odd_tensor,
                self.odd_tensor_gain,
            ),
        )


class _LowOrderTensorClosure(nn.Module):
    """Low-rank Cartesian realization of all l<=2 output sectors."""

    def __init__(
        self,
        *,
        scalar_width: int,
        num_heads: int,
        rank: int,
        multipole_rank: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.eps = eps

        self.polar_left = _ChannelMix(num_heads, rank)
        self.polar_right = _ChannelMix(num_heads, rank)
        self.axial_left = _ChannelMix(num_heads, rank)
        self.axial_right = _ChannelMix(num_heads, rank)
        self.even_tensor_left = _ChannelMix(num_heads, rank)
        self.even_tensor_right = _ChannelMix(num_heads, rank)
        self.odd_tensor_left = _ChannelMix(num_heads, rank)
        self.odd_tensor_right = _ChannelMix(num_heads, rank)

        self.static_polar = _ChannelMix(multipole_rank, rank)
        self.static_axial = _ChannelMix(multipole_rank, rank)
        self.static_even_tensor = _ChannelMix(multipole_rank, rank)
        self.static_odd_tensor = _ChannelMix(multipole_rank, rank)
        self.static_odd_scalar = _ChannelMix(multipole_rank, rank)

        self.even_scalar_out = nn.Linear(rank, scalar_width, bias=False)
        nn.init.zeros_(self.even_scalar_out.weight)
        self.odd_scalar_out = _ChannelMix(rank, num_heads, zero_init=True)
        self.polar_out = _ChannelMix(rank, num_heads, zero_init=True)
        self.axial_out = _ChannelMix(rank, num_heads, zero_init=True)
        self.even_tensor_out = _ChannelMix(rank, num_heads, zero_init=True)
        self.odd_tensor_out = _ChannelMix(rank, num_heads, zero_init=True)

    def forward(
        self,
        state: _ParityState,
        multipoles: NodeMultipoles,
    ) -> _ParityState:
        polar_left = self.polar_left(state.polar_vector) + self.static_polar(
            multipoles.polar.to(dtype=state.polar_vector.dtype)
        )
        polar_right = self.polar_right(state.polar_vector) + self.static_polar(
            multipoles.polar.roll(1, dims=1).to(dtype=state.polar_vector.dtype)
        )
        axial_left = self.axial_left(state.axial_vector) + self.static_axial(
            multipoles.axial.to(dtype=state.axial_vector.dtype)
        )
        axial_right = self.axial_right(state.axial_vector) + self.static_axial(
            multipoles.axial.roll(1, dims=1).to(dtype=state.axial_vector.dtype)
        )
        even_left = self.even_tensor_left(state.even_tensor) + self.static_even_tensor(
            multipoles.even_tensor.to(dtype=state.even_tensor.dtype)
        )
        even_right = (
            self.even_tensor_right(state.even_tensor)
            + self.static_even_tensor(
                multipoles.even_tensor.roll(1, dims=1).to(
                    dtype=state.even_tensor.dtype
                )
            )
        )
        odd_left = self.odd_tensor_left(state.odd_tensor) + self.static_odd_tensor(
            multipoles.odd_tensor.to(dtype=state.odd_tensor.dtype)
        )
        odd_right = self.odd_tensor_right(state.odd_tensor) + self.static_odd_tensor(
            multipoles.odd_tensor.roll(1, dims=1).to(dtype=state.odd_tensor.dtype)
        )
        static_odd = self.static_odd_scalar(
            multipoles.odd_scalar.to(dtype=state.odd_scalar.dtype)
        )

        even_scalar = (
            (polar_left * polar_right).sum(dim=-1)
            + (axial_left * axial_right).sum(dim=-1)
            + _st_inner(even_left, even_right)
            + _st_inner(odd_left, odd_right)
        )
        odd_scalar = (
            (polar_left * axial_left).sum(dim=-1)
            + _st_inner(even_left, odd_left)
            + static_odd
        )
        axial = (
            torch.cross(polar_left, polar_right, dim=-1)
            + _st_commutator_vector(even_left, even_right)
            + _st_commutator_vector(odd_left, odd_right)
        )
        polar = (
            torch.cross(axial_left, polar_left, dim=-1)
            + _st_commutator_vector(even_left, odd_left)
        )
        even_tensor = (
            _st_cross(polar_left, polar_right)
            + _st_cross(axial_left, axial_right)
            + _st_jordan_product(even_left, even_right)
            + _st_jordan_product(odd_left, odd_right)
        )
        odd_tensor = (
            _st_cross(polar_left, axial_left)
            + _st_jordan_product(even_left, odd_left)
        )

        return _ParityState(
            even_scalar=self.even_scalar_out(even_scalar),
            odd_scalar=self.odd_scalar_out(odd_scalar),
            polar_vector=self.polar_out(polar),
            axial_vector=self.axial_out(axial),
            even_tensor=self.even_tensor_out(even_tensor),
            odd_tensor=self.odd_tensor_out(odd_tensor),
        )


__all__ = [
    "NodeMultipoleBank",
    "NodeMultipoles",
]
