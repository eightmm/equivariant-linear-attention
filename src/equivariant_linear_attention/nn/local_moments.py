"""Compactly supported local geometry cumulants through order four."""

from __future__ import annotations

import torch
from torch import nn

from .geometry import MomentFeatures, compact_third_trace
from .local_support import LocalSupport, wendland_c2
from .ops import (
    SYMMETRIC_DEGREE_SLICES,
    SYMMETRIC_EXPONENTS,
    bounded_compact_stf3,
    bounded_compact_stf4,
    bounded_scalar,
    bounded_st,
    compact_stf3,
    compact_stf4,
    matrix_to_st,
    segment_sum,
    st_commutator_vector,
    st_cross,
    symmetric2_to_matrix,
    symmetric_monomials,
    unit_ball,
    work_dtype,
)
from .state import ChannelMix

_DEGREE4_AXES = tuple(
    tuple([0] * exponent[0] + [1] * exponent[1] + [2] * exponent[2])
    for exponent in SYMMETRIC_EXPONENTS[20:35]
)


def _logit(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(1e-5, 1.0 - 1e-5)
    return torch.log(value) - torch.log1p(-value)


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(value.clamp_min(1e-5)))


def _fourth_cumulant(
    fourth_moment: torch.Tensor,
    covariance: torch.Tensor,
) -> torch.Tensor:
    correction: list[torch.Tensor] = []
    for axes in _DEGREE4_AXES:
        a, b, c, d = axes
        correction.append(
            covariance[..., a, b] * covariance[..., c, d]
            + covariance[..., a, c] * covariance[..., b, d]
            + covariance[..., a, d] * covariance[..., b, c]
        )
    return fourth_moment - torch.stack(correction, dim=-1)


def _fourth_features(
    fourth: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    (
        xxxx,
        xxxy,
        xxxz,
        xxyy,
        xxyz,
        xxzz,
        xyyy,
        xyyz,
        xyzz,
        xzzz,
        yyyy,
        yyyz,
        yyzz,
        yzzz,
        zzzz,
    ) = fourth.unbind(dim=-1)
    trace_xx = xxxx + xxyy + xxzz
    trace_xy = xxxy + xyyy + xyzz
    trace_xz = xxxz + xyyz + xzzz
    trace_yy = xxyy + yyyy + yyzz
    trace_yz = xxyz + yyyz + yzzz
    trace_zz = xxzz + yyzz + zzzz
    trace_matrix = torch.stack(
        (
            torch.stack((trace_xx, trace_xy, trace_xz), dim=-1),
            torch.stack((trace_xy, trace_yy, trace_yz), dim=-1),
            torch.stack((trace_xz, trace_yz, trace_zz), dim=-1),
        ),
        dim=-2,
    )
    scalar = (xxxx + yyyy + zzzz + 2.0 * (xxyy + xxzz + yyzz)) / 5.0
    return (
        bounded_scalar(scalar, eps),
        bounded_st(matrix_to_st(trace_matrix), eps),
        bounded_compact_stf4(compact_stf4(fourth), eps),
    )


class LocalCumulantBank(nn.Module):
    """Learn multi-scale local densities and extract irreducible cumulants."""

    def __init__(self, *, scalar_width: int, rank: int, eps: float) -> None:
        super().__init__()
        if rank < 4:
            raise ValueError("rank must be at least four")
        self.rank = int(rank)
        self.eps = float(eps)
        self.source_weight = nn.Linear(scalar_width, rank)
        self.target_weight = nn.Linear(scalar_width, rank)
        self.radial_weight = nn.Linear(4, rank, bias=False)
        self.raw_temperature = nn.Parameter(torch.zeros(rank))

        relative = torch.linspace(0.20, 0.95, rank)
        absolute = torch.logspace(-0.30, 0.60, rank)
        physical_mix = torch.full((rank,), -4.0)
        physical_mix[rank // 2 :] = 4.0
        self.raw_relative_scale = nn.Parameter(_logit(relative))
        self.raw_absolute_scale = nn.Parameter(_inverse_softplus(absolute))
        self.raw_physical_mix = nn.Parameter(physical_mix)

        self.lane_u = ChannelMix(rank, rank)
        self.lane_v = ChannelMix(rank, rank)
        self.lane_w = ChannelMix(rank, rank)
        self.tensor_u = ChannelMix(rank, rank)
        self.tensor_v = ChannelMix(rank, rank)

    def scales(
        self,
        support_scale: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        relative = 0.05 + 0.95 * torch.sigmoid(self.raw_relative_scale.to(dtype=dtype))
        absolute = (
            torch.nn.functional.softplus(self.raw_absolute_scale.to(dtype=dtype))
            + self.eps
        )
        physical = torch.sigmoid(self.raw_physical_mix.to(dtype=dtype))
        return (
            (1.0 - physical)[None, :] * support_scale[:, None] * relative[None, :]
            + physical[None, :] * absolute[None, :]
        ).clamp_min(self.eps)

    def weights(
        self,
        scalar: torch.Tensor,
        support: LocalSupport,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        dtype = scale.dtype
        adaptive_q = support.distance / support.scale[support.receiver].clamp_min(
            self.eps
        )
        radial = torch.stack(
            (
                adaptive_q,
                adaptive_q.square(),
                torch.log1p(support.distance),
                support.distance / (1.0 + support.distance),
            ),
            dim=-1,
        )
        temperature = (
            torch.nn.functional.softplus(self.raw_temperature.to(dtype=dtype)) + 0.25
        )
        logit = (
            self.source_weight(scalar)[support.source].to(dtype=dtype)
            + self.target_weight(scalar)[support.receiver].to(dtype=dtype)
            + self.radial_weight(radial.to(dtype=scalar.dtype)).to(dtype=dtype)
        ) / temperature
        amplitude = torch.nn.functional.softplus(logit) + self.eps
        outer_window = wendland_c2(adaptive_q)[:, None]
        lane_q = support.distance[:, None] / scale[support.receiver]
        return amplitude * outer_window * wendland_c2(lane_q)

    def forward(
        self,
        scalar: torch.Tensor,
        support: LocalSupport,
    ) -> MomentFeatures:
        dtype = work_dtype(scalar, support.displacement, self.raw_absolute_scale)
        scale = self.scales(support.scale.to(dtype=dtype), dtype=dtype)
        weight = self.weights(scalar, support, scale)
        receiver = support.receiver
        num_nodes = scalar.shape[0]
        displacement = support.displacement[:, None, :] / scale[receiver, :, None]
        mass = segment_sum(weight, receiver, num_nodes).clamp_min(self.eps)
        polar = (
            segment_sum(
                weight[..., None] * displacement,
                receiver,
                num_nodes,
            )
            / mass[..., None]
        )

        central = displacement - polar[receiver]
        monomials = symmetric_monomials(central.reshape(-1, 3)).reshape(
            central.shape[0], central.shape[1], -1
        )
        central_moment = (
            segment_sum(
                weight[..., None] * monomials,
                receiver,
                num_nodes,
            )
            / mass[..., None]
        )
        second = central_moment[..., SYMMETRIC_DEGREE_SLICES[2]]
        third = central_moment[..., SYMMETRIC_DEGREE_SLICES[3]]
        fourth = central_moment[..., SYMMETRIC_DEGREE_SLICES[4]]

        covariance = symmetric2_to_matrix(second)
        second_scalar = covariance.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / 3.0
        even_tensor = matrix_to_st(covariance)
        third_trace = compact_third_trace(third)
        fourth = _fourth_cumulant(fourth, covariance)
        fourth_scalar, fourth_tensor, fourth_rank4 = _fourth_features(
            fourth, eps=self.eps
        )

        u = self.lane_u(polar)
        v = self.lane_v(polar)
        w = self.lane_w(polar)
        tensor_u = self.tensor_u(even_tensor)
        tensor_v = self.tensor_v(even_tensor)
        axial = torch.cross(u, v, dim=-1) + st_commutator_vector(tensor_u, tensor_v)
        odd_scalar = (axial * w).sum(dim=-1)
        odd_tensor = st_cross(w, axial)
        return MomentFeatures(
            mass=torch.log1p(mass),
            polar=unit_ball(polar, self.eps),
            second_scalar=bounded_scalar(second_scalar, self.eps),
            even_tensor=bounded_st(even_tensor, self.eps),
            axial=unit_ball(axial, self.eps),
            odd_scalar=bounded_scalar(odd_scalar, self.eps),
            odd_tensor=bounded_st(odd_tensor, self.eps),
            third_trace=unit_ball(third_trace, self.eps),
            third_tensor=bounded_compact_stf3(compact_stf3(third), self.eps),
            fourth_scalar=fourth_scalar,
            fourth_tensor=fourth_tensor,
            fourth_rank4=fourth_rank4,
        )


__all__ = ["LocalCumulantBank"]
