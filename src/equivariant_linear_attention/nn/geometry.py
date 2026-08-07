"""Edge-free relative moments through fourth order."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .ops import (
    bounded_scalar,
    bounded_st,
    bounded_stf3,
    bounded_stf4,
    centered_geometry,
    matrix_to_st,
    segment_sum,
    st_cross,
    stf3,
    stf4,
    work_dtype,
)


@dataclass(frozen=True)
class GeometryContext:
    positions: torch.Tensor
    centered: torch.Tensor
    normalized: torch.Tensor
    radius: torch.Tensor
    index: torch.Tensor
    num_segments: int
    counts: torch.Tensor

    @classmethod
    def build(
        cls,
        positions: torch.Tensor,
        index: torch.Tensor,
        *,
        num_segments: int,
        eps: float,
    ) -> GeometryContext:
        centered, radius, normalized = centered_geometry(
            positions,
            index,
            num_segments,
            eps=eps,
        )
        return cls(
            positions=positions,
            centered=centered,
            normalized=normalized,
            radius=radius,
            index=index.to(dtype=torch.long),
            num_segments=num_segments,
            counts=torch.bincount(index.to(dtype=torch.long), minlength=num_segments),
        )


@dataclass(frozen=True)
class MomentFeatures:
    mass: torch.Tensor
    polar: torch.Tensor
    second_scalar: torch.Tensor
    even_tensor: torch.Tensor
    axial: torch.Tensor
    odd_scalar: torch.Tensor
    odd_tensor: torch.Tensor
    third_tensor: torch.Tensor
    fourth_scalar: torch.Tensor
    fourth_tensor: torch.Tensor
    fourth_rank4: torch.Tensor


def _powers(position: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    p2 = torch.einsum("na,nb->nab", position, position)
    p3 = torch.einsum("na,nb,nc->nabc", position, position, position)
    p4 = torch.einsum("na,nb,nc,nd->nabcd", position, position, position, position)
    return position, p2, p3, p4


def explicit_centered_moments(
    position: torch.Tensor,
    weight: torch.Tensor,
    index: torch.Tensor,
    num_segments: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact weighted means of ``(x_j-x_i)^k`` for k=1..4.

    ``weight`` has shape ``(N,R)`` and is source-only. The identities are exact
    for those separable weights and use only graphwise raw sums.
    """

    dtype = work_dtype(position, weight)
    x = position.to(dtype=dtype)
    w = weight.to(dtype=dtype)
    p1, p2, p3, p4 = _powers(x)
    index = index.to(dtype=torch.long)

    s0 = segment_sum(w, index, num_segments)[index]
    s1 = segment_sum(w[..., None] * p1[:, None, :], index, num_segments)[index]
    s2 = segment_sum(w[..., None, None] * p2[:, None, :, :], index, num_segments)[index]
    s3 = segment_sum(w[..., None, None, None] * p3[:, None, :, :, :], index, num_segments)[index]
    s4 = segment_sum(w[..., None, None, None, None] * p4[:, None, :, :, :, :], index, num_segments)[index]

    xr = x[:, None, :]
    x2 = p2[:, None, :, :]
    x3 = p3[:, None, :, :, :]
    x4 = p4[:, None, :, :, :, :]

    m1 = s1 - s0[..., None] * xr

    m2 = (
        s2
        - torch.einsum("nra,nrb->nrab", xr, s1)
        - torch.einsum("nra,nrb->nrab", s1, xr)
        + s0[..., None, None] * x2
    )

    first3 = (
        torch.einsum("nra,nrbc->nrabc", xr, s2)
        + torch.einsum("nrb,nrac->nrabc", xr, s2)
        + torch.einsum("nrc,nrab->nrabc", xr, s2)
    )
    second3 = (
        torch.einsum("nra,nrb,nrc->nrabc", xr, xr, s1)
        + torch.einsum("nra,nrc,nrb->nrabc", xr, xr, s1)
        + torch.einsum("nrb,nrc,nra->nrabc", xr, xr, s1)
    )
    m3 = s3 - first3 + second3 - s0[..., None, None, None] * x3

    first4 = (
        torch.einsum("nra,nrbcd->nrabcd", xr, s3)
        + torch.einsum("nrb,nracd->nrabcd", xr, s3)
        + torch.einsum("nrc,nrabd->nrabcd", xr, s3)
        + torch.einsum("nrd,nrabc->nrabcd", xr, s3)
    )
    second4 = (
        torch.einsum("nra,nrb,nrcd->nrabcd", xr, xr, s2)
        + torch.einsum("nra,nrc,nrbd->nrabcd", xr, xr, s2)
        + torch.einsum("nra,nrd,nrbc->nrabcd", xr, xr, s2)
        + torch.einsum("nrb,nrc,nrad->nrabcd", xr, xr, s2)
        + torch.einsum("nrb,nrd,nrac->nrabcd", xr, xr, s2)
        + torch.einsum("nrc,nrd,nrab->nrabcd", xr, xr, s2)
    )
    third4 = (
        torch.einsum("nra,nrb,nrc,nrd->nrabcd", xr, xr, xr, s1)
        + torch.einsum("nra,nrb,nrd,nrc->nrabcd", xr, xr, xr, s1)
        + torch.einsum("nra,nrc,nrd,nrb->nrabcd", xr, xr, xr, s1)
        + torch.einsum("nrb,nrc,nrd,nra->nrabcd", xr, xr, xr, s1)
    )
    m4 = s4 - first4 + second4 - third4 + s0[..., None, None, None, None] * x4

    denominator = s0.clamp_min(torch.finfo(dtype).eps)
    m1 = m1 / denominator[..., None]
    m2 = m2 / denominator[..., None, None]
    m3 = m3 / denominator[..., None, None, None]
    m4 = m4 / denominator[..., None, None, None, None]
    return s0, m1, m2, m3, m4


class AdaptiveMomentBank(nn.Module):
    """Learn separable positive source lanes and exact moments through order four."""

    def __init__(self, *, scalar_width: int, rank: int, eps: float) -> None:
        super().__init__()
        if rank < 4:
            raise ValueError("moment rank must be at least four")
        self.rank = rank
        self.eps = float(eps)
        self.content_weight = nn.Linear(scalar_width, rank)
        self.radial_weight = nn.Linear(3, rank, bias=False)
        self.raw_temperature = nn.Parameter(torch.zeros(rank))

    def forward(self, scalar: torch.Tensor, geometry: GeometryContext) -> MomentFeatures:
        radius2 = geometry.normalized.square().sum(dim=-1)
        radial = torch.stack((radius2, torch.log1p(radius2), radius2 / (1.0 + radius2)), dim=-1)
        temperature = torch.nn.functional.softplus(self.raw_temperature) + 0.25
        log_weight = (self.content_weight(scalar) + self.radial_weight(radial)) / temperature
        weight = torch.nn.functional.softplus(log_weight) + self.eps
        mass, m1, m2, m3, m4 = explicit_centered_moments(
            geometry.normalized,
            weight,
            geometry.index,
            geometry.num_segments,
        )

        trace2 = m2.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / 3.0
        even_tensor = matrix_to_st(m2)
        third_tensor = bounded_stf3(stf3(m3), self.eps)
        fourth_scalar = torch.einsum("...aabb->...", m4) / 5.0
        fourth_trace = torch.einsum("...aabc->...bc", m4)
        fourth_tensor = matrix_to_st(fourth_trace)
        fourth_rank4 = bounded_stf4(stf4(m4), self.eps)

        second = m1.roll(1, dims=1)
        third = m1.roll(2, dims=1)
        axial = torch.cross(m1, second, dim=-1)
        odd_scalar = (axial * third).sum(dim=-1)
        odd_tensor = st_cross(third, axial)
        return MomentFeatures(
            mass=torch.log1p(mass),
            polar=m1,
            second_scalar=bounded_scalar(trace2, self.eps),
            even_tensor=bounded_st(even_tensor, self.eps),
            axial=axis_bounded(axial, self.eps),
            odd_scalar=bounded_scalar(odd_scalar, self.eps),
            odd_tensor=bounded_st(odd_tensor, self.eps),
            third_tensor=third_tensor,
            fourth_scalar=bounded_scalar(fourth_scalar, self.eps),
            fourth_tensor=bounded_st(fourth_tensor, self.eps),
            fourth_rank4=fourth_rank4,
        )


def axis_bounded(value: torch.Tensor, eps: float) -> torch.Tensor:
    return value / torch.sqrt(1.0 + value.square().sum(dim=-1, keepdim=True) + eps)


__all__ = [
    "AdaptiveMomentBank",
    "GeometryContext",
    "MomentFeatures",
    "explicit_centered_moments",
]
