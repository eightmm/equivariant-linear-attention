"""Edge-free relative moments through fourth order in compact symmetric bases."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, factorial

import torch
from torch import nn

from .ops import (
    SYMMETRIC_DEGREE_SLICES,
    SYMMETRIC_DEGREES,
    SYMMETRIC_EXPONENTS,
    SYMMETRIC_MULTINOMIAL_SQRT,
    bounded_compact_stf3,
    bounded_compact_stf4,
    bounded_scalar,
    bounded_st,
    centered_geometry,
    compact_stf3,
    compact_stf4,
    matrix_to_st,
    segment_mean,
    segment_softmax,
    segment_sum,
    st_cross,
    symmetric2_to_matrix,
    symmetric_monomials,
    work_dtype,
)


def greedy_farthest_seeds(
    position: torch.Tensor,
    index: torch.Tensor,
    num_segments: int,
    *,
    num_seeds: int,
    sharpness: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft farthest-point seeds, one spread set of chart anchors per segment.

    A differentiable, O(3)-equivariant relaxation of greedy k-means++ seeding:
    each seed is the soft-argmax-weighted mean of the positions farthest from
    every seed chosen so far. Scores are invariant and normalized by the
    segment's mean square radius, so the sharpness is scale-free; the weighted
    mean of positions carries the rotation. Learned chart logits alone leave
    charts clustered near the centroid, which caps the local kernel's
    selectivity; these seeds supply the spatial spread instead.

    Also returns the mean square distance from a node to its nearest seed,
    which is the natural unit of chart spacing. Expressing the assignment
    sharpness in that unit keeps the partition equally soft on compact and on
    grossly expanded structures, instead of collapsing to a few hard cells
    whenever a decoy is broken.
    """

    scale = segment_mean(
        position.square().sum(dim=-1), index, num_segments
    ).clamp_min(eps)[index]
    seeds: list[torch.Tensor] = []
    nearest2: torch.Tensor | None = None
    for slot in range(num_seeds):
        score = position.square().sum(dim=-1) if slot == 0 else nearest2
        weight = segment_softmax(sharpness * score / scale, index, num_segments)
        seed = segment_sum(weight[:, None] * position, index, num_segments)
        seeds.append(seed)
        distance2 = (position - seed[index]).square().sum(dim=-1)
        nearest2 = (
            distance2 if nearest2 is None else torch.minimum(nearest2, distance2)
        )
    spacing2 = segment_mean(nearest2, index, num_segments).clamp_min(eps)
    return torch.stack(seeds, dim=1), spacing2


def chart_density(
    positions: torch.Tensor,
    index: torch.Tensor,
    num_segments: int,
    *,
    num_charts: int,
    bandwidths: tuple[float, ...],
    length_scale: float,
    seed_sharpness: float = 16.0,
    assignment_scale: float = 3.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Node-linear soft neighbour counts at fixed Angstrom bandwidths.

    Returns ``log1p`` of ``sum_j k(i,j)`` for each bandwidth, where ``k`` is
    the chart-recentered truncated-Gaussian kernel. Because the kernel is a
    feature inner product, the row sum is one segment reduction rather than a
    pairwise scan, so this is an ``O(N G)`` in-model replacement for the dense
    ``O(N^2)`` neighbour counts that the offline diagnostic feature used.

    Charts come from geometry alone, so the channel is available whether or
    not the local relation sector is enabled.
    """

    dtype = work_dtype(positions)
    position = positions.to(dtype=dtype)
    index = index.to(dtype=torch.long)
    center = segment_mean(position, index, num_segments)
    absolute = (position - center[index]) / float(length_scale)
    seeds, spacing2 = greedy_farthest_seeds(
        absolute,
        index,
        num_segments,
        num_seeds=num_charts,
        sharpness=seed_sharpness,
        eps=eps,
    )
    delta = absolute[:, None, :] - seeds.detach()[index]
    distance2 = delta.square().sum(dim=-1)
    assignment = torch.softmax(
        -assignment_scale * distance2 / spacing2.detach()[index, None],
        dim=-1,
    )
    weight = torch.sqrt(assignment + eps)
    monomials = symmetric_monomials(delta.reshape(-1, 3)).reshape(
        delta.shape[0], delta.shape[1], -1
    )
    degree = torch.tensor(SYMMETRIC_DEGREES, device=position.device, dtype=dtype)
    degree_factorial = torch.tensor(
        tuple(float(factorial(value)) for value in SYMMETRIC_DEGREES),
        device=position.device,
        dtype=dtype,
    )
    multinomial = torch.tensor(
        SYMMETRIC_MULTINOMIAL_SQRT, device=position.device, dtype=dtype
    )

    channels: list[torch.Tensor] = []
    for bandwidth in bandwidths:
        gamma = 0.5 / (float(bandwidth) / float(length_scale)) ** 2
        coefficient = (
            torch.sqrt((2.0 * gamma) ** degree / degree_factorial) * multinomial
        )
        feature = (
            weight[..., None]
            * torch.exp(-gamma * distance2)[..., None]
            * coefficient
            * monomials
        )
        summary = segment_sum(feature, index, num_segments)
        total = (feature * summary[index]).sum(dim=(-2, -1))
        # The Gram row sum includes the node's own term; a neighbour count
        # must not.
        total = total - feature.square().sum(dim=(-2, -1))
        channels.append(torch.log1p(total.clamp_min(0.0)))
    return torch.stack(channels, dim=-1)


@dataclass(frozen=True)
class GeometryContext:
    positions: torch.Tensor
    centered: torch.Tensor
    normalized: torch.Tensor
    monomials: torch.Tensor
    radius: torch.Tensor
    index: torch.Tensor
    num_segments: int
    counts: torch.Tensor
    absolute: torch.Tensor
    absolute_radius2: torch.Tensor
    chart_seeds: torch.Tensor | None = None
    chart_spacing2: torch.Tensor | None = None

    @classmethod
    def build(
        cls,
        positions: torch.Tensor,
        index: torch.Tensor,
        *,
        num_segments: int,
        eps: float,
        length_scale: float = 10.0,
        num_seeds: int = 0,
        seed_sharpness: float = 16.0,
    ) -> GeometryContext:
        centered, radius, normalized = centered_geometry(
            positions,
            index,
            num_segments,
            eps=eps,
        )
        basis_dtype = work_dtype(normalized)
        monomials = symmetric_monomials(normalized.to(dtype=basis_dtype))
        absolute = centered / float(length_scale)
        long_index = index.to(dtype=torch.long)
        chart_seeds = None
        chart_spacing2 = None
        if num_seeds > 0:
            # Seeds are a geometric prior, not a learned quantity: detaching
            # keeps the soft-argmax construction out of the backward pass.
            chart_seeds, chart_spacing2 = greedy_farthest_seeds(
                absolute,
                long_index,
                num_segments,
                num_seeds=num_seeds,
                sharpness=seed_sharpness,
                eps=eps,
            )
            chart_seeds = chart_seeds.detach()
            chart_spacing2 = chart_spacing2.detach()
        return cls(
            positions=positions,
            centered=centered,
            normalized=normalized,
            monomials=monomials,
            radius=radius,
            index=long_index,
            num_segments=num_segments,
            counts=torch.bincount(long_index, minlength=num_segments),
            absolute=absolute,
            absolute_radius2=absolute.square().sum(dim=-1),
            chart_seeds=chart_seeds,
            chart_spacing2=chart_spacing2,
        )


def radial_invariants(geometry: GeometryContext, dtype: torch.dtype) -> torch.Tensor:
    """Six invariant radial channels: unit-free shape plus absolute scale.

    The absolute channels are bounded or log-slow: broken structures reach
    per-node absolute radius^2 of ~5e4 (length-scale units), so a raw channel
    would saturate downstream softmax gates before learning.
    """

    normalized2 = geometry.normalized.square().sum(dim=-1).to(dtype=dtype)
    absolute2 = geometry.absolute_radius2.to(dtype=dtype)
    absolute = torch.sqrt(absolute2 + 1e-12)
    return torch.stack(
        (
            normalized2,
            torch.log1p(normalized2),
            normalized2 / (1.0 + normalized2),
            torch.log1p(absolute2),
            absolute2 / (1.0 + absolute2),
            absolute / (1.0 + absolute),
        ),
        dim=-1,
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


def _powers(
    position: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    """Reference full-Cartesian weighted moments through order four.

    This function is retained as an oracle for mathematical tests. The model
    executes :func:`compact_centered_moments`, which uses one packed reduction
    over the 35 independent symmetric monomials.
    """

    dtype = work_dtype(position, weight)
    x = position.to(dtype=dtype)
    w = weight.to(dtype=dtype)
    p1, p2, p3, p4 = _powers(x)
    index = index.to(dtype=torch.long)

    s0 = segment_sum(w, index, num_segments)[index]
    s1 = segment_sum(w[..., None] * p1[:, None, :], index, num_segments)[index]
    s2 = segment_sum(w[..., None, None] * p2[:, None, :, :], index, num_segments)[index]
    s3 = segment_sum(
        w[..., None, None, None] * p3[:, None, :, :, :],
        index,
        num_segments,
    )[index]
    s4 = segment_sum(
        w[..., None, None, None, None] * p4[:, None, :, :, :, :],
        index,
        num_segments,
    )[index]

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
    return (
        s0,
        m1 / denominator[..., None],
        m2 / denominator[..., None, None],
        m3 / denominator[..., None, None, None],
        m4 / denominator[..., None, None, None, None],
    )


def _translation_table() -> tuple[tuple[int, ...], ...]:
    exponent_to_index = {
        exponent: index for index, exponent in enumerate(SYMMETRIC_EXPONENTS)
    }
    target: list[int] = []
    source: list[int] = []
    shift: list[int] = []
    coefficient: list[float] = []
    for target_index, alpha in enumerate(SYMMETRIC_EXPONENTS):
        for beta_x in range(alpha[0] + 1):
            for beta_y in range(alpha[1] + 1):
                for beta_z in range(alpha[2] + 1):
                    beta = (beta_x, beta_y, beta_z)
                    delta = (
                        alpha[0] - beta_x,
                        alpha[1] - beta_y,
                        alpha[2] - beta_z,
                    )
                    target.append(target_index)
                    source.append(exponent_to_index[beta])
                    shift.append(exponent_to_index[delta])
                    sign = -1.0 if sum(delta) % 2 else 1.0
                    coefficient.append(
                        sign
                        * comb(alpha[0], beta_x)
                        * comb(alpha[1], beta_y)
                        * comb(alpha[2], beta_z)
                    )
    return tuple(target), tuple(source), tuple(shift), tuple(coefficient)


(
    _TRANSLATION_TARGET,
    _TRANSLATION_SOURCE,
    _TRANSLATION_SHIFT,
    _TRANSLATION_COEFFICIENT,
) = _translation_table()


def _compact_centered_core(
    monomials: torch.Tensor,
    weight: torch.Tensor,
    index: torch.Tensor,
    num_segments: int,
    *,
    target_index: torch.Tensor,
    source_index: torch.Tensor,
    shift_index: torch.Tensor,
    coefficient: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = work_dtype(monomials, weight)
    basis = monomials.to(dtype=dtype)
    source_weight = weight.to(dtype=dtype)
    index = index.to(dtype=torch.long)
    raw = segment_sum(
        source_weight[..., None] * basis[:, None, :],
        index,
        num_segments,
    )[index]
    contribution = (
        raw[:, :, source_index]
        * basis[:, shift_index].unsqueeze(1)
        * coefficient.to(dtype=dtype).reshape(1, 1, -1)
    )
    scatter_index = target_index.reshape(1, 1, -1).expand(
        raw.shape[0],
        raw.shape[1],
        -1,
    )
    centered = raw.new_zeros(raw.shape).scatter_add(2, scatter_index, contribution)
    mass = raw[..., 0].clamp_min(torch.finfo(dtype).eps)
    return mass, centered / mass[..., None]


def compact_centered_moments(
    position: torch.Tensor,
    weight: torch.Tensor,
    index: torch.Tensor,
    num_segments: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact compact moments via one reduction over 35 symmetric monomials."""

    monomials = symmetric_monomials(position.to(dtype=work_dtype(position, weight)))
    device = position.device
    return _compact_centered_core(
        monomials,
        weight,
        index,
        num_segments,
        target_index=torch.tensor(_TRANSLATION_TARGET, device=device, dtype=torch.long),
        source_index=torch.tensor(_TRANSLATION_SOURCE, device=device, dtype=torch.long),
        shift_index=torch.tensor(_TRANSLATION_SHIFT, device=device, dtype=torch.long),
        coefficient=torch.tensor(_TRANSLATION_COEFFICIENT, device=device),
    )


class AdaptiveMomentBank(nn.Module):
    """Learn source lanes and compute exact compact moments through order four."""

    def __init__(self, *, scalar_width: int, rank: int, eps: float) -> None:
        super().__init__()
        if rank < 4:
            raise ValueError("moment rank must be at least four")
        self.rank = rank
        self.eps = float(eps)
        self.content_weight = nn.Linear(scalar_width, rank)
        self.radial_weight = nn.Linear(6, rank, bias=False)
        self.raw_temperature = nn.Parameter(torch.zeros(rank))
        self.register_buffer(
            "translation_target",
            torch.tensor(_TRANSLATION_TARGET, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "translation_source",
            torch.tensor(_TRANSLATION_SOURCE, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "translation_shift",
            torch.tensor(_TRANSLATION_SHIFT, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "translation_coefficient",
            torch.tensor(_TRANSLATION_COEFFICIENT, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self, scalar: torch.Tensor, geometry: GeometryContext
    ) -> MomentFeatures:
        radial = radial_invariants(geometry, scalar.dtype)
        temperature = torch.nn.functional.softplus(self.raw_temperature) + 0.25
        log_weight = (
            self.content_weight(scalar)
            + self.radial_weight(radial.to(dtype=scalar.dtype))
        ) / temperature
        weight = torch.nn.functional.softplus(log_weight) + self.eps
        mass, centered = _compact_centered_core(
            geometry.monomials,
            weight,
            geometry.index,
            geometry.num_segments,
            target_index=self.translation_target,
            source_index=self.translation_source,
            shift_index=self.translation_shift,
            coefficient=self.translation_coefficient,
        )

        m1 = centered[..., SYMMETRIC_DEGREE_SLICES[1]]
        m2 = centered[..., SYMMETRIC_DEGREE_SLICES[2]]
        m3 = centered[..., SYMMETRIC_DEGREE_SLICES[3]]
        m4 = centered[..., SYMMETRIC_DEGREE_SLICES[4]]

        second_matrix = symmetric2_to_matrix(m2)
        trace2 = second_matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / 3.0
        even_tensor = matrix_to_st(second_matrix)
        third_tensor = bounded_compact_stf3(compact_stf3(m3), self.eps)

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
        ) = m4.unbind(dim=-1)
        trace_xx = xxxx + xxyy + xxzz
        trace_xy = xxxy + xyyy + xyzz
        trace_xz = xxxz + xyyz + xzzz
        trace_yy = xxyy + yyyy + yyzz
        trace_yz = xxyz + yyyz + yzzz
        trace_zz = xxzz + yyzz + zzzz
        fourth_trace_matrix = torch.stack(
            (
                torch.stack((trace_xx, trace_xy, trace_xz), dim=-1),
                torch.stack((trace_xy, trace_yy, trace_yz), dim=-1),
                torch.stack((trace_xz, trace_yz, trace_zz), dim=-1),
            ),
            dim=-2,
        )
        fourth_scalar = (xxxx + yyyy + zzzz + 2.0 * (xxyy + xxzz + yyzz)) / 5.0
        fourth_tensor = matrix_to_st(fourth_trace_matrix)
        fourth_rank4 = bounded_compact_stf4(compact_stf4(m4), self.eps)

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
            axial=axial
            / torch.sqrt(1.0 + axial.square().sum(dim=-1, keepdim=True) + self.eps),
            odd_scalar=bounded_scalar(odd_scalar, self.eps),
            odd_tensor=bounded_st(odd_tensor, self.eps),
            third_tensor=third_tensor,
            fourth_scalar=bounded_scalar(fourth_scalar, self.eps),
            fourth_tensor=bounded_st(fourth_tensor, self.eps),
            fourth_rank4=fourth_rank4,
        )


__all__ = [
    "AdaptiveMomentBank",
    "GeometryContext",
    "MomentFeatures",
    "chart_density",
    "compact_centered_moments",
    "explicit_centered_moments",
    "greedy_farthest_seeds",
    "radial_invariants",
]
