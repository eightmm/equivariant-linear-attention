"""Soft-local equivariant geometry without explicit edges or pair states."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, factorial

import torch
from torch import nn

from .geometry import GeometryContext, MomentFeatures
from .ops import (
    SYMMETRIC_DEGREES,
    SYMMETRIC_EXPONENTS,
    SYMMETRIC_MULTINOMIAL_SQRT,
    bounded_scalar,
    bounded_st,
    matrix_to_st,
    segment_sum,
    st_cross,
    symmetric2_to_matrix,
    symmetric_monomials,
    unit_ball,
    work_dtype,
)

_LOCAL_BASIS_SIZE = 10  # complete symmetric Cartesian basis through degree two
_LOCAL_EXPONENTS = SYMMETRIC_EXPONENTS[:_LOCAL_BASIS_SIZE]


def _degree_two_translation_table() -> tuple[tuple[int, ...], ...]:
    exponent_to_index = {
        exponent: index for index, exponent in enumerate(_LOCAL_EXPONENTS)
    }
    target: list[int] = []
    source: list[int] = []
    shift: list[int] = []
    coefficient: list[float] = []
    for target_index, alpha in enumerate(_LOCAL_EXPONENTS):
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
    _LOCAL_TRANSLATION_TARGET,
    _LOCAL_TRANSLATION_SOURCE,
    _LOCAL_TRANSLATION_SHIFT,
    _LOCAL_TRANSLATION_COEFFICIENT,
) = _degree_two_translation_table()


@dataclass(frozen=True)
class LocalMomentFeatures:
    """Receiver-local density moments and their low-order correlations."""

    mass: torch.Tensor
    polar: torch.Tensor
    second_scalar: torch.Tensor
    even_tensor: torch.Tensor
    pair_scalar: torch.Tensor
    pair_tensor: torch.Tensor
    axial: torch.Tensor
    odd_scalar: torch.Tensor
    odd_tensor: torch.Tensor


class _LaneMix(nn.Module):
    """Mix multiplicity lanes without touching geometric axes."""

    def __init__(self, rank: int, *, shift: int) -> None:
        super().__init__()
        initial = torch.eye(rank).roll(shifts=shift, dims=0)
        self.weight = nn.Parameter(initial)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.einsum(
            "or,nr...->no...",
            self.weight.to(dtype=value.dtype),
            value,
        )


class LocalMercerMomentBank(nn.Module):
    """Gaussian-soft local moments through degree two in node-linear memory.

    A complete degree-two Gaussian Mercer map supplies the scalar receiver-source
    kernel. Weighted feature-monomial summaries are reduced once per interaction
    segment, then contracted at each receiver. This yields receiver-dependent
    local moments without materializing an ``N x N`` matrix or an edge list.
    """

    def __init__(self, *, scalar_width: int, rank: int, eps: float) -> None:
        super().__init__()
        if rank < 4:
            raise ValueError("local moment rank must be at least four")
        self.rank = int(rank)
        self.eps = float(eps)
        self.content_weight = nn.Linear(scalar_width, rank)
        self.radial_weight = nn.Linear(3, rank, bias=False)
        self.raw_temperature = nn.Parameter(torch.zeros(rank))
        # Normalized-coordinate bandwidths span broad to sharply local kernels.
        self.raw_gamma = nn.Parameter(torch.linspace(-1.5, 2.0, rank))
        self.lane_u = _LaneMix(rank, shift=0)
        self.lane_v = _LaneMix(rank, shift=1)
        self.lane_w = _LaneMix(rank, shift=2)

        degrees = SYMMETRIC_DEGREES[:_LOCAL_BASIS_SIZE]
        self.register_buffer(
            "degree",
            torch.tensor(degrees, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "degree_factorial",
            torch.tensor(
                tuple(float(factorial(degree)) for degree in degrees),
                dtype=torch.float64,
            ),
            persistent=False,
        )
        self.register_buffer(
            "multinomial_sqrt",
            torch.tensor(
                SYMMETRIC_MULTINOMIAL_SQRT[:_LOCAL_BASIS_SIZE],
                dtype=torch.float64,
            ),
            persistent=False,
        )
        self.register_buffer(
            "translation_target",
            torch.tensor(_LOCAL_TRANSLATION_TARGET, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "translation_source",
            torch.tensor(_LOCAL_TRANSLATION_SOURCE, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "translation_shift",
            torch.tensor(_LOCAL_TRANSLATION_SHIFT, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "translation_coefficient",
            torch.tensor(_LOCAL_TRANSLATION_COEFFICIENT, dtype=torch.float64),
            persistent=False,
        )

    def bandwidth(self) -> torch.Tensor:
        """Positive normalized-coordinate inverse length scales, one per lane."""

        return torch.nn.functional.softplus(self.raw_gamma) + 0.05

    def feature_map(
        self,
        position: torch.Tensor,
        monomials: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the complete degree-two Gaussian Mercer map ``[N,R,10]``."""

        if position.ndim != 2 or position.shape[-1] != 3:
            raise ValueError("position must have shape (N,3)")
        dtype = work_dtype(position, self.raw_gamma)
        coordinate = position.to(dtype=dtype)
        if monomials is None:
            monomials = symmetric_monomials(coordinate)
        basis = monomials[:, :_LOCAL_BASIS_SIZE].to(dtype=dtype)
        gamma = torch.nn.functional.softplus(
            self.raw_gamma.to(dtype=dtype)
        ) + 0.05
        radius2 = coordinate.square().sum(dim=-1)
        base = torch.exp(-radius2[:, None] * gamma[None, :])
        degree = self.degree.to(dtype=dtype)
        factorial_value = self.degree_factorial.to(dtype=dtype)
        multinomial = self.multinomial_sqrt.to(dtype=dtype)
        coefficient = (
            torch.sqrt(
                (2.0 * gamma[:, None]).pow(degree[None, :])
                / factorial_value[None, :]
            )
            * multinomial[None, :]
        )
        return base[..., None] * coefficient[None, :, :] * basis[:, None, :]

    def _centered_moments(
        self,
        raw: torch.Tensor,
        receiver_basis: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mass = raw[..., 0].clamp_min(self.eps)
        contribution = (
            raw[:, :, self.translation_source]
            * receiver_basis[:, self.translation_shift].unsqueeze(1)
            * self.translation_coefficient.to(dtype=raw.dtype).reshape(1, 1, -1)
        )
        scatter_index = self.translation_target.reshape(1, 1, -1).expand(
            raw.shape[0],
            raw.shape[1],
            -1,
        )
        centered = raw.new_zeros(raw.shape).scatter_add(
            2,
            scatter_index,
            contribution,
        )
        return mass, centered / mass[..., None]

    def forward(
        self,
        scalar: torch.Tensor,
        geometry: GeometryContext,
    ) -> LocalMomentFeatures:
        dtype = work_dtype(scalar, geometry.normalized, self.raw_gamma)
        position = geometry.normalized.to(dtype=dtype)
        basis = geometry.monomials[:, :_LOCAL_BASIS_SIZE].to(dtype=dtype)
        feature = self.feature_map(position, basis)

        radius2 = position.square().sum(dim=-1)
        radial = torch.stack(
            (
                radius2,
                torch.log1p(radius2),
                radius2 / (1.0 + radius2),
            ),
            dim=-1,
        )
        temperature = torch.nn.functional.softplus(
            self.raw_temperature.to(dtype=dtype)
        ) + 0.25
        content_logit = self.content_weight(scalar).to(dtype=dtype)
        radial_logit = self.radial_weight(
            radial.to(dtype=self.radial_weight.weight.dtype)
        ).to(dtype=dtype)
        source_weight = torch.nn.functional.softplus(
            (content_logit + radial_logit) / temperature
        ) + self.eps

        # S[g,r,p,q] = sum_j w[j,r] phi[j,r,p] psi[j,q].
        packed = (
            source_weight[..., None, None]
            * feature[..., :, None]
            * basis[:, None, None, :]
        )
        summary = segment_sum(
            packed,
            geometry.index,
            geometry.num_segments,
        )
        # Receiver contraction produces sum_j w_j K_r(i,j) psi(x_j).
        raw = torch.einsum(
            "nrp,nrpq->nrq",
            feature,
            summary[geometry.index],
        )
        mass, centered = self._centered_moments(raw, basis)
        m1 = centered[..., 1:4]
        m2 = centered[..., 4:10]
        second_matrix = symmetric2_to_matrix(m2)
        second_scalar = (
            second_matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / 3.0
        )
        even_tensor = matrix_to_st(second_matrix)

        # Aggregation precedes products: these terms implicitly sum over local
        # neighbor pairs/triples without enumerating explicit tuples.
        u = self.lane_u(m1)
        v = self.lane_v(m1)
        w = self.lane_w(m1)
        pair_scalar = (u * v).sum(dim=-1)
        pair_tensor = st_cross(u, v)
        axial = torch.cross(u, v, dim=-1)
        odd_scalar = (axial * w).sum(dim=-1)
        odd_tensor = st_cross(w, axial)

        return LocalMomentFeatures(
            mass=torch.log1p(mass),
            polar=unit_ball(m1, self.eps),
            second_scalar=bounded_scalar(second_scalar, self.eps),
            even_tensor=bounded_st(even_tensor, self.eps),
            pair_scalar=bounded_scalar(pair_scalar, self.eps),
            pair_tensor=bounded_st(pair_tensor, self.eps),
            axial=unit_ball(axial, self.eps),
            odd_scalar=bounded_scalar(odd_scalar, self.eps),
            odd_tensor=bounded_st(odd_tensor, self.eps),
        )


class LocalMomentFusion(nn.Module):
    """Inject soft-local carriers into the existing global moment interface."""

    def __init__(self, *, rank: int) -> None:
        super().__init__()
        if rank < 4:
            raise ValueError("moment rank must be at least four")
        # Zero initialization preserves the previous ELA operator exactly at
        # initialization while allowing the learned gates to open the local path
        # during training.
        self.raw_gate = nn.Parameter(torch.zeros(9, rank))

    def forward(
        self,
        global_moments: MomentFeatures,
        local_moments: LocalMomentFeatures,
    ) -> MomentFeatures:
        gate = torch.tanh(self.raw_gate).to(dtype=global_moments.mass.dtype)
        mass = global_moments.mass + gate[0] * local_moments.mass
        polar = global_moments.polar + gate[1, :, None] * local_moments.polar
        second_scalar = (
            global_moments.second_scalar
            + gate[2] * local_moments.second_scalar
            + gate[4] * local_moments.pair_scalar
        )
        even_tensor = (
            global_moments.even_tensor
            + gate[3, :, None] * local_moments.even_tensor
            + gate[5, :, None] * local_moments.pair_tensor
        )
        axial = global_moments.axial + gate[6, :, None] * local_moments.axial
        odd_scalar = global_moments.odd_scalar + gate[7] * local_moments.odd_scalar
        odd_tensor = (
            global_moments.odd_tensor
            + gate[8, :, None] * local_moments.odd_tensor
        )
        return MomentFeatures(
            mass=mass,
            polar=polar,
            second_scalar=second_scalar,
            even_tensor=even_tensor,
            axial=axial,
            odd_scalar=odd_scalar,
            odd_tensor=odd_tensor,
            third_tensor=global_moments.third_tensor,
            fourth_scalar=global_moments.fourth_scalar,
            fourth_tensor=global_moments.fourth_tensor,
            fourth_rank4=global_moments.fourth_rank4,
        )


__all__ = [
    "LocalMercerMomentBank",
    "LocalMomentFeatures",
    "LocalMomentFusion",
]
