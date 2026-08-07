"""Fused self-adjoint relation operators and packed Krylov filtering."""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import torch
from torch import nn

from .geometry import GeometryContext
from .ops import (
    SYMMETRIC_DEGREES,
    SYMMETRIC_MULTINOMIAL_SQRT,
    segment_mean,
    segment_sum,
    st_inner,
    st_orthonormal,
    symmetric2_to_matrix,
    symmetric_monomials,
    unit_ball,
)
from .state import ChannelMix, PairedChannelMix, ParityState


class RelationMessage:
    """All transported irreps packed into one ``[N,H,D]`` tensor.

    The scalar value width is retained as metadata; all other sector sizes are
    fixed by the persistent carrier. Packing lets one scalar relation operator
    transport every sector with a single segmented Gram contraction.
    """

    def __init__(
        self,
        scalar: torch.Tensor,
        odd_scalar: torch.Tensor,
        polar_vector: torch.Tensor,
        axial_vector: torch.Tensor,
        even_tensor: torch.Tensor,
        odd_tensor: torch.Tensor,
    ) -> None:
        if scalar.ndim != 3:
            raise ValueError("scalar must have shape (N,H,D)")
        nodes, heads, head_dim = scalar.shape
        expected = {
            "odd_scalar": (nodes, heads),
            "polar_vector": (nodes, heads, 3),
            "axial_vector": (nodes, heads, 3),
            "even_tensor": (nodes, heads, 5),
            "odd_tensor": (nodes, heads, 5),
        }
        values = {
            "odd_scalar": odd_scalar,
            "polar_vector": polar_vector,
            "axial_vector": axial_vector,
            "even_tensor": even_tensor,
            "odd_tensor": odd_tensor,
        }
        for name, shape in expected.items():
            if values[name].shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        self.head_dim = int(head_dim)
        self.data = torch.cat(
            (
                scalar,
                odd_scalar.unsqueeze(-1),
                polar_vector,
                axial_vector,
                even_tensor,
                odd_tensor,
            ),
            dim=-1,
        )

    @classmethod
    def from_packed(cls, data: torch.Tensor, *, head_dim: int) -> RelationMessage:
        if data.ndim != 3 or data.shape[-1] != head_dim + 17:
            raise ValueError("packed relation data has an invalid final dimension")
        scalar, odd, polar, axial, even_tensor, odd_tensor = torch.split(
            data,
            (head_dim, 1, 3, 3, 5, 5),
            dim=-1,
        )
        return cls(
            scalar,
            odd.squeeze(-1),
            polar,
            axial,
            even_tensor,
            odd_tensor,
        )

    @property
    def scalar(self) -> torch.Tensor:
        return self.data[..., : self.head_dim]

    @property
    def odd_scalar(self) -> torch.Tensor:
        return self.data[..., self.head_dim]

    @property
    def polar_vector(self) -> torch.Tensor:
        start = self.head_dim + 1
        return self.data[..., start : start + 3]

    @property
    def axial_vector(self) -> torch.Tensor:
        start = self.head_dim + 4
        return self.data[..., start : start + 3]

    @property
    def even_tensor(self) -> torch.Tensor:
        start = self.head_dim + 7
        return self.data[..., start : start + 5]

    @property
    def odd_tensor(self) -> torch.Tensor:
        start = self.head_dim + 12
        return self.data[..., start : start + 5]

    def as_tuple(self) -> tuple[torch.Tensor, ...]:
        return (
            self.scalar,
            self.odd_scalar,
            self.polar_vector,
            self.axial_vector,
            self.even_tensor,
            self.odd_tensor,
        )

    def add(self, other: RelationMessage) -> RelationMessage:
        self._check_compatible(other)
        return RelationMessage.from_packed(
            self.data + other.data,
            head_dim=self.head_dim,
        )

    def subtract(self, other: RelationMessage) -> RelationMessage:
        self._check_compatible(other)
        return RelationMessage.from_packed(
            self.data - other.data,
            head_dim=self.head_dim,
        )

    def scale(self, scale: torch.Tensor) -> RelationMessage:
        suffix = (1,) * (self.data.ndim - scale.ndim)
        return RelationMessage.from_packed(
            self.data * scale.reshape(*scale.shape, *suffix),
            head_dim=self.head_dim,
        )

    def _check_compatible(self, other: RelationMessage) -> None:
        if self.head_dim != other.head_dim or self.data.shape != other.data.shape:
            raise ValueError("relation messages are not compatible")


class ValueProjection(nn.Module):
    """Standalone packed value projection retained for advanced use."""

    def __init__(self, *, scalar_width: int, num_heads: int) -> None:
        super().__init__()
        if scalar_width % num_heads:
            raise ValueError("scalar_width must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = scalar_width // num_heads
        self.scalar = nn.Linear(scalar_width, scalar_width)
        self.odd = ChannelMix(num_heads, num_heads)
        self.vector = PairedChannelMix(num_heads, num_heads)
        self.tensor = PairedChannelMix(num_heads, num_heads)

    def forward(self, state: ParityState) -> RelationMessage:
        polar, axial = self.vector(state.polar_vector, state.axial_vector)
        even_tensor, odd_tensor = self.tensor(state.even_tensor, state.odd_tensor)
        return RelationMessage(
            self.scalar(state.even_scalar).reshape(
                state.num_nodes,
                self.num_heads,
                self.head_dim,
            ),
            self.odd(state.odd_scalar),
            polar,
            axial,
            even_tensor,
            odd_tensor,
        )


class ContentGramFeatures(nn.Module):
    """Standalone all-irrep invariant Gram feature projection."""

    def __init__(
        self, *, scalar_width: int, num_heads: int, feature_width: int
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.feature_width = feature_width
        self.scalar = nn.Linear(scalar_width, num_heads * feature_width)
        self.odd = ChannelMix(num_heads, num_heads)
        self.vector = PairedChannelMix(num_heads, num_heads)
        self.tensor = PairedChannelMix(num_heads, num_heads)

    def forward(self, state: ParityState) -> torch.Tensor:
        scalar = self.scalar(state.even_scalar).reshape(
            state.num_nodes,
            self.num_heads,
            self.feature_width,
        )
        polar, axial = self.vector(state.polar_vector, state.axial_vector)
        even_tensor, odd_tensor = self.tensor(state.even_tensor, state.odd_tensor)
        return torch.cat(
            (
                scalar,
                self.odd(state.odd_scalar).unsqueeze(-1),
                unit_ball(polar, 1e-8),
                unit_ball(axial, 1e-8),
                st_orthonormal(even_tensor),
                st_orthonormal(odd_tensor),
            ),
            dim=-1,
        )


class RelationProjection(nn.Module):
    """Fuse value and content projections into three large tensor operations."""

    def __init__(
        self, *, scalar_width: int, num_heads: int, feature_width: int
    ) -> None:
        super().__init__()
        if scalar_width % num_heads:
            raise ValueError("scalar_width must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = scalar_width // num_heads
        self.feature_width = feature_width
        self.scalar = nn.Linear(
            scalar_width,
            scalar_width + num_heads * feature_width,
        )
        self.odd = ChannelMix(num_heads, 2 * num_heads)
        self.vector = PairedChannelMix(num_heads, 2 * num_heads)
        self.tensor = PairedChannelMix(num_heads, 2 * num_heads)

    def forward(self, state: ParityState) -> tuple[RelationMessage, torch.Tensor]:
        scalar = self.scalar(state.even_scalar)
        value_scalar, content_scalar = torch.split(
            scalar,
            (self.num_heads * self.head_dim, self.num_heads * self.feature_width),
            dim=-1,
        )
        odd = self.odd(state.odd_scalar)
        value_odd, content_odd = odd.split(self.num_heads, dim=1)
        polar, axial = self.vector(state.polar_vector, state.axial_vector)
        value_polar, content_polar = polar.split(self.num_heads, dim=1)
        value_axial, content_axial = axial.split(self.num_heads, dim=1)
        even_tensor, odd_tensor = self.tensor(state.even_tensor, state.odd_tensor)
        value_even_tensor, content_even_tensor = even_tensor.split(
            self.num_heads, dim=1
        )
        value_odd_tensor, content_odd_tensor = odd_tensor.split(self.num_heads, dim=1)

        value = RelationMessage(
            value_scalar.reshape(state.num_nodes, self.num_heads, self.head_dim),
            value_odd,
            value_polar,
            value_axial,
            value_even_tensor,
            value_odd_tensor,
        )
        content = torch.cat(
            (
                content_scalar.reshape(
                    state.num_nodes,
                    self.num_heads,
                    self.feature_width,
                ),
                content_odd.unsqueeze(-1),
                unit_ball(content_polar, 1e-8),
                unit_ball(content_axial, 1e-8),
                st_orthonormal(content_even_tensor),
                st_orthonormal(content_odd_tensor),
            ),
            dim=-1,
        )
        return value, content


class MercerFeatures(nn.Module):
    """Compact 35-dimensional Gaussian Mercer feature map through degree four."""

    def __init__(self, *, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.raw_gamma = nn.Parameter(torch.linspace(-1.0, 1.0, num_heads))
        self.register_buffer(
            "degree",
            torch.tensor(SYMMETRIC_DEGREES, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "degree_factorial",
            torch.tensor(
                tuple(float(factorial(degree)) for degree in SYMMETRIC_DEGREES),
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "multinomial_sqrt",
            torch.tensor(SYMMETRIC_MULTINOMIAL_SQRT, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        position: torch.Tensor,
        monomials: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gamma = torch.nn.functional.softplus(self.raw_gamma) + 0.05
        coordinate = position
        if monomials is None:
            monomials = symmetric_monomials(coordinate)
        monomials = monomials.to(dtype=coordinate.dtype)
        radius2 = coordinate.square().sum(dim=-1)
        base = torch.exp(-radius2[:, None] * gamma[None, :])
        degree = self.degree.to(dtype=coordinate.dtype)
        factorial_value = self.degree_factorial.to(dtype=coordinate.dtype)
        multinomial = self.multinomial_sqrt.to(dtype=coordinate.dtype)
        coefficient = (
            torch.sqrt(
                (2.0 * gamma[:, None]).pow(degree[None, :]) / factorial_value[None, :]
            )
            * multinomial[None, :]
        )
        return base[..., None] * coefficient[None, :, :] * monomials[:, None, :]


@dataclass(frozen=True)
class AtlasFactors:
    assignment: torch.Tensor
    mass: torch.Tensor
    metric: torch.Tensor
    chart_weight: torch.Tensor
    node_metric: torch.Tensor
    effective_dimension: torch.Tensor


class ManifoldAtlas(nn.Module):
    """Partition of unity with two packed chart-statistic reductions."""

    def __init__(
        self, *, scalar_width: int, num_heads: int, num_charts: int, eps: float
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_charts = num_charts
        self.eps = float(eps)
        self.logits = nn.Linear(scalar_width, num_charts)
        self.radial_logits = nn.Linear(3, num_charts, bias=False)
        self.raw_distance_scale = nn.Parameter(torch.zeros(num_charts))
        self.raw_ridge = nn.Parameter(torch.full((num_charts,), -2.0))
        self.head_bias = nn.Parameter(torch.zeros(num_heads, num_charts))
        self.head_slope = nn.Parameter(torch.zeros(num_heads, num_charts))

    def _statistics(
        self,
        assignment: torch.Tensor,
        geometry: GeometryContext,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [1, x, sym(x x)] is the first ten entries of the shared compact basis.
        basis = geometry.monomials[:, :10].to(dtype=assignment.dtype)
        statistics = segment_sum(
            assignment[..., None] * basis[:, None, :],
            geometry.index,
            geometry.num_segments,
        )
        mass = statistics[..., 0].clamp_min(self.eps)
        center = statistics[..., 1:4] / mass[..., None]
        second = statistics[..., 4:10] / mass[..., None]
        raw_second = symmetric2_to_matrix(second)
        covariance = raw_second - torch.einsum("gka,gkb->gkab", center, center)
        covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
        return mass, center, covariance

    def forward(self, state: ParityState, geometry: GeometryContext) -> AtlasFactors:
        position = geometry.normalized.to(dtype=state.even_scalar.dtype)
        radius2 = position.square().sum(dim=-1)
        radial = torch.stack(
            (radius2, torch.log1p(radius2), radius2 / (1.0 + radius2)),
            dim=-1,
        )
        raw = self.logits(state.even_scalar) + self.radial_logits(radial)
        initial = torch.softmax(raw, dim=-1)
        mass, center, covariance = self._statistics(initial, geometry)
        identity = torch.eye(3, device=position.device, dtype=position.dtype)
        trace = covariance.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / 3.0
        ridge = torch.nn.functional.softplus(self.raw_ridge)[None, :] + self.eps
        metric = covariance + (ridge * trace.clamp_min(1.0))[..., None, None] * identity
        delta = position[:, None, :] - center[geometry.index]
        solved = torch.linalg.solve(
            metric[geometry.index], delta.unsqueeze(-1)
        ).squeeze(-1)
        distance2 = (delta * solved).sum(dim=-1).clamp(min=0.0, max=64.0)
        scale = torch.nn.functional.softplus(self.raw_distance_scale)
        balance = torch.log(mass[geometry.index] + self.eps)
        assignment = torch.softmax(
            raw - scale[None, :] * distance2 - 0.25 * balance,
            dim=-1,
        )
        mass, _, covariance = self._statistics(assignment, geometry)
        trace = covariance.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / 3.0
        metric = (
            covariance
            + (
                (torch.nn.functional.softplus(self.raw_ridge)[None, :] + self.eps)
                * trace.clamp_min(1.0)
            )[..., None, None]
            * identity
        )
        metric_scale = metric.diagonal(dim1=-2, dim2=-1).sum(dim=-1).div(3.0)
        metric = metric / metric_scale.clamp_min(self.eps)[..., None, None]
        covariance_trace = covariance.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        covariance_square = covariance.square().sum(dim=(-2, -1))
        effective_dimension = (
            covariance_trace.square() / (covariance_square + self.eps)
        ).clamp(1.0, 3.0)
        chart_weight = torch.sigmoid(
            self.head_bias[None, :, :]
            + self.head_slope[None, :, :] * (effective_dimension[:, None, :] - 2.0)
        )
        node_metric = (assignment[..., None, None] * metric[geometry.index]).sum(dim=1)
        return AtlasFactors(
            assignment=assignment,
            mass=mass,
            metric=metric,
            chart_weight=chart_weight,
            node_metric=node_metric,
            effective_dimension=effective_dimension,
        )

    def feature(self, factors: AtlasFactors, index: torch.Tensor) -> torch.Tensor:
        scale = torch.sqrt(
            factors.chart_weight / factors.mass[:, None, :].clamp_min(self.eps)
        )
        return factors.assignment[:, None, :] * scale[index]

    def apply(
        self,
        factors: AtlasFactors,
        value: torch.Tensor,
        index: torch.Tensor,
        num_segments: int,
    ) -> torch.Tensor:
        feature = self.feature(factors, index)
        summary = segment_sum(
            feature[..., None] * value[..., None, :],
            index,
            num_segments,
        )
        return (feature[..., None] * summary[index]).sum(dim=-2)


@dataclass(frozen=True)
class RelationFactors:
    feature: torch.Tensor
    content_feature: torch.Tensor
    content_trace: torch.Tensor
    mercer_feature: torch.Tensor
    mercer_trace: torch.Tensor
    atlas: AtlasFactors
    mixture: torch.Tensor
    index: torch.Tensor
    num_segments: int


class SelfAdjointRelation(nn.Module):
    """One unified Gram factor for content, Mercer, and atlas relations."""

    def __init__(
        self,
        *,
        scalar_width: int,
        num_heads: int,
        feature_width: int,
        num_charts: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.eps = float(eps)
        self.projection = RelationProjection(
            scalar_width=scalar_width,
            num_heads=num_heads,
            feature_width=feature_width,
        )
        self.mercer = MercerFeatures(num_heads=num_heads)
        self.atlas = ManifoldAtlas(
            scalar_width=scalar_width,
            num_heads=num_heads,
            num_charts=num_charts,
            eps=eps,
        )
        self.mixture = nn.Linear(scalar_width, num_heads * 3)

    def project(self, state: ParityState) -> tuple[RelationMessage, torch.Tensor]:
        return self.projection(state)

    def build(
        self,
        state: ParityState,
        geometry: GeometryContext,
        *,
        content_feature: torch.Tensor | None = None,
    ) -> RelationFactors:
        if content_feature is None:
            _, content_feature = self.project(state)
        content = content_feature
        mercer = self.mercer(
            geometry.normalized.to(dtype=state.even_scalar.dtype),
            geometry.monomials,
        )
        content_trace = segment_sum(
            content.square().sum(dim=-1),
            geometry.index,
            geometry.num_segments,
        ).clamp_min(self.eps)
        mercer_trace = segment_sum(
            mercer.square().sum(dim=-1),
            geometry.index,
            geometry.num_segments,
        ).clamp_min(self.eps)
        atlas = self.atlas(state, geometry)
        graph_scalar = segment_mean(
            state.even_scalar,
            geometry.index,
            geometry.num_segments,
        )
        mixture = torch.softmax(
            self.mixture(graph_scalar).reshape(
                geometry.num_segments,
                self.num_heads,
                3,
            ),
            dim=-1,
        )
        node_mixture = mixture[geometry.index]
        content_scale = torch.sqrt(
            node_mixture[:, :, 0] / content_trace[geometry.index]
        )
        mercer_scale = torch.sqrt(node_mixture[:, :, 1] / mercer_trace[geometry.index])
        atlas_feature = self.atlas.feature(atlas, geometry.index)
        feature = torch.cat(
            (
                content * content_scale[..., None],
                mercer * mercer_scale[..., None],
                atlas_feature * torch.sqrt(node_mixture[:, :, 2, None]),
            ),
            dim=-1,
        )
        return RelationFactors(
            feature=feature,
            content_feature=content,
            content_trace=content_trace,
            mercer_feature=mercer,
            mercer_trace=mercer_trace,
            atlas=atlas,
            mixture=mixture,
            index=geometry.index,
            num_segments=geometry.num_segments,
        )

    @staticmethod
    def _gram_apply(
        feature: torch.Tensor,
        value: torch.Tensor,
        index: torch.Tensor,
        num_segments: int,
    ) -> torch.Tensor:
        summary = segment_sum(
            feature[..., None] * value[..., None, :],
            index,
            num_segments,
        )
        return (feature[..., None] * summary[index]).sum(dim=-2)

    def apply_tensor(
        self, factors: RelationFactors, value: torch.Tensor
    ) -> torch.Tensor:
        return self._gram_apply(
            factors.feature,
            value,
            factors.index,
            factors.num_segments,
        )

    def apply(
        self, factors: RelationFactors, message: RelationMessage
    ) -> RelationMessage:
        return RelationMessage.from_packed(
            self.apply_tensor(factors, message.data),
            head_dim=message.head_dim,
        )


def message_inner(left: RelationMessage, right: RelationMessage) -> torch.Tensor:
    left._check_compatible(right)
    return (
        (left.scalar * right.scalar).mean(dim=-1)
        + left.odd_scalar * right.odd_scalar
        + (left.polar_vector * right.polar_vector).mean(dim=-1)
        + (left.axial_vector * right.axial_vector).mean(dim=-1)
        + st_inner(left.even_tensor, right.even_tensor) / 5.0
        + st_inner(left.odd_tensor, right.odd_tensor) / 5.0
    ) / 6.0


def orthogonalize(
    candidate: RelationMessage,
    bases: tuple[RelationMessage, ...],
    *,
    index: torch.Tensor,
    num_segments: int,
    counts: torch.Tensor,
    eps: float,
) -> RelationMessage:
    if not bases:
        output = candidate
    else:
        # Krylov bases are already mutually orthogonal, so all projection
        # numerators and denominators can be reduced together. This replaces
        # two segment reductions per basis with one packed reduction.
        node_statistics = torch.stack(
            tuple(message_inner(basis, candidate) for basis in bases)
            + tuple(message_inner(basis, basis) for basis in bases),
            dim=-1,
        )
        statistics = segment_sum(node_statistics, index, num_segments)
        basis_count = len(bases)
        numerator = statistics[..., :basis_count]
        denominator = statistics[..., basis_count:].clamp_min(eps)
        coefficient = (numerator / denominator)[index]
        packed_projection = sum(
            (
                basis.data * coefficient[..., basis_index].unsqueeze(-1)
                for basis_index, basis in enumerate(bases)
            ),
            start=torch.zeros_like(candidate.data),
        )
        output = RelationMessage.from_packed(
            candidate.data - packed_projection,
            head_dim=candidate.head_dim,
        )
    norm = segment_sum(
        message_inner(output, output).clamp_min(0.0),
        index,
        num_segments,
    )
    norm = norm / counts.to(dtype=norm.dtype).clamp_min(1.0).unsqueeze(-1)
    inverse = torch.rsqrt(norm + eps).clamp(max=8.0)
    return output.scale(inverse[index])


class KrylovMixer(nn.Module):
    def __init__(self, *, scalar_width: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.coefficient = nn.Linear(scalar_width, num_heads * 2)
        nn.init.zeros_(self.coefficient.weight)
        nn.init.zeros_(self.coefficient.bias)

    def forward(
        self,
        state: ParityState,
        first: RelationMessage,
        second: RelationMessage,
        third: RelationMessage,
        geometry: GeometryContext,
    ) -> RelationMessage:
        graph_scalar = segment_mean(
            state.even_scalar,
            geometry.index,
            geometry.num_segments,
        )
        coefficient = torch.tanh(
            self.coefficient(graph_scalar).reshape(
                geometry.num_segments,
                self.num_heads,
                2,
            )
        )[geometry.index]
        packed = (
            first.data
            + coefficient[:, :, 0, None] * second.data
            + coefficient[:, :, 1, None] * third.data
        )
        return RelationMessage.from_packed(packed, head_dim=first.head_dim)


__all__ = [
    "AtlasFactors",
    "ContentGramFeatures",
    "KrylovMixer",
    "MercerFeatures",
    "RelationFactors",
    "RelationMessage",
    "RelationProjection",
    "SelfAdjointRelation",
    "ValueProjection",
    "message_inner",
    "orthogonalize",
]
