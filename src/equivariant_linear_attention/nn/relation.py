"""Self-adjoint edge-free relation operators and orthogonal Krylov filtering."""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import torch
from torch import nn

from .geometry import GeometryContext
from .ops import (
    segment_mean,
    segment_sum,
    st_inner,
    st_orthonormal,
    unit_ball,
)
from .state import ChannelMix, ParityState


@dataclass(frozen=True)
class RelationMessage:
    scalar: torch.Tensor
    odd_scalar: torch.Tensor
    polar_vector: torch.Tensor
    axial_vector: torch.Tensor
    even_tensor: torch.Tensor
    odd_tensor: torch.Tensor

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
        return RelationMessage(*(a + b for a, b in zip(self.as_tuple(), other.as_tuple(), strict=True)))

    def subtract(self, other: RelationMessage) -> RelationMessage:
        return RelationMessage(*(a - b for a, b in zip(self.as_tuple(), other.as_tuple(), strict=True)))

    def scale(self, scale: torch.Tensor) -> RelationMessage:
        output: list[torch.Tensor] = []
        for value in self.as_tuple():
            suffix = (1,) * (value.ndim - scale.ndim)
            output.append(value * scale.reshape(*scale.shape, *suffix))
        return RelationMessage(*output)


class ValueProjection(nn.Module):
    def __init__(self, *, scalar_width: int, num_heads: int) -> None:
        super().__init__()
        if scalar_width % num_heads:
            raise ValueError("scalar_width must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = scalar_width // num_heads
        self.scalar = nn.Linear(scalar_width, scalar_width)
        self.odd = ChannelMix(num_heads, num_heads)
        self.polar = ChannelMix(num_heads, num_heads)
        self.axial = ChannelMix(num_heads, num_heads)
        self.even_tensor = ChannelMix(num_heads, num_heads)
        self.odd_tensor = ChannelMix(num_heads, num_heads)

    def forward(self, state: ParityState) -> RelationMessage:
        n = state.num_nodes
        return RelationMessage(
            self.scalar(state.even_scalar).reshape(n, self.num_heads, self.head_dim),
            self.odd(state.odd_scalar),
            self.polar(state.polar_vector),
            self.axial(state.axial_vector),
            self.even_tensor(state.even_tensor),
            self.odd_tensor(state.odd_tensor),
        )


class ContentGramFeatures(nn.Module):
    """All-irrep feature map whose Gram matrix is invariant and PSD."""

    def __init__(self, *, scalar_width: int, num_heads: int, feature_width: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.feature_width = feature_width
        self.scalar = nn.Linear(scalar_width, num_heads * feature_width)
        self.odd = ChannelMix(num_heads, num_heads)
        self.polar = ChannelMix(num_heads, num_heads)
        self.axial = ChannelMix(num_heads, num_heads)
        self.even_tensor = ChannelMix(num_heads, num_heads)
        self.odd_tensor = ChannelMix(num_heads, num_heads)

    def forward(self, state: ParityState) -> torch.Tensor:
        n = state.num_nodes
        scalar = self.scalar(state.even_scalar).reshape(n, self.num_heads, self.feature_width)
        odd = self.odd(state.odd_scalar).unsqueeze(-1)
        polar = unit_ball(self.polar(state.polar_vector), 1e-8)
        axial = unit_ball(self.axial(state.axial_vector), 1e-8)
        even_tensor = st_orthonormal(self.even_tensor(state.even_tensor))
        odd_tensor = st_orthonormal(self.odd_tensor(state.odd_tensor))
        return torch.cat((scalar, odd, polar, axial, even_tensor, odd_tensor), dim=-1)


class MercerFeatures(nn.Module):
    """Finite PSD Gaussian Mercer feature map through Cartesian order four."""

    def __init__(self, *, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.raw_gamma = nn.Parameter(torch.linspace(-1.0, 1.0, num_heads))

    def forward(self, position: torch.Tensor) -> torch.Tensor:
        gamma = torch.nn.functional.softplus(self.raw_gamma) + 0.05
        r2 = position.square().sum(dim=-1)
        base = torch.exp(-r2[:, None] * gamma[None, :])
        powers = [position]
        powers.append(torch.einsum("na,nb->nab", position, position).flatten(1))
        powers.append(torch.einsum("na,nb,nc->nabc", position, position, position).flatten(1))
        powers.append(
            torch.einsum("na,nb,nc,nd->nabcd", position, position, position, position).flatten(1)
        )
        features = [base.unsqueeze(-1)]
        for order, power in enumerate(powers, start=1):
            coefficient = torch.sqrt((2.0 * gamma) ** order / float(factorial(order)))
            features.append(base[..., None] * coefficient[None, :, None] * power[:, None, :])
        return torch.cat(features, dim=-1)


@dataclass(frozen=True)
class AtlasFactors:
    assignment: torch.Tensor
    mass: torch.Tensor
    metric: torch.Tensor
    chart_weight: torch.Tensor
    node_metric: torch.Tensor
    effective_dimension: torch.Tensor


class ManifoldAtlas(nn.Module):
    """Learn a partition of unity and its symmetric PSD chart relation."""

    def __init__(self, *, scalar_width: int, num_heads: int, num_charts: int, eps: float) -> None:
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
        position: torch.Tensor,
        index: torch.Tensor,
        num_segments: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mass = segment_sum(assignment, index, num_segments).clamp_min(self.eps)
        center = segment_sum(assignment[..., None] * position[:, None, :], index, num_segments)
        center = center / mass[..., None]
        delta = position[:, None, :] - center[index]
        covariance = segment_sum(
            assignment[..., None, None] * torch.einsum("nka,nkb->nkab", delta, delta),
            index,
            num_segments,
        ) / mass[..., None, None]
        return mass, center, covariance

    def forward(self, state: ParityState, geometry: GeometryContext) -> AtlasFactors:
        position = geometry.normalized.to(dtype=state.even_scalar.dtype)
        radius2 = position.square().sum(dim=-1)
        radial = torch.stack((radius2, torch.log1p(radius2), radius2 / (1.0 + radius2)), dim=-1)
        raw = self.logits(state.even_scalar) + self.radial_logits(radial)
        initial = torch.softmax(raw, dim=-1)
        mass, center, covariance = self._statistics(
            initial, position, geometry.index, geometry.num_segments
        )
        identity = torch.eye(3, device=position.device, dtype=position.dtype)
        trace = covariance.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / 3.0
        ridge = torch.nn.functional.softplus(self.raw_ridge)[None, :] + self.eps
        metric = covariance + (ridge * trace.clamp_min(1.0))[..., None, None] * identity
        delta = position[:, None, :] - center[geometry.index]
        solved = torch.linalg.solve(metric[geometry.index], delta.unsqueeze(-1)).squeeze(-1)
        distance2 = (delta * solved).sum(dim=-1).clamp(min=0.0, max=64.0)
        scale = torch.nn.functional.softplus(self.raw_distance_scale)
        balance = torch.log(mass[geometry.index] + self.eps)
        assignment = torch.softmax(raw - scale[None, :] * distance2 - 0.25 * balance, dim=-1)
        mass, center, covariance = self._statistics(
            assignment, position, geometry.index, geometry.num_segments
        )
        trace = covariance.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / 3.0
        metric = covariance + (
            (torch.nn.functional.softplus(self.raw_ridge)[None, :] + self.eps)
            * trace.clamp_min(1.0)
        )[..., None, None] * identity
        metric = metric / metric.diagonal(dim1=-2, dim2=-1).sum(dim=-1).div(3.0).clamp_min(self.eps)[..., None, None]
        covariance_trace = covariance.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        covariance_square = covariance.square().sum(dim=(-2, -1))
        effective_dimension = (
            covariance_trace.square() / (covariance_square + self.eps)
        ).clamp(1.0, 3.0)
        chart_weight = torch.sigmoid(
            self.head_bias[None, :, :]
            + self.head_slope[None, :, :] * (effective_dimension[:, None, :] - 2.0)
        )
        node_metric = (
            assignment[..., None, None] * metric[geometry.index]
        ).sum(dim=1)
        return AtlasFactors(
            assignment=assignment,
            mass=mass,
            metric=metric,
            chart_weight=chart_weight,
            node_metric=node_metric,
            effective_dimension=effective_dimension,
        )

    def apply(self, factors: AtlasFactors, value: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
        # value: [N,H,D]
        assignment = factors.assignment.to(dtype=value.dtype)
        chart_sum = segment_sum(
            assignment[:, :, None, None] * value[:, None, :, :],
            index,
            num_segments,
        )
        chart_mean = chart_sum / factors.mass.to(dtype=value.dtype)[:, :, None, None]
        weighted = chart_mean * factors.chart_weight.to(dtype=value.dtype).permute(0, 2, 1)[:, :, :, None]
        return (assignment[:, :, None, None] * weighted[index]).sum(dim=1)


@dataclass(frozen=True)
class RelationFactors:
    content_feature: torch.Tensor
    content_trace: torch.Tensor
    mercer_feature: torch.Tensor
    mercer_trace: torch.Tensor
    atlas: AtlasFactors
    mixture: torch.Tensor
    index: torch.Tensor
    num_segments: int


class SelfAdjointRelation(nn.Module):
    """Convex mixture of content, Mercer, and manifold PSD operators."""

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
        self.content = ContentGramFeatures(
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

    def build(self, state: ParityState, geometry: GeometryContext) -> RelationFactors:
        content = self.content(state)
        mercer = self.mercer(geometry.normalized.to(dtype=state.even_scalar.dtype))
        content_trace = segment_sum(content.square().sum(dim=-1), geometry.index, geometry.num_segments).clamp_min(self.eps)
        mercer_trace = segment_sum(mercer.square().sum(dim=-1), geometry.index, geometry.num_segments).clamp_min(self.eps)
        atlas = self.atlas(state, geometry)
        graph_scalar = segment_mean(state.even_scalar, geometry.index, geometry.num_segments)
        mixture = torch.softmax(
            self.mixture(graph_scalar).reshape(geometry.num_segments, self.num_heads, 3),
            dim=-1,
        )
        return RelationFactors(
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
        trace: torch.Tensor,
        value: torch.Tensor,
        index: torch.Tensor,
        num_segments: int,
    ) -> torch.Tensor:
        summary = segment_sum(
            feature[..., None] * value[..., None, :],
            index,
            num_segments,
        )
        output = (feature[..., None] * summary[index]).sum(dim=-2)
        return output / trace[index, :, None]

    def apply_tensor(self, factors: RelationFactors, value: torch.Tensor) -> torch.Tensor:
        content = self._gram_apply(
            factors.content_feature,
            factors.content_trace,
            value,
            factors.index,
            factors.num_segments,
        )
        mercer = self._gram_apply(
            factors.mercer_feature,
            factors.mercer_trace,
            value,
            factors.index,
            factors.num_segments,
        )
        atlas = self.atlas.apply(
            factors.atlas,
            value,
            factors.index,
            factors.num_segments,
        )
        weight = factors.mixture[factors.index]
        return (
            weight[:, :, 0, None] * content
            + weight[:, :, 1, None] * mercer
            + weight[:, :, 2, None] * atlas
        )

    def apply(self, factors: RelationFactors, message: RelationMessage) -> RelationMessage:
        odd = self.apply_tensor(factors, message.odd_scalar.unsqueeze(-1)).squeeze(-1)
        return RelationMessage(
            self.apply_tensor(factors, message.scalar),
            odd,
            self.apply_tensor(factors, message.polar_vector),
            self.apply_tensor(factors, message.axial_vector),
            self.apply_tensor(factors, message.even_tensor),
            self.apply_tensor(factors, message.odd_tensor),
        )


def message_inner(left: RelationMessage, right: RelationMessage) -> torch.Tensor:
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
    output = candidate
    for basis in bases:
        numerator = segment_sum(message_inner(basis, output), index, num_segments)
        denominator = segment_sum(message_inner(basis, basis), index, num_segments).clamp_min(eps)
        output = output.subtract(basis.scale((numerator / denominator)[index]))
    norm = segment_sum(message_inner(output, output).clamp_min(0.0), index, num_segments)
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
        graph_scalar = segment_mean(state.even_scalar, geometry.index, geometry.num_segments)
        coefficient = torch.tanh(
            self.coefficient(graph_scalar).reshape(geometry.num_segments, self.num_heads, 2)
        )[geometry.index]
        return first.add(second.scale(coefficient[:, :, 0])).add(third.scale(coefficient[:, :, 1]))


__all__ = [
    "AtlasFactors",
    "KrylovMixer",
    "RelationFactors",
    "RelationMessage",
    "SelfAdjointRelation",
    "ValueProjection",
    "message_inner",
    "orthogonalize",
]
