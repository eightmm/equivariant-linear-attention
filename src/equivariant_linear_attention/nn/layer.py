"""One ELA layer: pointwise local jets, global relation, and feed-forward."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .closure import EquivariantClosure
from .equivariant_ffn import EquivariantFeedForward
from .geometry import AdaptiveMomentBank, GeometryContext, MomentFeatures
from .local_closure import LocalEquivariantClosure
from .local_geometry import PointwiseLocalGeometry
from .local_support import LocalSupport
from .relation import KrylovMixer, SelfAdjointRelation, orthogonalize
from .state import EquivariantRMSNorm, ParityState


@dataclass(frozen=True)
class LayerOutput:
    state: ParityState
    node_metric: torch.Tensor
    moments: MomentFeatures


class EdgeFreeELALayer(nn.Module):
    """Pointwise local analysis followed by one global self-adjoint operator."""

    def __init__(
        self,
        *,
        scalar_width: int,
        num_heads: int,
        moment_rank: int,
        relation_width: int,
        num_charts: int,
        residual_scale: float,
        eps: float,
        local_probe_rank: int | None = None,
        local_scales: int = 3,
        local_points: int | None = None,
        local_chunk_size: int = 128,
    ) -> None:
        super().__init__()
        if scalar_width % num_heads:
            raise ValueError("scalar_width must be divisible by num_heads")
        self.eps = float(eps)
        probe_rank = (
            max(2, min(8, scalar_width // 32))
            if local_probe_rank is None
            else int(local_probe_rank)
        )
        point_count = (
            max(8, min(32, scalar_width // 4))
            if local_points is None
            else int(local_points)
        )
        self.local_norm = EquivariantRMSNorm(
            scalar_width=scalar_width, num_heads=num_heads, eps=eps
        )
        self.attention_norm = EquivariantRMSNorm(
            scalar_width=scalar_width, num_heads=num_heads, eps=eps
        )
        self.ffn_norm = EquivariantRMSNorm(
            scalar_width=scalar_width, num_heads=num_heads, eps=eps
        )
        self.local_geometry = PointwiseLocalGeometry(
            scalar_width=scalar_width,
            moment_rank=moment_rank,
            probe_rank=probe_rank,
            num_scales=local_scales,
            max_points=point_count,
            chunk_size=local_chunk_size,
            eps=eps,
        )
        self.local_closure = LocalEquivariantClosure(
            scalar_width=scalar_width,
            num_heads=num_heads,
            moment_rank=moment_rank,
            probe_rank=probe_rank,
            num_scales=local_scales,
            eps=eps,
        )
        self.moments = AdaptiveMomentBank(
            scalar_width=scalar_width, rank=moment_rank, eps=eps
        )
        self.relation = SelfAdjointRelation(
            scalar_width=scalar_width,
            num_heads=num_heads,
            feature_width=relation_width,
            num_charts=num_charts,
            eps=eps,
        )
        self.krylov = KrylovMixer(scalar_width=scalar_width, num_heads=num_heads)
        self.closure = EquivariantClosure(
            scalar_width=scalar_width,
            num_heads=num_heads,
            head_dim=scalar_width // num_heads,
            moment_rank=moment_rank,
            eps=eps,
        )
        self.ffn = EquivariantFeedForward(
            scalar_width=scalar_width, num_heads=num_heads, eps=eps
        )
        self.local_scale = nn.Parameter(
            torch.tensor(max(0.01, 0.25 * float(residual_scale)))
        )
        self.attention_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        self.ffn_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(
        self,
        state: ParityState,
        geometry: GeometryContext,
        support: LocalSupport | None = None,
    ) -> LayerOutput:
        local_state = self.local_norm(state)
        local = self.local_geometry(local_state.even_scalar, geometry, support)
        state = state.add(
            self.local_closure(local_state, local).scale(self.local_scale)
        )

        normalized = self.attention_norm(state)
        moments = self.moments(normalized.even_scalar, geometry)
        value, content_feature = self.relation.project(normalized)
        factors = self.relation.build(
            normalized, geometry, content_feature=content_feature
        )
        order_one = self.relation.apply(factors, value)
        order_two_raw = self.relation.apply(factors, order_one)
        order_two = orthogonalize(
            order_two_raw,
            (order_one,),
            index=geometry.index,
            num_segments=geometry.num_segments,
            counts=geometry.counts,
            eps=self.eps,
        )
        order_three_raw = self.relation.apply(factors, order_two_raw)
        order_three = orthogonalize(
            order_three_raw,
            (order_one, order_two),
            index=geometry.index,
            num_segments=geometry.num_segments,
            counts=geometry.counts,
            eps=self.eps,
        )
        message = self.krylov(
            normalized, order_one, order_two, order_three, geometry
        )
        state = state.add(
            self.closure(normalized, message, moments).scale(self.attention_scale)
        )
        state = state.add(self.ffn(self.ffn_norm(state)).scale(self.ffn_scale))
        return LayerOutput(
            state=state,
            node_metric=factors.atlas.node_metric,
            moments=moments,
        )


__all__ = ["EdgeFreeELALayer", "LayerOutput"]
