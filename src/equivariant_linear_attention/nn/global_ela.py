"""Global equivariant linear transport used inside every TriELA pair block."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .closure import EquivariantClosure
from .geometry import AdaptiveMomentBank, GeometryContext, MomentFeatures
from .relation import KrylovMixer, SelfAdjointRelation, orthogonalize
from .state import EquivariantRMSNorm, ParityState


@dataclass(frozen=True)
class GlobalELAOutput:
    state: ParityState
    node_metric: torch.Tensor
    moments: MomentFeatures


class GlobalELABlock(nn.Module):
    """Node-linear O(3)-equivariant transport conditioned by pair summaries.

    Pair context is invariant.  It is injected only into even scalars through a
    zero-initialized projection, so it may condition relation factors without
    changing the transformation law of any geometric carrier.
    """

    def __init__(
        self,
        *,
        scalar_width: int,
        pair_width: int,
        num_heads: int,
        moment_rank: int,
        relation_width: int,
        num_charts: int,
        residual_scale: float,
        eps: float,
    ) -> None:
        super().__init__()
        if scalar_width % num_heads:
            raise ValueError("scalar_width must be divisible by num_heads")
        self.eps = float(eps)
        self.context_projection = nn.Linear(pair_width, scalar_width)
        nn.init.zeros_(self.context_projection.weight)
        nn.init.zeros_(self.context_projection.bias)
        self.norm = EquivariantRMSNorm(
            scalar_width=scalar_width,
            num_heads=num_heads,
            eps=eps,
        )
        self.moments = AdaptiveMomentBank(
            scalar_width=scalar_width,
            rank=moment_rank,
            eps=eps,
        )
        # The old optional chart/local relation is deliberately absent.  The
        # exact local operator is a separate, pair-conditioned block.
        self.relation = SelfAdjointRelation(
            scalar_width=scalar_width,
            num_heads=num_heads,
            feature_width=relation_width,
            num_charts=num_charts,
            eps=eps,
        )
        self.krylov = KrylovMixer(
            scalar_width=scalar_width,
            num_heads=num_heads,
        )
        self.closure = EquivariantClosure(
            scalar_width=scalar_width,
            num_heads=num_heads,
            head_dim=scalar_width // num_heads,
            moment_rank=moment_rank,
            eps=eps,
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(
        self,
        state: ParityState,
        geometry: GeometryContext,
        context: torch.Tensor,
    ) -> GlobalELAOutput:
        if context.shape != (state.num_nodes, self.context_projection.in_features):
            raise ValueError("context must have shape (N,pair_width)")
        conditioned = ParityState(
            state.even_scalar + self.context_projection(context),
            state.odd_scalar,
            state.polar_vector,
            state.axial_vector,
            state.even_tensor,
            state.odd_tensor,
        )
        normalized = self.norm(conditioned)
        moments = self.moments(normalized.even_scalar, geometry)
        value, content_feature = self.relation.project(normalized)
        factors = self.relation.build(
            normalized,
            geometry,
            content_feature=content_feature,
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
            normalized,
            order_one,
            order_two,
            order_three,
            geometry,
        )
        delta = self.closure(normalized, message, moments)
        return GlobalELAOutput(
            state=conditioned.add(delta.scale(self.residual_scale)),
            node_metric=factors.atlas.node_metric,
            moments=moments,
        )


__all__ = ["GlobalELABlock", "GlobalELAOutput"]
