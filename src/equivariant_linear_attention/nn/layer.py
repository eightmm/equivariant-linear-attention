"""One tensor-fused edge-free ELA layer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .closure import EquivariantClosure
from .geometry import AdaptiveMomentBank, GeometryContext, MomentFeatures
from .local_geometry import LocalMercerMomentBank, LocalMomentFusion
from .ops import bounded_scalar, bounded_st, unit_ball
from .relation import KrylovMixer, SelfAdjointRelation, orthogonalize
from .state import ChannelMix, EquivariantRMSNorm, ParityState, state_invariants


@dataclass(frozen=True)
class LayerOutput:
    state: ParityState
    node_metric: torch.Tensor
    moments: MomentFeatures


class EquivariantFeedForward(nn.Module):
    def __init__(self, *, scalar_width: int, num_heads: int, eps: float) -> None:
        super().__init__()
        hidden = max(64, 2 * scalar_width)
        invariant_width = scalar_width + 5 * num_heads
        self.even = nn.Sequential(
            nn.Linear(invariant_width, hidden),
            nn.SiLU(),
            nn.Linear(hidden, scalar_width),
        )
        self.gates = nn.Sequential(
            nn.Linear(invariant_width, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 5 * num_heads),
        )
        nn.init.zeros_(self.gates[-1].weight)
        nn.init.zeros_(self.gates[-1].bias)
        self.odd = ChannelMix(num_heads, num_heads)
        self.polar = ChannelMix(num_heads, num_heads)
        self.axial = ChannelMix(num_heads, num_heads)
        self.even_tensor = ChannelMix(num_heads, num_heads)
        self.odd_tensor = ChannelMix(num_heads, num_heads)
        self.polar_cross = ChannelMix(num_heads, num_heads)
        self.axial_cross = ChannelMix(num_heads, num_heads)
        self.even_tensor_cross = ChannelMix(num_heads, num_heads)
        self.odd_tensor_cross = ChannelMix(num_heads, num_heads)
        self.num_heads = num_heads
        self.eps = float(eps)

    def forward(self, state: ParityState) -> ParityState:
        invariants = state_invariants(state, self.eps)
        gates = 2.0 * torch.sigmoid(self.gates(invariants)).reshape(
            state.num_nodes,
            5,
            self.num_heads,
        )
        odd = gates[:, 0] * self.odd(state.odd_scalar)
        polar = gates[:, 1, :, None] * (
            self.polar(state.polar_vector)
            + state.odd_scalar[..., None] * self.polar_cross(state.axial_vector)
        )
        axial = gates[:, 2, :, None] * (
            self.axial(state.axial_vector)
            + state.odd_scalar[..., None] * self.axial_cross(state.polar_vector)
        )
        even_tensor = gates[:, 3, :, None] * (
            self.even_tensor(state.even_tensor)
            + state.odd_scalar[..., None] * self.even_tensor_cross(state.odd_tensor)
        )
        odd_tensor = gates[:, 4, :, None] * (
            self.odd_tensor(state.odd_tensor)
            + state.odd_scalar[..., None] * self.odd_tensor_cross(state.even_tensor)
        )
        return ParityState(
            bounded_scalar(self.even(invariants), self.eps),
            bounded_scalar(odd, self.eps),
            unit_ball(polar, self.eps),
            unit_ball(axial, self.eps),
            bounded_st(even_tensor, self.eps),
            bounded_st(odd_tensor, self.eps),
        )


class EdgeFreeELALayer(nn.Module):
    """Global/local moments + one fused PSD relation + Krylov + closure."""

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
    ) -> None:
        super().__init__()
        if scalar_width % num_heads:
            raise ValueError("scalar_width must be divisible by num_heads")
        self.eps = float(eps)
        self.attention_norm = EquivariantRMSNorm(
            scalar_width=scalar_width,
            num_heads=num_heads,
            eps=eps,
        )
        self.ffn_norm = EquivariantRMSNorm(
            scalar_width=scalar_width,
            num_heads=num_heads,
            eps=eps,
        )
        self.moments = AdaptiveMomentBank(
            scalar_width=scalar_width,
            rank=moment_rank,
            eps=eps,
        )
        self.local_moments = LocalMercerMomentBank(
            scalar_width=scalar_width,
            rank=moment_rank,
            eps=eps,
        )
        self.moment_fusion = LocalMomentFusion(rank=moment_rank)
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
        self.ffn = EquivariantFeedForward(
            scalar_width=scalar_width,
            num_heads=num_heads,
            eps=eps,
        )
        self.attention_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        self.ffn_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, state: ParityState, geometry: GeometryContext) -> LayerOutput:
        normalized = self.attention_norm(state)
        global_moments = self.moments(normalized.even_scalar, geometry)
        local_moments = self.local_moments(normalized.even_scalar, geometry)
        moments = self.moment_fusion(global_moments, local_moments)
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
        attention_delta = self.closure(normalized, message, moments)
        state = state.add(attention_delta.scale(self.attention_scale))
        ffn_delta = self.ffn(self.ffn_norm(state))
        state = state.add(ffn_delta.scale(self.ffn_scale))
        return LayerOutput(
            state=state,
            node_metric=factors.atlas.node_metric,
            moments=moments,
        )


__all__ = ["EdgeFreeELALayer", "LayerOutput"]
