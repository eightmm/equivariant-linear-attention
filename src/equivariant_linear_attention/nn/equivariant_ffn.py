"""Equivariant feed-forward block shared by ELA layers."""

from __future__ import annotations

import torch
from torch import nn

from .ops import bounded_scalar, bounded_st, unit_ball
from .state import ChannelMix, ParityState, state_invariants


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


__all__ = ["EquivariantFeedForward"]
