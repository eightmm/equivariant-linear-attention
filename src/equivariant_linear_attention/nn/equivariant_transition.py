"""Parity-valid node transition separated from global and local transport."""

from __future__ import annotations

import torch
from torch import nn

from .equivariant_ffn import EquivariantFeedForward
from .state import EquivariantRMSNorm, ParityState


class EquivariantTransitionBlock(nn.Module):
    def __init__(
        self,
        *,
        scalar_width: int,
        num_heads: int,
        residual_scale: float,
        eps: float,
    ) -> None:
        super().__init__()
        self.norm = EquivariantRMSNorm(
            scalar_width=scalar_width,
            num_heads=num_heads,
            eps=eps,
        )
        self.ffn = EquivariantFeedForward(
            scalar_width=scalar_width,
            num_heads=num_heads,
            eps=eps,
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, state: ParityState) -> ParityState:
        return state.add(self.ffn(self.norm(state)).scale(self.residual_scale))


__all__ = ["EquivariantTransitionBlock"]
