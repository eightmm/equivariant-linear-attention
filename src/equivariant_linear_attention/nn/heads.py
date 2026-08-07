"""Generic O(3)-equivariant output heads."""

from __future__ import annotations

from math import sqrt

import torch
from torch import nn


class EquivariantVectorHead(nn.Module):
    """Map invariant scalars and polar-vector carriers to polar outputs."""

    def __init__(
        self,
        scalar_channels: int,
        vector_channels: int,
        *,
        output_channels: int = 1,
        hidden_channels: int | None = None,
        zero_init: bool = False,
    ) -> None:
        super().__init__()
        hidden_channels = (
            max(16, scalar_channels) if hidden_channels is None else hidden_channels
        )
        self.scalar_channels = scalar_channels
        self.vector_channels = vector_channels
        self.output_channels = output_channels
        self.base_weight = nn.Parameter(torch.empty(output_channels, vector_channels))
        if zero_init:
            nn.init.zeros_(self.base_weight)
        else:
            nn.init.normal_(self.base_weight, std=1.0 / sqrt(max(1, vector_channels)))
        self.scalar_mixer = nn.Sequential(
            nn.Linear(scalar_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, output_channels * vector_channels),
        )
        nn.init.zeros_(self.scalar_mixer[-1].weight)
        nn.init.zeros_(self.scalar_mixer[-1].bias)

    def forward(self, scalars: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
        if scalars.ndim != 2 or scalars.shape[1] != self.scalar_channels:
            raise ValueError(f"scalars must have shape (N,{self.scalar_channels})")
        if vectors.shape != (scalars.shape[0], self.vector_channels, 3):
            raise ValueError(f"vectors must have shape (N,{self.vector_channels},3)")
        learned = torch.tanh(self.scalar_mixer(scalars)).reshape(
            scalars.shape[0], self.output_channels, self.vector_channels
        )
        weight = learned + self.base_weight.to(dtype=scalars.dtype)[None, :, :]
        return torch.einsum("noi,nid->nod", weight, vectors)


__all__ = ["EquivariantVectorHead"]
