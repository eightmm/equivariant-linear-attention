"""Scalar-potential and auxiliary force heads."""

from __future__ import annotations

from typing import ClassVar

import torch
from torch import nn

from .nn.heads import EquivariantVectorHead
from .nn.pooling import MaskedInvariantPooling


class ScalarEnergyHead(nn.Module):
    def __init__(
        self, scalar_channels: int, *, hidden_channels: int | None = None
    ) -> None:
        super().__init__()
        hidden_channels = (
            max(8, scalar_channels) if hidden_channels is None else hidden_channels
        )
        self.scalar_channels = scalar_channels
        self.node_energy = nn.Sequential(
            nn.Linear(scalar_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )
        self.pool = MaskedInvariantPooling(reduction="sum")

    def node_energies(self, scalars: torch.Tensor) -> torch.Tensor:
        if scalars.ndim != 2 or scalars.shape[-1] != self.scalar_channels:
            raise ValueError(f"scalars must have shape (N,{self.scalar_channels})")
        return self.node_energy(scalars).squeeze(-1)

    def forward(
        self,
        scalars: torch.Tensor,
        batch: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        num_graphs: int | None = None,
    ) -> torch.Tensor:
        return self.pool(
            self.node_energies(scalars), batch, mask=mask, num_graphs=num_graphs
        )


def conservative_forces(
    energy: torch.Tensor,
    positions: torch.Tensor,
    *,
    create_graph: bool = False,
    retain_graph: bool | None = None,
) -> torch.Tensor:
    if not positions.requires_grad:
        raise ValueError("positions must require gradients")
    if retain_graph is None:
        retain_graph = create_graph
    gradient = torch.autograd.grad(
        energy,
        positions,
        grad_outputs=torch.ones_like(energy),
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=False,
    )[0]
    return -gradient


class DirectVectorForceHead(nn.Module):
    metadata: ClassVar[dict[str, object]] = {
        "force_semantics": "non_conservative_auxiliary",
        "conservative": False,
    }

    def __init__(
        self,
        scalar_channels: int,
        vector_channels: int,
        *,
        hidden_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.vector_head = EquivariantVectorHead(
            scalar_channels,
            vector_channels,
            output_channels=1,
            hidden_channels=hidden_channels,
        )

    def forward(self, scalars: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
        return self.vector_head(scalars, vectors).squeeze(1)


__all__ = ["DirectVectorForceHead", "ScalarEnergyHead", "conservative_forces"]
