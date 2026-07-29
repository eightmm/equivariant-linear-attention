"""Generic scalar-potential and force heads for 3D systems."""

from __future__ import annotations

from types import MappingProxyType
from typing import ClassVar, Mapping

import torch
from torch import nn

from .heads import EquivariantVectorHead
from .pooling import MaskedInvariantPooling


class ScalarEnergyHead(nn.Module):
    """Map invariant node states to an additive scalar graph potential."""

    energy_semantics: ClassVar[str] = "additive_scalar_potential"

    def __init__(
        self,
        scalar_channels: int,
        *,
        hidden_channels: int | None = None,
    ) -> None:
        super().__init__()
        if (
            isinstance(scalar_channels, bool)
            or not isinstance(scalar_channels, int)
            or scalar_channels <= 0
        ):
            raise ValueError("scalar_channels must be a positive integer")
        if hidden_channels is None:
            hidden_channels = max(8, scalar_channels)
        if (
            isinstance(hidden_channels, bool)
            or not isinstance(hidden_channels, int)
            or hidden_channels <= 0
        ):
            raise ValueError("hidden_channels must be a positive integer")
        self.scalar_channels = scalar_channels
        self.node_energy = nn.Sequential(
            nn.Linear(scalar_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )
        self.pool = MaskedInvariantPooling(
            reduction="sum",
            empty_policy="zero",
        )

    def node_energies(self, scalars: torch.Tensor) -> torch.Tensor:
        """Return one learned invariant energy contribution per node."""

        if not isinstance(scalars, torch.Tensor):
            raise TypeError("scalars must be a tensor")
        if not torch.is_floating_point(scalars):
            raise TypeError("scalars must use a floating-point dtype")
        if (
            scalars.ndim != 2
            or scalars.shape[1] != self.scalar_channels
        ):
            raise ValueError(
                f"scalars must have shape (N, {self.scalar_channels})"
            )
        return self.node_energy(scalars).squeeze(-1)

    def forward(
        self,
        scalars: torch.Tensor,
        batch: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        num_graphs: int | None = None,
    ) -> torch.Tensor:
        """Return one extensive scalar energy per packed graph."""

        contributions = self.node_energies(scalars)
        return self.pool(
            contributions,
            batch,
            mask=mask,
            num_graphs=num_graphs,
        )


def conservative_forces(
    energy: torch.Tensor,
    positions: torch.Tensor,
    *,
    create_graph: bool = False,
    retain_graph: bool | None = None,
) -> torch.Tensor:
    """Return conservative forces ``-d sum(energy) / d positions``.

    Set ``create_graph=True`` when a force loss must differentiate through the
    force computation.  A disconnected energy is rejected rather than being
    mislabeled as a learned zero-force potential.
    """

    if not isinstance(energy, torch.Tensor):
        raise TypeError("energy must be a tensor")
    if not torch.is_floating_point(energy):
        raise TypeError("energy must use a floating-point dtype")
    if not isinstance(positions, torch.Tensor):
        raise TypeError("positions must be a tensor")
    if (
        positions.ndim < 2
        or positions.shape[-1] != 3
        or not torch.is_floating_point(positions)
    ):
        raise ValueError(
            "positions must be a floating-point tensor ending in dimension 3"
        )
    if energy.device != positions.device:
        raise ValueError("energy and positions must be on the same device")
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
    """Auxiliary direct vector prediction with explicit nonconservative status."""

    metadata: ClassVar[Mapping[str, object]] = MappingProxyType(
        {
            "force_semantics": "non_conservative_auxiliary",
            "conservative": False,
        }
    )

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

    def forward(
        self,
        scalars: torch.Tensor,
        vectors: torch.Tensor,
    ) -> torch.Tensor:
        return self.vector_head(scalars, vectors).squeeze(1)
