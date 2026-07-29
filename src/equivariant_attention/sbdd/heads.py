"""Task-specific heads that consume representations from the generic core."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import nn
import torch.nn.functional as F

from .schema import LabelDirection


@dataclass(frozen=True)
class AffinityHeadOutput:
    affinity: torch.Tensor
    base: torch.Tensor
    interaction_residual: torch.Tensor
    strain_contribution: torch.Tensor


class AffinityHead(nn.Module):
    """Separate global/interface base, bound-geometry residual, and strain.

    The output convention is "higher means stronger".  Label conversion stays
    in the scientific data contract; docking scores are not accepted there as
    affinity labels.
    """

    output_direction = LabelDirection.HIGHER_IS_STRONGER

    def __init__(
        self,
        interface_dim: int,
        global_dim: int,
        *,
        interaction_dim: int | None = None,
        include_strain: bool = False,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(include_strain, bool):
            raise TypeError("include_strain must be boolean")
        for name, value in (
            ("interface_dim", interface_dim),
            ("global_dim", global_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if interaction_dim is not None and (
            isinstance(interaction_dim, bool)
            or not isinstance(interaction_dim, int)
            or interaction_dim <= 0
        ):
            raise ValueError("interaction_dim must be a positive integer")
        if hidden_dim is None:
            hidden_dim = max(8, interface_dim + global_dim)
        if (
            isinstance(hidden_dim, bool)
            or not isinstance(hidden_dim, int)
            or hidden_dim <= 0
        ):
            raise ValueError("hidden_dim must be a positive integer")
        self.interface_dim = interface_dim
        self.global_dim = global_dim
        self.interaction_dim = interaction_dim
        self.include_strain = include_strain
        self.base_head = nn.Sequential(
            nn.Linear(interface_dim + global_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.interaction_head = (
            None if interaction_dim is None else nn.Linear(interaction_dim, 1)
        )
        self.raw_strain_scale = (
            None if not include_strain else nn.Parameter(torch.tensor(0.0))
        )

    def forward(
        self,
        interface_representation: torch.Tensor,
        global_representation: torch.Tensor,
        *,
        same_bound_geometry_interaction: torch.Tensor | None = None,
        strain: torch.Tensor | None = None,
    ) -> AffinityHeadOutput:
        _matrix(
            "interface_representation",
            interface_representation,
            self.interface_dim,
        )
        _matrix(
            "global_representation",
            global_representation,
            self.global_dim,
        )
        if (
            interface_representation.shape[0] != global_representation.shape[0]
            or interface_representation.device != global_representation.device
            or interface_representation.dtype != global_representation.dtype
        ):
            raise ValueError(
                "interface and global representations must share batch/dtype/device"
            )
        base = self.base_head(
            torch.cat(
                [interface_representation, global_representation],
                dim=-1,
            )
        ).squeeze(-1)
        if self.interaction_head is None:
            if same_bound_geometry_interaction is not None:
                raise ValueError("head was not configured for same-bound interaction")
            interaction = base * 0.0
        else:
            if same_bound_geometry_interaction is None:
                raise ValueError(
                    "configured interaction lane requires same-bound geometry"
                )
            assert self.interaction_dim is not None
            _matrix(
                "same_bound_geometry_interaction",
                same_bound_geometry_interaction,
                self.interaction_dim,
            )
            if (
                same_bound_geometry_interaction.shape[0] != base.shape[0]
                or same_bound_geometry_interaction.device != base.device
                or same_bound_geometry_interaction.dtype != base.dtype
            ):
                raise ValueError("same-bound interaction must share batch/dtype/device")
            interaction = self.interaction_head(
                same_bound_geometry_interaction
            ).squeeze(-1)
        if self.raw_strain_scale is None:
            if strain is not None:
                raise ValueError("head was not configured for strain")
            strain_contribution = base * 0.0
        else:
            if strain is None:
                raise ValueError("configured strain lane requires strain")
            if (
                strain.ndim != 1
                or strain.shape != base.shape
                or not torch.is_floating_point(strain)
                or not bool(torch.isfinite(strain).all())
                or strain.device != base.device
                or strain.dtype != base.dtype
            ):
                raise ValueError(
                    "strain must be finite and match affinity batch/dtype/device"
                )
            if bool((strain < 0).any()):
                raise ValueError("strain must be nonnegative")
            strain_contribution = -F.softplus(self.raw_strain_scale) * strain
        return AffinityHeadOutput(
            affinity=base + interaction + strain_contribution,
            base=base,
            interaction_residual=interaction,
            strain_contribution=strain_contribution,
        )


class RefinementOutputKind(StrEnum):
    DISPLACEMENT = "displacement"
    VELOCITY = "velocity"
    SCORE = "score"


@dataclass(frozen=True)
class PoseRefinementRequest:
    """Explicit movable-node and optional diffusion/flow-time contract."""

    ligand_mask: torch.Tensor
    protein_mask: torch.Tensor
    flexible_protein_mask: torch.Tensor | None = None
    time: torch.Tensor | None = None

    def __post_init__(self) -> None:
        for name in ("ligand_mask", "protein_mask"):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a tensor")
            if value.ndim != 1 or value.dtype != torch.bool:
                raise ValueError(f"{name} must be a one-dimensional boolean tensor")
        if (
            self.ligand_mask.shape != self.protein_mask.shape
            or self.ligand_mask.device != self.protein_mask.device
        ):
            raise ValueError("ligand and protein masks must share shape/device")
        if bool((self.ligand_mask & self.protein_mask).any()):
            raise ValueError("ligand and protein masks must be disjoint")
        if not bool(self.ligand_mask.any()):
            raise ValueError("pose refinement requires at least one ligand atom")
        if self.flexible_protein_mask is not None:
            value = self.flexible_protein_mask
            if not isinstance(value, torch.Tensor):
                raise TypeError("flexible_protein_mask must be a tensor")
            if (
                value.ndim != 1
                or value.dtype != torch.bool
                or value.shape != self.protein_mask.shape
                or value.device != self.protein_mask.device
            ):
                raise ValueError(
                    "flexible_protein_mask must match protein mask shape/device"
                )
            if bool((value & ~self.protein_mask).any()):
                raise ValueError(
                    "flexible_protein_mask must be a subset of protein_mask"
                )
        if self.time is not None:
            if not isinstance(self.time, torch.Tensor):
                raise TypeError("time must be a tensor")
            if (
                not torch.is_floating_point(self.time)
                or not bool(torch.isfinite(self.time).all())
                or self.time.numel() not in {1, self.ligand_mask.numel()}
            ):
                raise ValueError("time must be finite scalar or one value per node")

    @property
    def update_mask(self) -> torch.Tensor:
        if self.flexible_protein_mask is None:
            return self.ligand_mask
        return self.ligand_mask | self.flexible_protein_mask


@dataclass(frozen=True)
class PoseRefinementOutput:
    vector: torch.Tensor
    update_mask: torch.Tensor
    kind: RefinementOutputKind

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RefinementOutputKind):
            raise TypeError("kind must be a RefinementOutputKind")
        if (
            not isinstance(self.vector, torch.Tensor)
            or self.vector.ndim != 2
            or self.vector.shape[-1] != 3
            or not torch.is_floating_point(self.vector)
            or not bool(torch.isfinite(self.vector).all())
        ):
            raise ValueError("vector must be a finite floating tensor of shape (N, 3)")
        if (
            not isinstance(self.update_mask, torch.Tensor)
            or self.update_mask.ndim != 1
            or self.update_mask.dtype != torch.bool
            or self.update_mask.shape[0] != self.vector.shape[0]
            or self.update_mask.device != self.vector.device
        ):
            raise ValueError("update_mask must match vector nodes/device")

    @property
    def displacement(self) -> torch.Tensor:
        if self.kind is not RefinementOutputKind.DISPLACEMENT:
            raise AttributeError("refinement output is not a displacement")
        return self.vector

    @property
    def velocity(self) -> torch.Tensor:
        if self.kind is not RefinementOutputKind.VELOCITY:
            raise AttributeError("refinement output is not a velocity")
        return self.vector

    @property
    def score(self) -> torch.Tensor:
        if self.kind is not RefinementOutputKind.SCORE:
            raise AttributeError("refinement output is not a score")
        return self.vector


class PoseRefinementHead(nn.Module):
    """Invariant gates over equivariant vector carriers."""

    def __init__(
        self,
        scalar_dim: int,
        vector_channels: int,
        *,
        time_conditioned: bool = False,
        output_kind: RefinementOutputKind = RefinementOutputKind.DISPLACEMENT,
    ) -> None:
        super().__init__()
        if not isinstance(time_conditioned, bool):
            raise TypeError("time_conditioned must be boolean")
        if not isinstance(output_kind, RefinementOutputKind):
            raise TypeError("output_kind must be a RefinementOutputKind")
        for name, value in (
            ("scalar_dim", scalar_dim),
            ("vector_channels", vector_channels),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.scalar_dim = scalar_dim
        self.vector_channels = vector_channels
        self.time_conditioned = time_conditioned
        self.output_kind = output_kind
        self.gate = nn.Linear(
            scalar_dim + int(time_conditioned),
            vector_channels,
        )

    def forward(
        self,
        node_scalars: torch.Tensor,
        node_vectors: torch.Tensor,
        request: PoseRefinementRequest,
    ) -> PoseRefinementOutput:
        _matrix("node_scalars", node_scalars, self.scalar_dim)
        if (
            node_vectors.ndim != 3
            or node_vectors.shape != (node_scalars.shape[0], self.vector_channels, 3)
            or not torch.is_floating_point(node_vectors)
            or not bool(torch.isfinite(node_vectors).all())
        ):
            raise ValueError(
                "node_vectors must have finite shape (N, vector_channels, 3)"
            )
        if (
            node_vectors.device != node_scalars.device
            or node_vectors.dtype != node_scalars.dtype
        ):
            raise ValueError("node scalar/vector carriers must share dtype/device")
        if (
            request.ligand_mask.shape[0] != node_scalars.shape[0]
            or request.ligand_mask.device != node_scalars.device
        ):
            raise ValueError("refinement masks must match node count/device")
        gate_input = node_scalars
        if self.time_conditioned:
            if request.time is None:
                raise ValueError("time-conditioned refinement requires time")
            time = request.time.to(
                device=node_scalars.device,
                dtype=node_scalars.dtype,
            )
            if time.numel() == 1:
                time = time.expand(node_scalars.shape[0])
            gate_input = torch.cat([node_scalars, time.reshape(-1, 1)], dim=-1)
        elif request.time is not None:
            raise ValueError("time was supplied to a non-time-conditioned head")
        weights = self.gate(gate_input)
        vector = torch.einsum("nc,ncd->nd", weights, node_vectors)
        update_mask = request.update_mask
        vector = vector * update_mask.unsqueeze(-1).to(dtype=vector.dtype)
        return PoseRefinementOutput(
            vector=vector,
            update_mask=update_mask,
            kind=self.output_kind,
        )


def _matrix(name: str, value: torch.Tensor, width: int) -> None:
    if (
        value.ndim != 2
        or value.shape[1] != width
        or not torch.is_floating_point(value)
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{name} must be finite with shape (B, {width})")
