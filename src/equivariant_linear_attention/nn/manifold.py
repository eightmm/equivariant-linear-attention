"""Natural-gradient coordinate updates on an SE(3) quotient configuration space."""

from __future__ import annotations

import torch
from torch import nn

from .heads import EquivariantVectorHead
from .ops import segment_sum
from .state import ParityState


def quotient_rigid_shape_step(
    raw: torch.Tensor,
    positions: torch.Tensor,
    index: torch.Tensor,
    *,
    num_segments: int,
    counts: torch.Tensor,
    selected: torch.Tensor,
    component_gates: torch.Tensor,
    max_step: float,
    eps: float,
) -> torch.Tensor:
    """Decompose a selected velocity into translation, rotation, and shape."""

    mask = selected.to(dtype=raw.dtype).unsqueeze(-1)
    selected_count_raw = segment_sum(selected.to(dtype=torch.long), index, num_segments)
    selected_count = selected_count_raw.clamp_min(1).to(dtype=raw.dtype)
    center = (
        segment_sum(positions * mask, index, num_segments) / selected_count[:, None]
    )
    translation = segment_sum(raw * mask, index, num_segments) / selected_count[:, None]
    relative = (positions - center[index]) * mask
    centered_velocity = (raw - translation[index]) * mask
    angular_momentum = segment_sum(
        torch.cross(relative, centered_velocity, dim=-1), index, num_segments
    )
    identity = torch.eye(3, device=raw.device, dtype=raw.dtype)
    inertia = segment_sum(
        relative.square().sum(dim=-1)[:, None, None] * identity
        - torch.einsum("na,nb->nab", relative, relative),
        index,
        num_segments,
    )
    scale = inertia.diagonal(dim1=-2, dim2=-1).sum(dim=-1).div(3.0).clamp_min(1.0)
    ridge = eps + 8.0 * torch.finfo(raw.dtype).eps * scale
    angular_velocity = torch.linalg.solve(
        inertia + ridge[:, None, None] * identity,
        angular_momentum.unsqueeze(-1),
    ).squeeze(-1)
    rotation = torch.cross(angular_velocity[index], relative, dim=-1)
    shape = (raw - translation[index] - rotation) * mask

    fully_selected = selected_count_raw == counts
    rigid_allowed = (~fully_selected).to(dtype=raw.dtype)
    translation_gate = component_gates[:, 0] * rigid_allowed
    rotation_gate = component_gates[:, 1] * rigid_allowed
    shape_gate = component_gates[:, 2]
    step = (
        translation_gate[index, None] * translation[index]
        + rotation_gate[index, None] * rotation
        + shape_gate[index, None] * shape
    ) * mask
    norm = torch.sqrt(step.square().sum(dim=-1) + eps)
    maximum = norm.new_zeros(num_segments)
    maximum.scatter_reduce_(0, index, norm, reduce="amax", include_self=True)
    rescale = (float(max_step) / maximum.clamp_min(float(max_step))).clamp(max=1.0)
    return step * rescale[index, None]


class QuotientCoordinateUpdate(nn.Module):
    """Atlas-preconditioned polar update with automatic rigid/shape semantics."""

    def __init__(self, *, scalar_width: int, num_heads: int, eps: float) -> None:
        super().__init__()
        self.eps = float(eps)
        self.vector = EquivariantVectorHead(
            scalar_width,
            num_heads,
            output_channels=1,
            zero_init=True,
        )
        self.node_gate = nn.Linear(scalar_width, 1)
        self.component_gate = nn.Linear(scalar_width, 3)
        nn.init.zeros_(self.node_gate.weight)
        nn.init.zeros_(self.node_gate.bias)
        nn.init.zeros_(self.component_gate.weight)
        nn.init.zeros_(self.component_gate.bias)

    def forward(
        self,
        state: ParityState,
        positions: torch.Tensor,
        index: torch.Tensor,
        *,
        num_segments: int,
        counts: torch.Tensor,
        node_metric: torch.Tensor,
        update_mask: torch.Tensor | None,
        max_step: float,
    ) -> torch.Tensor:
        selected = (
            torch.ones(state.num_nodes, device=positions.device, dtype=torch.bool)
            if update_mask is None
            else update_mask
        )
        raw = self.vector(state.even_scalar, state.polar_vector).squeeze(1)
        raw = torch.sigmoid(self.node_gate(state.even_scalar)) * raw
        identity = torch.eye(3, device=raw.device, dtype=raw.dtype)
        metric = node_metric.to(dtype=raw.dtype) + self.eps * identity
        natural = torch.linalg.solve(metric, raw.unsqueeze(-1)).squeeze(-1)

        mask = selected.to(dtype=state.even_scalar.dtype).unsqueeze(-1)
        selected_count = segment_sum(
            selected.to(dtype=torch.long), index, num_segments
        ).clamp_min(1)
        component_state = segment_sum(state.even_scalar * mask, index, num_segments)
        component_state = (
            component_state / selected_count.to(dtype=state.even_scalar.dtype)[:, None]
        )
        raw_gate = self.component_gate(component_state)
        component_gates = torch.cat(
            (torch.tanh(raw_gate[:, :2]), 1.0 + torch.tanh(raw_gate[:, 2:])), dim=-1
        )
        return quotient_rigid_shape_step(
            natural,
            positions,
            index,
            num_segments=num_segments,
            counts=counts,
            selected=selected,
            component_gates=component_gates,
            max_step=max_step,
            eps=self.eps,
        )


__all__ = ["QuotientCoordinateUpdate", "quotient_rigid_shape_step"]
