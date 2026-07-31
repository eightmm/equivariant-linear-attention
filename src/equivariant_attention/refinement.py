from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal, Protocol

import torch
from torch import nn

from .heads import EquivariantVectorHead
from .layered_se3 import UnifiedSE3State
from .unified import Prepared3DGraph, UnifiedEquivariantAttention


CenteringMode = Literal["none", "graph", "selected"]


class GeometryRebuilder(Protocol):
    """Rebuild a prepared sparse graph after a coordinate update."""

    def __call__(
        self,
        positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> Prepared3DGraph: ...


@dataclass(frozen=True, slots=True)
class CoordinateRefinementConfig:
    """Outer-loop coordinate refinement separated from the ELA core."""

    steps: int = 1
    max_step: float = 0.25
    centering: CenteringMode = "selected"

    def __post_init__(self) -> None:
        if isinstance(self.steps, bool) or not isinstance(self.steps, int):
            raise TypeError("steps must be an integer")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if isinstance(self.max_step, bool) or not isinstance(
            self.max_step, (int, float)
        ):
            raise TypeError("max_step must be a real number")
        numeric = float(self.max_step)
        if not isfinite(numeric) or numeric <= 0.0:
            raise ValueError("max_step must be finite and positive")
        if self.centering not in {"none", "graph", "selected"}:
            raise ValueError("centering must be none, graph, or selected")


class ELACoordinateRefiner(nn.Module):
    """Apply a static-geometry ELA backbone in an explicit refinement loop.

    The backbone itself does not mutate coordinates. A zero-initialized
    equivariant vector head predicts one bounded polar displacement per outer
    step. The caller may provide a graph rebuilder; otherwise the prepared
    candidate topology is reused while continuous geometry is recomputed.
    """

    def __init__(
        self,
        backbone: UnifiedEquivariantAttention,
        config: CoordinateRefinementConfig = CoordinateRefinementConfig(),
    ) -> None:
        super().__init__()
        if not isinstance(backbone, UnifiedEquivariantAttention):
            raise TypeError("backbone must be a UnifiedEquivariantAttention")
        if not isinstance(config, CoordinateRefinementConfig):
            raise TypeError("config must be a CoordinateRefinementConfig")
        if bool(getattr(backbone.config, "coordinate_updates", False)):
            raise ValueError(
                "backbone coordinate_updates must be disabled; refinement is external"
            )
        self.backbone = backbone
        self.config = config
        scalar_width = backbone.core.hidden_dim
        vector_width = backbone.core.num_heads
        self.vector_head = EquivariantVectorHead(
            scalar_width,
            vector_width,
            output_channels=1,
        )
        self.gate = nn.Linear(scalar_width, 1)
        with torch.no_grad():
            self.vector_head.base_weight.zero_()
            final = self.vector_head.scalar_mixer[-1]
            if not isinstance(final, nn.Linear):
                raise RuntimeError("unexpected vector-head output module")
            final.weight.zero_()
            final.bias.zero_()
            self.gate.weight.zero_()
            self.gate.bias.zero_()

    @staticmethod
    def _mask(
        update_mask: torch.Tensor | None,
        *,
        num_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        if update_mask is None:
            return torch.ones(num_nodes, device=device, dtype=torch.bool)
        if not isinstance(update_mask, torch.Tensor):
            raise TypeError("update_mask must be a tensor")
        if update_mask.dtype != torch.bool:
            raise TypeError("update_mask must use boolean dtype")
        if update_mask.shape != (num_nodes,):
            raise ValueError("update_mask must have shape (N,)")
        if update_mask.device != device:
            raise ValueError("update_mask and positions must share one device")
        return update_mask

    def _center_and_bound(
        self,
        raw: torch.Tensor,
        batch: torch.Tensor,
        selected: torch.Tensor,
    ) -> torch.Tensor:
        batch_index = batch.to(dtype=torch.long)
        if batch_index.ndim != 1 or batch_index.shape[0] != raw.shape[0]:
            raise ValueError("batch must have shape (N,)")
        if batch_index.numel() == 0:
            raise ValueError("coordinate refinement requires at least one node")
        num_graphs = int(batch_index.max().item()) + 1
        masked = torch.where(selected[:, None], raw, torch.zeros_like(raw))

        if self.config.centering == "none":
            centered = masked
        elif self.config.centering == "graph":
            graph_sum = raw.new_zeros((num_graphs, 3)).index_add(
                0, batch_index, raw
            )
            counts = torch.bincount(
                batch_index, minlength=num_graphs
            ).clamp_min(1)
            centered = raw - graph_sum[batch_index] / counts[
                batch_index, None
            ].to(dtype=raw.dtype)
            centered = torch.where(
                selected[:, None], centered, torch.zeros_like(centered)
            )
        else:
            selected_sum = raw.new_zeros((num_graphs, 3)).index_add(
                0, batch_index, masked
            )
            selected_count = torch.bincount(
                batch_index[selected], minlength=num_graphs
            ).clamp_min(1)
            centered = raw - selected_sum[batch_index] / selected_count[
                batch_index, None
            ].to(dtype=raw.dtype)
            centered = torch.where(
                selected[:, None], centered, torch.zeros_like(centered)
            )

        norms = torch.linalg.vector_norm(centered, dim=-1)
        graph_max = norms.new_zeros((num_graphs,))
        graph_max.scatter_reduce_(
            0,
            batch_index,
            norms,
            reduce="amax",
            include_self=True,
        )
        scale = (self.config.max_step / graph_max.clamp_min(
            self.config.max_step
        )).clamp(max=1.0)
        return centered * scale[batch_index, None]

    def displacement(
        self,
        state: UnifiedSE3State,
        batch: torch.Tensor,
        *,
        update_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        selected = self._mask(
            update_mask,
            num_nodes=state.even_scalar.shape[0],
            device=state.even_scalar.device,
        )
        raw = self.vector_head(
            state.even_scalar,
            state.polar_vector,
        ).squeeze(1)
        raw = torch.sigmoid(self.gate(state.even_scalar)) * raw
        return self._center_and_bound(raw, batch, selected)

    def forward(
        self,
        node_irreps: torch.Tensor,
        positions: torch.Tensor,
        graph: Prepared3DGraph,
        *,
        graph_rebuilder: GeometryRebuilder | None = None,
        update_mask: torch.Tensor | None = None,
        node_role_id: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        current_positions = positions
        current_graph = graph
        total_delta = torch.zeros_like(positions)
        for _ in range(self.config.steps):
            state, _, _ = self.backbone.forward_features(
                node_irreps,
                current_positions,
                current_graph,
                node_role_id=node_role_id,
                condition=condition,
            )
            delta = self.displacement(
                state,
                current_graph.batch,
                update_mask=update_mask,
            ).to(dtype=current_positions.dtype)
            current_positions = current_positions + delta
            total_delta = total_delta + delta
            if graph_rebuilder is not None:
                current_graph = graph_rebuilder(
                    current_positions,
                    current_graph.batch,
                )

        output = dict(
            self.backbone(
                node_irreps,
                current_positions,
                current_graph,
                node_role_id=node_role_id,
                condition=condition,
            )
        )
        output["positions"] = current_positions
        output["coordinate_delta"] = total_delta
        return output


__all__ = [
    "CoordinateRefinementConfig",
    "ELACoordinateRefiner",
    "GeometryRebuilder",
]
