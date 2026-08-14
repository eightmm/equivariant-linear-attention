"""Canonical tensor-fused edge-free equivariant linear-attention network."""

from __future__ import annotations

from math import sqrt

import torch
from torch import nn

from ..context import ELAFeatures, InvariantContextEncoder
from ..graph import ELAGraph
from ..irreps import IrrepLayout, split_irreps
from ..nn.geometry import GeometryContext, chart_density
from ..nn.layer import EdgeFreeELALayer
from ..nn.local_support import LocalSupport, build_local_support
from ..nn.manifold import QuotientCoordinateUpdate
from ..nn.ops import interaction_index, segment_count, segment_sum
from ..nn.state import InputProjection, OutputProjection, ParityState
from .config import ELAConfig


class ELA(nn.Module):
    """Parity-complete edge-free O(3)-equivariant linear-attention network."""

    def __init__(
        self,
        input_irreps: str,
        output_irreps: str = "1x0e",
        *,
        width: int = 128,
        depth: int = 8,
        condition_dim: int = 0,
        order_dim: int = 0,
        update_positions: bool = False,
        max_coordinate_step: float = 0.25,
        num_local_charts: int = 16,
        length_scale: float = 10.0,
        density_bandwidths: tuple[float, ...] = (),
        density_charts: int = 16,
        local_points: int = 0,
    ) -> None:
        super().__init__()
        self.config = ELAConfig(
            input_irreps=input_irreps,
            output_irreps=output_irreps,
            width=width,
            depth=depth,
            features=ELAFeatures(condition_dim=condition_dim, order_dim=order_dim),
            update_positions=update_positions,
            max_coordinate_step=max_coordinate_step,
            num_local_charts=num_local_charts,
            length_scale=length_scale,
            density_bandwidths=tuple(density_bandwidths),
            density_charts=density_charts,
            local_points=local_points,
        )
        config = self.config
        self.input_projection = InputProjection(
            config.input_layout,
            scalar_width=config.width,
            num_heads=config.num_heads,
        )
        self.context_encoder = InvariantContextEncoder(
            features=config.features,
            width=config.width,
        )
        block_scale = 0.1 / sqrt(config.depth)
        self.layers = nn.ModuleList(
            [
                EdgeFreeELALayer(
                    scalar_width=config.width,
                    num_heads=config.num_heads,
                    moment_rank=config.moment_rank,
                    relation_width=config.relation_width,
                    num_charts=config.num_charts,
                    num_local_charts=config.num_local_charts,
                    residual_scale=block_scale,
                    eps=config.eps,
                    local_points=config.local_points,
                    local_probe_rank=config.local_probe_rank,
                    local_scales=config.local_scales,
                    local_chunk_size=config.local_chunk_size,
                )
                for _ in range(config.depth)
            ]
        )
        self.output_projection = OutputProjection(
            config.output_layout,
            scalar_width=config.width,
            num_heads=config.num_heads,
        )
        self.density_projection: nn.Linear | None = None
        if config.density_bandwidths:
            self.density_projection = nn.Linear(
                len(config.density_bandwidths), config.width
            )
            # Start as an exact no-op so the channel has to earn its use.
            nn.init.zeros_(self.density_projection.weight)
            nn.init.zeros_(self.density_projection.bias)
        self.coordinate_update: QuotientCoordinateUpdate | None = None
        if config.update_positions:
            self.coordinate_update = QuotientCoordinateUpdate(
                scalar_width=config.width,
                num_heads=config.num_heads,
                eps=config.eps,
            )

    @classmethod
    def from_config(cls, config: ELAConfig) -> ELA:
        if not isinstance(config, ELAConfig):
            raise TypeError("config must be ELAConfig")
        return cls(
            config.input_irreps,
            config.output_irreps,
            width=config.width,
            depth=config.depth,
            condition_dim=config.features.condition_dim,
            order_dim=config.features.order_dim,
            update_positions=config.update_positions,
            max_coordinate_step=config.max_coordinate_step,
            num_local_charts=config.num_local_charts,
            length_scale=config.length_scale,
            density_bandwidths=config.density_bandwidths,
            density_charts=config.density_charts,
            local_points=config.local_points,
        )

    @property
    def input_irreps(self) -> IrrepLayout:
        return self.config.input_layout

    @property
    def output_irreps(self) -> IrrepLayout:
        return self.config.output_layout

    @property
    def updates_positions(self) -> bool:
        return self.coordinate_update is not None

    def split_input(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        return split_irreps(self.input_irreps, value)

    def split_output(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        return split_irreps(self.output_irreps, value)

    def describe(self) -> dict[str, object]:
        return {
            "model": "ELA",
            "graph": "ELAGraph",
            "input_irreps": str(self.input_irreps),
            "output_irreps": str(self.output_irreps),
            "width": self.config.width,
            "depth": self.config.depth,
            "condition_dim": self.config.features.condition_dim,
            "order_dim": self.config.features.order_dim,
            "update_positions": self.updates_positions,
            "max_coordinate_step": self.config.max_coordinate_step,
            "num_parameters": sum(parameter.numel() for parameter in self.parameters()),
            **self.config.contract(),
        }

    def extra_repr(self) -> str:
        return (
            f"input_irreps={str(self.input_irreps)!r}, "
            f"output_irreps={str(self.output_irreps)!r}, "
            f"width={self.config.width}, depth={self.config.depth}, "
            f"update_positions={self.updates_positions}"
        )

    def _initial_state(
        self, graph: ELAGraph, density: torch.Tensor | None = None
    ) -> ParityState:
        state = self.input_projection(graph.x)
        context = self.context_encoder(graph)
        if density is not None and self.density_projection is not None:
            projected = self.density_projection(density.to(dtype=state.even_scalar.dtype))
            context = projected if context is None else context + projected
        if context is None:
            return state
        return ParityState(
            state.even_scalar + context,
            state.odd_scalar,
            state.polar_vector,
            state.axial_vector,
            state.even_tensor,
            state.odd_tensor,
        )

    def _local_support(self, geometry: GeometryContext) -> LocalSupport | None:
        """Transient kNN support for the non-canonical local-jet branch."""

        if not self.config.uses_local_jet:
            return None
        return build_local_support(
            geometry,
            max_points=self.config.local_points,
            chunk_size=self.config.local_chunk_size,
            eps=self.config.eps,
        )

    def forward(self, graph: ELAGraph) -> ELAGraph:
        if not isinstance(graph, ELAGraph):
            raise TypeError("ELA accepts exactly one ELAGraph")
        if graph.num_nodes == 0:
            raise ValueError("ELA requires at least one node")
        if graph.x.shape[-1] != self.input_irreps.dim:
            raise ValueError(f"graph.x final dimension must be {self.input_irreps.dim}")

        batch = graph.batch_index
        interactions, num_interactions, interaction_counts = interaction_index(
            batch,
            graph.group,
        )
        positions = graph.pos
        density = None
        if self.config.density_bandwidths:
            density = chart_density(
                positions,
                interactions,
                num_segments=num_interactions,
                num_charts=self.config.density_charts,
                bandwidths=self.config.density_bandwidths,
                length_scale=self.config.length_scale,
                eps=self.config.eps,
            )
        state = self._initial_state(graph, density)
        total_delta = torch.zeros_like(positions)

        if self.coordinate_update is None:
            # Coordinates are immutable, so their compact monomial basis, all
            # component metadata, and the transient local support are shared by
            # every layer.
            geometry = GeometryContext.build(
                positions,
                interactions,
                num_segments=num_interactions,
                eps=self.config.eps,
                length_scale=self.config.length_scale,
                num_seeds=self.config.num_local_charts,
            )
            support = self._local_support(geometry)
            for layer in self.layers:
                state = layer(state, geometry, support).state
        else:
            stage_step = self.config.max_coordinate_step / self.config.depth
            for layer in self.layers:
                geometry = GeometryContext.build(
                    positions,
                    interactions,
                    num_segments=num_interactions,
                    eps=self.config.eps,
                    length_scale=self.config.length_scale,
                    num_seeds=self.config.num_local_charts,
                )
                output = layer(state, geometry, self._local_support(geometry))
                state = output.state
                delta = self.coordinate_update(
                    state,
                    positions,
                    interactions,
                    num_segments=num_interactions,
                    counts=interaction_counts,
                    node_metric=output.node_metric,
                    update_mask=graph.update_mask,
                    max_step=stage_step,
                )
                positions = positions + delta
                total_delta = total_delta + delta

        node_output = self.output_projection(state)
        graph_sum = segment_sum(node_output, batch, graph.num_graphs)
        graph_count = segment_count(batch, graph.num_graphs, dtype=node_output.dtype)
        graph_mean = graph_sum / graph_count.clamp_min(1.0).unsqueeze(-1)
        return graph.with_output(
            x=node_output,
            pos=positions,
            graph_x=graph_mean,
            graph_sum=graph_sum,
            delta=total_delta,
        )


__all__ = ["ELA"]
