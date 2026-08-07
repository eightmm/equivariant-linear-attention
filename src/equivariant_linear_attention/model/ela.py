"""Canonical tensor-fused edge-free equivariant linear-attention model."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, sqrt

import torch
from torch import nn

from ..context import ELAFeatures, InvariantContextEncoder
from ..graph import ELAGraph
from ..irreps import IrrepLayout, split_irreps
from ..nn.geometry import GeometryContext
from ..nn.layer import EdgeFreeELALayer
from ..nn.manifold import QuotientCoordinateUpdate
from ..nn.ops import interaction_index, segment_count, segment_sum
from ..nn.state import InputProjection, OutputProjection, ParityState


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    numeric = float(value)
    if not isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return numeric


@dataclass(frozen=True, slots=True)
class ELAConfig:
    input_irreps: str
    output_irreps: str = "1x0e"
    width: int = 128
    depth: int = 8
    features: ELAFeatures = field(default_factory=ELAFeatures)
    update_positions: bool = False
    max_coordinate_step: float = 0.25

    def __post_init__(self) -> None:
        _positive_integer("width", self.width)
        _positive_integer("depth", self.depth)
        if self.width < 16:
            raise ValueError("width must be at least 16")
        if not isinstance(self.features, ELAFeatures):
            raise TypeError("features must be ELAFeatures")
        if not isinstance(self.update_positions, bool):
            raise TypeError("update_positions must be a bool")
        _positive_real("max_coordinate_step", self.max_coordinate_step)
        if self.input_layout.max_degree > 2 or self.output_layout.max_degree > 2:
            raise ValueError("persistent input and output irreps must have l<=2")

    @property
    def input_layout(self) -> IrrepLayout:
        return IrrepLayout.parse(self.input_irreps)

    @property
    def output_layout(self) -> IrrepLayout:
        return IrrepLayout.parse(self.output_irreps)

    @property
    def num_heads(self) -> int:
        target = max(1, min(16, self.width // 16))
        for candidate in range(target, 0, -1):
            if self.width % candidate == 0:
                return candidate
        return 1

    @property
    def moment_rank(self) -> int:
        return max(4, min(12, self.width // 16))

    @property
    def relation_width(self) -> int:
        return max(8, self.width // self.num_heads)

    @property
    def num_charts(self) -> int:
        return max(2, min(8, self.width // 24))

    @property
    def eps(self) -> float:
        return 1e-8

    def contract(self) -> dict[str, object]:
        return {
            "architecture": "edge_free_equivariant_linear_attention",
            "public_contract": "ELAGraph -> ELA -> ELAGraph",
            "persistent_irreps": "0e + 0o + 1o + 1e + 2e + 2o",
            "relative_moment_order": 4,
            "transient_irreps": ("3o", "4e"),
            "relation_operator": "single_fused_self_adjoint_gram_factor",
            "relation_value_layout": "packed_all_irreps",
            "coordinate_basis": "35d_compact_symmetric_degree_0_to_4",
            "krylov_basis": "three_term_graphwise_irrep_orthogonal",
            "coordinate_manifold": "natural_gradient_SE3_quotient_plus_shape",
            "explicit_edges": False,
            "pair_state": False,
            "derived_num_heads": self.num_heads,
            "derived_moment_rank": self.moment_rank,
            "derived_num_charts": self.num_charts,
        }


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
                    residual_scale=block_scale,
                    eps=config.eps,
                )
                for _ in range(config.depth)
            ]
        )
        self.output_projection = OutputProjection(
            config.output_layout,
            scalar_width=config.width,
            num_heads=config.num_heads,
        )
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

    def _initial_state(self, graph: ELAGraph) -> ParityState:
        state = self.input_projection(graph.x)
        context = self.context_encoder(graph)
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
        state = self._initial_state(graph)
        total_delta = torch.zeros_like(positions)

        if self.coordinate_update is None:
            # Coordinates are immutable, so their compact monomial basis and all
            # component metadata are shared by every layer.
            geometry = GeometryContext.build(
                positions,
                interactions,
                num_segments=num_interactions,
                eps=self.config.eps,
            )
            for layer in self.layers:
                state = layer(state, geometry).state
        else:
            stage_step = self.config.max_coordinate_step / self.config.depth
            for layer in self.layers:
                geometry = GeometryContext.build(
                    positions,
                    interactions,
                    num_segments=num_interactions,
                    eps=self.config.eps,
                )
                output = layer(state, geometry)
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


__all__ = ["ELA", "ELAConfig"]
