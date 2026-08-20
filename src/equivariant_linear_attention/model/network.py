"""The one canonical pair-centric equivariant TriELA network."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
from torch import nn

from ..context import InvariantContextEncoder
from ..graph import ELAGraph
from ..heads import DistogramHead
from ..irreps import IrrepLayout, split_irreps
from ..nn.geometry import GeometryContext
from ..nn.ops import interaction_index, segment_count, segment_sum
from ..nn.pair_embedding import PairEmbedding
from ..nn.pair_state import (
    BiomolecularPairContext,
    DensePairState,
    build_dense_pair_layout,
)
from ..nn.state import InputProjection, OutputProjection, ParityState
from .config import TriELAConfig
from .stage import TriELAStage


@dataclass(frozen=True)
class TriELAOutput:
    graph: ELAGraph
    pair_state: DensePairState
    distogram_logits: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


class TriELA(nn.Module):
    """Pair-centric O(3)-equivariant model for token-level 3D data."""

    def __init__(
        self,
        input_irreps: str,
        output_irreps: str = "1x0e",
        *,
        width: int = 128,
        pair_width: int = 64,
        triangle_hidden: int = 64,
        num_stages: int = 3,
        pair_blocks_per_stage: int = 4,
        local_blocks_per_stage: int = 2,
        pair_transition_factor: int = 4,
        pair_dropout: float = 0.1,
        local_points: int = 32,
        max_pair_tokens: int = 512,
        condition_dim: int = 0,
        order_dim: int = 0,
        pair_feature_dim: int = 0,
        distance_rbf_bins: int = 32,
        distance_max: float = 32.0,
        distogram_bins: int = 64,
        distogram_max: float = 32.0,
        update_positions: bool = False,
        max_coordinate_step: float = 0.25,
    ) -> None:
        super().__init__()
        self.config = TriELAConfig(
            input_irreps=input_irreps,
            output_irreps=output_irreps,
            width=width,
            pair_width=pair_width,
            triangle_hidden=triangle_hidden,
            num_stages=num_stages,
            pair_blocks_per_stage=pair_blocks_per_stage,
            local_blocks_per_stage=local_blocks_per_stage,
            pair_transition_factor=pair_transition_factor,
            pair_dropout=pair_dropout,
            local_points=local_points,
            max_pair_tokens=max_pair_tokens,
            condition_dim=condition_dim,
            order_dim=order_dim,
            pair_feature_dim=pair_feature_dim,
            distance_rbf_bins=distance_rbf_bins,
            distance_max=distance_max,
            distogram_bins=distogram_bins,
            distogram_max=distogram_max,
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
        feature_kwargs = {
            "scalar_width": config.width,
            "num_heads": config.num_heads,
            "pair_width": config.pair_width,
            "rbf_bins": config.distance_rbf_bins,
            "max_distance": config.distance_max,
            "pair_feature_dim": config.pair_feature_dim,
            "eps": config.eps,
        }
        self.pair_embedding = PairEmbedding(**feature_kwargs)
        total_node_blocks = config.num_stages * (
            config.pair_blocks_per_stage + config.local_blocks_per_stage
        )
        residual_scale = 0.1 / sqrt(total_node_blocks)
        self.stages = nn.ModuleList(
            [
                TriELAStage(
                    scalar_width=config.width,
                    pair_width=config.pair_width,
                    triangle_hidden=config.triangle_hidden,
                    num_heads=config.num_heads,
                    moment_rank=config.moment_rank,
                    relation_width=config.relation_width,
                    num_charts=config.num_charts,
                    pair_blocks=config.pair_blocks_per_stage,
                    local_blocks=config.local_blocks_per_stage,
                    pair_transition_factor=config.pair_transition_factor,
                    pair_dropout=config.pair_dropout,
                    local_points=config.local_points,
                    local_probe_rank=config.local_probe_rank,
                    local_scales=config.local_scales,
                    local_chunk_size=config.local_chunk_size,
                    rbf_bins=config.distance_rbf_bins,
                    max_distance=config.distance_max,
                    pair_feature_dim=config.pair_feature_dim,
                    residual_scale=residual_scale,
                    update_positions=config.update_positions,
                    eps=config.eps,
                )
                for _ in range(config.num_stages)
            ]
        )
        self.output_projection = OutputProjection(
            config.output_layout,
            scalar_width=config.width,
            num_heads=config.num_heads,
        )
        self.distogram_head = DistogramHead(
            pair_width=config.pair_width,
            num_bins=config.distogram_bins,
            max_distance=config.distogram_max,
        )

    @classmethod
    def from_config(cls, config: TriELAConfig) -> TriELA:
        if not isinstance(config, TriELAConfig):
            raise TypeError("config must be TriELAConfig")
        return cls(
            config.input_irreps,
            config.output_irreps,
            width=config.width,
            pair_width=config.pair_width,
            triangle_hidden=config.triangle_hidden,
            num_stages=config.num_stages,
            pair_blocks_per_stage=config.pair_blocks_per_stage,
            local_blocks_per_stage=config.local_blocks_per_stage,
            pair_transition_factor=config.pair_transition_factor,
            pair_dropout=config.pair_dropout,
            local_points=config.local_points,
            max_pair_tokens=config.max_pair_tokens,
            condition_dim=config.condition_dim,
            order_dim=config.order_dim,
            pair_feature_dim=config.pair_feature_dim,
            distance_rbf_bins=config.distance_rbf_bins,
            distance_max=config.distance_max,
            distogram_bins=config.distogram_bins,
            distogram_max=config.distogram_max,
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
        return self.config.update_positions

    def split_input(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        return split_irreps(self.input_irreps, value)

    def split_output(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        return split_irreps(self.output_irreps, value)

    def describe(self) -> dict[str, object]:
        return {
            "model": "TriELA",
            "graph": "ELAGraph",
            "input_irreps": str(self.input_irreps),
            "output_irreps": str(self.output_irreps),
            "width": self.config.width,
            "pair_width": self.config.pair_width,
            "num_stages": self.config.num_stages,
            "pair_blocks_per_stage": self.config.pair_blocks_per_stage,
            "local_blocks_per_stage": self.config.local_blocks_per_stage,
            "num_parameters": sum(parameter.numel() for parameter in self.parameters()),
            **self.config.contract(),
        }

    def extra_repr(self) -> str:
        return (
            f"input_irreps={str(self.input_irreps)!r}, "
            f"output_irreps={str(self.output_irreps)!r}, "
            f"width={self.config.width}, pair_width={self.config.pair_width}, "
            f"num_stages={self.config.num_stages}"
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

    def _run(
        self,
        graph: ELAGraph,
        pair_context: BiomolecularPairContext | None,
        *,
        with_aux: bool,
    ) -> ELAGraph | TriELAOutput:
        if not isinstance(graph, ELAGraph):
            raise TypeError("TriELA accepts exactly one ELAGraph")
        if graph.num_nodes == 0:
            raise ValueError("TriELA requires at least one node")
        if graph.x.shape[-1] != self.input_irreps.dim:
            raise ValueError(f"graph.x final dimension must be {self.input_irreps.dim}")
        batch = graph.batch_index
        interactions, num_interactions, _ = interaction_index(
            batch,
            graph.group,
            num_graphs=graph.num_graphs,
        )
        layout = build_dense_pair_layout(
            batch,
            graph.group,
            max_pair_tokens=self.config.max_pair_tokens,
        )
        if pair_context is not None:
            if not isinstance(pair_context, BiomolecularPairContext):
                raise TypeError("pair_context must be BiomolecularPairContext")
            pair_context.validate(
                num_nodes=graph.num_nodes,
                device=graph.x.device,
                dtype=graph.x.dtype,
            )
        positions = graph.pos
        geometry = GeometryContext.build(
            positions,
            interactions,
            num_segments=num_interactions,
            eps=self.config.eps,
        )
        state = self._initial_state(graph)
        pair = self.pair_embedding(state, positions, layout, pair_context)
        support = None
        coordinate_step = self.config.max_coordinate_step / (
            self.config.num_stages * self.config.local_blocks_per_stage
        )
        initial_positions = positions
        for stage in self.stages:
            output = stage(
                state,
                pair,
                positions,
                geometry,
                support,
                pair_context,
                update_mask=graph.update_mask,
                coordinate_step=coordinate_step,
            )
            state = output.state
            pair = output.pair
            positions = output.positions
            geometry = output.geometry
            support = output.support

        # Autocast is an execution policy, not a public graph-schema change.
        # Rejoin the input floating dtype before constructing the output graph
        # so x, positions, pooled outputs, and coordinate deltas stay coherent.
        node_output = self.output_projection(state).to(dtype=graph.x.dtype)
        graph_sum = segment_sum(node_output, batch, graph.num_graphs)
        graph_count = segment_count(batch, graph.num_graphs, dtype=node_output.dtype)
        graph_mean = graph_sum / graph_count.clamp_min(1.0).unsqueeze(-1)
        coordinate_delta = positions - initial_positions
        result = graph.with_output(
            x=node_output,
            pos=positions,
            graph_x=graph_mean,
            graph_sum=graph_sum,
            delta=coordinate_delta,
        )
        if not with_aux:
            return result
        logits = self.distogram_head(pair)
        coordinate_work_dtype = (
            torch.float64 if coordinate_delta.dtype == torch.float64 else torch.float32
        )
        coordinate_mean_square = (
            coordinate_delta.to(dtype=coordinate_work_dtype).square().mean()
        )
        positive_coordinate_motion = coordinate_mean_square > 0.0
        safe_coordinate_square = torch.where(
            positive_coordinate_motion,
            coordinate_mean_square,
            torch.ones_like(coordinate_mean_square),
        )
        coordinate_rms = torch.where(
            positive_coordinate_motion,
            torch.sqrt(safe_coordinate_square),
            torch.zeros_like(coordinate_mean_square),
        )
        diagnostics = {
            "pair_slots": pair.lengths,
            "active_pairs": pair.pair_mask.sum(dim=(1, 2)),
            "coordinate_rms": coordinate_rms.reshape(1),
        }
        return TriELAOutput(
            graph=result,
            pair_state=pair,
            distogram_logits=logits,
            diagnostics=diagnostics,
        )

    def forward(
        self,
        graph: ELAGraph,
        pair_context: BiomolecularPairContext | None = None,
    ) -> ELAGraph:
        output = self._run(graph, pair_context, with_aux=False)
        assert isinstance(output, ELAGraph)
        return output

    def forward_with_aux(
        self,
        graph: ELAGraph,
        pair_context: BiomolecularPairContext | None = None,
    ) -> TriELAOutput:
        output = self._run(graph, pair_context, with_aux=True)
        assert isinstance(output, TriELAOutput)
        return output


__all__ = ["TriELA", "TriELAOutput"]
