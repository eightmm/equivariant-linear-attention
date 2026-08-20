"""One pair-centric TriELA stage."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..nn.equivariant_transition import EquivariantTransitionBlock
from ..nn.geometry import GeometryContext
from ..nn.global_ela import GlobalELABlock
from ..nn.local_ela import LocalELABlock
from ..nn.local_support import LocalSupport, build_local_support
from ..nn.manifold import QuotientCoordinateUpdate
from ..nn.pair_adapters import PairContextInjection, PairToNodeSummary
from ..nn.pair_embedding import NodeGeometryToPair
from ..nn.pair_state import BiomolecularPairContext, DensePairState
from ..nn.state import ParityState
from ..nn.triangle import TrianglePairBlock


@dataclass(frozen=True)
class TriELAStageOutput:
    state: ParityState
    pair: DensePairState
    positions: torch.Tensor
    geometry: GeometryContext
    support: LocalSupport | None
    node_metric: torch.Tensor


class TriELAStage(nn.Module):
    """Pair refresh, repeated pair/global reasoning, then local geometry."""

    def __init__(
        self,
        *,
        scalar_width: int,
        pair_width: int,
        triangle_hidden: int,
        num_heads: int,
        moment_rank: int,
        relation_width: int,
        num_charts: int,
        pair_blocks: int,
        local_blocks: int,
        pair_transition_factor: int,
        pair_dropout: float,
        local_points: int,
        local_probe_rank: int,
        local_scales: int,
        local_chunk_size: int,
        rbf_bins: int,
        max_distance: float,
        pair_feature_dim: int,
        residual_scale: float,
        update_positions: bool,
        eps: float,
    ) -> None:
        super().__init__()
        feature_kwargs = {
            "scalar_width": scalar_width,
            "num_heads": num_heads,
            "pair_width": pair_width,
            "rbf_bins": rbf_bins,
            "max_distance": max_distance,
            "pair_feature_dim": pair_feature_dim,
            "eps": eps,
        }
        self.node_to_pair = NodeGeometryToPair(**feature_kwargs)
        self.pair_blocks = nn.ModuleList(
            [
                TrianglePairBlock(
                    pair_width=pair_width,
                    triangle_hidden=triangle_hidden,
                    transition_factor=pair_transition_factor,
                    dropout=pair_dropout,
                )
                for _ in range(pair_blocks)
            ]
        )
        self.pair_to_node = nn.ModuleList(
            [
                PairToNodeSummary(
                    pair_width=pair_width,
                    context_width=pair_width,
                    eps=eps,
                )
                for _ in range(pair_blocks)
            ]
        )
        self.pair_injection = nn.ModuleList(
            [
                PairContextInjection(
                    context_width=pair_width,
                    scalar_width=scalar_width,
                    num_heads=num_heads,
                )
                for _ in range(pair_blocks)
            ]
        )
        self.global_blocks = nn.ModuleList(
            [
                GlobalELABlock(
                    scalar_width=scalar_width,
                    pair_width=pair_width,
                    num_heads=num_heads,
                    moment_rank=moment_rank,
                    relation_width=relation_width,
                    num_charts=num_charts,
                    residual_scale=residual_scale,
                    eps=eps,
                )
                for _ in range(pair_blocks)
            ]
        )
        self.global_transitions = nn.ModuleList(
            [
                EquivariantTransitionBlock(
                    scalar_width=scalar_width,
                    num_heads=num_heads,
                    residual_scale=residual_scale,
                    eps=eps,
                )
                for _ in range(pair_blocks)
            ]
        )
        self.local_blocks = nn.ModuleList(
            [
                LocalELABlock(
                    scalar_width=scalar_width,
                    pair_width=pair_width,
                    num_heads=num_heads,
                    moment_rank=moment_rank,
                    probe_rank=local_probe_rank,
                    num_scales=local_scales,
                    max_points=local_points,
                    chunk_size=local_chunk_size,
                    residual_scale=residual_scale,
                    eps=eps,
                )
                for _ in range(local_blocks)
            ]
        )
        self.local_transitions = nn.ModuleList(
            [
                EquivariantTransitionBlock(
                    scalar_width=scalar_width,
                    num_heads=num_heads,
                    residual_scale=residual_scale,
                    eps=eps,
                )
                for _ in range(local_blocks)
            ]
        )
        self.coordinate_updates: nn.ModuleList | None = None
        if update_positions:
            self.coordinate_updates = nn.ModuleList(
                [
                    QuotientCoordinateUpdate(
                        scalar_width=scalar_width,
                        num_heads=num_heads,
                        eps=eps,
                    )
                    for _ in range(local_blocks)
                ]
            )
        self.local_points = int(local_points)
        self.local_chunk_size = int(local_chunk_size)
        self.eps = float(eps)

    def _support(self, geometry: GeometryContext) -> LocalSupport:
        return build_local_support(
            geometry,
            max_points=self.local_points,
            chunk_size=self.local_chunk_size,
            eps=self.eps,
        )

    def forward(
        self,
        state: ParityState,
        pair: DensePairState,
        positions: torch.Tensor,
        geometry: GeometryContext,
        support: LocalSupport | None,
        metadata: BiomolecularPairContext | None,
        *,
        update_mask: torch.Tensor | None,
        coordinate_step: float,
    ) -> TriELAStageOutput:
        pair = pair.with_z(
            pair.z
            + self.node_to_pair(
                state,
                positions,
                pair,
                metadata,
            )
        )
        node_metric: torch.Tensor | None = None
        for pair_block, summarize, inject, global_block, transition in zip(
            self.pair_blocks,
            self.pair_to_node,
            self.pair_injection,
            self.global_blocks,
            self.global_transitions,
            strict=True,
        ):
            pair = pair.with_z(pair_block(pair.z, pair.pair_mask))
            context = summarize(pair)
            state = inject(state, context)
            global_output = global_block(state, geometry, context)
            state = transition(global_output.state)
            node_metric = global_output.node_metric
        if node_metric is None:
            raise RuntimeError("TriELAStage requires at least one pair/global block")

        if support is None:
            support = self._support(geometry)
        for local_index, (local_block, transition) in enumerate(
            zip(self.local_blocks, self.local_transitions, strict=True)
        ):
            state = transition(local_block(state, geometry, support, pair).state)
            if self.coordinate_updates is not None:
                updater = self.coordinate_updates[local_index]
                delta = updater(
                    state,
                    positions,
                    geometry.index,
                    num_segments=geometry.num_segments,
                    counts=geometry.counts,
                    node_metric=node_metric,
                    update_mask=update_mask,
                    max_step=coordinate_step,
                )
                positions = positions + delta
                geometry = GeometryContext.build(
                    positions,
                    geometry.index,
                    num_segments=geometry.num_segments,
                    eps=self.eps,
                )
                support = (
                    self._support(geometry)
                    if local_index + 1 < len(self.local_blocks)
                    else None
                )
        return TriELAStageOutput(
            state=state,
            pair=pair,
            positions=positions,
            geometry=geometry,
            support=support,
            node_metric=node_metric,
        )


__all__ = ["TriELAStage", "TriELAStageOutput"]
