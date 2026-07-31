from __future__ import annotations

import torch
from torch import nn

from .equivariant_linear_attention import EquivariantLinearAttentionConfig
from .implicit_spatial import ImplicitSpatialKernelConfig
from .spatial_ablation import (
    SpatialOperatorAblationConfig,
    SpatialOperatorAblationModel,
    SpatialOperatorArm,
    empty_prepared_graph_like,
)
from .unified import prepare_3d_graph


class SpatialOperatorRegressionModel(nn.Module):
    """Graph-regression adapter for matched explicit/implicit/hybrid ELA arms."""

    supports_graph_layout = False

    def __init__(
        self,
        *,
        arm: SpatialOperatorArm,
        node_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        local_rank: int,
        local_cutoff: float,
        num_rbf: int = 16,
        num_node_roles: int = 0,
        implicit_scales: tuple[float, ...] | None = None,
        implicit_residual_scale_init: float = 0.0,
        implicit_every: int = 1,
        implicit_chunk_size: int = 2048,
    ) -> None:
        super().__init__()
        if arm not in {"explicit", "implicit", "hybrid"}:
            raise ValueError("arm must be explicit, implicit, or hybrid")
        scales = (
            implicit_scales
            if implicit_scales is not None
            else (local_cutoff, 2.0 * local_cutoff, 4.0 * local_cutoff)
        )
        self.arm: SpatialOperatorArm = arm
        self.consumes_external_neighbors = arm in {"explicit", "hybrid"}
        self.config = SpatialOperatorAblationConfig(
            model=EquivariantLinearAttentionConfig(
                input_irreps=f"{node_dim}x0e",
                output_irreps="1x0e",
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                local_rank=local_rank,
                local_cutoff=local_cutoff,
                num_rbf=num_rbf,
                num_node_roles=num_node_roles,
                coordinate_updates=False,
            ),
            implicit=ImplicitSpatialKernelConfig(
                scales=scales,
                order=2,
                exclude_self=True,
                normalization="one_plus_mass",
                learnable_scale_weights=True,
                chunk_size=implicit_chunk_size,
            ),
            implicit_residual_scale_init=implicit_residual_scale_init,
            implicit_every=implicit_every,
        )
        self.model = SpatialOperatorAblationModel(self.config, arm=arm)

    def forward(
        self,
        node_feats: torch.Tensor,
        pos: torch.Tensor,
        *,
        batch: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        edge_index_is_validated: bool = False,
        edge_relation_id: torch.Tensor | None = None,
        node_role_id: torch.Tensor | None = None,
        readout_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del edge_index_is_validated
        if self.consumes_external_neighbors and edge_index is None:
            raise ValueError(f"{self.arm} spatial arm requires sparse edge_index")
        if edge_index is None:
            edge_index = torch.empty((2, 0), device=batch.device, dtype=torch.long)
        graph = prepare_3d_graph(
            batch,
            edge_index,
            edge_relation_id=edge_relation_id,
        )
        no_edge_graph = empty_prepared_graph_like(graph)
        output = self.model(
            node_feats,
            pos,
            graph,
            no_edge_graph=no_edge_graph,
            node_role_id=node_role_id,
        )
        if readout_mask is None:
            graph_scalars = output["graph_irreps"]
        else:
            if readout_mask.shape != batch.shape:
                raise ValueError("readout_mask must have shape (N,)")
            selected = readout_mask.to(dtype=output["node_irreps"].dtype)
            num_graphs = graph.graph_layout.num_graphs
            graph_scalars = output["node_irreps"].new_zeros((num_graphs, 1))
            graph_scalars.index_add_(
                0,
                graph.batch,
                output["node_irreps"] * selected[:, None],
            )
            counts = output["node_irreps"].new_zeros((num_graphs, 1))
            counts.index_add_(0, graph.batch, selected[:, None])
            graph_scalars = graph_scalars / counts.clamp_min(1.0)
        return {
            **output,
            "graph_scalars": graph_scalars,
        }


__all__ = ["SpatialOperatorRegressionModel"]
