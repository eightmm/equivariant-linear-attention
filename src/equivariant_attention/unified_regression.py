from __future__ import annotations

import torch
from torch import nn

from .equivariant_linear_attention import (
    EquivariantLinearAttention,
    EquivariantLinearAttentionConfig,
)
from .unified import prepare_3d_graph


class UnifiedRegressionModel(nn.Module):
    """Scalar graph-regression adapter for equivariant linear attention."""

    supports_graph_layout = False
    consumes_external_neighbors = True

    def __init__(
        self,
        *,
        node_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        local_rank: int,
        local_cutoff: float,
        num_rbf: int = 16,
        num_node_roles: int = 0,
        residual_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.config = EquivariantLinearAttentionConfig(
            input_irreps=f"{node_dim}x0e",
            output_irreps="1x0e",
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            local_rank=local_rank,
            local_cutoff=local_cutoff,
            num_rbf=num_rbf,
            num_node_roles=num_node_roles,
            residual_dropout=residual_dropout,
            drop_path_rate=drop_path_rate,
            norm_eps=norm_eps,
        )
        self.model = EquivariantLinearAttention(self.config)

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
        if edge_index is None:
            raise ValueError("UnifiedRegressionModel requires sparse edge_index")
        graph = prepare_3d_graph(
            batch,
            edge_index,
            edge_relation_id=edge_relation_id,
        )
        output = self.model(
            node_feats,
            pos,
            graph,
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
                batch,
                output["node_irreps"] * selected[:, None],
            )
            counts = output["node_irreps"].new_zeros((num_graphs, 1))
            counts.index_add_(0, batch, selected[:, None])
            graph_scalars = graph_scalars / counts.clamp_min(1.0)
        return {
            **output,
            "graph_scalars": graph_scalars,
        }


__all__ = ["UnifiedRegressionModel"]
