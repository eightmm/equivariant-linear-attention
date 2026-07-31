from __future__ import annotations

import torch
from torch import nn

from .canonical_se3 import CanonicalMultipoleSE3Core
from .equivariant_linear_attention import (
    EquivariantLinearAttention,
    EquivariantLinearAttentionConfig,
)
from .unified import (
    Prepared3DGraph,
    Unified3DConfig,
    prepare_3d_graph,
)


class _FrozenCanonicalUnifiedAttention(nn.Module):
    """The 2026-07-30 canonical unified core used by historical receipts."""

    attention_kind = "canonical_multipole_parity_factorized_moment"

    def __init__(self, config: Unified3DConfig) -> None:
        super().__init__()
        self.config = config
        self.core = CanonicalMultipoleSE3Core(
            input_irreps=config.input_layout,
            output_irreps=config.output_layout,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            local_rank=config.local_rank,
            local_cutoff=config.local_cutoff,
            num_rbf=config.num_rbf,
            num_node_roles=config.num_node_roles,
            num_edge_relations=config.num_edge_relations,
            relation_cutoffs=config.relation_cutoffs,
            residual_scale_init=config.residual_scale_init,
            eps=config.eps,
        )
        with torch.no_grad():
            for block in self.core.blocks:
                weight = block.local_chiral_scalar_out.weight
                weight.zero_()
                rows = torch.arange(weight.shape[0], device=weight.device)
                weight[rows, rows.remainder(weight.shape[1])] = 1.0

    def forward(
        self,
        node_irreps: torch.Tensor,
        pos: torch.Tensor,
        graph: Prepared3DGraph,
        *,
        node_role_id: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if not isinstance(graph, Prepared3DGraph) or not graph._validated:
            raise TypeError("graph must be a validated Prepared3DGraph")
        if node_irreps.shape[0] != graph.num_nodes:
            raise ValueError("node_irreps node count must match graph")
        if pos.shape != (graph.num_nodes, 3):
            raise ValueError("pos must have shape (N, 3)")
        if node_irreps.device != graph.device or pos.device != graph.device:
            raise ValueError("model inputs and graph must share one device")
        return self.core(
            node_irreps,
            pos,
            graph.batch,
            graph.graph_layout,
            graph.neighbors,
            node_role_id=node_role_id,
        )


class _RegressionAdapter(nn.Module):
    supports_graph_layout = False
    consumes_external_neighbors = True

    model: nn.Module

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
            raise ValueError(f"{type(self).__name__} requires sparse edge_index")
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


class UnifiedRegressionModel(_RegressionAdapter):
    """Frozen scalar adapter for the historical ``unified_multipole`` arm."""

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
    ) -> None:
        super().__init__()
        self.config = Unified3DConfig(
            input_irreps=f"{node_dim}x0e",
            output_irreps="1x0e",
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            local_rank=local_rank,
            local_cutoff=local_cutoff,
            num_rbf=num_rbf,
            num_node_roles=num_node_roles,
        )
        self.model = _FrozenCanonicalUnifiedAttention(self.config)


class EquivariantLinearAttentionRegressionModel(_RegressionAdapter):
    """Scalar adapter for the refined equivariant linear-attention stack."""

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


__all__ = [
    "EquivariantLinearAttentionRegressionModel",
    "UnifiedRegressionModel",
]
