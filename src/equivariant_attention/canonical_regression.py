from __future__ import annotations

import torch
from torch import nn

from .canonical import ELA, ELAConfig, SparseGeometry
from .unified import prepare_3d_graph


class ELARegressionModel(nn.Module):
    """Minimal scalar graph-regression adapter for the canonical ELA model."""

    supports_graph_layout = False
    consumes_external_neighbors = True

    def __init__(
        self,
        *,
        node_dim: int,
        width: int = 128,
        depth: int = 8,
        cutoff: float = 5.0,
        num_rbf: int = 16,
    ) -> None:
        super().__init__()
        self.config = ELAConfig(
            input_irreps=f"{node_dim}x0e",
            output_irreps="1x0e",
            width=width,
            depth=depth,
            geometry=SparseGeometry(cutoff=cutoff, num_rbf=num_rbf),
        )
        self.model = ELA(self.config)

    def forward(
        self,
        node_feats: torch.Tensor,
        pos: torch.Tensor,
        *,
        batch: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        edge_index_is_validated: bool = False,
        edge_relation_id: torch.Tensor | None = None,
        readout_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del edge_index_is_validated
        if edge_index is None:
            raise ValueError("ELARegressionModel requires sparse edge_index")
        graph = prepare_3d_graph(
            batch,
            edge_index,
            edge_relation_id=edge_relation_id,
        )
        output = self.model(node_feats, pos, graph)
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
        return {**output, "graph_scalars": graph_scalars}


__all__ = ["ELARegressionModel"]
