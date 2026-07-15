from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .model import _scatter_mean


@dataclass(frozen=True)
class EGNNBaselineConfig:
    node_dim: int
    hidden_dim: int = 64
    num_layers: int = 3
    coord_update_scale: float = 0.1


class EGNNBaseline(nn.Module):
    """Small EGNN-style scalar regression baseline."""

    def __init__(self, config: EGNNBaselineConfig) -> None:
        super().__init__()
        _validate_config(config)
        self.config = config
        self.input = nn.Linear(config.node_dim, config.hidden_dim)
        self.layers = nn.ModuleList(
            [
                _EGNNLayer(
                    hidden_dim=config.hidden_dim,
                    coord_update_scale=config.coord_update_scale,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.node_scalar = nn.Linear(config.hidden_dim, 1)
        self.graph_scalar = nn.Linear(config.hidden_dim, 1)

    def forward(
        self,
        node_feats: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor | None = None,
        neighbor_index: torch.Tensor | None = None,
        neighbor_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if node_feats.ndim != 2 or node_feats.shape[1] != self.config.node_dim:
            msg = f"node_feats must have shape (N, {self.config.node_dim})"
            raise ValueError(msg)
        if pos.shape != (node_feats.shape[0], 3):
            msg = f"pos must have shape (N, 3), got {tuple(pos.shape)}"
            raise ValueError(msg)
        if node_feats.dtype != pos.dtype:
            pos = pos.to(dtype=node_feats.dtype)
        if batch is None:
            batch = torch.zeros(node_feats.shape[0], dtype=torch.long, device=node_feats.device)
        else:
            batch = batch.to(device=node_feats.device, dtype=torch.long)
        if neighbor_index is None:
            raise ValueError("EGNNBaseline requires neighbor_index")
        neighbor_index = neighbor_index.to(device=node_feats.device, dtype=torch.long)
        if neighbor_mask is None:
            neighbor_mask = torch.ones(neighbor_index.shape, dtype=torch.bool, device=node_feats.device)
        else:
            neighbor_mask = neighbor_mask.to(device=node_feats.device, dtype=torch.bool)

        h = self.input(node_feats)
        x = pos
        for layer in self.layers:
            h, x = layer(h, x, neighbor_index, neighbor_mask)

        graph_hidden = _scatter_mean(h, batch)
        return {
            "node_scalar": self.node_scalar(h),
            "graph_scalar": self.graph_scalar(graph_hidden),
            "node_vector": x - pos,
            "graph_vector": _scatter_mean(x - pos, batch),
            "node_tensor": pos.new_zeros((pos.shape[0], 3, 3)),
            "graph_tensor": pos.new_zeros((graph_hidden.shape[0], 3, 3)),
        }


class _EGNNLayer(nn.Module):
    def __init__(self, hidden_dim: int, coord_update_scale: float) -> None:
        super().__init__()
        self.coord_update_scale = coord_update_scale
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.coord_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.node_mlp = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        neighbor_index: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_i = h.unsqueeze(1).expand(-1, neighbor_index.shape[1], -1)
        h_j = h[neighbor_index]
        rel = pos.unsqueeze(1) - pos[neighbor_index]
        dist2 = rel.square().sum(dim=-1, keepdim=True)
        message = self.edge_mlp(torch.cat([h_i, h_j, dist2], dim=-1))
        message = message * neighbor_mask.unsqueeze(-1).to(message.dtype)
        count = neighbor_mask.sum(dim=1, keepdim=True).clamp_min(1).to(message.dtype)

        coord_gate = torch.tanh(self.coord_mlp(message)) * neighbor_mask.unsqueeze(-1).to(message.dtype)
        coord_delta = (rel * coord_gate).sum(dim=1) / count
        pos = pos + self.coord_update_scale * coord_delta

        pooled = message.sum(dim=1) / count
        h = h + self.node_mlp(torch.cat([h, pooled], dim=-1))
        return h, pos


def _validate_config(config: EGNNBaselineConfig) -> None:
    if config.node_dim <= 0:
        raise ValueError("node_dim must be positive")
    if config.hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if config.num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if config.coord_update_scale < 0:
        raise ValueError("coord_update_scale must be nonnegative")
