from __future__ import annotations

import torch
from torch import nn

from .moment import (
    _batched_complete_graph_edges,
    _bounded_centered_displacement,
    _graph_metadata,
    _local_edge_index_components,
    _readout_metadata,
    _scatter_mean,
    _validated_local_edge_index,
)


class _StaticEGNNLayer(nn.Module):
    """Static-coordinate QM9 EGNN layer for internal benchmark comparisons."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        _validate_positive_int("hidden_dim", hidden_dim)
        self.hidden_dim = hidden_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.edge_gate = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        value: torch.Tensor,
        pos: torch.Tensor,
        receiver: torch.Tensor,
        sender: torch.Tensor,
    ) -> torch.Tensor:
        edge, aggregate = self._edge_and_aggregate(
            value,
            pos,
            receiver,
            sender,
        )
        del edge
        return value + self.node_mlp(torch.cat([value, aggregate], dim=-1))

    def forward_with_edge(
        self,
        value: torch.Tensor,
        pos: torch.Tensor,
        receiver: torch.Tensor,
        sender: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edge, aggregate = self._edge_and_aggregate(
            value,
            pos,
            receiver,
            sender,
        )
        updated = value + self.node_mlp(torch.cat([value, aggregate], dim=-1))
        return updated, edge

    def _edge_and_aggregate(
        self,
        value: torch.Tensor,
        pos: torch.Tensor,
        receiver: torch.Tensor,
        sender: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        displacement = pos[receiver] - pos[sender]
        squared_distance = displacement.square().sum(dim=-1, keepdim=True)
        edge_input = torch.cat(
            [value[receiver], value[sender], squared_distance.to(dtype=value.dtype)],
            dim=-1,
        )
        edge = self.edge_mlp(edge_input)
        message = edge * self.edge_gate(edge)
        aggregate = value.new_zeros(value.shape).index_add(0, receiver, message)
        return edge, aggregate


class _StaticEGNNBaseline(nn.Module):
    """Private, parameter-matched EGNN-style baseline; not a public layer family."""

    attention_kind = "internal_static_egnn_baseline"
    symmetry = "O3_invariant"
    benchmark_only = True

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 91,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        _validate_positive_int("node_dim", node_dim)
        _validate_positive_int("hidden_dim", hidden_dim)
        _validate_positive_int("num_layers", num_layers)
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [_StaticEGNNLayer(hidden_dim) for _ in range(num_layers)]
        )
        self.scalar_out_norm = nn.LayerNorm(hidden_dim)
        self.scalar_out = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.scalar_out.weight)
        nn.init.zeros_(self.scalar_out.bias)

    def forward(
        self,
        node_feats: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor | None = None,
        *,
        edge_index: torch.Tensor | None = None,
        edge_index_is_validated: bool = False,
        readout_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        node_feats, pos, batch, num_graphs, graph_counts = self._check_inputs(
            node_feats, pos, batch
        )
        receiver, sender = _resolve_egnn_edges(
            batch,
            graph_counts,
            edge_index=edge_index,
            edge_index_is_validated=edge_index_is_validated,
            num_nodes=node_feats.shape[0],
            device=node_feats.device,
        )

        value = self.embedding(node_feats)
        for layer in self.layers:
            value = layer(value, pos, receiver, sender)
        node_scalar = self.scalar_out(self.scalar_out_norm(value))
        pool_mask, pool_batch, pool_counts = _readout_metadata(
            readout_mask,
            batch,
            num_graphs=num_graphs,
            graph_counts=graph_counts,
        )
        pooled_scalar = (
            node_scalar if pool_mask is None else node_scalar[pool_mask]
        )
        graph_scalar = _scatter_mean(
            pooled_scalar,
            pool_batch,
            num_graphs,
            pool_counts,
        )
        return {"graph_scalars": graph_scalar}

    def _check_inputs(
        self,
        node_feats: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
        if node_feats.ndim != 2 or node_feats.shape[1] != self.node_dim:
            raise ValueError(f"node_feats must have shape (N, {self.node_dim})")
        if node_feats.shape[0] == 0:
            raise ValueError("at least one node is required")
        if pos.shape != (node_feats.shape[0], 3):
            raise ValueError(f"pos must have shape (N, 3), got {tuple(pos.shape)}")
        if not torch.is_floating_point(node_feats):
            raise TypeError("node_feats must be a floating point tensor")
        if pos.dtype not in {torch.float32, torch.float64}:
            raise TypeError("pos must use float32 or float64 coordinates")
        if node_feats.device != pos.device:
            raise ValueError("node_feats and pos must be on the same device")
        if not torch.isfinite(node_feats).all() or not torch.isfinite(pos).all():
            raise ValueError("node_feats and pos must be finite")
        if batch is None:
            batch = torch.zeros(
                node_feats.shape[0], dtype=torch.long, device=node_feats.device
            )
        elif batch.shape != (node_feats.shape[0],):
            raise ValueError(f"batch must have shape (N,), got {tuple(batch.shape)}")
        else:
            if batch.dtype not in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }:
                raise TypeError("batch must use an integer dtype")
            if batch.device != node_feats.device:
                raise ValueError(
                    "batch, node_feats, and pos must be on the same device"
                )
            batch = batch.to(dtype=torch.long)
        num_graphs, graph_counts = _graph_metadata(batch)
        return node_feats, pos, batch, num_graphs, graph_counts


class _EGNNCoordinateUpdater(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.coordinate_gate = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        edge: torch.Tensor,
        pos: torch.Tensor,
        receiver: torch.Tensor,
        sender: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        graph_counts: torch.Tensor,
    ) -> torch.Tensor:
        relative = pos[receiver] - pos[sender]
        weight = torch.tanh(self.coordinate_gate(edge)).to(dtype=pos.dtype)
        messages = relative * weight
        aggregate = pos.new_zeros(pos.shape).index_add(0, receiver, messages)
        neighbor_count = pos.new_zeros(pos.shape[0]).index_add(
            0,
            receiver,
            pos.new_ones(receiver.shape[0]),
        ).clamp_min(1)
        raw_displacement = aggregate / neighbor_count.unsqueeze(-1)
        return _bounded_centered_displacement(
            raw_displacement,
            batch,
            num_graphs=num_graphs,
            graph_counts=graph_counts,
            maximum=0.25,
        )


class _DynamicEGNNBaseline(_StaticEGNNBaseline):
    """Private bounded-coordinate EGNN control for matched benchmark runs."""

    attention_kind = "internal_dynamic_egnn_baseline"

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 91,
        num_layers: int = 3,
    ) -> None:
        if num_layers < 2:
            raise ValueError("dynamic EGNN requires at least two layers")
        super().__init__(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        self.coordinate_updaters = nn.ModuleList(
            [_EGNNCoordinateUpdater(hidden_dim) for _ in range(num_layers - 1)]
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor | None = None,
        *,
        edge_index: torch.Tensor | None = None,
        edge_index_is_validated: bool = False,
        readout_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        node_feats, pos, batch, num_graphs, graph_counts = self._check_inputs(
            node_feats,
            pos,
            batch,
        )
        receiver, sender = _resolve_egnn_edges(
            batch,
            graph_counts,
            edge_index=edge_index,
            edge_index_is_validated=edge_index_is_validated,
            num_nodes=node_feats.shape[0],
            device=node_feats.device,
        )

        value = self.embedding(node_feats)
        for layer_index, layer in enumerate(self.layers):
            value, edge = layer.forward_with_edge(value, pos, receiver, sender)
            if layer_index < len(self.coordinate_updaters):
                pos = pos + self.coordinate_updaters[layer_index](
                    edge,
                    pos,
                    receiver,
                    sender,
                    batch,
                    num_graphs,
                    graph_counts,
                )
        node_scalar = self.scalar_out(self.scalar_out_norm(value))
        pool_mask, pool_batch, pool_counts = _readout_metadata(
            readout_mask,
            batch,
            num_graphs=num_graphs,
            graph_counts=graph_counts,
        )
        pooled_scalar = (
            node_scalar if pool_mask is None else node_scalar[pool_mask]
        )
        graph_scalar = _scatter_mean(
            pooled_scalar,
            pool_batch,
            num_graphs,
            pool_counts,
        )
        return {
            "graph_scalars": graph_scalar,
            "node_positions": pos,
        }


def _directed_complete_edges_without_self(
    batch: torch.Tensor,
    graph_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    receiver, sender = _batched_complete_graph_edges(batch, graph_counts)
    nonself = receiver != sender
    return receiver[nonself], sender[nonself]


def _resolve_egnn_edges(
    batch: torch.Tensor,
    graph_counts: torch.Tensor,
    *,
    edge_index: torch.Tensor | None,
    edge_index_is_validated: bool,
    num_nodes: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(edge_index_is_validated, bool):
        raise TypeError("edge_index_is_validated must be boolean")
    if edge_index is None:
        if edge_index_is_validated:
            raise ValueError("validated edge mode requires edge_index")
        return _directed_complete_edges_without_self(batch, graph_counts)
    if edge_index_is_validated:
        receiver, sender = _local_edge_index_components(
            edge_index,
            device=device,
        )
    else:
        receiver, sender = _validated_local_edge_index(
            edge_index,
            batch,
            num_nodes=num_nodes,
            device=device,
        )
    nonself = receiver != sender
    return receiver[nonself], sender[nonself]


def _validate_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
