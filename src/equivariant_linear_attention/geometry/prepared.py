from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .layout import PackedGraphLayout, pack_graph_layout
from .neighbors import PackedNeighborGraph, build_receiver_csr


_INTEGER_DTYPES = frozenset(
    {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
)


@dataclass(frozen=True, slots=True)
class Prepared3DGraph:
    """One immutable graph and execution layout validated before model forward."""

    batch: torch.Tensor
    graph_layout: PackedGraphLayout
    neighbors: PackedNeighborGraph
    _validated: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.batch, torch.Tensor):
            raise TypeError("batch must be a tensor")
        if self.batch.dtype != torch.long:
            raise TypeError("batch must use torch.long")
        if self.batch.ndim != 1:
            raise ValueError("batch must be one-dimensional")
        if not isinstance(self.graph_layout, PackedGraphLayout):
            raise TypeError("graph_layout must be a PackedGraphLayout")
        if not isinstance(self.neighbors, PackedNeighborGraph):
            raise TypeError("neighbors must be a PackedNeighborGraph")
        self.graph_layout.validate_batch(self.batch)
        if self.graph_layout.device != self.batch.device:
            raise ValueError("graph_layout and batch must share one device")
        if self.neighbors.device != self.batch.device:
            raise ValueError("neighbors and batch must share one device")
        if self.graph_layout.num_nodes != self.batch.numel():
            raise ValueError("graph_layout node count must match batch")
        if self.neighbors.num_nodes != self.batch.numel():
            raise ValueError("neighbor node count must match batch")
        if self.neighbors.num_edges:
            receiver = self.neighbors.receiver_index().to(dtype=torch.long)
            sender = self.neighbors.sender.to(dtype=torch.long)
            if not torch.equal(self.batch[receiver], self.batch[sender]):
                raise ValueError("neighbors must not connect different graphs")
        object.__setattr__(self, "_validated", True)

    @classmethod
    def _from_trusted(
        cls,
        *,
        batch: torch.Tensor,
        graph_layout: PackedGraphLayout,
        neighbors: PackedNeighborGraph,
    ) -> Prepared3DGraph:
        value = object.__new__(cls)
        object.__setattr__(value, "batch", batch)
        object.__setattr__(value, "graph_layout", graph_layout)
        object.__setattr__(value, "neighbors", neighbors)
        object.__setattr__(value, "_validated", True)
        return value

    @property
    def device(self) -> torch.device:
        return self.batch.device

    @property
    def num_nodes(self) -> int:
        return self.batch.numel()

    @property
    def num_edges(self) -> int:
        return self.neighbors.num_edges

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> Prepared3DGraph:
        target = torch.device(device)
        if target.type == "cuda" and target.index is None:
            target = torch.device("cuda", torch.cuda.current_device())
        if target == self.device:
            return self
        graph_layout = self.graph_layout.to(target, non_blocking=non_blocking)
        neighbors = self.neighbors.to(target, non_blocking=non_blocking)
        return Prepared3DGraph._from_trusted(
            batch=graph_layout.batch,
            graph_layout=graph_layout,
            neighbors=neighbors,
        )


def prepare_3d_graph(
    batch: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    edge_relation_id: torch.Tensor | None = None,
    prefer_int32: bool = True,
) -> Prepared3DGraph:
    """Validate and pack one receiver-major sparse candidate graph."""

    if not isinstance(batch, torch.Tensor):
        raise TypeError("batch must be a tensor")
    if batch.ndim != 1:
        raise ValueError("batch must be one-dimensional")
    if batch.dtype not in _INTEGER_DTYPES:
        raise TypeError("batch must use an integer dtype")
    if not isinstance(edge_index, torch.Tensor):
        raise TypeError("edge_index must be a tensor")
    if edge_index.device != batch.device:
        raise ValueError("batch and edge_index must share one device")

    batch_long = batch.to(dtype=torch.long)
    neighbors = build_receiver_csr(
        edge_index,
        num_nodes=batch_long.numel(),
        edge_relation_id=edge_relation_id,
        prefer_int32=prefer_int32,
        build_ell=False,
    )
    graph_layout = pack_graph_layout(batch_long, assume_grouped=False)
    return Prepared3DGraph(
        batch=graph_layout.batch,
        graph_layout=graph_layout,
        neighbors=neighbors,
    )


__all__ = ["Prepared3DGraph", "prepare_3d_graph"]
