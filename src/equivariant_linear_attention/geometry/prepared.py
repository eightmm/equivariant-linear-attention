from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Literal

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
GraphSource = Literal["explicit", "radius"]


@dataclass(frozen=True, slots=True)
class PreparationSpec:
    """Provenance needed to decide whether a prepared topology is reusable."""

    source: GraphSource = "explicit"
    cutoff: float | None = None
    max_neighbors: int | None = None
    include_self: bool = False
    num_edge_relations: int = 0
    skin: float = 0.0
    reference_positions: torch.Tensor | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.source not in {"explicit", "radius"}:
            raise ValueError("source must be explicit or radius")
        if not isinstance(self.include_self, bool):
            raise TypeError("include_self must be a bool")
        if (
            isinstance(self.num_edge_relations, bool)
            or not isinstance(self.num_edge_relations, int)
            or self.num_edge_relations < 0
        ):
            raise ValueError("num_edge_relations must be a nonnegative integer")
        if self.max_neighbors is not None and (
            isinstance(self.max_neighbors, bool)
            or not isinstance(self.max_neighbors, int)
            or self.max_neighbors <= 0
        ):
            raise ValueError("max_neighbors must be a positive integer or None")
        if isinstance(self.skin, bool) or not isinstance(self.skin, (int, float)):
            raise TypeError("skin must be a real number")
        skin = float(self.skin)
        if not isfinite(skin) or skin < 0.0:
            raise ValueError("skin must be finite and nonnegative")
        object.__setattr__(self, "skin", skin)

        if self.source == "explicit":
            if self.reference_positions is not None:
                raise ValueError("explicit topology must not store reference positions")
            if self.cutoff is not None:
                raise ValueError("explicit topology does not have a discovery cutoff")
            return

        if isinstance(self.cutoff, bool) or not isinstance(
            self.cutoff, (int, float)
        ):
            raise TypeError("radius topology cutoff must be a real number")
        cutoff = float(self.cutoff)
        if not isfinite(cutoff) or cutoff <= 0.0:
            raise ValueError("radius topology cutoff must be finite and positive")
        object.__setattr__(self, "cutoff", cutoff)
        reference = self.reference_positions
        if (
            not isinstance(reference, torch.Tensor)
            or reference.ndim != 2
            or reference.shape[-1] != 3
            or not reference.is_floating_point()
        ):
            raise ValueError(
                "radius topology requires floating reference_positions with shape (N,3)"
            )

    @classmethod
    def explicit(cls, *, num_edge_relations: int = 0) -> PreparationSpec:
        return cls(source="explicit", num_edge_relations=num_edge_relations)

    @classmethod
    def radius(
        cls,
        positions: torch.Tensor,
        *,
        cutoff: float,
        max_neighbors: int | None,
        include_self: bool,
        num_edge_relations: int,
        skin: float,
    ) -> PreparationSpec:
        return cls(
            source="radius",
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            include_self=include_self,
            num_edge_relations=num_edge_relations,
            skin=skin,
            reference_positions=positions.detach().clone(),
        )

    @property
    def candidate_cutoff(self) -> float | None:
        if self.cutoff is None:
            return None
        return self.cutoff + self.skin

    def matches(
        self,
        *,
        source: GraphSource,
        cutoff: float | None,
        max_neighbors: int | None,
        include_self: bool,
        num_edge_relations: int,
        skin: float,
    ) -> bool:
        if self.source != source:
            return False
        if self.num_edge_relations != num_edge_relations:
            return False
        if source == "explicit":
            return True
        return (
            self.cutoff == float(cutoff) if cutoff is not None else False
        ) and (
            self.max_neighbors == max_neighbors
            and self.include_self == include_self
            and self.skin == float(skin)
        )

    def can_reuse_positions(self, positions: torch.Tensor) -> bool:
        if self.source == "explicit":
            return True
        reference = self.reference_positions
        if reference is None:
            return False
        if (
            positions.shape != reference.shape
            or positions.device != reference.device
            or positions.dtype != reference.dtype
        ):
            return False
        if not positions.is_floating_point():
            return False
        if self.skin == 0.0:
            return torch.equal(positions.detach(), reference)
        displacement = positions.detach().to(dtype=torch.float64) - reference.to(
            dtype=torch.float64
        )
        max_displacement = torch.linalg.vector_norm(displacement, dim=-1).amax()
        return bool(max_displacement <= 0.5 * self.skin)

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> PreparationSpec:
        reference = self.reference_positions
        return PreparationSpec(
            source=self.source,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
            include_self=self.include_self,
            num_edge_relations=self.num_edge_relations,
            skin=self.skin,
            reference_positions=None
            if reference is None
            else reference.to(device=device, non_blocking=non_blocking),
        )


@dataclass(frozen=True, slots=True)
class Prepared3DGraph:
    """One immutable graph and execution layout validated before model forward."""

    batch: torch.Tensor
    graph_layout: PackedGraphLayout
    neighbors: PackedNeighborGraph
    spec: PreparationSpec = field(default_factory=PreparationSpec.explicit)
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
        if not isinstance(self.spec, PreparationSpec):
            raise TypeError("spec must be a PreparationSpec")
        self.graph_layout.validate_batch(self.batch)
        if self.graph_layout.device != self.batch.device:
            raise ValueError("graph_layout and batch must share one device")
        if self.neighbors.device != self.batch.device:
            raise ValueError("neighbors and batch must share one device")
        if self.graph_layout.num_nodes != self.batch.numel():
            raise ValueError("graph_layout node count must match batch")
        if self.neighbors.num_nodes != self.batch.numel():
            raise ValueError("neighbor node count must match batch")
        if self.spec.reference_positions is not None:
            if self.spec.reference_positions.device != self.batch.device:
                raise ValueError("reference positions and graph must share one device")
            if self.spec.reference_positions.shape != (self.batch.numel(), 3):
                raise ValueError("reference positions do not match graph node count")
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
        spec: PreparationSpec,
    ) -> Prepared3DGraph:
        value = object.__new__(cls)
        object.__setattr__(value, "batch", batch)
        object.__setattr__(value, "graph_layout", graph_layout)
        object.__setattr__(value, "neighbors", neighbors)
        object.__setattr__(value, "spec", spec)
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
            spec=self.spec.to(target, non_blocking=non_blocking),
        )


def prepare_3d_graph(
    batch: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    edge_relation_id: torch.Tensor | None = None,
    prefer_int32: bool = True,
    spec: PreparationSpec | None = None,
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
    if spec is None:
        relation_count = 0
        if edge_relation_id is not None and edge_relation_id.numel():
            relation_count = int(edge_relation_id.max().item()) + 1
        spec = PreparationSpec.explicit(num_edge_relations=relation_count)
    return Prepared3DGraph(
        batch=graph_layout.batch,
        graph_layout=graph_layout,
        neighbors=neighbors,
        spec=spec,
    )


__all__ = ["PreparationSpec", "Prepared3DGraph", "prepare_3d_graph"]
