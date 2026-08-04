"""Sparse-geometry preparation and execution metadata."""

from .layout import PackedGraphLayout, pack_graph_layout
from .neighbors import PackedNeighborGraph, build_receiver_csr
from .prepared import PreparationSpec, Prepared3DGraph, prepare_3d_graph
from .radius import radius_graph

__all__ = [
    "PackedGraphLayout",
    "PackedNeighborGraph",
    "PreparationSpec",
    "Prepared3DGraph",
    "build_receiver_csr",
    "pack_graph_layout",
    "prepare_3d_graph",
    "radius_graph",
]
