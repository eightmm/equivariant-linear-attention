from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

import torch
from torch import nn

from .canonical_se3 import CanonicalMultipoleSE3Core
from .graph_layout import PackedGraphLayout, pack_graph_layout
from .irreps import IrrepLayout
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


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _positive_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    numeric = float(value)
    if not isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return numeric


def _nonnegative_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    numeric = float(value)
    if not isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return numeric


@dataclass(frozen=True, slots=True)
class Unified3DConfig:
    """Canonical parity-complete SE(3) configuration.

    Internal angular degree, parity, multipole construction, normalization,
    local/global composition, and fallback policy are fixed. Users declare only
    final ``output_irreps``; width, depth, rank, and cutoff remain ordinary
    resource/capacity hyperparameters.
    """

    node_dim: int
    output_irreps: str = "1x0e"
    hidden_dim: int = 64
    num_layers: int = 4
    num_heads: int = 4
    local_rank: int = 2
    local_cutoff: float = 5.0
    num_rbf: int = 16
    input_vector_dim: int = 0
    input_tensor_dim: int = 0
    num_node_roles: int = 0
    relation_cutoffs: tuple[float, ...] = ()
    residual_scale_init: float = 0.1
    eps: float = 1e-12

    def __post_init__(self) -> None:
        for name in (
            "node_dim",
            "hidden_dim",
            "num_layers",
            "num_heads",
            "local_rank",
            "num_rbf",
        ):
            _positive_integer(name, getattr(self, name))
        for name in (
            "input_vector_dim",
            "input_tensor_dim",
            "num_node_roles",
        ):
            _nonnegative_integer(name, getattr(self, name))
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")

        layout = IrrepLayout.parse(self.output_irreps)
        if not layout.blocks:
            raise ValueError("output_irreps must not be empty")
        unsupported = [
            block.irrep
            for block in layout.blocks
            if block.irrep.degree > 2
        ]
        if unsupported:
            raise ValueError(
                "output_irreps supports l<=2 in the optimized unified core; "
                f"unsupported={unsupported}"
            )

        local_cutoff = _positive_real("local_cutoff", self.local_cutoff)
        _positive_real("eps", self.eps)
        _nonnegative_real("residual_scale_init", self.residual_scale_init)
        if not isinstance(self.relation_cutoffs, tuple):
            raise TypeError("relation_cutoffs must be a tuple")
        for cutoff in self.relation_cutoffs:
            relation_cutoff = _positive_real("relation_cutoffs", cutoff)
            if relation_cutoff > local_cutoff:
                raise ValueError(
                    "relation_cutoffs may only narrow the shared local cutoff"
                )

    @property
    def num_edge_relations(self) -> int:
        return len(self.relation_cutoffs)

    @property
    def output_layout(self) -> IrrepLayout:
        return IrrepLayout.parse(self.output_irreps)

    @property
    def internal_irreps(self) -> IrrepLayout:
        heads = self.num_heads
        return IrrepLayout.parse(
            f"{self.hidden_dim}x0e + {heads}x0o + "
            f"{heads}x1o + {heads}x1e + "
            f"{heads}x2e + {heads}x2o"
        )

    def canonical_contract(self) -> dict[str, object]:
        return {
            "public_symmetry": "SE3",
            "internal_symmetry": "O3_parity_complete",
            "internal_irreps": str(self.internal_irreps),
            "user_representation_control": "output_irreps_only",
            "node_geometry": "static_l0_l1_l2_radial_multipoles",
            "global_operator": "exact_positive_feature_gemm_l0_l1_l2",
            "global_balancing": "one_cycle",
            "local_operator": "single_positive_mass_damped_rank_r",
            "local_routing": "scalar_vector_axial_tensor_parity_complete",
            "local_reduction": "single_receiver_csr",
            "tensor_product_closure": "low_rank_lte2_cartesian",
            "irrep_normalization": "sector_rms_pre_norm",
            "residual_scaling": "per_copy_layerscale",
            "cutoff_regularity": "C2",
            "scale_density_context": True,
            "chiral_construction": "aggregate_cross_triple_product",
            "chiral_initialization": "deterministic_rank_to_head_bridge",
            "neighbor_input": "prepacked_receiver_csr",
            "local_refresh": "every_block",
            "fallbacks": (),
        }


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
    """Validate and pack one receiver-major sparse graph outside model forward."""

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
    graph_layout = pack_graph_layout(
        batch_long,
        assume_grouped=False,
    )
    return Prepared3DGraph(
        batch=graph_layout.batch,
        graph_layout=graph_layout,
        neighbors=neighbors,
    )


class UnifiedEquivariantAttention(nn.Module):
    """One multipole-complete parity-aware SE(3) execution path."""

    attention_kind = "canonical_multipole_parity_factorized_moment"
    symmetry = "SE3"
    internal_symmetry = "O3_parity_complete"
    supports_graph_layout = True

    def __init__(self, config: Unified3DConfig) -> None:
        super().__init__()
        if not isinstance(config, Unified3DConfig):
            raise TypeError("config must be a Unified3DConfig")
        self.config = config
        self.core = CanonicalMultipoleSE3Core(
            node_dim=config.node_dim,
            output_irreps=config.output_layout,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            local_rank=config.local_rank,
            local_cutoff=config.local_cutoff,
            num_rbf=config.num_rbf,
            input_vector_dim=config.input_vector_dim,
            input_tensor_dim=config.input_tensor_dim,
            num_node_roles=config.num_node_roles,
            num_edge_relations=config.num_edge_relations,
            relation_cutoffs=config.relation_cutoffs,
            residual_scale_init=config.residual_scale_init,
            eps=config.eps,
        )
        self._initialize_chiral_bridge()
        self.internal_irreps = self.core.internal_irreps
        self.output_irreps = self.core.output_irreps

    def _initialize_chiral_bridge(self) -> None:
        """Keep an immediate even-objective gradient path into odd geometry."""

        with torch.no_grad():
            for block in self.core.blocks:
                weight = block.local_chiral_scalar_out.weight
                weight.zero_()
                rows = torch.arange(weight.shape[0], device=weight.device)
                weight[rows, rows.remainder(weight.shape[1])] = 1.0

    def forward(
        self,
        node_feats: torch.Tensor,
        pos: torch.Tensor,
        graph: Prepared3DGraph,
        *,
        node_role_id: torch.Tensor | None = None,
        node_vectors: torch.Tensor | None = None,
        node_tensors: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if not isinstance(graph, Prepared3DGraph) or not graph._validated:
            raise TypeError("graph must be a validated Prepared3DGraph")
        if node_feats.shape[0] != graph.num_nodes:
            raise ValueError("node_feats node count must match graph")
        if pos.shape[0] != graph.num_nodes:
            raise ValueError("pos node count must match graph")
        if node_feats.device != graph.device or pos.device != graph.device:
            raise ValueError("model inputs and graph must share one device")
        return self.core(
            node_feats,
            pos,
            graph.batch,
            graph.graph_layout,
            graph.neighbors,
            node_role_id=node_role_id,
            node_vectors=node_vectors,
            node_tensors=node_tensors,
        )

    def split_output(
        self,
        value: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.core.split_output(value)

    def extra_repr(self) -> str:
        return (
            f"output_irreps={self.output_irreps}, hidden_dim={self.config.hidden_dim}, "
            f"layers={self.config.num_layers}, heads={self.config.num_heads}, "
            f"local_rank={self.config.local_rank}, "
            f"multipole_rank={self.core.multipole_rank}"
        )


__all__ = [
    "Prepared3DGraph",
    "Unified3DConfig",
    "UnifiedEquivariantAttention",
    "prepare_3d_graph",
]
