from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

import torch
from torch import nn

from .graph_layout import PackedGraphLayout, pack_graph_layout
from .irreps import IrrepLayout, split_irreps
from .layered_se3 import (
    LayeredCanonicalSE3Core,
    UnifiedEquivariantLayer,
    UnifiedSE3Context,
    UnifiedSE3State,
)
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
    """Canonical layer-stack configuration.

    Input and output representations are public. Hidden parity and angular
    degree stay fixed to the canonical low-order parity-complete carrier. An
    optional invariant condition enables DiT-style adaptive modulation, and an
    optional coordinate path applies a bounded ``1o`` displacement after every
    layer while recomputing geometry on the same prepared candidate topology.
    """

    input_irreps: str
    output_irreps: str = "1x0e"
    hidden_dim: int = 64
    num_layers: int = 4
    num_heads: int = 4
    local_rank: int = 2
    local_cutoff: float = 5.0
    num_rbf: int = 16
    num_node_roles: int = 0
    relation_cutoffs: tuple[float, ...] = ()
    residual_scale_init: float = 0.1
    condition_dim: int = 0
    coordinate_updates: bool = False
    max_coordinate_step: float = 0.25
    eps: float = 1e-12

    def __post_init__(self) -> None:
        for name in (
            "hidden_dim",
            "num_layers",
            "num_heads",
            "local_rank",
            "num_rbf",
        ):
            _positive_integer(name, getattr(self, name))
        for name in ("num_node_roles", "condition_dim"):
            _nonnegative_integer(name, getattr(self, name))
        if not isinstance(self.coordinate_updates, bool):
            raise TypeError("coordinate_updates must be a bool")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")

        input_layout = IrrepLayout.parse(self.input_irreps)
        output_layout = IrrepLayout.parse(self.output_irreps)
        if not output_layout.blocks:
            raise ValueError("output_irreps must not be empty")
        unsupported_input = [
            block.irrep for block in input_layout.blocks if block.irrep.degree > 2
        ]
        unsupported_output = [
            block.irrep for block in output_layout.blocks if block.irrep.degree > 2
        ]
        if unsupported_input or unsupported_output:
            raise ValueError(
                "input_irreps and output_irreps support l<=2 in the optimized "
                "unified core; "
                f"unsupported={unsupported_input + unsupported_output}"
            )

        local_cutoff = _positive_real("local_cutoff", self.local_cutoff)
        _positive_real("max_coordinate_step", self.max_coordinate_step)
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
    def input_layout(self) -> IrrepLayout:
        return IrrepLayout.parse(self.input_irreps)

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
            "user_representation_control": "input_and_output_irreps",
            "input_irreps": str(self.input_layout),
            "layer_api": "attention_residual_tensor_closure_ffn_residual",
            "conditioning": (
                "dit_invariant_adaptive_modulation"
                if self.condition_dim
                else "none"
            ),
            "condition_irrep": "0e",
            "coordinate_output": "bounded_polar_residual",
            "coordinate_updates": self.coordinate_updates,
            "coordinate_topology": (
                "fixed_candidate_recompute_geometry_each_layer"
                if self.coordinate_updates
                else "fixed_input_geometry"
            ),
            "node_geometry": (
                "dynamic_l0_l1_l2_radial_multipoles"
                if self.coordinate_updates
                else "static_l0_l1_l2_radial_multipoles"
            ),
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


class UnifiedEquivariantAttention(nn.Module):
    """Layered parity-complete SE(3) stack with optional condition and coordinates."""

    attention_kind = "layered_multipole_parity_factorized_moment"
    symmetry = "SE3"
    internal_symmetry = "O3_parity_complete"
    supports_graph_layout = True

    def __init__(self, config: Unified3DConfig) -> None:
        super().__init__()
        if not isinstance(config, Unified3DConfig):
            raise TypeError("config must be a Unified3DConfig")
        self.config = config
        self.core = LayeredCanonicalSE3Core(
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
            condition_dim=config.condition_dim,
            coordinate_updates=config.coordinate_updates,
            max_coordinate_step=config.max_coordinate_step,
            eps=config.eps,
        )
        self._initialize_chiral_bridge()
        self.internal_irreps = self.core.internal_irreps
        self.output_irreps = self.core.output_irreps

    @property
    def layers(self) -> nn.ModuleList:
        return self.core.layers

    def _initialize_chiral_bridge(self) -> None:
        """Keep an immediate even-objective gradient path into odd geometry."""

        with torch.no_grad():
            for layer in self.core.layers:
                weight = layer.local_chiral_scalar_out.weight
                weight.zero_()
                rows = torch.arange(weight.shape[0], device=weight.device)
                weight[rows, rows.remainder(weight.shape[1])] = 1.0

    def _validate_graph_inputs(
        self,
        node_irreps: torch.Tensor,
        pos: torch.Tensor,
        graph: Prepared3DGraph,
    ) -> None:
        if not isinstance(graph, Prepared3DGraph) or not graph._validated:
            raise TypeError("graph must be a validated Prepared3DGraph")
        if node_irreps.shape[0] != graph.num_nodes:
            raise ValueError("node_irreps node count must match graph")
        if pos.shape != (graph.num_nodes, 3):
            raise ValueError("pos must have shape (N, 3)")
        if node_irreps.device != graph.device or pos.device != graph.device:
            raise ValueError("model inputs and graph must share one device")

    def prepare_context(
        self,
        pos: torch.Tensor,
        graph: Prepared3DGraph,
    ) -> UnifiedSE3Context:
        dummy = pos.new_zeros((graph.num_nodes, self.config.input_layout.dim))
        self._validate_graph_inputs(dummy, pos, graph)
        self.core._assert_finite("pos", pos)
        return self.core.prepare_context(
            pos,
            graph.batch,
            graph.graph_layout,
            graph.neighbors,
        )

    def embed_input(
        self,
        node_irreps: torch.Tensor,
        pos: torch.Tensor,
        graph: Prepared3DGraph,
        *,
        node_role_id: torch.Tensor | None = None,
    ) -> tuple[UnifiedSE3State, UnifiedSE3Context]:
        self._validate_graph_inputs(node_irreps, pos, graph)
        self.core._validate_inputs(
            node_irreps,
            pos,
            graph.batch,
            graph.graph_layout,
            graph.neighbors,
            node_role_id=node_role_id,
        )
        context = self.core.prepare_context(
            pos,
            graph.batch,
            graph.graph_layout,
            graph.neighbors,
        )
        return (
            self.core.embed_input(
                node_irreps,
                context,
                node_role_id=node_role_id,
            ),
            context,
        )

    def project_state(self, state: UnifiedSE3State) -> torch.Tensor:
        return self.core.project_state(state)

    def split_input(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        return split_irreps(self.config.input_layout, value)

    def split_output(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        return split_irreps(self.output_irreps, value)

    def forward_features(
        self,
        node_irreps: torch.Tensor,
        pos: torch.Tensor,
        graph: Prepared3DGraph,
        *,
        node_role_id: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> tuple[UnifiedSE3State, torch.Tensor, torch.Tensor]:
        self._validate_graph_inputs(node_irreps, pos, graph)
        return self.core.forward_features(
            node_irreps,
            pos,
            graph.batch,
            graph.graph_layout,
            graph.neighbors,
            node_role_id=node_role_id,
            condition=condition,
        )

    def forward(
        self,
        node_irreps: torch.Tensor,
        pos: torch.Tensor,
        graph: Prepared3DGraph,
        *,
        node_role_id: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        self._validate_graph_inputs(node_irreps, pos, graph)
        return self.core(
            node_irreps,
            pos,
            graph.batch,
            graph.graph_layout,
            graph.neighbors,
            node_role_id=node_role_id,
            condition=condition,
        )

    def extra_repr(self) -> str:
        return (
            f"input_irreps={self.config.input_layout}, "
            f"output_irreps={self.output_irreps}, hidden_dim={self.config.hidden_dim}, "
            f"layers={self.config.num_layers}, heads={self.config.num_heads}, "
            f"local_rank={self.config.local_rank}, "
            f"condition_dim={self.config.condition_dim}, "
            f"coordinate_updates={self.config.coordinate_updates}"
        )


__all__ = [
    "Prepared3DGraph",
    "Unified3DConfig",
    "UnifiedEquivariantAttention",
    "UnifiedEquivariantLayer",
    "UnifiedSE3Context",
    "UnifiedSE3State",
    "prepare_3d_graph",
]
