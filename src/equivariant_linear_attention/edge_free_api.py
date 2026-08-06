from __future__ import annotations

import torch
from torch import nn

from .api import ELA as _SparseELA
from .batch import ELABatch
from .geometry.neighbors import build_receiver_csr
from .geometry.prepared import (
    PreparationSpec,
    Prepared3DGraph,
    _prepare_trusted_3d_graph,
)
from .model.edge_free import EdgeFreeELACore
from .model.ela import ELAConfig
from .nn.layers import _ELAHiddenState


def _quotient_rigid_shape_step(
    raw: torch.Tensor,
    positions: torch.Tensor,
    batch: torch.Tensor,
    selected: torch.Tensor,
    component_gates: torch.Tensor,
    *,
    max_step: float,
    eps: float,
) -> torch.Tensor:
    """Split a velocity into translation, rotation, and internal shape tangent.

    Fully selected interaction components are interpreted modulo the global
    ``SE(3)`` gauge and therefore keep only the internal shape tangent. Partly
    selected components retain their rigid translation and rotation so a ligand
    or movable domain can change pose relative to fixed context.
    """

    if raw.shape != positions.shape or raw.ndim != 2 or raw.shape[-1] != 3:
        raise ValueError("raw and positions must have shape (N, 3)")
    if batch.shape != (raw.shape[0],):
        raise ValueError("batch must have shape (N,)")
    if selected.shape != (raw.shape[0],) or selected.dtype != torch.bool:
        raise ValueError("selected must be a boolean tensor with shape (N,)")
    if component_gates.ndim != 2 or component_gates.shape[-1] != 3:
        raise ValueError("component_gates must have shape (G, 3)")

    batch_index = batch.to(dtype=torch.long)
    num_graphs = component_gates.shape[0]
    if batch_index.numel() and int(batch_index.max().item()) >= num_graphs:
        raise ValueError("component_gates do not cover every graph")

    work_positions = positions.to(dtype=raw.dtype)
    mask = selected.to(dtype=raw.dtype).unsqueeze(-1)
    selected_count_raw = torch.bincount(
        batch_index[selected],
        minlength=num_graphs,
    )
    selected_count = selected_count_raw.clamp_min(1)
    selected_count_float = selected_count.to(dtype=raw.dtype)
    graph_count = torch.bincount(batch_index, minlength=num_graphs)

    center = raw.new_zeros((num_graphs, 3)).index_add(
        0,
        batch_index,
        work_positions * mask,
    ) / selected_count_float.unsqueeze(-1)
    translation = raw.new_zeros((num_graphs, 3)).index_add(
        0,
        batch_index,
        raw * mask,
    ) / selected_count_float.unsqueeze(-1)

    relative = (work_positions - center[batch_index]) * mask
    centered_velocity = (raw - translation[batch_index]) * mask
    angular_momentum = raw.new_zeros((num_graphs, 3)).index_add(
        0,
        batch_index,
        torch.cross(relative, centered_velocity, dim=-1),
    )

    identity = torch.eye(3, device=raw.device, dtype=raw.dtype)
    relative_outer = torch.einsum("na,nb->nab", relative, relative)
    radius_square = relative.square().sum(dim=-1)
    inertia = raw.new_zeros((num_graphs, 3, 3)).index_add(
        0,
        batch_index,
        radius_square[:, None, None] * identity - relative_outer,
    )
    inertia_scale = (
        inertia.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / 3.0
    ).clamp_min(1.0)
    regularizer = (
        float(eps)
        + 8.0 * torch.finfo(raw.dtype).eps * inertia_scale
    )
    angular_velocity = torch.linalg.solve(
        inertia + regularizer[:, None, None] * identity,
        angular_momentum.unsqueeze(-1),
    ).squeeze(-1)
    rotation = torch.cross(
        angular_velocity[batch_index],
        relative,
        dim=-1,
    )
    shape = (raw - translation[batch_index] - rotation) * mask

    fully_selected = selected_count_raw == graph_count
    rigid_allowed = (~fully_selected).to(dtype=raw.dtype)
    translation_gate = component_gates[:, 0] * rigid_allowed
    rotation_gate = component_gates[:, 1] * rigid_allowed
    shape_gate = component_gates[:, 2]
    step = (
        translation_gate[batch_index, None] * translation[batch_index]
        + rotation_gate[batch_index, None] * rotation
        + shape_gate[batch_index, None] * shape
    ) * mask

    norms = torch.linalg.vector_norm(step, dim=-1)
    graph_max = norms.new_zeros((num_graphs,))
    graph_max.scatter_reduce_(
        0,
        batch_index,
        norms,
        reduce="amax",
        include_self=True,
    )
    graph_scale = (
        float(max_step) / graph_max.clamp_min(float(max_step))
    ).clamp(max=1.0)
    return step * graph_scale[batch_index, None]


class ELA(_SparseELA):
    """The public ``ELAGraph -> ELA -> ELAGraph`` edge-free model.

    The canonical path uses exact edge-free relative moments through order three,
    an invariant orthogonal Krylov basis, and a learned low-rank latent atlas
    relation. Explicit input edges remain only an optional supervised residual;
    no edge set is inferred or stored by the default path.
    """

    def __init__(
        self,
        input_irreps: str,
        output_irreps: str = "1x0e",
        *,
        width: int = 128,
        depth: int = 8,
        cutoff: float = 5.0,
        max_neighbors: int | None = None,
        edge_types: int = 0,
        condition_dim: int = 0,
        order_dim: int = 0,
        update_positions: bool = False,
        max_coordinate_step: float = 0.25,
    ) -> None:
        super().__init__(
            input_irreps,
            output_irreps,
            width=width,
            depth=depth,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            edge_types=edge_types,
            condition_dim=condition_dim,
            order_dim=order_dim,
            update_positions=update_positions,
            max_coordinate_step=max_coordinate_step,
        )
        self._install_edge_free_core()

    @classmethod
    def from_config(cls, config: ELAConfig) -> ELA:
        model = super().from_config(config)
        model._install_edge_free_core()
        return model

    def _install_edge_free_core(self) -> None:
        self.core = EdgeFreeELACore(self.advanced_config)
        self._initialize_chiral_bridge()
        self.internal_irreps = self.core.internal_irreps
        self.output_irreps = self.core.output_irreps

        self.coordinate_component_gate: nn.Linear | None = None
        if self.config.coordinate_updates > 0:
            self.coordinate_component_gate = nn.Linear(self.config.width, 3)
            nn.init.zeros_(self.coordinate_component_gate.weight)
            nn.init.zeros_(self.coordinate_component_gate.bias)

    def _preparation_matches(self, batch: ELABatch) -> bool:
        if batch.edge_index is not None:
            return super()._preparation_matches(batch)

        graph = batch._prepared_graph
        if graph is None:
            return False
        if not graph.spec.matches(
            source="explicit",
            cutoff=None,
            max_neighbors=None,
            include_self=False,
            num_edge_relations=self.config.geometry.num_edge_relations,
            skin=0.0,
        ):
            return False
        if graph.neighbors.num_edges:
            return False
        if (
            graph.neighbors.relation_id is not None
            and graph.neighbors.relation_id.numel()
        ):
            return False
        if batch._trusted_prepared:
            return True
        return torch.equal(graph.batch, batch.interaction_batch)

    def _prepare_packed(
        self,
        batch: ELABatch,
        *,
        prefer_int32: bool = True,
        force: bool = False,
    ) -> ELABatch:
        if batch.edge_index is not None:
            return super()._prepare_packed(
                batch,
                prefer_int32=prefer_int32,
                force=force,
            )
        if not isinstance(batch, ELABatch):
            raise TypeError("internal preparation expects ELABatch")
        if batch.edge_relation_id is not None:
            raise ValueError("edge_type cannot be used without edge_index")
        if batch.is_prepared and not force and self._preparation_matches(batch):
            return batch
        if batch.is_prepared:
            batch = batch.without_prepared_graph()

        graph_counts = None
        if batch.interaction_group is None:
            if batch.ptr is None:
                raise RuntimeError("packed graph pointer normalization failed")
            graph_counts = (batch.ptr[1:] - batch.ptr[:-1]).to(dtype=torch.long)

        empty_edges = torch.empty(
            (2, 0),
            device=batch.positions.device,
            dtype=torch.long,
        )
        relation_count = self.config.geometry.num_edge_relations
        empty_relation = (
            torch.empty(0, device=batch.positions.device, dtype=torch.long)
            if relation_count
            else None
        )
        neighbors = build_receiver_csr(
            empty_edges,
            num_nodes=batch.num_nodes,
            edge_relation_id=empty_relation,
            prefer_int32=prefer_int32,
            build_ell=False,
        )
        prepared = _prepare_trusted_3d_graph(
            batch.interaction_batch,
            neighbors,
            graph_counts=graph_counts,
            spec=PreparationSpec.explicit(num_edge_relations=relation_count),
        )
        return batch._with_prepared_graph_trusted(prepared)

    def _quotient_coordinate_delta(
        self,
        state: _ELAHiddenState,
        positions: torch.Tensor,
        graph: Prepared3DGraph,
        update_mask: torch.Tensor | None,
        *,
        max_step: float,
    ) -> torch.Tensor:
        if (
            self.coordinate_head is None
            or self.coordinate_gate is None
            or self.coordinate_component_gate is None
        ):
            raise RuntimeError("coordinate updates are not enabled on this model")
        selected = self._selected_mask(
            update_mask,
            num_nodes=graph.num_nodes,
            device=graph.device,
        )
        raw = self.coordinate_head(
            state.even_scalar,
            state.polar_vector,
        ).squeeze(1)
        raw = torch.sigmoid(self.coordinate_gate(state.even_scalar)) * raw

        batch_index = graph.batch.to(dtype=torch.long)
        num_graphs = graph.graph_layout.num_graphs
        selected_float = selected.to(dtype=state.even_scalar.dtype).unsqueeze(-1)
        selected_count = torch.bincount(
            batch_index[selected],
            minlength=num_graphs,
        ).clamp_min(1).to(dtype=state.even_scalar.dtype)
        component_state = state.even_scalar.new_zeros(
            (num_graphs, state.even_scalar.shape[-1])
        ).index_add(
            0,
            batch_index,
            state.even_scalar * selected_float,
        ) / selected_count.unsqueeze(-1)
        raw_component_gates = self.coordinate_component_gate(component_state)
        component_gates = torch.cat(
            [
                torch.tanh(raw_component_gates[:, :2]),
                1.0 + torch.tanh(raw_component_gates[:, 2:]),
            ],
            dim=-1,
        )
        return _quotient_rigid_shape_step(
            raw,
            positions,
            graph.batch,
            selected,
            component_gates,
            max_step=max_step,
            eps=max(self.config.eps, 1e-12),
        )

    def _stagewise_forward_features(
        self,
        node_irreps: torch.Tensor,
        pos: torch.Tensor,
        graph: Prepared3DGraph,
        *,
        condition: torch.Tensor | None,
        update_mask: torch.Tensor | None,
    ) -> tuple[
        _ELAHiddenState,
        torch.Tensor,
        torch.Tensor,
        Prepared3DGraph,
    ]:
        """Carry hidden state on the quotient-aware coordinate manifold."""

        self._validate_graph_inputs(node_irreps, pos, graph)
        self.core._validate_inputs(
            node_irreps,
            pos,
            graph.batch,
            graph.graph_layout,
            graph.neighbors,
            node_role_id=None,
        )
        current_positions = pos
        current_graph = graph
        layer_context = self.core.prepare_context(
            current_positions,
            current_graph.batch,
            current_graph.graph_layout,
            current_graph.neighbors,
        )
        state = self.core.embed_input(node_irreps, layer_context)
        total_delta = torch.zeros_like(pos)
        update_layers = frozenset(self.config.coordinate_update_layers)
        stage_max_step = float(self.config.max_coordinate_step) / len(update_layers)

        for layer_number, layer in enumerate(self.layers, start=1):
            layer_output = layer(state, layer_context, condition)
            state = layer_output.state
            if layer_number not in update_layers:
                continue

            delta = self._quotient_coordinate_delta(
                state,
                current_positions,
                current_graph,
                update_mask,
                max_step=stage_max_step,
            ).to(dtype=current_positions.dtype)
            current_positions = current_positions + delta
            total_delta = total_delta + delta

            if layer_number == self.config.depth:
                continue
            # Edge-free moments are recomputed from the new coordinates. If the
            # task supplied explicit topology, the same topology is retained and
            # only its geometric values are refreshed.
            layer_context = self.core.prepare_context(
                current_positions,
                current_graph.batch,
                current_graph.graph_layout,
                current_graph.neighbors,
            )

        return state, current_positions, total_delta, current_graph

    def describe(self) -> dict[str, object]:
        description = super().describe()
        description.update(
            {
                "default_geometry": "edge_free_relative_moments",
                "relative_moment_order": 3,
                "implicit_relation_orders": (1, 2, 3),
                "krylov_basis": "graphwise_irrep_orthogonal",
                "latent_edge_relation": "soft_atlas_incidence",
                "coordinate_manifold": "SE3_quotient_auto",
                "explicit_edge_residual": True,
                "automatic_radius_graph": False,
            }
        )
        return description

    def canonical_contract(self) -> dict[str, object]:
        contract = self.config.canonical_contract()
        contract.update(
            {
                "spatial_policy": (
                    "edge_free_l3_moments_plus_orthogonal_krylov_and_latent_atlas"
                ),
                "message_fusion": (
                    "orthogonal_krylov_plus_latent_soft_edges"
                    "_plus_optional_explicit_residual"
                ),
                "implicit_spatial": "receiver_centered_l0_l1_l2_l3_graph_moments",
                "implicit_relation_orders": (1, 2, 3),
                "krylov_basis": "graphwise_irrep_orthogonal",
                "latent_edge_relation": "A_Dinv_AT_soft_atlas",
                "coordinate_manifold": "rigid_pose_plus_shape_quotient",
                "automatic_radius_graph": False,
            }
        )
        return contract


__all__ = ["ELA"]
