from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
from torch import nn

from .canonical_se3 import CanonicalMultipoleSE3Core, _CanonicalMultipoleBlock
from .graph_layout import PackedGraphLayout
from .irreps import IrrepLayout
from .multipole_ops import NodeMultipoles
from .neighbors import PackedNeighborGraph
from .parity_se3 import (
    _ChannelMix,
    _ParityCompleteBlock,
    _ParityState,
    _StaticGeometry,
    _bounded_scalar,
    _bounded_st,
    _segment_sum,
    _st_from_vector,
    _st_square,
    _unit_ball,
)


@dataclass(frozen=True)
class UnifiedSE3State:
    """Public parity-complete hidden state used by one layer.

    The component dimensions are fixed by the canonical hidden carrier:
    ``0e + 0o + 1o + 1e + 2e + 2o``. Positions are deliberately not part of
    this object because they transform affinely under translations.
    """

    even_scalar: torch.Tensor
    odd_scalar: torch.Tensor
    polar_vector: torch.Tensor
    axial_vector: torch.Tensor
    even_tensor: torch.Tensor
    odd_tensor: torch.Tensor

    @classmethod
    def _from_internal(cls, state: _ParityState) -> UnifiedSE3State:
        return cls(
            even_scalar=state.even_scalar,
            odd_scalar=state.odd_scalar,
            polar_vector=state.polar_vector,
            axial_vector=state.axial_vector,
            even_tensor=state.even_tensor,
            odd_tensor=state.odd_tensor,
        )

    def _to_internal(self) -> _ParityState:
        return _ParityState(
            even_scalar=self.even_scalar,
            odd_scalar=self.odd_scalar,
            polar_vector=self.polar_vector,
            axial_vector=self.axial_vector,
            even_tensor=self.even_tensor,
            odd_tensor=self.odd_tensor,
        )


@dataclass(frozen=True)
class UnifiedSE3Context:
    """Geometry and batching data shared by one or more layer calls."""

    positions: torch.Tensor
    normalized_positions: torch.Tensor
    radius: torch.Tensor
    centered_norm: torch.Tensor
    geometry: _StaticGeometry
    multipoles: NodeMultipoles
    batch: torch.Tensor
    graph_layout: PackedGraphLayout


@dataclass(frozen=True)
class UnifiedSE3LayerOutput:
    """Output of one layer-level step."""

    state: UnifiedSE3State
    positions: torch.Tensor
    coordinate_delta: torch.Tensor


@dataclass(frozen=True)
class _BranchModulation:
    shift_even: torch.Tensor
    scale_even: torch.Tensor
    scale_other: torch.Tensor
    gate_even: torch.Tensor
    gate_other: torch.Tensor


@dataclass(frozen=True)
class _LayerModulation:
    attention: _BranchModulation
    ffn: _BranchModulation
    coordinate_gate: torch.Tensor


class _DiTConditioner(nn.Module):
    """Zero-initialized invariant conditioning for attention and FFN residuals.

    Conditions are ordinary ``0e`` features. A condition may be graph-level,
    node-level, or a single vector broadcast to the full batch. Even scalars
    receive adaptive shift and scale; all non-scalar sectors receive only
    copy-wise scales so equivariance is preserved. Residual gates are neutral
    at initialization, preserving the unconditioned function exactly.
    """

    def __init__(
        self,
        *,
        condition_dim: int,
        scalar_width: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        if condition_dim <= 0:
            raise ValueError("condition_dim must be positive")
        self.condition_dim = condition_dim
        self.scalar_width = scalar_width
        self.num_heads = num_heads
        self.branch_width = 3 * scalar_width + 10 * num_heads
        self.projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, 2 * self.branch_width + 1),
        )
        output = self.projection[-1]
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def _broadcast(
        self,
        condition: torch.Tensor,
        *,
        batch: torch.Tensor,
        num_graphs: int,
        num_nodes: int,
    ) -> torch.Tensor:
        if condition.ndim == 1:
            condition = condition.unsqueeze(0)
        if condition.ndim != 2 or condition.shape[1] != self.condition_dim:
            raise ValueError(
                f"condition must have shape (N, {self.condition_dim}), "
                f"(G, {self.condition_dim}), or ({self.condition_dim},)"
            )
        if condition.shape[0] == num_nodes:
            return condition
        if condition.shape[0] == num_graphs:
            return condition[batch]
        if condition.shape[0] == 1:
            return condition.expand(num_nodes, -1)
        raise ValueError(
            "condition leading dimension must be one, the node count, or the graph count"
        )

    def _branch(self, value: torch.Tensor) -> _BranchModulation:
        c = self.scalar_width
        h = self.num_heads
        shift_even, scale_even, gate_even, scale_other, gate_other = torch.split(
            value,
            [c, c, c, 5 * h, 5 * h],
            dim=-1,
        )
        return _BranchModulation(
            shift_even=shift_even,
            scale_even=scale_even,
            scale_other=scale_other.reshape(value.shape[0], 5, h),
            gate_even=gate_even,
            gate_other=gate_other.reshape(value.shape[0], 5, h),
        )

    def forward(
        self,
        condition: torch.Tensor,
        *,
        batch: torch.Tensor,
        num_graphs: int,
        num_nodes: int,
        dtype: torch.dtype,
    ) -> _LayerModulation:
        node_condition = self._broadcast(
            condition,
            batch=batch,
            num_graphs=num_graphs,
            num_nodes=num_nodes,
        ).to(device=batch.device, dtype=dtype)
        projected = self.projection(node_condition)
        attention_raw, ffn_raw, coordinate_gate = torch.split(
            projected,
            [self.branch_width, self.branch_width, 1],
            dim=-1,
        )
        return _LayerModulation(
            attention=self._branch(attention_raw),
            ffn=self._branch(ffn_raw),
            coordinate_gate=coordinate_gate,
        )


def _state_add(left: _ParityState, right: _ParityState) -> _ParityState:
    return _ParityState(
        even_scalar=left.even_scalar + right.even_scalar,
        odd_scalar=left.odd_scalar + right.odd_scalar,
        polar_vector=left.polar_vector + right.polar_vector,
        axial_vector=left.axial_vector + right.axial_vector,
        even_tensor=left.even_tensor + right.even_tensor,
        odd_tensor=left.odd_tensor + right.odd_tensor,
    )


def _state_subtract(left: _ParityState, right: _ParityState) -> _ParityState:
    return _ParityState(
        even_scalar=left.even_scalar - right.even_scalar,
        odd_scalar=left.odd_scalar - right.odd_scalar,
        polar_vector=left.polar_vector - right.polar_vector,
        axial_vector=left.axial_vector - right.axial_vector,
        even_tensor=left.even_tensor - right.even_tensor,
        odd_tensor=left.odd_tensor - right.odd_tensor,
    )


def _modulate_state(
    state: _ParityState,
    modulation: _BranchModulation,
) -> _ParityState:
    scales = 1.0 + modulation.scale_other
    return _ParityState(
        even_scalar=state.even_scalar * (1.0 + modulation.scale_even)
        + modulation.shift_even,
        odd_scalar=state.odd_scalar * scales[:, 0],
        polar_vector=state.polar_vector * scales[:, 1].unsqueeze(-1),
        axial_vector=state.axial_vector * scales[:, 2].unsqueeze(-1),
        even_tensor=state.even_tensor * scales[:, 3].unsqueeze(-1),
        odd_tensor=state.odd_tensor * scales[:, 4].unsqueeze(-1),
    )


def _gate_delta(
    delta: _ParityState,
    modulation: _BranchModulation,
) -> _ParityState:
    # Neutral zero initialization and bounded modulation in [0, 2].
    even_gate = 1.0 + torch.tanh(modulation.gate_even)
    other_gate = 1.0 + torch.tanh(modulation.gate_other)
    return _ParityState(
        even_scalar=delta.even_scalar * even_gate,
        odd_scalar=delta.odd_scalar * other_gate[:, 0],
        polar_vector=delta.polar_vector * other_gate[:, 1].unsqueeze(-1),
        axial_vector=delta.axial_vector * other_gate[:, 2].unsqueeze(-1),
        even_tensor=delta.even_tensor * other_gate[:, 3].unsqueeze(-1),
        odd_tensor=delta.odd_tensor * other_gate[:, 4].unsqueeze(-1),
    )


class UnifiedEquivariantLayer(_CanonicalMultipoleBlock):
    """One canonical SE(3) layer with explicit attention and FFN residuals.

    The layer preserves the same global/local/tensor-closure equations as the
    canonical stack. Optional invariant DiT-style conditioning modulates the
    normalized hidden sectors and independently gates attention and FFN
    residuals. Optional coordinate refinement predicts a bounded polar-vector
    displacement and returns ``x + delta_x``; it never treats raw positions as
    ordinary irrep features.
    """

    def __init__(
        self,
        *,
        scalar_width: int,
        num_heads: int,
        local_rank: int,
        num_rbf: int,
        num_edge_relations: int,
        multipole_rank: int,
        residual_scale_init: float,
        condition_dim: int = 0,
        coordinate_updates: bool = False,
        max_coordinate_step: float = 0.25,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(
            scalar_width=scalar_width,
            num_heads=num_heads,
            local_rank=local_rank,
            num_rbf=num_rbf,
            num_edge_relations=num_edge_relations,
            multipole_rank=multipole_rank,
            residual_scale_init=residual_scale_init,
            eps=eps,
        )
        if condition_dim < 0:
            raise ValueError("condition_dim must be nonnegative")
        if max_coordinate_step <= 0.0:
            raise ValueError("max_coordinate_step must be positive")
        self.condition_dim = condition_dim
        self.coordinate_updates = bool(coordinate_updates)
        self.max_coordinate_step = float(max_coordinate_step)
        self.conditioner = (
            _DiTConditioner(
                condition_dim=condition_dim,
                scalar_width=scalar_width,
                num_heads=num_heads,
            )
            if condition_dim
            else None
        )
        self.coordinate_vector = (
            _ChannelMix(num_heads, 1, zero_init=True)
            if self.coordinate_updates
            else None
        )
        self.coordinate_gate = (
            nn.Linear(scalar_width, 1) if self.coordinate_updates else None
        )
        if self.coordinate_gate is not None:
            nn.init.zeros_(self.coordinate_gate.weight)
            nn.init.zeros_(self.coordinate_gate.bias)

    def _resolve_modulation(
        self,
        condition: torch.Tensor | None,
        context: UnifiedSE3Context,
        state: _ParityState,
    ) -> _LayerModulation | None:
        if self.conditioner is None:
            if condition is not None:
                raise ValueError("condition requires positive condition_dim")
            return None
        if condition is None:
            raise ValueError("condition is required when condition_dim is positive")
        return self.conditioner(
            condition,
            batch=context.batch,
            num_graphs=context.graph_layout.num_graphs,
            num_nodes=state.even_scalar.shape[0],
            dtype=state.even_scalar.dtype,
        )

    def attention_residual(
        self,
        state: UnifiedSE3State,
        context: UnifiedSE3Context,
        condition: torch.Tensor | None = None,
    ) -> UnifiedSE3State:
        internal = state._to_internal()
        modulation = self._resolve_modulation(condition, context, internal)
        updated = self._attention_residual_internal(
            internal,
            context,
            None if modulation is None else modulation.attention,
        )
        return UnifiedSE3State._from_internal(updated)

    def ffn_residual(
        self,
        state: UnifiedSE3State,
        context: UnifiedSE3Context,
        condition: torch.Tensor | None = None,
    ) -> UnifiedSE3State:
        internal = state._to_internal()
        modulation = self._resolve_modulation(condition, context, internal)
        updated = self._ffn_residual_internal(
            internal,
            None if modulation is None else modulation.ffn,
        )
        return UnifiedSE3State._from_internal(updated)

    def forward(
        self,
        state: UnifiedSE3State,
        context: UnifiedSE3Context,
        condition: torch.Tensor | None = None,
    ) -> UnifiedSE3LayerOutput:
        internal = state._to_internal()
        updated, positions, coordinate_delta = self._forward_internal(
            internal,
            context,
            condition,
        )
        return UnifiedSE3LayerOutput(
            state=UnifiedSE3State._from_internal(updated),
            positions=positions,
            coordinate_delta=coordinate_delta,
        )

    def _attention_residual_internal(
        self,
        state: _ParityState,
        context: UnifiedSE3Context,
        modulation: _BranchModulation | None,
    ) -> _ParityState:
        normalized = self.pre_norm(state)
        if modulation is not None:
            normalized = _modulate_state(normalized, modulation)
        global_message = self._global_message(
            normalized,
            context.normalized_positions,
            context.batch,
            context.graph_layout,
        )
        local_message = self._local_message(normalized, context.geometry)
        candidate = _ParityCompleteBlock._update_state(
            self,
            state,
            global_message,
            local_message,
        )
        attention_delta = _state_subtract(candidate, state)
        if modulation is not None:
            attention_delta = _gate_delta(attention_delta, modulation)
        updated = _state_add(state, attention_delta)

        closure = self.tensor_closure(self.closure_norm(updated), context.multipoles)
        closure_delta = _ParityState(
            even_scalar=self.closure_scalar_scale.to(
                dtype=updated.even_scalar.dtype
            )
            * closure.even_scalar,
            odd_scalar=self.closure_odd_scale.to(dtype=updated.odd_scalar.dtype)
            * _bounded_scalar(closure.odd_scalar, self.eps),
            polar_vector=self.closure_polar_scale.to(
                dtype=updated.polar_vector.dtype
            )
            * _unit_ball(closure.polar_vector, self.eps),
            axial_vector=self.closure_axial_scale.to(
                dtype=updated.axial_vector.dtype
            )
            * _unit_ball(closure.axial_vector, self.eps),
            even_tensor=self.closure_even_tensor_scale.to(
                dtype=updated.even_tensor.dtype
            )
            * _bounded_st(closure.even_tensor, self.eps),
            odd_tensor=self.closure_odd_tensor_scale.to(
                dtype=updated.odd_tensor.dtype
            )
            * _bounded_st(closure.odd_tensor, self.eps),
        )
        if modulation is not None:
            closure_delta = _gate_delta(closure_delta, modulation)
        return _state_add(updated, closure_delta)

    def _ffn_residual_internal(
        self,
        state: _ParityState,
        modulation: _BranchModulation | None,
    ) -> _ParityState:
        if modulation is None:
            return self._enhanced_ffn(state)

        normalized = self.pre_norm(state)
        normalized = _modulate_state(normalized, modulation)
        scalar_state = self.ffn_norm(normalized.even_scalar)
        invariants = torch.cat(
            [
                scalar_state,
                normalized.odd_scalar.square(),
                normalized.polar_vector.square().sum(dim=-1),
                normalized.axial_vector.square().sum(dim=-1),
                _st_square(normalized.even_tensor),
                _st_square(normalized.odd_tensor),
            ],
            dim=-1,
        )
        even_delta = self.ffn_even(invariants)
        gates = torch.tanh(self.ffn_gates(scalar_state)).reshape(
            scalar_state.shape[0],
            self.num_heads,
            5,
        )
        odd_delta = gates[..., 0] * self.ffn_odd(normalized.odd_scalar)
        polar_delta = gates[..., 1, None] * (
            self.ffn_polar(normalized.polar_vector)
            + normalized.odd_scalar.unsqueeze(-1)
            * self.ffn_polar_cross(normalized.axial_vector)
        )
        axial_delta = gates[..., 2, None] * (
            self.ffn_axial(normalized.axial_vector)
            + normalized.odd_scalar.unsqueeze(-1)
            * self.ffn_axial_cross(normalized.polar_vector)
        )
        even_tensor_delta = gates[..., 3, None] * (
            self.ffn_even_tensor(normalized.even_tensor)
            + normalized.odd_scalar.unsqueeze(-1)
            * self.ffn_even_tensor_cross(normalized.odd_tensor)
        )
        odd_tensor_delta = gates[..., 4, None] * (
            self.ffn_odd_tensor(normalized.odd_tensor)
            + normalized.odd_scalar.unsqueeze(-1)
            * self.ffn_odd_tensor_cross(normalized.even_tensor)
        )
        delta = _ParityState(
            even_scalar=self.ffn_scalar_scale.to(dtype=state.even_scalar.dtype)
            * even_delta,
            odd_scalar=self.ffn_odd_scale.to(dtype=state.odd_scalar.dtype)
            * _bounded_scalar(odd_delta, self.eps),
            polar_vector=self.ffn_polar_scale.to(dtype=state.polar_vector.dtype)
            * _unit_ball(polar_delta, self.eps),
            axial_vector=self.ffn_axial_scale.to(dtype=state.axial_vector.dtype)
            * _unit_ball(axial_delta, self.eps),
            even_tensor=self.ffn_even_tensor_scale.to(
                dtype=state.even_tensor.dtype
            )
            * _bounded_st(even_tensor_delta, self.eps),
            odd_tensor=self.ffn_odd_tensor_scale.to(dtype=state.odd_tensor.dtype)
            * _bounded_st(odd_tensor_delta, self.eps),
        )
        return _state_add(state, _gate_delta(delta, modulation))

    def _coordinate_delta_internal(
        self,
        state: _ParityState,
        modulation: _LayerModulation | None,
    ) -> torch.Tensor:
        if self.coordinate_vector is None or self.coordinate_gate is None:
            return state.polar_vector.new_zeros((state.polar_vector.shape[0], 3))
        normalized = self.pre_norm(state)
        raw = self.coordinate_vector(normalized.polar_vector).squeeze(1)
        gate_logit = self.coordinate_gate(self.scalar_norm(state.even_scalar))
        if modulation is not None:
            gate_logit = gate_logit + modulation.coordinate_gate.to(
                dtype=gate_logit.dtype
            )
        gate = torch.sigmoid(gate_logit)
        return self.max_coordinate_step * gate * _unit_ball(raw, self.eps)

    def _forward_internal(
        self,
        state: _ParityState,
        context: UnifiedSE3Context,
        condition: torch.Tensor | None,
    ) -> tuple[_ParityState, torch.Tensor, torch.Tensor]:
        modulation = self._resolve_modulation(condition, context, state)
        updated = self._attention_residual_internal(
            state,
            context,
            None if modulation is None else modulation.attention,
        )
        updated = self._ffn_residual_internal(
            updated,
            None if modulation is None else modulation.ffn,
        )
        coordinate_delta = self._coordinate_delta_internal(updated, modulation)
        positions = context.positions + coordinate_delta
        return updated, positions, coordinate_delta


class LayeredCanonicalSE3Core(CanonicalMultipoleSE3Core):
    """Canonical stack implemented as reusable conditioned layer primitives."""

    def __init__(
        self,
        *,
        input_irreps: str | IrrepLayout,
        output_irreps: str | IrrepLayout,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        local_rank: int,
        local_cutoff: float,
        num_rbf: int,
        num_node_roles: int = 0,
        num_edge_relations: int = 0,
        relation_cutoffs: tuple[float, ...] = (),
        residual_scale_init: float = 0.1,
        condition_dim: int = 0,
        coordinate_updates: bool = False,
        max_coordinate_step: float = 0.25,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(
            input_irreps=input_irreps,
            output_irreps=output_irreps,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            local_rank=local_rank,
            local_cutoff=local_cutoff,
            num_rbf=num_rbf,
            num_node_roles=num_node_roles,
            num_edge_relations=num_edge_relations,
            relation_cutoffs=relation_cutoffs,
            residual_scale_init=residual_scale_init,
            eps=eps,
        )
        self.condition_dim = int(condition_dim)
        self.coordinate_updates = bool(coordinate_updates)
        self.max_coordinate_step = float(max_coordinate_step)
        block_scale = residual_scale_init / sqrt(max(1, num_layers))
        self.blocks = nn.ModuleList(
            [
                UnifiedEquivariantLayer(
                    scalar_width=hidden_dim,
                    num_heads=num_heads,
                    local_rank=local_rank,
                    num_rbf=num_rbf,
                    num_edge_relations=num_edge_relations,
                    multipole_rank=self.multipole_rank,
                    residual_scale_init=block_scale,
                    condition_dim=condition_dim,
                    coordinate_updates=coordinate_updates,
                    max_coordinate_step=max_coordinate_step,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )

    @property
    def layers(self) -> nn.ModuleList:
        return self.blocks

    def prepare_context(
        self,
        positions: torch.Tensor,
        batch: torch.Tensor,
        graph_layout: PackedGraphLayout,
        neighbors: PackedNeighborGraph,
    ) -> UnifiedSE3Context:
        normalized_positions, radius, centered_norm = self._normalized_geometry(
            positions,
            batch,
            graph_layout,
        )
        geometry = self._build_geometry(positions, neighbors)
        return UnifiedSE3Context(
            positions=positions,
            normalized_positions=normalized_positions,
            radius=radius,
            centered_norm=centered_norm,
            geometry=geometry,
            multipoles=self.node_multipoles(geometry),
            batch=batch,
            graph_layout=graph_layout,
        )

    def embed_input(
        self,
        node_irreps: torch.Tensor,
        context: UnifiedSE3Context,
        *,
        node_role_id: torch.Tensor | None = None,
    ) -> UnifiedSE3State:
        external = self.input_projection(node_irreps)
        batch = context.batch
        even_scalar = external.even_scalar
        if self.role_embedding is not None:
            if node_role_id is None:
                raise RuntimeError("validated role IDs are missing")
            even_scalar = even_scalar + self.role_embedding(
                node_role_id.to(dtype=torch.long)
            )

        density_features = torch.stack(
            [
                torch.log1p(context.radius[batch]),
                torch.log1p(context.centered_norm),
                torch.log1p(context.multipoles.mass.mean(dim=-1)),
                torch.log1p(context.multipoles.mass_square.mean(dim=-1)),
            ],
            dim=-1,
        ).to(dtype=even_scalar.dtype)
        even_scalar = even_scalar + self.scale_density_projection(density_features)

        polar = torch.tanh(self.initial_vector_gate(even_scalar)).unsqueeze(-1)
        polar = polar * context.normalized_positions[:, None, :].to(
            dtype=even_scalar.dtype
        )
        polar = polar + self.initial_multipole_polar(
            context.multipoles.polar.to(dtype=polar.dtype)
        )
        polar = polar + external.polar_vector.to(dtype=polar.dtype)

        even_tensor = torch.tanh(self.initial_tensor_gate(even_scalar)).unsqueeze(-1)
        even_tensor = even_tensor * _st_from_vector(
            context.normalized_positions[:, None, :].expand(-1, self.num_heads, -1)
        ).to(dtype=even_scalar.dtype)
        even_tensor = even_tensor + self.initial_multipole_even_tensor(
            context.multipoles.even_tensor.to(dtype=even_tensor.dtype)
        )
        even_tensor = even_tensor + external.even_tensor.to(dtype=even_tensor.dtype)

        return UnifiedSE3State(
            even_scalar=even_scalar,
            odd_scalar=external.odd_scalar.to(dtype=even_scalar.dtype)
            + self.initial_multipole_odd_scalar(
                context.multipoles.odd_scalar.to(dtype=even_scalar.dtype)
            ),
            polar_vector=polar,
            axial_vector=external.axial_vector.to(dtype=polar.dtype)
            + self.initial_multipole_axial(
                context.multipoles.axial.to(dtype=polar.dtype)
            ),
            even_tensor=even_tensor,
            odd_tensor=external.odd_tensor.to(dtype=even_tensor.dtype)
            + self.initial_multipole_odd_tensor(
                context.multipoles.odd_tensor.to(dtype=even_tensor.dtype)
            ),
        )

    def project_state(self, state: UnifiedSE3State) -> torch.Tensor:
        return self.output_projection(state._to_internal())

    def forward_features(
        self,
        node_irreps: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        graph_layout: PackedGraphLayout,
        neighbors: PackedNeighborGraph,
        *,
        node_role_id: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> tuple[UnifiedSE3State, torch.Tensor, torch.Tensor]:
        self._validate_inputs(
            node_irreps,
            positions,
            batch,
            graph_layout,
            neighbors,
            node_role_id=node_role_id,
        )
        context = self.prepare_context(positions, batch, graph_layout, neighbors)
        state = self.embed_input(
            node_irreps,
            context,
            node_role_id=node_role_id,
        )
        current_positions = positions
        total_delta = positions.new_zeros(positions.shape)
        for index, layer in enumerate(self.blocks):
            internal, next_positions, coordinate_delta = layer._forward_internal(
                state._to_internal(),
                context,
                condition,
            )
            state = UnifiedSE3State._from_internal(internal)
            current_positions = next_positions
            total_delta = total_delta + coordinate_delta.to(dtype=total_delta.dtype)
            if self.coordinate_updates and index + 1 < len(self.blocks):
                context = self.prepare_context(
                    current_positions,
                    batch,
                    graph_layout,
                    neighbors,
                )
            elif not self.coordinate_updates:
                current_positions = positions
        return state, current_positions, total_delta

    def forward(
        self,
        node_irreps: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        graph_layout: PackedGraphLayout,
        neighbors: PackedNeighborGraph,
        *,
        node_role_id: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        state, positions, coordinate_delta = self.forward_features(
            node_irreps,
            pos,
            batch,
            graph_layout,
            neighbors,
            node_role_id=node_role_id,
            condition=condition,
        )
        node_output = self.project_state(state)
        graph_output = _segment_sum(
            node_output,
            batch,
            graph_layout.num_graphs,
        ) / graph_layout.graph_counts.to(dtype=node_output.dtype).clamp_min(
            1.0
        ).unsqueeze(-1)
        return {
            "node_irreps": node_output,
            "graph_irreps": graph_output,
            "positions": positions,
            "coordinate_delta": coordinate_delta,
        }


__all__ = [
    "LayeredCanonicalSE3Core",
    "UnifiedEquivariantLayer",
    "UnifiedSE3Context",
    "UnifiedSE3LayerOutput",
    "UnifiedSE3State",
]
