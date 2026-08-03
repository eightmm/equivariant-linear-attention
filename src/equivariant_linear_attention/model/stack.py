from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt

import torch
from torch import nn

from ..irreps import IrrepLayout
from ..nn.layers import (
    LayeredCanonicalSE3Core,
    UnifiedEquivariantLayer,
    UnifiedSE3Context,
    UnifiedSE3State,
    _BranchModulation,
    _LayerModulation,
    _gate_delta,
    _state_add,
    _state_subtract,
)
from ..nn.parity import (
    _ParityCompleteBlock,
    _ParityState,
    _bounded_scalar,
    _bounded_st,
    _st_square,
    _unit_ball,
)
from .runtime import Unified3DConfig, UnifiedEquivariantAttention


def _probability(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    numeric = float(value)
    if not isfinite(numeric) or not 0.0 <= numeric < 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1)")
    return numeric


@dataclass(frozen=True, slots=True)
class EquivariantLinearAttentionConfig(Unified3DConfig):
    """Canonical configuration for the refined equivariant linear-attention stack.

    The exact positive feature-map global operator and the single sparse local
    residual remain the only attention routes. The additions in this class are
    branch regularization and normalization controls, not alternative message
    passing backends.
    """

    residual_dropout: float = 0.0
    drop_path_rate: float = 0.0
    norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        Unified3DConfig.__post_init__(self)
        _probability("residual_dropout", self.residual_dropout)
        _probability("drop_path_rate", self.drop_path_rate)
        if isinstance(self.norm_eps, bool) or not isinstance(
            self.norm_eps, (int, float)
        ):
            raise TypeError("norm_eps must be a real number")
        norm_eps = float(self.norm_eps)
        if not isfinite(norm_eps) or norm_eps <= 0.0:
            raise ValueError("norm_eps must be finite and positive")

    def canonical_contract(self) -> dict[str, object]:
        contract = Unified3DConfig.canonical_contract(self)
        contract.update(
            {
                "architecture": "equivariant_linear_attention",
                "attention_qk_normalization": "headwise_rms_before_positive_feature",
                "branch_normalization": "all_sector_equivariant_rms_pre_norm",
                "activation": "identity_initialized_norm_gated_irrep_activation",
                "residual_dropout": "direction_preserving_irrep_copy_dropout",
                "stochastic_depth": "graphwise_equivariant_drop_path",
                "condition_modulation": "bounded_equivariant_adaln",
            }
        )
        return contract


class _EquivariantRMSNorm(nn.Module):
    """RMS normalization for the fixed 0e+0o+1o+1e+2e+2o carrier.

    One invariant RMS is computed per node and irrep sector. Learnable gains act
    only on multiplicity channels, so the operation commutes with every O(3)
    action tracked by the SE(3) core.
    """

    def __init__(self, *, scalar_width: int, num_heads: int, eps: float) -> None:
        super().__init__()
        self.eps = max(float(eps), 1e-8)
        self.even_gain = nn.Parameter(torch.ones(scalar_width))
        self.odd_gain = nn.Parameter(torch.ones(num_heads))
        self.polar_gain = nn.Parameter(torch.ones(num_heads))
        self.axial_gain = nn.Parameter(torch.ones(num_heads))
        self.even_tensor_gain = nn.Parameter(torch.ones(num_heads))
        self.odd_tensor_gain = nn.Parameter(torch.ones(num_heads))

    @staticmethod
    def _work_dtype(value: torch.Tensor) -> torch.dtype:
        return torch.float64 if value.dtype == torch.float64 else torch.float32

    def _scalar(self, value: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
        dtype = self._work_dtype(value)
        work = value.to(dtype=dtype)
        rms = torch.sqrt(work.square().mean(dim=-1, keepdim=True) + self.eps)
        result = work / rms * gain.to(device=value.device, dtype=dtype).unsqueeze(0)
        return result.to(dtype=value.dtype)

    def _vector(self, value: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
        dtype = self._work_dtype(value)
        work = value.to(dtype=dtype)
        rms = torch.sqrt(work.square().mean(dim=(-2, -1), keepdim=True) + self.eps)
        result = work / rms * gain.to(
            device=value.device, dtype=dtype
        ).reshape(1, -1, 1)
        return result.to(dtype=value.dtype)

    def _tensor(self, value: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
        dtype = self._work_dtype(value)
        work = value.to(dtype=dtype)
        rms = torch.sqrt(
            _st_square(work).mean(dim=-1, keepdim=True).unsqueeze(-1) / 5.0
            + self.eps
        )
        result = work / rms * gain.to(
            device=value.device, dtype=dtype
        ).reshape(1, -1, 1)
        return result.to(dtype=value.dtype)

    def forward(self, state: _ParityState) -> _ParityState:
        return _ParityState(
            even_scalar=self._scalar(state.even_scalar, self.even_gain),
            odd_scalar=self._scalar(state.odd_scalar, self.odd_gain),
            polar_vector=self._vector(state.polar_vector, self.polar_gain),
            axial_vector=self._vector(state.axial_vector, self.axial_gain),
            even_tensor=self._tensor(state.even_tensor, self.even_tensor_gain),
            odd_tensor=self._tensor(state.odd_tensor, self.odd_tensor_gain),
        )


class _NormGatedIrrepActivation(nn.Module):
    """Direction-preserving nonlinearity driven by invariant sector magnitudes.

    The last projection is zero initialized and gates are ``2*sigmoid``; the
    module is therefore exactly the identity at initialization. Odd scalars,
    vectors, and tensors are only multiplied by even scalar gates.
    """

    def __init__(self, *, scalar_width: int, num_heads: int, eps: float) -> None:
        super().__init__()
        self.eps = max(float(eps), 1e-8)
        hidden = max(32, scalar_width, 4 * num_heads)
        self.projection = nn.Sequential(
            nn.Linear(scalar_width + 5 * num_heads, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 5 * num_heads),
        )
        output = self.projection[-1]
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        self.num_heads = num_heads

    def _invariants(self, reference: _ParityState) -> torch.Tensor:
        odd = torch.sqrt(reference.odd_scalar.square() + self.eps)
        polar = torch.sqrt(reference.polar_vector.square().mean(dim=-1) + self.eps)
        axial = torch.sqrt(reference.axial_vector.square().mean(dim=-1) + self.eps)
        even_tensor = torch.sqrt(_st_square(reference.even_tensor) / 5.0 + self.eps)
        odd_tensor = torch.sqrt(_st_square(reference.odd_tensor) / 5.0 + self.eps)
        return torch.cat(
            [reference.even_scalar, odd, polar, axial, even_tensor, odd_tensor],
            dim=-1,
        )

    def forward(self, delta: _ParityState, reference: _ParityState) -> _ParityState:
        gates = 2.0 * torch.sigmoid(self.projection(self._invariants(reference)))
        gates = gates.reshape(delta.even_scalar.shape[0], 5, self.num_heads)
        return _ParityState(
            even_scalar=delta.even_scalar,
            odd_scalar=delta.odd_scalar * gates[:, 0],
            polar_vector=delta.polar_vector * gates[:, 1].unsqueeze(-1),
            axial_vector=delta.axial_vector * gates[:, 2].unsqueeze(-1),
            even_tensor=delta.even_tensor * gates[:, 3].unsqueeze(-1),
            odd_tensor=delta.odd_tensor * gates[:, 4].unsqueeze(-1),
        )


class _EquivariantDropout(nn.Module):
    """Drop complete irrep copies, never individual vector/tensor components."""

    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = _probability("dropout", p)

    def _mask(self, value: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        keep = 1.0 - self.p
        return (
            (torch.rand(shape, device=value.device) < keep).to(dtype=value.dtype)
            / keep
        )

    def forward(self, state: _ParityState) -> _ParityState:
        if not self.training or self.p == 0.0:
            return state
        n, c = state.even_scalar.shape
        h = state.odd_scalar.shape[1]
        return _ParityState(
            even_scalar=state.even_scalar * self._mask(state.even_scalar, (n, c)),
            odd_scalar=state.odd_scalar * self._mask(state.odd_scalar, (n, h)),
            polar_vector=state.polar_vector
            * self._mask(state.polar_vector, (n, h, 1)),
            axial_vector=state.axial_vector
            * self._mask(state.axial_vector, (n, h, 1)),
            even_tensor=state.even_tensor
            * self._mask(state.even_tensor, (n, h, 1)),
            odd_tensor=state.odd_tensor
            * self._mask(state.odd_tensor, (n, h, 1)),
        )


class _GraphDropPath(nn.Module):
    """Graph-wise stochastic depth shared by every node and irrep sector."""

    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = _probability("drop_path", p)

    def forward(
        self,
        state: _ParityState,
        *,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> _ParityState:
        if not self.training or self.p == 0.0:
            return state
        keep = 1.0 - self.p
        graph_mask = (
            (torch.rand((num_graphs, 1), device=batch.device) < keep)
            .to(dtype=state.even_scalar.dtype)
            / keep
        )
        node_mask = graph_mask[batch]
        return _ParityState(
            even_scalar=state.even_scalar * node_mask,
            odd_scalar=state.odd_scalar * node_mask,
            polar_vector=state.polar_vector * node_mask.unsqueeze(-1),
            axial_vector=state.axial_vector * node_mask.unsqueeze(-1),
            even_tensor=state.even_tensor * node_mask.unsqueeze(-1),
            odd_tensor=state.odd_tensor * node_mask.unsqueeze(-1),
        )


class _GroupedRMSLinear(nn.Module):
    """Linear projection followed by grouped RMS query/key normalization."""

    def __init__(
        self,
        linear: nn.Module,
        *,
        groups: int,
        width: int,
        eps: float,
        bounded: bool = False,
    ) -> None:
        super().__init__()
        if groups <= 0 or width <= 0:
            raise ValueError("groups and width must be positive")
        self.linear = linear
        self.groups = groups
        self.width = width
        self.eps = max(float(eps), 1e-8)
        self.bounded = bool(bounded)
        self.gain = nn.Parameter(torch.ones(groups))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        projected = self.linear(value)
        shape = (*projected.shape[:-1], self.groups, self.width)
        grouped = projected.reshape(shape)
        dtype = torch.float64 if grouped.dtype == torch.float64 else torch.float32
        work = grouped.to(dtype=dtype)
        square = work.square().mean(dim=-1, keepdim=True)
        denominator = torch.sqrt(
            (1.0 + square if self.bounded else square) + self.eps
        )
        work = work / denominator * self.gain.to(dtype=dtype).reshape(
            *((1,) * (work.ndim - 2)), self.groups, 1
        )
        return work.to(dtype=projected.dtype).reshape_as(projected)


def _bounded_modulate_state(
    state: _ParityState,
    modulation: _BranchModulation,
    *,
    strength: float = 0.1,
) -> _ParityState:
    scalar_scale = 1.0 + strength * torch.tanh(modulation.scale_even)
    scalar_shift = strength * torch.tanh(modulation.shift_even)
    other_scale = 1.0 + strength * torch.tanh(modulation.scale_other)
    return _ParityState(
        even_scalar=state.even_scalar * scalar_scale + scalar_shift,
        odd_scalar=state.odd_scalar * other_scale[:, 0],
        polar_vector=state.polar_vector * other_scale[:, 1].unsqueeze(-1),
        axial_vector=state.axial_vector * other_scale[:, 2].unsqueeze(-1),
        even_tensor=state.even_tensor * other_scale[:, 3].unsqueeze(-1),
        odd_tensor=state.odd_tensor * other_scale[:, 4].unsqueeze(-1),
    )


class EquivariantLinearAttentionLayer(UnifiedEquivariantLayer):
    """Pre-norm equivariant linear-attention layer with norm-gated residuals."""

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
        residual_dropout: float = 0.0,
        drop_path_probability: float = 0.0,
        norm_eps: float = 1e-6,
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
            condition_dim=condition_dim,
            coordinate_updates=coordinate_updates,
            max_coordinate_step=max_coordinate_step,
            eps=eps,
        )
        self.attention_norm = _EquivariantRMSNorm(
            scalar_width=scalar_width,
            num_heads=num_heads,
            eps=norm_eps,
        )
        self.ffn_state_norm = _EquivariantRMSNorm(
            scalar_width=scalar_width,
            num_heads=num_heads,
            eps=norm_eps,
        )
        self.closure_norm = _EquivariantRMSNorm(
            scalar_width=scalar_width,
            num_heads=num_heads,
            eps=norm_eps,
        )
        self.coordinate_norm = _EquivariantRMSNorm(
            scalar_width=scalar_width,
            num_heads=num_heads,
            eps=norm_eps,
        )
        # Every scalar route now receives explicit RMS-normalized state once.
        # The inherited shared pre-norm is replaced by branch-specific norms.
        self.pre_norm = nn.Identity()
        self.scalar_norm = nn.Identity()
        self.ffn_norm = nn.Identity()
        self.even_update_norm = nn.RMSNorm(
            scalar_width + 6 * num_heads,
            eps=max(norm_eps, 1e-8),
        )

        self.query_scalar = _GroupedRMSLinear(
            self.query_scalar,
            groups=num_heads,
            width=self.head_dim,
            eps=norm_eps,
        )
        self.key_scalar = _GroupedRMSLinear(
            self.key_scalar,
            groups=num_heads,
            width=self.head_dim,
            eps=norm_eps,
        )
        self.local_scalar_query = _GroupedRMSLinear(
            self.local_scalar_query,
            groups=1,
            width=local_rank,
            eps=norm_eps,
            bounded=True,
        )
        self.local_scalar_key = _GroupedRMSLinear(
            self.local_scalar_key,
            groups=1,
            width=local_rank,
            eps=norm_eps,
            bounded=True,
        )

        self.attention_activation = _NormGatedIrrepActivation(
            scalar_width=scalar_width,
            num_heads=num_heads,
            eps=norm_eps,
        )
        self.ffn_activation = _NormGatedIrrepActivation(
            scalar_width=scalar_width,
            num_heads=num_heads,
            eps=norm_eps,
        )
        self.attention_dropout = _EquivariantDropout(residual_dropout)
        self.ffn_dropout = _EquivariantDropout(residual_dropout)
        self.attention_drop_path = _GraphDropPath(drop_path_probability)
        self.ffn_drop_path = _GraphDropPath(drop_path_probability)

    def _regularize(
        self,
        delta: _ParityState,
        reference: _ParityState,
        *,
        activation: _NormGatedIrrepActivation,
        dropout: _EquivariantDropout,
        drop_path: _GraphDropPath,
        context: UnifiedSE3Context,
    ) -> _ParityState:
        delta = activation(delta, reference)
        delta = dropout(delta)
        return drop_path(
            delta,
            batch=context.batch,
            num_graphs=context.graph_layout.num_graphs,
        )

    def _attention_branch(
        self,
        state: _ParityState,
        context: UnifiedSE3Context,
        modulation: _BranchModulation | None,
    ) -> _ParityState:
        normalized = self.attention_norm(state)
        if modulation is not None:
            normalized = _bounded_modulate_state(normalized, modulation)
        global_message = self._global_message(
            normalized,
            context.normalized_positions,
            context.batch,
            context.graph_layout,
        )
        local_message = self._local_message(normalized, context.geometry)
        candidate = _ParityCompleteBlock._update_state(
            self,
            normalized,
            global_message,
            local_message,
        )
        attention_delta = _state_subtract(candidate, normalized)
        if modulation is not None:
            attention_delta = _gate_delta(attention_delta, modulation)
        intermediate = _state_add(state, attention_delta)

        closure = self.tensor_closure(
            self.closure_norm(intermediate),
            context.multipoles,
        )
        closure_delta = _ParityState(
            even_scalar=self.closure_scalar_scale.to(dtype=state.even_scalar.dtype)
            * closure.even_scalar,
            odd_scalar=self.closure_odd_scale.to(dtype=state.odd_scalar.dtype)
            * _bounded_scalar(closure.odd_scalar, self.eps),
            polar_vector=self.closure_polar_scale.to(dtype=state.polar_vector.dtype)
            * _unit_ball(closure.polar_vector, self.eps),
            axial_vector=self.closure_axial_scale.to(dtype=state.axial_vector.dtype)
            * _unit_ball(closure.axial_vector, self.eps),
            even_tensor=self.closure_even_tensor_scale.to(dtype=state.even_tensor.dtype)
            * _bounded_st(closure.even_tensor, self.eps),
            odd_tensor=self.closure_odd_tensor_scale.to(dtype=state.odd_tensor.dtype)
            * _bounded_st(closure.odd_tensor, self.eps),
        )
        if modulation is not None:
            closure_delta = _gate_delta(closure_delta, modulation)
        total_delta = _state_add(attention_delta, closure_delta)
        total_delta = self._regularize(
            total_delta,
            normalized,
            activation=self.attention_activation,
            dropout=self.attention_dropout,
            drop_path=self.attention_drop_path,
            context=context,
        )
        return _state_add(state, total_delta)

    def _ffn_branch(
        self,
        state: _ParityState,
        context: UnifiedSE3Context,
        modulation: _BranchModulation | None,
    ) -> _ParityState:
        normalized = self.ffn_state_norm(state)
        if modulation is not None:
            normalized = _bounded_modulate_state(normalized, modulation)
        scalar_state = normalized.even_scalar
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
        scalar_gates = torch.tanh(self.ffn_gates(scalar_state)).reshape(
            scalar_state.shape[0],
            self.num_heads,
            5,
        )
        odd_delta = scalar_gates[..., 0] * self.ffn_odd(normalized.odd_scalar)
        polar_delta = scalar_gates[..., 1, None] * (
            self.ffn_polar(normalized.polar_vector)
            + normalized.odd_scalar.unsqueeze(-1)
            * self.ffn_polar_cross(normalized.axial_vector)
        )
        axial_delta = scalar_gates[..., 2, None] * (
            self.ffn_axial(normalized.axial_vector)
            + normalized.odd_scalar.unsqueeze(-1)
            * self.ffn_axial_cross(normalized.polar_vector)
        )
        even_tensor_delta = scalar_gates[..., 3, None] * (
            self.ffn_even_tensor(normalized.even_tensor)
            + normalized.odd_scalar.unsqueeze(-1)
            * self.ffn_even_tensor_cross(normalized.odd_tensor)
        )
        odd_tensor_delta = scalar_gates[..., 4, None] * (
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
            even_tensor=self.ffn_even_tensor_scale.to(dtype=state.even_tensor.dtype)
            * _bounded_st(even_tensor_delta, self.eps),
            odd_tensor=self.ffn_odd_tensor_scale.to(dtype=state.odd_tensor.dtype)
            * _bounded_st(odd_tensor_delta, self.eps),
        )
        if modulation is not None:
            delta = _gate_delta(delta, modulation)
        delta = self._regularize(
            delta,
            normalized,
            activation=self.ffn_activation,
            dropout=self.ffn_dropout,
            drop_path=self.ffn_drop_path,
            context=context,
        )
        return _state_add(state, delta)

    def attention_residual(
        self,
        state: UnifiedSE3State,
        context: UnifiedSE3Context,
        condition: torch.Tensor | None = None,
    ) -> UnifiedSE3State:
        internal = state._to_internal()
        modulation = self._resolve_modulation(condition, context, internal)
        return UnifiedSE3State._from_internal(
            self._attention_branch(
                internal,
                context,
                None if modulation is None else modulation.attention,
            )
        )

    def ffn_residual(
        self,
        state: UnifiedSE3State,
        context: UnifiedSE3Context,
        condition: torch.Tensor | None = None,
    ) -> UnifiedSE3State:
        internal = state._to_internal()
        modulation = self._resolve_modulation(condition, context, internal)
        return UnifiedSE3State._from_internal(
            self._ffn_branch(
                internal,
                context,
                None if modulation is None else modulation.ffn,
            )
        )

    def _coordinate_delta_internal(
        self,
        state: _ParityState,
        modulation: _LayerModulation | None,
    ) -> torch.Tensor:
        if self.coordinate_vector is None or self.coordinate_gate is None:
            return state.polar_vector.new_zeros((state.polar_vector.shape[0], 3))
        normalized = self.coordinate_norm(state)
        raw = self.coordinate_vector(normalized.polar_vector).squeeze(1)
        gate_logit = self.coordinate_gate(normalized.even_scalar)
        if modulation is not None:
            gate_logit = gate_logit + modulation.coordinate_gate.to(
                dtype=gate_logit.dtype
            )
        return self.max_coordinate_step * torch.sigmoid(gate_logit) * _unit_ball(
            raw,
            self.eps,
        )

    def _forward_internal(
        self,
        state: _ParityState,
        context: UnifiedSE3Context,
        condition: torch.Tensor | None,
    ) -> tuple[_ParityState, torch.Tensor, torch.Tensor]:
        modulation = self._resolve_modulation(condition, context, state)
        updated = self._attention_branch(
            state,
            context,
            None if modulation is None else modulation.attention,
        )
        updated = self._ffn_branch(
            updated,
            context,
            None if modulation is None else modulation.ffn,
        )
        coordinate_delta = self._coordinate_delta_internal(updated, modulation)
        return updated, context.positions + coordinate_delta, coordinate_delta


class EquivariantLinearAttentionCore(LayeredCanonicalSE3Core):
    """Stack of refined equivariant linear-attention layers."""

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
        residual_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_eps: float = 1e-6,
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
            condition_dim=condition_dim,
            coordinate_updates=coordinate_updates,
            max_coordinate_step=max_coordinate_step,
            eps=eps,
        )
        block_scale = residual_scale_init / sqrt(max(1, num_layers))
        layers: list[nn.Module] = []
        for index in range(num_layers):
            probability = float(drop_path_rate) * index / max(1, num_layers - 1)
            layers.append(
                EquivariantLinearAttentionLayer(
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
                    residual_dropout=residual_dropout,
                    drop_path_probability=probability,
                    norm_eps=norm_eps,
                    eps=eps,
                )
            )
        self.blocks = nn.ModuleList(layers)


class EquivariantLinearAttention(UnifiedEquivariantAttention):
    """Canonical equivariant linear-attention stack.

    Exact global linear attention remains the defining operation. Node
    multipoles and the sparse local residual augment that operator without
    introducing a dense pair matrix or a persistent edge hidden state.
    """

    attention_kind = "equivariant_linear_attention"

    def __init__(self, config: EquivariantLinearAttentionConfig) -> None:
        nn.Module.__init__(self)
        if not isinstance(config, EquivariantLinearAttentionConfig):
            raise TypeError("config must be an EquivariantLinearAttentionConfig")
        self.config = config
        self.core = EquivariantLinearAttentionCore(
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
            residual_dropout=config.residual_dropout,
            drop_path_rate=config.drop_path_rate,
            norm_eps=config.norm_eps,
            eps=config.eps,
        )
        self._initialize_chiral_bridge()
        self.internal_irreps = self.core.internal_irreps
        self.output_irreps = self.core.output_irreps


__all__ = [
    "EquivariantLinearAttention",
    "EquivariantLinearAttentionConfig",
    "EquivariantLinearAttentionCore",
    "EquivariantLinearAttentionLayer",
]
