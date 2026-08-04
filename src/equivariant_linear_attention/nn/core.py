from __future__ import annotations

from math import sqrt

import torch
import torch.nn.functional as F
from torch import nn

from ..geometry.layout import PackedGraphLayout
from ..geometry.neighbors import PackedNeighborGraph
from ..irreps import IrrepLayout
from ..kernels.local import dispatch_local_message
from .multipoles import (
    NodeMultipoleBank,
    NodeMultipoles,
    _LowOrderTensorClosure,
    _ParitySectorNorm,
    _c2_cutoff,
    _normalize_st,
    _safe_unit_direction,
    _st_orthonormal,
)
from .parity import (
    ParityCompleteSE3Core,
    _ChannelMix,
    _ParityCompleteBlock,
    _ParityState,
    _StaticGeometry,
    _bounded_scalar,
    _bounded_st,
    _compute_dtype,
    _csr_sum,
    _exact_balanced_attention,
    _positive_feature,
    _radial_basis,
    _segment_sum,
    _st_cross,
    _st_from_vector,
    _st_inner,
    _st_matvec,
    _st_square,
    _unit_ball,
)


class _CanonicalMultipoleBlock(_ParityCompleteBlock):
    """One exact-global plus one receiver-normalized sparse local operator."""

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
        eps: float,
    ) -> None:
        super().__init__(
            scalar_width=scalar_width,
            num_heads=num_heads,
            local_rank=local_rank,
            num_rbf=num_rbf,
            num_edge_relations=num_edge_relations,
            residual_scale_init=residual_scale_init,
            eps=eps,
        )
        self.pre_norm = _ParitySectorNorm(
            scalar_width=scalar_width,
            num_heads=num_heads,
            eps=eps,
        )
        self.closure_norm = _ParitySectorNorm(
            scalar_width=scalar_width,
            num_heads=num_heads,
            eps=eps,
        )

        self.tensor_query_even = _ChannelMix(num_heads, num_heads)
        self.tensor_key_even = _ChannelMix(num_heads, num_heads)
        self.tensor_query_odd = _ChannelMix(num_heads, num_heads)
        self.tensor_key_odd = _ChannelMix(num_heads, num_heads)
        self.query_odd_scalar = _ChannelMix(num_heads, num_heads)
        self.key_odd_scalar = _ChannelMix(num_heads, num_heads)
        self.raw_tensor_alignment = nn.Parameter(
            torch.full((2, num_heads), -2.0)
        )
        self.raw_odd_alignment = nn.Parameter(torch.full((num_heads,), -2.0))
        self.register_buffer(
            "global_radial_centers",
            torch.linspace(0.0, 1.5, 4),
            persistent=True,
        )
        self.raw_global_radial_alignment = nn.Parameter(
            torch.full((num_heads, 4), -3.0)
        )

        self.local_odd_scalar_query = _ChannelMix(num_heads, local_rank)
        self.local_odd_scalar_key = _ChannelMix(num_heads, local_rank)
        self.local_even_tensor_query = _ChannelMix(num_heads, local_rank)
        self.local_even_tensor_key = _ChannelMix(num_heads, local_rank)
        self.local_odd_tensor_query = _ChannelMix(num_heads, local_rank)
        self.local_odd_tensor_key = _ChannelMix(num_heads, local_rank)
        self.local_tensor_radial_score = nn.Linear(
            num_rbf,
            local_rank,
            bias=False,
        )
        self.local_tensor_mix = nn.Parameter(torch.zeros(5, local_rank))
        self.relation_radial_scale = (
            nn.Parameter(torch.zeros(num_edge_relations, local_rank))
            if num_edge_relations
            else None
        )
        self.relation_value_gate = (
            nn.Parameter(torch.zeros(num_edge_relations, local_rank, 9))
            if num_edge_relations
            else None
        )
        self.second_moment_chiral_mix = nn.Parameter(
            torch.zeros(local_rank)
        )
        nn.init.zeros_(self.local_tensor_radial_score.weight)

        self.tensor_closure = _LowOrderTensorClosure(
            scalar_width=scalar_width,
            num_heads=num_heads,
            rank=max(2, local_rank),
            multipole_rank=multipole_rank,
            eps=eps,
        )

        scale = float(residual_scale_init)
        self.scalar_scale = nn.Parameter(torch.full((scalar_width,), scale))
        self.odd_scale = nn.Parameter(torch.full((num_heads,), scale))
        self.polar_scale = nn.Parameter(torch.full((num_heads, 1), scale))
        self.axial_scale = nn.Parameter(torch.full((num_heads, 1), scale))
        self.even_tensor_scale = nn.Parameter(torch.full((num_heads, 1), scale))
        self.odd_tensor_scale = nn.Parameter(torch.full((num_heads, 1), scale))

        self.closure_scalar_scale = nn.Parameter(
            torch.full((scalar_width,), scale)
        )
        self.closure_odd_scale = nn.Parameter(torch.full((num_heads,), scale))
        self.closure_polar_scale = nn.Parameter(
            torch.full((num_heads, 1), scale)
        )
        self.closure_axial_scale = nn.Parameter(
            torch.full((num_heads, 1), scale)
        )
        self.closure_even_tensor_scale = nn.Parameter(
            torch.full((num_heads, 1), scale)
        )
        self.closure_odd_tensor_scale = nn.Parameter(
            torch.full((num_heads, 1), scale)
        )

        self.ffn_scalar_scale = nn.Parameter(torch.full((scalar_width,), scale))
        self.ffn_odd_scale = nn.Parameter(torch.full((num_heads,), scale))
        self.ffn_polar_scale = nn.Parameter(torch.full((num_heads, 1), scale))
        self.ffn_axial_scale = nn.Parameter(torch.full((num_heads, 1), scale))
        self.ffn_even_tensor_scale = nn.Parameter(
            torch.full((num_heads, 1), scale)
        )
        self.ffn_odd_tensor_scale = nn.Parameter(
            torch.full((num_heads, 1), scale)
        )
        del self.ffn_scale

    def forward(
        self,
        state: _ParityState,
        normalized_pos: torch.Tensor,
        geometry: _StaticGeometry,
        multipoles: NodeMultipoles,
        batch: torch.Tensor,
        graph_layout: PackedGraphLayout,
    ) -> _ParityState:
        normalized = self.pre_norm(state)
        global_message = self._global_message(
            normalized,
            normalized_pos,
            batch,
            graph_layout,
        )
        local_message = self._local_message(normalized, geometry)
        updated = super()._update_state(state, global_message, local_message)

        closure = self.tensor_closure(self.closure_norm(updated), multipoles)
        updated = _ParityState(
            even_scalar=updated.even_scalar
            + self.closure_scalar_scale.to(dtype=updated.even_scalar.dtype)
            * closure.even_scalar,
            odd_scalar=updated.odd_scalar
            + self.closure_odd_scale.to(dtype=updated.odd_scalar.dtype)
            * _bounded_scalar(closure.odd_scalar, self.eps),
            polar_vector=updated.polar_vector
            + self.closure_polar_scale.to(dtype=updated.polar_vector.dtype)
            * _unit_ball(closure.polar_vector, self.eps),
            axial_vector=updated.axial_vector
            + self.closure_axial_scale.to(dtype=updated.axial_vector.dtype)
            * _unit_ball(closure.axial_vector, self.eps),
            even_tensor=updated.even_tensor
            + self.closure_even_tensor_scale.to(dtype=updated.even_tensor.dtype)
            * _bounded_st(closure.even_tensor, self.eps),
            odd_tensor=updated.odd_tensor
            + self.closure_odd_tensor_scale.to(dtype=updated.odd_tensor.dtype)
            * _bounded_st(closure.odd_tensor, self.eps),
        )
        return self._enhanced_ffn(updated)

    def _global_message(
        self,
        state: _ParityState,
        normalized_pos: torch.Tensor,
        batch: torch.Tensor,
        graph_layout: PackedGraphLayout,
    ) -> tuple[torch.Tensor, ...]:
        scalar_state = self.scalar_norm(state.even_scalar)
        num_nodes = scalar_state.shape[0]
        query_scalar = _positive_feature(
            self.query_scalar(scalar_state).reshape(
                num_nodes,
                self.num_heads,
                self.head_dim,
            )
        )
        key_scalar = _positive_feature(
            self.key_scalar(scalar_state).reshape(
                num_nodes,
                self.num_heads,
                self.head_dim,
            )
        )
        gates = torch.tanh(self.query_key_gates(scalar_state)).reshape(
            num_nodes,
            4,
            self.num_heads,
            1,
        )
        query_polar = _unit_ball(
            self.query_polar(state.polar_vector) * gates[:, 0],
            self.eps,
        )
        key_polar = _unit_ball(
            self.key_polar(state.polar_vector) * gates[:, 1],
            self.eps,
        )
        query_axial = _unit_ball(
            self.query_axial(state.axial_vector) * gates[:, 2],
            self.eps,
        )
        key_axial = _unit_ball(
            self.key_axial(state.axial_vector) * gates[:, 3],
            self.eps,
        )
        query_even_tensor = _st_orthonormal(
            _normalize_st(
                self.tensor_query_even(state.even_tensor),
                self.eps,
            )
        )
        key_even_tensor = _st_orthonormal(
            _normalize_st(
                self.tensor_key_even(state.even_tensor),
                self.eps,
            )
        )
        query_odd_tensor = _st_orthonormal(
            _normalize_st(
                self.tensor_query_odd(state.odd_tensor),
                self.eps,
            )
        )
        key_odd_tensor = _st_orthonormal(
            _normalize_st(
                self.tensor_key_odd(state.odd_tensor),
                self.eps,
            )
        )
        query_odd_scalar = _bounded_scalar(
            self.query_odd_scalar(state.odd_scalar),
            self.eps,
        )
        key_odd_scalar = _bounded_scalar(
            self.key_odd_scalar(state.odd_scalar),
            self.eps,
        )

        vector_alignment = F.softplus(self.raw_alignment_scale)
        tensor_alignment = F.softplus(self.raw_tensor_alignment)
        polar_scale = vector_alignment[0]
        axial_scale = vector_alignment[1]
        even_tensor_scale = tensor_alignment[0]
        odd_tensor_scale = tensor_alignment[1]
        odd_scalar_scale = F.softplus(self.raw_odd_alignment)
        odd_query_feature = torch.stack(
            (1.0 + query_odd_scalar, 1.0 - query_odd_scalar),
            dim=-1,
        ) * torch.sqrt(0.5 * odd_scalar_scale)[None, :, None]
        odd_key_feature = torch.stack(
            (1.0 + key_odd_scalar, 1.0 - key_odd_scalar),
            dim=-1,
        ) * torch.sqrt(0.5 * odd_scalar_scale)[None, :, None]

        radius_squared = normalized_pos.to(
            dtype=state.even_scalar.dtype
        ).square().sum(dim=-1)
        centers = self.global_radial_centers.to(
            device=radius_squared.device,
            dtype=radius_squared.dtype,
        ).square()
        # Squared-radius shells avoid the non-differentiable norm at the graph
        # centre while remaining invariant and positive.
        radial_shell = torch.exp(
            -4.0 * (radius_squared[:, None] - centers[None, :]).square()
        )
        radial_scale = F.softplus(self.raw_global_radial_alignment).to(
            dtype=radius_squared.dtype
        )
        radial_feature = radial_shell[:, None, :] * torch.sqrt(radial_scale)[None, :, :]

        constant = torch.sqrt(
            1.0
            + polar_scale
            + axial_scale
            + even_tensor_scale
            + odd_tensor_scale
        )[None, :, None]
        query_feature = torch.cat(
            [
                query_scalar,
                constant.expand(num_nodes, -1, -1),
                odd_query_feature,
                radial_feature,
                torch.sqrt(polar_scale)[None, :, None] * query_polar,
                torch.sqrt(axial_scale)[None, :, None] * query_axial,
                torch.sqrt(even_tensor_scale)[None, :, None]
                * query_even_tensor,
                torch.sqrt(odd_tensor_scale)[None, :, None]
                * query_odd_tensor,
            ],
            dim=-1,
        )
        key_feature = torch.cat(
            [
                key_scalar,
                constant.expand(num_nodes, -1, -1),
                odd_key_feature,
                radial_feature,
                torch.sqrt(polar_scale)[None, :, None] * key_polar,
                torch.sqrt(axial_scale)[None, :, None] * key_axial,
                torch.sqrt(even_tensor_scale)[None, :, None]
                * key_even_tensor,
                torch.sqrt(odd_tensor_scale)[None, :, None]
                * key_odd_tensor,
            ],
            dim=-1,
        )

        scalar_value = self.global_scalar_value(state.even_scalar).reshape(
            num_nodes,
            self.num_heads,
            self.head_dim,
        )
        odd_value = self.global_odd_value(state.odd_scalar).unsqueeze(-1)
        polar_value = self.global_polar_value(state.polar_vector)
        axial_value = self.global_axial_value(state.axial_vector)
        even_tensor_value = self.global_even_tensor_value(state.even_tensor)
        odd_tensor_value = self.global_odd_tensor_value(state.odd_tensor)

        geometry_gate = torch.tanh(
            self.global_geometry_gate(scalar_state)
        ).reshape(num_nodes, 2, self.num_heads)
        position = normalized_pos[:, None, :].expand(
            -1,
            self.num_heads,
            -1,
        ).to(dtype=state.even_scalar.dtype)
        position_tensor = _st_from_vector(position)
        vector_mass = geometry_gate[:, 0].unsqueeze(-1)
        tensor_mass = geometry_gate[:, 1].unsqueeze(-1)
        value = torch.cat(
            [
                scalar_value,
                odd_value,
                polar_value,
                axial_value,
                even_tensor_value,
                odd_tensor_value,
                vector_mass,
                vector_mass * position,
                tensor_mass,
                tensor_mass * position,
                tensor_mass * position_tensor,
            ],
            dim=-1,
        )
        transported = _exact_balanced_attention(
            query_feature,
            key_feature,
            value,
            batch,
            graph_layout,
            eps=self.eps,
        )
        (
            scalar_message,
            odd_message,
            polar_message,
            axial_message,
            even_tensor_message,
            odd_tensor_message,
            vector_gate_mass,
            vector_first,
            tensor_gate_mass,
            tensor_first,
            tensor_second,
        ) = torch.split(
            transported,
            [
                self.head_dim,
                1,
                3,
                3,
                5,
                5,
                1,
                3,
                1,
                3,
                5,
            ],
            dim=-1,
        )
        relative_vector = vector_first - vector_gate_mass * position
        relative_tensor = (
            tensor_second
            + tensor_gate_mass * position_tensor
            - 2.0 * _st_cross(tensor_first, position)
        )
        return (
            scalar_message,
            odd_message.squeeze(-1),
            polar_message + relative_vector,
            axial_message,
            even_tensor_message + relative_tensor,
            odd_tensor_message,
        )

    def _local_message(
        self,
        state: _ParityState,
        geometry: _StaticGeometry,
    ) -> tuple[torch.Tensor, ...]:
        """Dispatch the canonical local operator through the active backend."""

        return dispatch_local_message(self, state, geometry)

    def _torch_local_message(
        self,
        state: _ParityState,
        geometry: _StaticGeometry,
    ) -> tuple[torch.Tensor, ...]:
        """PyTorch numerical reference for the canonical local operator."""

        receiver = geometry.receiver
        sender = geometry.sender
        direction = geometry.direction
        rbf = geometry.rbf.to(dtype=state.even_scalar.dtype)
        dtype = _compute_dtype(
            state.even_scalar,
            state.polar_vector,
            state.even_tensor,
            direction,
        )

        scalar_state = self.scalar_norm(state.even_scalar)
        scalar_query = self.local_scalar_query(scalar_state).to(dtype=dtype)
        scalar_key = self.local_scalar_key(scalar_state).to(dtype=dtype)
        gates = torch.tanh(
            self.local_query_key_gates(scalar_state)
        ).reshape(
            scalar_state.shape[0],
            4,
            self.local_rank,
            1,
        )
        polar_query = (
            self.local_polar_query(state.polar_vector) * gates[:, 0]
        ).to(dtype=dtype)
        polar_key = (
            self.local_polar_key(state.polar_vector) * gates[:, 1]
        ).to(dtype=dtype)
        axial_query = (
            self.local_axial_query(state.axial_vector) * gates[:, 2]
        ).to(dtype=dtype)
        axial_key = (
            self.local_axial_key(state.axial_vector) * gates[:, 3]
        ).to(dtype=dtype)

        edge_direction = direction[:, None, :].to(dtype=dtype)
        unit_direction = _safe_unit_direction(
            direction.to(dtype=dtype),
            geometry.squared_distance.to(dtype=dtype),
            self.eps,
        )[:, None, :]
        receiver_polar = polar_query[receiver]
        sender_polar = polar_key[sender]
        receiver_axial = axial_query[receiver]
        sender_axial = axial_key[sender]
        polar_dot = (receiver_polar * sender_polar).sum(dim=-1)
        axial_dot = (receiver_axial * sender_axial).sum(dim=-1)
        receiver_polar_axis = (receiver_polar * unit_direction).sum(dim=-1)
        sender_polar_axis = (sender_polar * unit_direction).sum(dim=-1)
        receiver_axial_axis = (receiver_axial * unit_direction).sum(dim=-1)
        sender_axial_axis = (sender_axial * unit_direction).sum(dim=-1)

        odd_query = self.local_odd_scalar_query(state.odd_scalar).to(dtype=dtype)
        odd_key = self.local_odd_scalar_key(state.odd_scalar).to(dtype=dtype)
        even_tensor_query = _normalize_st(
            self.local_even_tensor_query(state.even_tensor).to(dtype=dtype),
            self.eps,
        )
        even_tensor_key = _normalize_st(
            self.local_even_tensor_key(state.even_tensor).to(dtype=dtype),
            self.eps,
        )
        odd_tensor_query = _normalize_st(
            self.local_odd_tensor_query(state.odd_tensor).to(dtype=dtype),
            self.eps,
        )
        odd_tensor_key = _normalize_st(
            self.local_odd_tensor_key(state.odd_tensor).to(dtype=dtype),
            self.eps,
        )
        even_query_axis = (
            _st_matvec(even_tensor_query[receiver], unit_direction)
            * unit_direction
        ).sum(dim=-1)
        even_key_axis = (
            _st_matvec(even_tensor_key[sender], unit_direction)
            * unit_direction
        ).sum(dim=-1)
        odd_query_axis = (
            _st_matvec(odd_tensor_query[receiver], unit_direction)
            * unit_direction
        ).sum(dim=-1)
        odd_key_axis = (
            _st_matvec(odd_tensor_key[sender], unit_direction)
            * unit_direction
        ).sum(dim=-1)
        relation_id = geometry.relation_id
        radial_score = self.local_radial_score(rbf).to(dtype=dtype)
        tensor_radial_score = self.local_tensor_radial_score(rbf).to(dtype=dtype)
        if self.relation_radial_scale is not None:
            if relation_id is None:
                raise ValueError("relation-aware ELA requires relation metadata")
            relation_scale = 1.0 + torch.tanh(
                self.relation_radial_scale.to(dtype=dtype)[relation_id]
            )
            radial_score = radial_score * relation_scale
            tensor_radial_score = tensor_radial_score * relation_scale

        tensor_mix = self.local_tensor_mix.to(dtype=dtype)
        tensor_score = (
            tensor_radial_score
            + tensor_mix[0][None, :] * odd_query[receiver] * odd_key[sender]
            + tensor_mix[1][None, :] * _st_inner(
                even_tensor_query[receiver],
                even_tensor_key[sender],
            )
            + tensor_mix[2][None, :] * _st_inner(
                odd_tensor_query[receiver],
                odd_tensor_key[sender],
            )
            + tensor_mix[3][None, :] * even_query_axis * even_key_axis
            + tensor_mix[4][None, :] * odd_query_axis * odd_key_axis
        )

        score = (
            scalar_query[receiver] * scalar_key[sender]
            + self.local_score_bias.to(dtype=dtype)[None, :]
            + radial_score
            + self.local_polar_mix[0].to(dtype=dtype)[None, :] * polar_dot
            + self.local_polar_mix[1].to(dtype=dtype)[None, :]
            * receiver_polar_axis
            * sender_polar_axis
            + self.local_axial_mix[0].to(dtype=dtype)[None, :] * axial_dot
            + self.local_axial_mix[1].to(dtype=dtype)[None, :]
            * receiver_axial_axis
            * sender_axial_axis
            + tensor_score
        )
        if self.relation_score_bias is not None:
            if relation_id is None:
                raise ValueError("relation-aware ELA requires relation metadata")
            score = score + self.relation_score_bias.to(dtype=dtype)[relation_id]
        positive_gate = torch.exp(3.0 * torch.tanh(score / 3.0))
        raw_weight = geometry.cutoff.to(dtype=dtype)[:, None] * positive_gate
        mass = _csr_sum(raw_weight, geometry.row_ptr)
        mass_square = _csr_sum(raw_weight.square(), geometry.row_ptr)
        denominator = 1.0 + mass

        num_nodes = state.even_scalar.shape[0]
        scalar_value = self.local_scalar_value(
            state.even_scalar
        ).reshape(num_nodes, self.local_rank, self.head_dim).to(dtype=dtype)
        odd_value = self.local_odd_value(state.odd_scalar).to(dtype=dtype)
        polar_value = self.local_polar_value(state.polar_vector).to(dtype=dtype)
        axial_value = self.local_axial_value(state.axial_vector).to(dtype=dtype)
        even_tensor_value = self.local_even_tensor_value(
            state.even_tensor
        ).to(dtype=dtype)
        odd_tensor_value = self.local_odd_tensor_value(
            state.odd_tensor
        ).to(dtype=dtype)
        direction_gate = torch.tanh(
            self.local_direction_gate(scalar_state)
        ).reshape(num_nodes, 3, self.local_rank).to(dtype=dtype)
        radial_gate = 2.0 * torch.sigmoid(
            self.local_radial_value(rbf).reshape(
                rbf.shape[0],
                self.local_rank,
                9,
            )
        ).to(dtype=dtype)
        if self.relation_value_gate is not None:
            if relation_id is None:
                raise ValueError("relation-aware ELA requires relation metadata")
            radial_gate = radial_gate * (
                1.0
                + torch.tanh(
                    self.relation_value_gate.to(dtype=dtype)[relation_id]
                )
            )

        def reduce_rank(value: torch.Tensor) -> torch.Tensor:
            return _csr_sum(value, geometry.row_ptr)

        scalar_rank = reduce_rank(
            raw_weight.unsqueeze(-1)
            * radial_gate[..., 0, None]
            * scalar_value[sender]
        ) / denominator.unsqueeze(-1)
        odd_rank = reduce_rank(
            raw_weight * radial_gate[..., 1] * odd_value[sender]
        ) / denominator
        polar_rank = reduce_rank(
            raw_weight.unsqueeze(-1)
            * radial_gate[..., 2, None]
            * polar_value[sender]
        ) / denominator.unsqueeze(-1)
        axial_rank = reduce_rank(
            raw_weight.unsqueeze(-1)
            * radial_gate[..., 3, None]
            * axial_value[sender]
        ) / denominator.unsqueeze(-1)
        even_tensor_rank = reduce_rank(
            raw_weight.unsqueeze(-1)
            * radial_gate[..., 4, None]
            * (
                even_tensor_value[sender]
                + _st_from_vector(edge_direction)
            )
        ) / denominator.unsqueeze(-1)
        odd_tensor_rank = reduce_rank(
            raw_weight.unsqueeze(-1)
            * radial_gate[..., 5, None]
            * odd_tensor_value[sender]
        ) / denominator.unsqueeze(-1)

        direction_ranks: list[torch.Tensor] = []
        for index in range(3):
            direction_ranks.append(
                reduce_rank(
                    raw_weight.unsqueeze(-1)
                    * radial_gate[..., 6 + index, None]
                    * direction_gate[sender, index].unsqueeze(-1)
                    * edge_direction
                )
                / denominator.unsqueeze(-1)
            )
        first_direction, second_direction, third_direction = direction_ranks
        chiral_axial = torch.cross(
            first_direction,
            second_direction,
            dim=-1,
        )
        tensor_direction = _st_matvec(even_tensor_rank, third_direction)
        second_axial = torch.cross(second_direction, tensor_direction, dim=-1)
        second_scalar = (first_direction * second_axial).sum(dim=-1)
        second_mix = torch.tanh(self.second_moment_chiral_mix).to(dtype=dtype)
        # Keep the proven first-moment axial/tensor channel unchanged.  The
        # second-moment pseudoscalar is an isolated zero-initialized residual,
        # so it cannot silently alter the canonical initialization.
        chiral_scalar = (
            chiral_axial * third_direction
        ).sum(dim=-1) + second_mix[None, :] * second_scalar
        chiral_tensor = _st_cross(third_direction, chiral_axial)

        scalar_message = self.local_scalar_out(scalar_rank)
        scalar_message = scalar_message + self.local_mass_out(
            torch.cat([torch.log1p(mass), torch.log1p(mass_square)], dim=-1)
            .to(dtype=state.even_scalar.dtype)
        ).reshape(num_nodes, self.num_heads, self.head_dim).to(dtype=dtype)
        chiral_scalar_message = self.local_chiral_scalar_out(chiral_scalar)
        return (
            scalar_message,
            self.local_odd_out(odd_rank)
            + chiral_scalar_message,
            self.local_polar_out(polar_rank),
            self.local_axial_out(axial_rank)
            + self.local_chiral_axial_out(chiral_axial),
            self.local_even_tensor_out(even_tensor_rank),
            self.local_odd_tensor_out(odd_tensor_rank)
            + self.local_chiral_tensor_out(chiral_tensor),
            chiral_scalar_message,
        )

    def _enhanced_ffn(self, state: _ParityState) -> _ParityState:
        normalized = self.pre_norm(state)
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
        return _ParityState(
            even_scalar=state.even_scalar
            + self.ffn_scalar_scale.to(dtype=state.even_scalar.dtype)
            * even_delta,
            odd_scalar=state.odd_scalar
            + self.ffn_odd_scale.to(dtype=state.odd_scalar.dtype)
            * _bounded_scalar(odd_delta, self.eps),
            polar_vector=state.polar_vector
            + self.ffn_polar_scale.to(dtype=state.polar_vector.dtype)
            * _unit_ball(polar_delta, self.eps),
            axial_vector=state.axial_vector
            + self.ffn_axial_scale.to(dtype=state.axial_vector.dtype)
            * _unit_ball(axial_delta, self.eps),
            even_tensor=state.even_tensor
            + self.ffn_even_tensor_scale.to(dtype=state.even_tensor.dtype)
            * _bounded_st(even_tensor_delta, self.eps),
            odd_tensor=state.odd_tensor
            + self.ffn_odd_tensor_scale.to(dtype=state.odd_tensor.dtype)
            * _bounded_st(odd_tensor_delta, self.eps),
        )


class CanonicalMultipoleSE3Core(ParityCompleteSE3Core):
    """Canonical node-multipole, tensor-routed, parity-complete SE(3) core."""

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
        self.multipole_rank = max(3, local_rank)
        self.node_multipoles = NodeMultipoleBank(
            num_rbf=num_rbf,
            rank=self.multipole_rank,
            eps=eps,
        )
        self.scale_density_projection = nn.Linear(4, hidden_dim, bias=False)
        self.initial_multipole_polar = _ChannelMix(
            self.multipole_rank,
            num_heads,
        )
        self.initial_multipole_axial = _ChannelMix(
            self.multipole_rank,
            num_heads,
        )
        self.initial_multipole_even_tensor = _ChannelMix(
            self.multipole_rank,
            num_heads,
        )
        self.initial_multipole_odd_scalar = _ChannelMix(
            self.multipole_rank,
            num_heads,
        )
        self.initial_multipole_odd_tensor = _ChannelMix(
            self.multipole_rank,
            num_heads,
        )
        block_scale = residual_scale_init / sqrt(max(1, num_layers))
        self.blocks = nn.ModuleList(
            [
                _CanonicalMultipoleBlock(
                    scalar_width=hidden_dim,
                    num_heads=num_heads,
                    local_rank=local_rank,
                    num_rbf=num_rbf,
                    num_edge_relations=num_edge_relations,
                    multipole_rank=self.multipole_rank,
                    residual_scale_init=block_scale,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.register_buffer(
            "_relation_cutoff_buffer",
            torch.tensor(relation_cutoffs, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        node_irreps: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        graph_layout: PackedGraphLayout,
        neighbors: PackedNeighborGraph,
        *,
        node_role_id: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        self._validate_inputs(
            node_irreps,
            pos,
            batch,
            graph_layout,
            neighbors,
            node_role_id=node_role_id,
        )
        normalized_pos, radius, centered_norm = self._normalized_geometry(
            pos,
            batch,
            graph_layout,
        )
        geometry = self._build_geometry(pos, neighbors)
        multipoles = self.node_multipoles(geometry)

        external = self.input_projection(node_irreps)
        even_scalar = external.even_scalar
        if self.role_embedding is not None:
            if node_role_id is None:
                raise RuntimeError("validated role IDs are missing")
            even_scalar = even_scalar + self.role_embedding(
                node_role_id.to(dtype=torch.long)
            )
        density_features = torch.stack(
            [
                torch.log1p(radius[batch]),
                torch.log1p(centered_norm),
                torch.log1p(multipoles.mass.mean(dim=-1)),
                torch.log1p(multipoles.mass_square.mean(dim=-1)),
            ],
            dim=-1,
        ).to(dtype=even_scalar.dtype)
        even_scalar = even_scalar + self.scale_density_projection(
            density_features
        )

        polar = torch.tanh(
            self.initial_vector_gate(even_scalar)
        ).unsqueeze(-1) * normalized_pos[:, None, :].to(
            dtype=even_scalar.dtype
        )
        polar = polar + self.initial_multipole_polar(
            multipoles.polar.to(dtype=polar.dtype)
        )
        polar = polar + external.polar_vector.to(dtype=polar.dtype)

        even_tensor = torch.tanh(
            self.initial_tensor_gate(even_scalar)
        ).unsqueeze(-1) * _st_from_vector(
            normalized_pos[:, None, :].expand(-1, self.num_heads, -1)
        ).to(dtype=even_scalar.dtype)
        even_tensor = even_tensor + self.initial_multipole_even_tensor(
            multipoles.even_tensor.to(dtype=even_tensor.dtype)
        )
        even_tensor = even_tensor + external.even_tensor.to(
            dtype=even_tensor.dtype
        )

        state = _ParityState(
            even_scalar=even_scalar,
            odd_scalar=external.odd_scalar.to(dtype=even_scalar.dtype)
            + self.initial_multipole_odd_scalar(
                multipoles.odd_scalar.to(dtype=even_scalar.dtype)
            ),
            polar_vector=polar,
            axial_vector=external.axial_vector.to(dtype=polar.dtype)
            + self.initial_multipole_axial(
                multipoles.axial.to(dtype=polar.dtype)
            ),
            even_tensor=even_tensor,
            odd_tensor=external.odd_tensor.to(dtype=even_tensor.dtype)
            + self.initial_multipole_odd_tensor(
                multipoles.odd_tensor.to(dtype=even_tensor.dtype)
            ),
        )
        for block in self.blocks:
            state = block(
                state,
                normalized_pos,
                geometry,
                multipoles,
                batch,
                graph_layout,
            )

        node_irreps = self.output_projection(state)
        graph_irreps = _segment_sum(
            node_irreps,
            batch,
            graph_layout.num_graphs,
        ) / graph_layout.graph_counts.to(
            dtype=node_irreps.dtype
        ).clamp_min(1.0).unsqueeze(-1)
        return {
            "node_irreps": node_irreps,
            "graph_irreps": graph_irreps,
        }

    def _normalized_geometry(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor,
        layout: PackedGraphLayout,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = _compute_dtype(pos)
        position = pos.to(dtype=dtype)
        counts = layout.graph_counts.to(dtype=dtype).clamp_min(1.0)
        center = _segment_sum(position, batch, layout.num_graphs) / counts[:, None]
        centered = position - center[batch]
        centered_square = centered.square().sum(dim=-1)
        centered_norm = torch.sqrt(centered_square + self.eps)
        radius = torch.sqrt(
            _segment_sum(centered_square, batch, layout.num_graphs)
            / counts
            + self.eps
        )
        normalized = centered / radius[batch, None].clamp_min(self.eps)
        return normalized, radius, centered_norm

    def _build_geometry(
        self,
        pos: torch.Tensor,
        neighbors: PackedNeighborGraph,
    ) -> _StaticGeometry:
        receiver = neighbors.receiver_index().to(dtype=torch.long)
        sender = neighbors.sender.to(dtype=torch.long)
        dtype = _compute_dtype(pos)
        position = pos.to(dtype=dtype)
        cutoff = position.new_tensor(self.local_cutoff)
        direction = (position[sender] - position[receiver]) / cutoff
        squared_distance = direction.square().sum(dim=-1)
        cutoff_argument = squared_distance
        relation_id = None
        if self.num_edge_relations:
            if neighbors.relation_id is None:
                raise ValueError(
                    "relation-aware ELA requires relation IDs"
                )
            relation_id = neighbors.relation_id.to(dtype=torch.long)
            relation_cutoff = self._relation_cutoff_buffer.to(
                device=pos.device,
                dtype=dtype,
            )[relation_id]
            cutoff_argument = squared_distance * (
                cutoff / relation_cutoff
            ).square()
        smooth = _c2_cutoff(cutoff_argument)
        smooth = smooth * (receiver != sender).to(dtype=smooth.dtype)
        centers = self._rbf_centers.to(device=pos.device, dtype=dtype)
        width = self._rbf_width.to(device=pos.device, dtype=dtype)
        rbf = _radial_basis(squared_distance, centers, width)
        return _StaticGeometry(
            receiver=receiver,
            sender=sender,
            direction=direction,
            squared_distance=squared_distance,
            rbf=rbf,
            cutoff=smooth,
            row_ptr=neighbors.row_ptr,
            relation_id=relation_id,
        )


__all__ = [
    "CanonicalMultipoleSE3Core",
    "NodeMultipoleBank",
    "NodeMultipoles",
]
