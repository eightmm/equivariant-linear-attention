from __future__ import annotations

from math import sqrt

import torch
import torch.nn.functional as F
from torch import nn

from ..geometry.layout import PackedGraphLayout
from ..geometry.neighbors import PackedNeighborGraph
from ..nn.layers import _ELALayerContext
from ..nn.multipoles import NodeMultipoles, _normalize_st, _st_orthonormal
from ..nn.parity import (
    _ParityState,
    _StaticGeometry,
    _bounded_scalar,
    _compute_dtype,
    _exact_balanced_attention,
    _positive_feature,
    _segment_sum,
    _st_cross,
    _st_from_vector,
    _unit_ball,
)
from .ela import ELALayer, _ELACore
from .stack import EquivariantLinearAttentionConfig


class _EdgeFreeRelativeMomentBank(nn.Module):
    """Graphwise receiver-centred ``l<=2`` moments without pair materialization.

    For positive source weights ``w_j`` this module evaluates, for every receiver
    ``i``, the exact graphwise sums

    ``sum_{j != i} w_j (x_j - x_i)`` and
    ``sum_{j != i} w_j ST((x_j - x_i) (x_j - x_i)^T)``.

    Only graph reductions over node, vector, and compact ST5 values are used.
    No radius graph, edge list, ``N x N`` matrix, or pair representation is
    constructed. Multiple deterministic radial profiles give the tensor closure
    independent geometric lanes from which axial and chiral sectors are formed.
    """

    def __init__(self, *, rank: int, eps: float) -> None:
        super().__init__()
        if rank < 3:
            raise ValueError("edge-free relative moments require rank>=3")
        self.rank = int(rank)
        self.eps = float(eps)
        self.register_buffer(
            "_radial_scales",
            torch.linspace(-4.0, 4.0, rank),
            persistent=False,
        )

    def forward(
        self,
        normalized_positions: torch.Tensor,
        batch: torch.Tensor,
        *,
        num_graphs: int,
    ) -> NodeMultipoles:
        if normalized_positions.ndim != 2 or normalized_positions.shape[-1] != 3:
            raise ValueError("normalized_positions must have shape (N, 3)")
        if batch.shape != (normalized_positions.shape[0],):
            raise ValueError("batch must have shape (N,)")

        dtype = _compute_dtype(normalized_positions)
        position = normalized_positions.to(dtype=dtype)
        batch_index = batch.to(dtype=torch.long)
        radius_squared = position.square().sum(dim=-1)
        radial_coordinate = radius_squared / (1.0 + radius_squared)
        radial = torch.exp(
            (radial_coordinate.unsqueeze(-1) - 0.5)
            * self._radial_scales.to(device=position.device, dtype=dtype)
        )

        graph_mass = _segment_sum(radial, batch_index, num_graphs)
        graph_mass_square = _segment_sum(radial.square(), batch_index, num_graphs)
        mass = (graph_mass[batch_index] - radial).clamp_min(0.0)
        mass_square = (
            graph_mass_square[batch_index] - radial.square()
        ).clamp_min(0.0)
        denominator = 1.0 + mass

        weighted_position = radial.unsqueeze(-1) * position.unsqueeze(1)
        graph_first = _segment_sum(weighted_position, batch_index, num_graphs)
        first_without_self = graph_first[batch_index] - weighted_position
        relative_first = first_without_self - mass.unsqueeze(-1) * position[:, None, :]
        polar = relative_first / denominator.unsqueeze(-1)

        position_tensor = _st_from_vector(position)
        weighted_tensor = radial.unsqueeze(-1) * position_tensor.unsqueeze(1)
        graph_second = _segment_sum(weighted_tensor, batch_index, num_graphs)
        second_without_self = graph_second[batch_index] - weighted_tensor
        relative_second = (
            second_without_self
            + mass.unsqueeze(-1) * position_tensor[:, None, :]
            - 2.0 * _st_cross(first_without_self, position[:, None, :])
        )
        even_tensor = relative_second / denominator.unsqueeze(-1)

        second = polar.roll(shifts=1, dims=1)
        third = polar.roll(shifts=2, dims=1)
        axial = torch.cross(polar, second, dim=-1)
        odd_scalar = (axial * third).sum(dim=-1)
        odd_tensor = _st_cross(third, axial)
        return NodeMultipoles(
            mass=mass,
            mass_square=mass_square,
            polar=polar,
            even_tensor=even_tensor,
            axial=axial,
            odd_scalar=odd_scalar,
            odd_tensor=odd_tensor,
        )


class EdgeFreeELALayer(ELALayer):
    """ELA layer with implicit Krylov relation closure and optional edge residual."""

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
            residual_dropout=residual_dropout,
            drop_path_probability=drop_path_probability,
            norm_eps=norm_eps,
            eps=eps,
        )
        self.global_krylov_gate = nn.Linear(scalar_width, 2 * num_heads)
        nn.init.zeros_(self.global_krylov_gate.weight)
        nn.init.zeros_(self.global_krylov_gate.bias)

    def _local_message(
        self,
        state: _ParityState,
        geometry: _StaticGeometry,
    ) -> tuple[torch.Tensor, ...]:
        # The default graph contains no implicit radius edges. Explicit user
        # edges still activate the established typed sparse local residual.
        if geometry.sender.numel():
            return super()._local_message(state, geometry)
        num_nodes = state.even_scalar.shape[0]
        scalar = state.even_scalar.new_zeros(
            (num_nodes, self.num_heads, self.head_dim)
        )
        scalar_sector = state.even_scalar.new_zeros((num_nodes, self.num_heads))
        vector = state.polar_vector.new_zeros((num_nodes, self.num_heads, 3))
        tensor = state.even_tensor.new_zeros((num_nodes, self.num_heads, 5))
        return (
            scalar,
            scalar_sector,
            vector,
            vector.clone(),
            tensor,
            tensor.clone(),
            scalar_sector.clone(),
        )

    @staticmethod
    def _decode_transport(
        transported: torch.Tensor,
        position: torch.Tensor,
        position_tensor: torch.Tensor,
        *,
        head_dim: int,
    ) -> tuple[torch.Tensor, ...]:
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
            [head_dim, 1, 3, 3, 5, 5, 1, 3, 1, 3, 5],
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

    @staticmethod
    def _mix_orders(
        first: tuple[torch.Tensor, ...],
        second: tuple[torch.Tensor, ...],
        third: tuple[torch.Tensor, ...],
        gates: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        second_gate = gates[:, 0]
        third_gate = gates[:, 1]
        outputs: list[torch.Tensor] = []
        for order_one, order_two, order_three in zip(
            first,
            second,
            third,
            strict=True,
        ):
            suffix = (1,) * (order_one.ndim - 2)
            gate_two = second_gate.reshape(*second_gate.shape, *suffix)
            gate_three = third_gate.reshape(*third_gate.shape, *suffix)
            outputs.append(
                order_one + gate_two * order_two + gate_three * order_three
            )
        return tuple(outputs)

    def _global_message(
        self,
        state: _ParityState,
        normalized_pos: torch.Tensor,
        batch: torch.Tensor,
        graph_layout: PackedGraphLayout,
    ) -> tuple[torch.Tensor, ...]:
        """Apply one invariant relation operator at Krylov orders one to three.

        Query and key features are constructed once from every retained irrep
        sector. The same normalized implicit relation is then applied three
        times to the value bank. This yields ``R V``, ``R^2 V``, and ``R^3 V``
        without constructing ``R``, an attention matrix, or a pair state.
        """

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
        query_key_gates = torch.tanh(self.query_key_gates(scalar_state)).reshape(
            num_nodes,
            4,
            self.num_heads,
            1,
        )
        query_polar = _unit_ball(
            self.query_polar(state.polar_vector) * query_key_gates[:, 0],
            self.eps,
        )
        key_polar = _unit_ball(
            self.key_polar(state.polar_vector) * query_key_gates[:, 1],
            self.eps,
        )
        query_axial = _unit_ball(
            self.query_axial(state.axial_vector) * query_key_gates[:, 2],
            self.eps,
        )
        key_axial = _unit_ball(
            self.key_axial(state.axial_vector) * query_key_gates[:, 3],
            self.eps,
        )
        query_even_tensor = _st_orthonormal(
            _normalize_st(self.tensor_query_even(state.even_tensor), self.eps)
        )
        key_even_tensor = _st_orthonormal(
            _normalize_st(self.tensor_key_even(state.even_tensor), self.eps)
        )
        query_odd_tensor = _st_orthonormal(
            _normalize_st(self.tensor_query_odd(state.odd_tensor), self.eps)
        )
        key_odd_tensor = _st_orthonormal(
            _normalize_st(self.tensor_key_odd(state.odd_tensor), self.eps)
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
        radial_shell = torch.exp(
            -4.0 * (radius_squared[:, None] - centers[None, :]).square()
        )
        radial_scale = F.softplus(self.raw_global_radial_alignment).to(
            dtype=radius_squared.dtype
        )
        radial_feature = (
            radial_shell[:, None, :] * torch.sqrt(radial_scale)[None, :, :]
        )

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
                torch.sqrt(odd_tensor_scale)[None, :, None] * query_odd_tensor,
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
                torch.sqrt(even_tensor_scale)[None, :, None] * key_even_tensor,
                torch.sqrt(odd_tensor_scale)[None, :, None] * key_odd_tensor,
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

        geometry_gate = torch.tanh(self.global_geometry_gate(scalar_state)).reshape(
            num_nodes,
            2,
            self.num_heads,
        )
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

        orders: list[tuple[torch.Tensor, ...]] = []
        transported = value
        for _ in range(3):
            transported = _exact_balanced_attention(
                query_feature,
                key_feature,
                transported,
                batch,
                graph_layout,
                eps=self.eps,
            )
            orders.append(
                self._decode_transport(
                    transported,
                    position,
                    position_tensor,
                    head_dim=self.head_dim,
                )
            )

        order_gates = torch.tanh(self.global_krylov_gate(scalar_state)).reshape(
            num_nodes,
            2,
            self.num_heads,
        )
        return self._mix_orders(orders[0], orders[1], orders[2], order_gates)


class EdgeFreeELACore(_ELACore):
    """Canonical ELA core with graphwise moments and no implicit sparse edges."""

    def __init__(self, config: EquivariantLinearAttentionConfig) -> None:
        super().__init__(config)
        self.node_multipoles = _EdgeFreeRelativeMomentBank(
            rank=self.multipole_rank,
            eps=config.eps,
        )
        block_scale = config.residual_scale_init / sqrt(max(1, config.num_layers))
        self.blocks = nn.ModuleList(
            [
                EdgeFreeELALayer(
                    scalar_width=config.hidden_dim,
                    num_heads=config.num_heads,
                    local_rank=config.local_rank,
                    num_rbf=config.num_rbf,
                    num_edge_relations=config.num_edge_relations,
                    multipole_rank=self.multipole_rank,
                    residual_scale_init=block_scale,
                    condition_dim=config.condition_dim,
                    coordinate_updates=False,
                    residual_dropout=0.0,
                    drop_path_probability=0.0,
                    norm_eps=config.norm_eps,
                    eps=config.eps,
                )
                for _ in range(config.num_layers)
            ]
        )

    def prepare_context(
        self,
        positions: torch.Tensor,
        batch: torch.Tensor,
        graph_layout: PackedGraphLayout,
        neighbors: PackedNeighborGraph,
    ) -> _ELALayerContext:
        normalized_positions, radius, centered_norm = self._normalized_geometry(
            positions,
            batch,
            graph_layout,
        )
        geometry = self._build_geometry(positions, neighbors)
        multipoles = self.node_multipoles(
            normalized_positions,
            batch,
            num_graphs=graph_layout.num_graphs,
        )
        return _ELALayerContext(
            positions=positions,
            normalized_positions=normalized_positions,
            radius=radius,
            centered_norm=centered_norm,
            geometry=geometry,
            multipoles=multipoles,
            batch=batch,
            graph_layout=graph_layout,
        )


__all__ = ["EdgeFreeELACore", "EdgeFreeELALayer"]
