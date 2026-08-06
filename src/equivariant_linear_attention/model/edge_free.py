from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
import torch.nn.functional as F
from torch import nn

from ..geometry.layout import PackedGraphLayout
from ..geometry.neighbors import PackedNeighborGraph
from ..nn.layers import (
    _BranchModulation,
    _ELALayerContext,
    _gate_delta,
    _state_add,
)
from ..nn.multipoles import (
    _matrix_to_st,
    _normalize_st,
    _st_orthonormal,
    _st_to_matrix,
)
from ..nn.parity import (
    _ChannelMix,
    _ParityState,
    _StaticGeometry,
    _bounded_scalar,
    _bounded_st,
    _compute_dtype,
    _exact_balanced_attention,
    _positive_feature,
    _segment_sum,
    _st_cross,
    _st_from_vector,
    _st_inner,
    _unit_ball,
)
from .ela import ELALayer, _ELACore
from .stack import EquivariantLinearAttentionConfig


_Message = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


def _stf3(value: torch.Tensor) -> torch.Tensor:
    """Project one symmetric rank-three Cartesian tensor onto the l=3 irrep."""

    trace = torch.einsum("...aac->...c", value)
    identity = torch.eye(3, device=value.device, dtype=value.dtype)
    correction = (
        torch.einsum("ab,...c->...abc", identity, trace)
        + torch.einsum("ac,...b->...abc", identity, trace)
        + torch.einsum("bc,...a->...abc", identity, trace)
    ) / 5.0
    return value - correction


def _bounded_stf3(value: torch.Tensor, eps: float) -> torch.Tensor:
    square = value.square().sum(dim=(-3, -2, -1), keepdim=True) / 7.0
    return value / torch.sqrt(1.0 + square + eps)


@dataclass(frozen=True)
class _EdgeFreeMultipoles:
    """Parity-complete graph moments with one transient Cartesian l=3 carrier."""

    mass: torch.Tensor
    mass_square: torch.Tensor
    polar: torch.Tensor
    even_tensor: torch.Tensor
    axial: torch.Tensor
    odd_scalar: torch.Tensor
    odd_tensor: torch.Tensor
    third_tensor: torch.Tensor


class _EdgeFreeRelativeMomentBank(nn.Module):
    """Receiver-centred ``l<=3`` moments without pair materialization.

    For positive source weights ``w_j`` this module evaluates exact graphwise
    first, second, and third relative coordinate moments. The third moment is
    projected onto the symmetric trace-free rank-three Cartesian irrep and is
    transient: it enriches each layer's closure but is not kept in the
    persistent hidden carrier.
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
    ) -> _EdgeFreeMultipoles:
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

        # Exact third relative moment from graphwise raw moments. Self terms may
        # remain in S0...S3 because (x_i-x_i)^3 is exactly zero after the
        # receiver-centred binomial expansion.
        position_rank = position[:, None, :].expand(-1, self.rank, -1)
        position_square = torch.einsum(
            "nra,nrb->nrab",
            position_rank,
            position_rank,
        )
        position_cube = torch.einsum(
            "nra,nrb,nrc->nrabc",
            position_rank,
            position_rank,
            position_rank,
        )
        graph_second_full = _segment_sum(
            radial[..., None, None] * position_square,
            batch_index,
            num_graphs,
        )
        graph_third_full = _segment_sum(
            radial[..., None, None, None] * position_cube,
            batch_index,
            num_graphs,
        )
        source_mass = graph_mass[batch_index]
        source_first = graph_first[batch_index]
        source_second = graph_second_full[batch_index]
        source_third = graph_third_full[batch_index]

        first_shift = (
            torch.einsum("nra,nrbc->nrabc", position_rank, source_second)
            + torch.einsum("nrb,nrac->nrabc", position_rank, source_second)
            + torch.einsum("nrc,nrab->nrabc", position_rank, source_second)
        )
        second_shift = (
            torch.einsum(
                "nra,nrb,nrc->nrabc",
                position_rank,
                position_rank,
                source_first,
            )
            + torch.einsum(
                "nra,nrc,nrb->nrabc",
                position_rank,
                position_rank,
                source_first,
            )
            + torch.einsum(
                "nrb,nrc,nra->nrabc",
                position_rank,
                position_rank,
                source_first,
            )
        )
        relative_third = (
            source_third
            - first_shift
            + second_shift
            - source_mass[..., None, None, None] * position_cube
        )
        relative_third = relative_third / denominator[..., None, None, None]
        third_tensor = _bounded_stf3(_stf3(relative_third), self.eps)

        second = polar.roll(shifts=1, dims=1)
        third = polar.roll(shifts=2, dims=1)
        axial = torch.cross(polar, second, dim=-1)
        odd_scalar = (axial * third).sum(dim=-1)
        odd_tensor = _st_cross(third, axial)
        return _EdgeFreeMultipoles(
            mass=mass,
            mass_square=mass_square,
            polar=polar,
            even_tensor=even_tensor,
            axial=axial,
            odd_scalar=odd_scalar,
            odd_tensor=odd_tensor,
            third_tensor=third_tensor,
        )


def _message_node_inner(left: _Message, right: _Message) -> torch.Tensor:
    """Invariant per-node/head inner product across every retained sector."""

    return (
        (left[0] * right[0]).mean(dim=-1)
        + left[1] * right[1]
        + (left[2] * right[2]).mean(dim=-1)
        + (left[3] * right[3]).mean(dim=-1)
        + _st_inner(left[4], right[4]) / 5.0
        + _st_inner(left[5], right[5]) / 5.0
    ) / 6.0


def _message_scale(value: _Message, scale: torch.Tensor) -> _Message:
    output: list[torch.Tensor] = []
    for sector in value:
        suffix = (1,) * (sector.ndim - 2)
        output.append(sector * scale.reshape(*scale.shape, *suffix))
    return tuple(output)  # type: ignore[return-value]


def _message_subtract(left: _Message, right: _Message) -> _Message:
    return tuple(
        left_sector - right_sector
        for left_sector, right_sector in zip(left, right, strict=True)
    )  # type: ignore[return-value]


def _orthogonalize_message(
    candidate: _Message,
    bases: tuple[_Message, ...],
    *,
    batch: torch.Tensor,
    num_graphs: int,
    graph_counts: torch.Tensor,
    eps: float,
) -> _Message:
    """Graph/head-wise modified Gram-Schmidt using invariant irrep norms."""

    batch_index = batch.to(dtype=torch.long)
    output = candidate
    for basis in bases:
        numerator = _segment_sum(
            _message_node_inner(basis, output),
            batch_index,
            num_graphs,
        )
        denominator = _segment_sum(
            _message_node_inner(basis, basis),
            batch_index,
            num_graphs,
        ).clamp_min(eps)
        coefficient = (numerator / denominator)[batch_index]
        output = _message_subtract(output, _message_scale(basis, coefficient))

    norm = _segment_sum(
        _message_node_inner(output, output).clamp_min(0.0),
        batch_index,
        num_graphs,
    )
    norm = norm / graph_counts.to(dtype=norm.dtype).clamp_min(1.0).unsqueeze(-1)
    inverse = torch.rsqrt(norm + eps).clamp(max=8.0)
    return _message_scale(output, inverse[batch_index])


@dataclass(frozen=True)
class _AtlasFactors:
    assignment: torch.Tensor
    mass: torch.Tensor
    effective_dimension: torch.Tensor


class _LatentAtlasOperator(nn.Module):
    """Learn a low-rank soft incidence relation instead of explicit edges.

    Node-to-chart assignments ``A`` define the symmetric PSD operator

    ``S = A D_chart^{-1} A^T``.

    ``S`` is a learned soft edge relation of rank at most ``num_charts``. It is
    applied as node->chart->node contractions, never materialized as an
    ``N x N`` matrix, and its chart geometry is refined by a learned
    Mahalanobis metric estimated from equivariant chart covariances.
    """

    def __init__(
        self,
        *,
        scalar_width: int,
        num_charts: int,
        eps: float,
    ) -> None:
        super().__init__()
        if num_charts < 2:
            raise ValueError("latent atlas requires at least two charts")
        self.num_charts = int(num_charts)
        self.eps = float(eps)
        self.content_logits = nn.Linear(scalar_width, num_charts)
        self.radial_logits = nn.Linear(2, num_charts, bias=False)
        self.raw_metric_scale = nn.Parameter(torch.full((num_charts,), -1.5))

    def factor(
        self,
        scalar_state: torch.Tensor,
        normalized_positions: torch.Tensor,
        batch: torch.Tensor,
        *,
        num_graphs: int,
    ) -> _AtlasFactors:
        dtype = _compute_dtype(scalar_state, normalized_positions)
        position = normalized_positions.to(dtype=dtype)
        batch_index = batch.to(dtype=torch.long)
        radius_squared = position.square().sum(dim=-1)
        radial_input = torch.stack(
            [radius_squared, torch.log1p(radius_squared)],
            dim=-1,
        ).to(dtype=scalar_state.dtype)
        raw_logits = (
            self.content_logits(scalar_state) + self.radial_logits(radial_input)
        ).to(dtype=dtype)

        initial = torch.softmax(raw_logits, dim=-1)
        initial_mass = _segment_sum(initial, batch_index, num_graphs).clamp_min(
            self.eps
        )
        initial_center = _segment_sum(
            initial.unsqueeze(-1) * position.unsqueeze(1),
            batch_index,
            num_graphs,
        ) / initial_mass.unsqueeze(-1)
        initial_delta = position[:, None, :] - initial_center[batch_index]
        initial_covariance = _segment_sum(
            initial[..., None, None]
            * torch.einsum("nka,nkb->nkab", initial_delta, initial_delta),
            batch_index,
            num_graphs,
        ) / initial_mass[..., None, None]

        trace = initial_covariance.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        machine_eps = torch.finfo(dtype).eps
        jitter = (
            self.eps + 16.0 * machine_eps * trace.clamp_min(1.0)
        )[..., None, None]
        identity = torch.eye(3, device=position.device, dtype=dtype)
        metric = initial_covariance + jitter * identity
        solved = torch.linalg.solve(
            metric[batch_index],
            initial_delta.unsqueeze(-1),
        ).squeeze(-1)
        distance_square = (initial_delta * solved).sum(dim=-1).clamp(
            min=0.0,
            max=64.0,
        )
        metric_scale = F.softplus(self.raw_metric_scale).to(dtype=dtype)
        balance = torch.log(initial_mass[batch_index] + self.eps)
        assignment = torch.softmax(
            raw_logits
            - metric_scale.unsqueeze(0) * distance_square
            - 0.25 * balance,
            dim=-1,
        )

        mass = _segment_sum(assignment, batch_index, num_graphs).clamp_min(
            self.eps
        )
        center = _segment_sum(
            assignment.unsqueeze(-1) * position.unsqueeze(1),
            batch_index,
            num_graphs,
        ) / mass.unsqueeze(-1)
        delta = position[:, None, :] - center[batch_index]
        covariance = _segment_sum(
            assignment[..., None, None]
            * torch.einsum("nka,nkb->nkab", delta, delta),
            batch_index,
            num_graphs,
        ) / mass[..., None, None]
        covariance_trace = covariance.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        covariance_square = covariance.square().sum(dim=(-2, -1))
        effective_dimension = (
            covariance_trace.square() / (covariance_square + self.eps)
        ).clamp(min=1.0, max=3.0)
        return _AtlasFactors(
            assignment=assignment,
            mass=mass,
            effective_dimension=effective_dimension,
        )

    def apply(
        self,
        factors: _AtlasFactors,
        value: torch.Tensor,
        batch: torch.Tensor,
        *,
        num_graphs: int,
    ) -> torch.Tensor:
        assignment = factors.assignment.to(dtype=value.dtype)
        batch_index = batch.to(dtype=torch.long)
        chart_sum = _segment_sum(
            assignment[..., None, None] * value[:, None, :, :],
            batch_index,
            num_graphs,
        )
        chart_mean = chart_sum / factors.mass.to(
            dtype=value.dtype
        )[..., None, None]
        return (
            assignment[..., None, None] * chart_mean[batch_index]
        ).sum(dim=1)


class _TransientL3Closure(nn.Module):
    """Contract a transient STF rank-three carrier back into persistent l<=2."""

    def __init__(
        self,
        *,
        multipole_rank: int,
        num_heads: int,
        rank: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.third_mix = _ChannelMix(multipole_rank, rank)
        self.polar_mix = _ChannelMix(num_heads, rank)
        self.axial_mix = _ChannelMix(num_heads, rank)
        self.even_tensor_mix = _ChannelMix(num_heads, rank)
        self.odd_tensor_mix = _ChannelMix(num_heads, rank)
        self.polar_out = _ChannelMix(rank, num_heads, zero_init=True)
        self.axial_out = _ChannelMix(rank, num_heads, zero_init=True)
        self.even_tensor_out = _ChannelMix(rank, num_heads, zero_init=True)
        self.odd_tensor_out = _ChannelMix(rank, num_heads, zero_init=True)

    def forward(
        self,
        state: _ParityState,
        third_tensor: torch.Tensor,
    ) -> _ParityState:
        third = _bounded_stf3(self.third_mix(third_tensor), self.eps)
        polar = self.polar_mix(state.polar_vector)
        axial = self.axial_mix(state.axial_vector)
        even_tensor = _st_to_matrix(self.even_tensor_mix(state.even_tensor))
        odd_tensor = _st_to_matrix(self.odd_tensor_mix(state.odd_tensor))

        # 3o x 2e -> 1o and 3o x 2o -> 1e by double contraction.
        polar_update = torch.einsum(
            "nrabc,nrbc->nra",
            third,
            even_tensor,
        )
        axial_update = torch.einsum(
            "nrabc,nrbc->nra",
            third,
            odd_tensor,
        )

        # 3o x 1o -> 2e and 3o x 1e -> 2o by one-index contraction.
        even_matrix = torch.einsum(
            "nrabc,nrc->nrab",
            third,
            polar,
        )
        odd_matrix = torch.einsum(
            "nrabc,nrc->nrab",
            third,
            axial,
        )
        zeros_even = state.even_scalar.new_zeros(state.even_scalar.shape)
        zeros_odd = state.odd_scalar.new_zeros(state.odd_scalar.shape)
        return _ParityState(
            even_scalar=zeros_even,
            odd_scalar=zeros_odd,
            polar_vector=self.polar_out(polar_update),
            axial_vector=self.axial_out(axial_update),
            even_tensor=self.even_tensor_out(_matrix_to_st(even_matrix)),
            odd_tensor=self.odd_tensor_out(_matrix_to_st(odd_matrix)),
        )


class EdgeFreeELALayer(ELALayer):
    """ELA layer with spectral Krylov closure and learned latent soft edges."""

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

        atlas_rank = max(2, min(8, num_heads))
        self.latent_atlas = _LatentAtlasOperator(
            scalar_width=scalar_width,
            num_charts=atlas_rank,
            eps=max(eps, 1e-8),
        )
        self.latent_edge_gate = nn.Linear(scalar_width, num_heads)
        nn.init.zeros_(self.latent_edge_gate.weight)
        nn.init.zeros_(self.latent_edge_gate.bias)

        third_rank = max(1, min(4, num_heads // 4))
        self.third_moment_closure = _TransientL3Closure(
            multipole_rank=multipole_rank,
            num_heads=num_heads,
            rank=third_rank,
            eps=eps,
        )
        self.third_moment_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init))
        )

    def _local_message(
        self,
        state: _ParityState,
        geometry: _StaticGeometry,
    ) -> tuple[torch.Tensor, ...]:
        # Explicit user topology remains a compatibility residual. The default
        # graph contains no inferred radius edges.
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
    ) -> _Message:
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
    def _mix_relations(
        first: _Message,
        second: _Message,
        third: _Message,
        atlas: _Message,
        order_gates: torch.Tensor,
        atlas_gate: torch.Tensor,
    ) -> _Message:
        second_gate = order_gates[:, 0]
        third_gate = order_gates[:, 1]
        outputs: list[torch.Tensor] = []
        for order_one, order_two, order_three, latent in zip(
            first,
            second,
            third,
            atlas,
            strict=True,
        ):
            suffix = (1,) * (order_one.ndim - 2)
            gate_two = second_gate.reshape(*second_gate.shape, *suffix)
            gate_three = third_gate.reshape(*third_gate.shape, *suffix)
            gate_atlas = atlas_gate.reshape(*atlas_gate.shape, *suffix)
            outputs.append(
                order_one
                + gate_two * order_two
                + gate_three * order_three
                + gate_atlas * latent
            )
        return tuple(outputs)  # type: ignore[return-value]

    def _global_message(
        self,
        state: _ParityState,
        normalized_pos: torch.Tensor,
        batch: torch.Tensor,
        graph_layout: PackedGraphLayout,
    ) -> _Message:
        """Build an orthogonal Krylov basis plus a learned latent-edge basis."""

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

        atlas_factors = self.latent_atlas.factor(
            scalar_state,
            normalized_pos,
            batch,
            num_graphs=graph_layout.num_graphs,
        )
        atlas_transport = self.latent_atlas.apply(
            atlas_factors,
            value,
            batch,
            num_graphs=graph_layout.num_graphs,
        )
        atlas_message = self._decode_transport(
            atlas_transport,
            position,
            position_tensor,
            head_dim=self.head_dim,
        )

        orders: list[_Message] = []
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

        orthogonal_eps = max(self.eps, 1e-8)
        first_basis = orders[0]
        second_basis = _orthogonalize_message(
            orders[1],
            (first_basis,),
            batch=batch,
            num_graphs=graph_layout.num_graphs,
            graph_counts=graph_layout.graph_counts,
            eps=orthogonal_eps,
        )
        third_basis = _orthogonalize_message(
            orders[2],
            (first_basis, second_basis),
            batch=batch,
            num_graphs=graph_layout.num_graphs,
            graph_counts=graph_layout.graph_counts,
            eps=orthogonal_eps,
        )
        atlas_basis = _orthogonalize_message(
            atlas_message,
            (first_basis, second_basis, third_basis),
            batch=batch,
            num_graphs=graph_layout.num_graphs,
            graph_counts=graph_layout.graph_counts,
            eps=orthogonal_eps,
        )

        order_gates = torch.tanh(self.global_krylov_gate(scalar_state)).reshape(
            num_nodes,
            2,
            self.num_heads,
        )
        atlas_gate = torch.tanh(self.latent_edge_gate(scalar_state))
        return self._mix_relations(
            first_basis,
            second_basis,
            third_basis,
            atlas_basis,
            order_gates,
            atlas_gate,
        )

    def _attention_branch(
        self,
        state: _ParityState,
        context: _ELALayerContext,
        modulation: _BranchModulation | None,
    ) -> _ParityState:
        updated = super()._attention_branch(state, context, modulation)
        third_tensor = getattr(context.multipoles, "third_tensor", None)
        if third_tensor is None:
            return updated

        reference = self.closure_norm(updated)
        third_delta = self.third_moment_closure(reference, third_tensor)
        scale = self.third_moment_scale.to(dtype=reference.even_scalar.dtype)
        third_delta = _ParityState(
            even_scalar=third_delta.even_scalar,
            odd_scalar=third_delta.odd_scalar,
            polar_vector=scale * _unit_ball(third_delta.polar_vector, self.eps),
            axial_vector=scale * _unit_ball(third_delta.axial_vector, self.eps),
            even_tensor=scale * _bounded_st(third_delta.even_tensor, self.eps),
            odd_tensor=scale * _bounded_st(third_delta.odd_tensor, self.eps),
        )
        if modulation is not None:
            third_delta = _gate_delta(third_delta, modulation)
        third_delta = self._regularize(
            third_delta,
            reference,
            activation=self.attention_activation,
            dropout=self.attention_dropout,
            drop_path=self.attention_drop_path,
            context=context,
        )
        return _state_add(updated, third_delta)


class EdgeFreeELACore(_ELACore):
    """Edge-free ELA core with l=3 moments and latent manifold relations."""

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
            multipoles=multipoles,  # type: ignore[arg-type]
            batch=batch,
            graph_layout=graph_layout,
        )


__all__ = [
    "EdgeFreeELACore",
    "EdgeFreeELALayer",
]
