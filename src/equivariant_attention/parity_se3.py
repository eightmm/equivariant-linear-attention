from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
import torch.nn.functional as F
from torch import nn

from .graph_layout import PackedGraphLayout
from .irreps import Irrep, IrrepLayout, split_irreps
from .neighbors import PackedNeighborGraph


def _compute_dtype(*values: torch.Tensor) -> torch.dtype:
    return (
        torch.float64
        if any(value.dtype == torch.float64 for value in values)
        else torch.float32
    )


def _segment_sum(
    value: torch.Tensor,
    index: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    output = value.new_zeros((num_segments, *value.shape[1:]))
    return output.index_add(0, index, value)


class _CsrSum(torch.autograd.Function):
    """CSR sum with an explicit backward that supports higher derivatives."""

    @staticmethod
    def forward(
        ctx: object,
        value: torch.Tensor,
        row_ptr: torch.Tensor,
    ) -> torch.Tensor:
        result = torch.segment_reduce(value, reduce="sum", offsets=row_ptr)
        ctx.save_for_backward(value, result, row_ptr)
        return result

    @staticmethod
    def backward(
        ctx: object,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        value, result, row_ptr = ctx.saved_tensors
        if torch.is_grad_enabled():
            counts = (row_ptr[1:] - row_ptr[:-1]).to(dtype=torch.long)
            grad_value = torch.repeat_interleave(
                grad_output,
                counts,
                dim=0,
                output_size=value.shape[0],
            )
        else:
            # The native kernel keeps ordinary training fast, while the
            # differentiable branch above supplies PyTorch's missing gradgrad.
            grad_value = torch.ops.aten._segment_reduce_backward.default(
                grad_output,
                result,
                value,
                "sum",
                offsets=row_ptr,
            )
        return grad_value, None


def _csr_sum(value: torch.Tensor, row_ptr: torch.Tensor) -> torch.Tensor:
    return _CsrSum.apply(value, row_ptr)


def _unit_ball(value: torch.Tensor, eps: float) -> torch.Tensor:
    square = value.square().sum(dim=-1, keepdim=True)
    return value / torch.sqrt(1.0 + square + eps)


def _bounded_scalar(value: torch.Tensor, eps: float) -> torch.Tensor:
    return value / torch.sqrt(1.0 + value.square() + eps)


def _st_from_vector(value: torch.Tensor) -> torch.Tensor:
    x, y, z = value.unbind(dim=-1)
    trace_third = (x.square() + y.square() + z.square()) / 3.0
    return torch.stack(
        [
            x.square() - trace_third,
            y.square() - trace_third,
            x * y,
            x * z,
            y * z,
        ],
        dim=-1,
    )


def _st_cross(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lx, ly, lz = left.unbind(dim=-1)
    rx, ry, rz = right.unbind(dim=-1)
    trace_third = (lx * rx + ly * ry + lz * rz) / 3.0
    return torch.stack(
        [
            lx * rx - trace_third,
            ly * ry - trace_third,
            0.5 * (lx * ry + ly * rx),
            0.5 * (lx * rz + lz * rx),
            0.5 * (ly * rz + lz * ry),
        ],
        dim=-1,
    )


def _st_to_matrix(value: torch.Tensor) -> torch.Tensor:
    xx, yy, xy, xz, yz = value.unbind(dim=-1)
    zz = -xx - yy
    return torch.stack(
        [
            torch.stack([xx, xy, xz], dim=-1),
            torch.stack([xy, yy, yz], dim=-1),
            torch.stack([xz, yz, zz], dim=-1),
        ],
        dim=-2,
    )


def _st_matvec(tensor: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...ab,...b->...a", _st_to_matrix(tensor), vector)


def _st_inner(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (
        left[..., 0] * right[..., 0]
        + left[..., 1] * right[..., 1]
        + (-left[..., 0] - left[..., 1])
        * (-right[..., 0] - right[..., 1])
        + 2.0
        * (
            left[..., 2] * right[..., 2]
            + left[..., 3] * right[..., 3]
            + left[..., 4] * right[..., 4]
        )
    )


def _st_square(value: torch.Tensor) -> torch.Tensor:
    return _st_inner(value, value)


def _bounded_st(value: torch.Tensor, eps: float) -> torch.Tensor:
    return value / torch.sqrt(1.0 + _st_square(value).unsqueeze(-1) / 5.0 + eps)


def _positive_feature(value: torch.Tensor) -> torch.Tensor:
    return (F.elu(value) + 1.0) / sqrt(max(1, value.shape[-1]))


def _radial_basis(
    squared_distance: torch.Tensor,
    centers: torch.Tensor,
    width: torch.Tensor,
) -> torch.Tensor:
    delta = squared_distance.unsqueeze(-1) - centers
    return torch.exp(-width * delta.square())


def _smooth_cutoff(squared_ratio: torch.Tensor) -> torch.Tensor:
    inside = squared_ratio < 1.0
    return torch.where(
        inside,
        0.5 * (1.0 + torch.cos(torch.pi * squared_ratio)),
        torch.zeros_like(squared_ratio),
    )


class _ChannelMix(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        zero_init: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        if zero_init:
            nn.init.zeros_(self.weight)
        else:
            nn.init.normal_(self.weight, std=1.0 / sqrt(max(1, in_channels)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.out_channels == 0:
            return value.new_zeros((value.shape[0], 0, *value.shape[2:]))
        return torch.einsum(
            "oc,nc...->no...",
            self.weight.to(dtype=value.dtype),
            value,
        )


@dataclass
class _ParityState:
    even_scalar: torch.Tensor
    odd_scalar: torch.Tensor
    polar_vector: torch.Tensor
    axial_vector: torch.Tensor
    even_tensor: torch.Tensor
    odd_tensor: torch.Tensor


class _InputProjection(nn.Module):
    """Same-irrep channel mixing from one flattened l<=2 input carrier."""

    def __init__(
        self,
        layout: IrrepLayout,
        *,
        scalar_width: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.layout = layout
        projectors: dict[str, nn.Module] = {}
        for block in layout.blocks:
            name = str(block.irrep)
            width = scalar_width if name == "0e" else num_heads
            if block.irrep.degree == 0:
                projectors[name] = nn.Linear(
                    block.multiplicity,
                    width,
                    bias=name == "0e",
                )
            else:
                projectors[name] = _ChannelMix(block.multiplicity, width)
        self.projectors = nn.ModuleDict(projectors)
        self.scalar_width = scalar_width
        self.num_heads = num_heads

    def forward(self, value: torch.Tensor) -> _ParityState:
        blocks = split_irreps(self.layout, value)
        num_nodes = value.shape[0]

        def project_scalar(name: str, width: int) -> torch.Tensor:
            if name not in self.projectors:
                return value.new_zeros((num_nodes, width))
            return self.projectors[name](blocks[name].squeeze(-1))

        def project_geometric(name: str, degree_dim: int) -> torch.Tensor:
            if name not in self.projectors:
                return value.new_zeros((num_nodes, self.num_heads, degree_dim))
            return self.projectors[name](blocks[name])

        return _ParityState(
            even_scalar=project_scalar("0e", self.scalar_width),
            odd_scalar=project_scalar("0o", self.num_heads),
            polar_vector=project_geometric("1o", 3),
            axial_vector=project_geometric("1e", 3),
            even_tensor=project_geometric("2e", 5),
            odd_tensor=project_geometric("2o", 5),
        )


@dataclass(frozen=True)
class _StaticGeometry:
    receiver: torch.Tensor
    sender: torch.Tensor
    direction: torch.Tensor
    squared_distance: torch.Tensor
    rbf: torch.Tensor
    cutoff: torch.Tensor
    row_ptr: torch.Tensor
    relation_id: torch.Tensor | None = None


def _layout_feature_gemm(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layout: PackedGraphLayout,
) -> torch.Tensor:
    if layout.num_graphs == 1:
        summary = torch.einsum("nhf,nhv->hfv", key, value)
        return torch.einsum("nhf,hfv->nhv", query, summary)

    if layout.dense_index is not None and layout.dense_mask is not None:
        query_dense = layout.gather_dense(query)
        key_dense = layout.gather_dense(key)
        value_dense = layout.gather_dense(value)
        mask = layout.dense_mask[..., None, None]
        key_dense = key_dense * mask.to(dtype=key_dense.dtype)
        value_dense = value_dense * mask.to(dtype=value_dense.dtype)
        summary = torch.einsum("gmhf,gmhv->ghfv", key_dense, value_dense)
        output_dense = torch.einsum("gmhf,ghfv->gmhv", query_dense, summary)
        grouped_output = output_dense[layout.dense_mask]
        return layout.ungroup_nodes(grouped_output)

    if layout.buckets:
        grouped_query = layout.group_nodes(query)
        grouped_key = layout.group_nodes(key)
        grouped_value = layout.group_nodes(value)
        grouped_output = value.new_zeros(value.shape)
        for bucket in layout.buckets:
            query_dense = bucket.gather(grouped_query)
            key_dense = bucket.gather(grouped_key)
            value_dense = bucket.gather(grouped_value)
            mask = bucket.mask[..., None, None]
            key_dense = key_dense * mask.to(dtype=key_dense.dtype)
            value_dense = value_dense * mask.to(dtype=value_dense.dtype)
            summary = torch.einsum("bmhf,bmhv->bhfv", key_dense, value_dense)
            output_dense = torch.einsum("bmhf,bhfv->bmhv", query_dense, summary)
            valid_index = bucket.node_index[bucket.mask].to(dtype=torch.long)
            grouped_output.index_copy_(0, valid_index, output_dense[bucket.mask])
        return layout.ungroup_nodes(grouped_output)

    grouped_query = layout.group_nodes(query)
    grouped_key = layout.group_nodes(key)
    grouped_value = layout.group_nodes(value)
    outputs: list[torch.Tensor] = []
    for start, count in layout.graph_spans:
        graph_query = grouped_query.narrow(0, start, count)
        graph_key = grouped_key.narrow(0, start, count)
        graph_value = grouped_value.narrow(0, start, count)
        summary = torch.einsum("nhf,nhv->hfv", graph_key, graph_value)
        outputs.append(torch.einsum("nhf,hfv->nhv", graph_query, summary))
    grouped_output = torch.cat(outputs, dim=0)
    return layout.ungroup_nodes(grouped_output)


def _exact_balanced_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    batch: torch.Tensor,
    layout: PackedGraphLayout,
    *,
    eps: float,
) -> torch.Tensor:
    dtype = _compute_dtype(query, key, value)
    query = query.to(dtype=dtype)
    key = key.to(dtype=dtype)
    value = value.to(dtype=dtype)

    query_sum = _segment_sum(query, batch, layout.num_graphs)
    key_mass = (key * query_sum[batch]).sum(dim=-1).clamp_min(eps)
    balanced_key = key / key_mass.unsqueeze(-1)

    augmented = torch.cat(
        [value, value.new_ones((*value.shape[:-1], 1))],
        dim=-1,
    )
    transported = _layout_feature_gemm(query, balanced_key, augmented, layout)
    numerator = transported[..., :-1]
    denominator = transported[..., -1:].clamp_min(eps)
    return numerator / denominator


class _ParityCompleteBlock(nn.Module):
    def __init__(
        self,
        *,
        scalar_width: int,
        num_heads: int,
        local_rank: int,
        num_rbf: int,
        num_edge_relations: int,
        residual_scale_init: float,
        eps: float,
    ) -> None:
        super().__init__()
        if scalar_width % num_heads:
            raise ValueError("scalar_width must be divisible by num_heads")
        self.scalar_width = scalar_width
        self.num_heads = num_heads
        self.local_rank = local_rank
        self.head_dim = scalar_width // num_heads
        self.num_edge_relations = num_edge_relations
        self.eps = eps

        self.scalar_norm = nn.LayerNorm(scalar_width)
        self.query_scalar = nn.Linear(scalar_width, scalar_width)
        self.key_scalar = nn.Linear(scalar_width, scalar_width)
        self.query_polar = _ChannelMix(num_heads, num_heads)
        self.key_polar = _ChannelMix(num_heads, num_heads)
        self.query_axial = _ChannelMix(num_heads, num_heads)
        self.key_axial = _ChannelMix(num_heads, num_heads)
        self.query_key_gates = nn.Linear(scalar_width, 4 * num_heads)
        self.raw_alignment_scale = nn.Parameter(
            torch.full((2, num_heads), -2.0)
        )

        self.global_scalar_value = nn.Linear(scalar_width, scalar_width)
        self.global_odd_value = _ChannelMix(num_heads, num_heads)
        self.global_polar_value = _ChannelMix(num_heads, num_heads)
        self.global_axial_value = _ChannelMix(num_heads, num_heads)
        self.global_even_tensor_value = _ChannelMix(num_heads, num_heads)
        self.global_odd_tensor_value = _ChannelMix(num_heads, num_heads)
        self.global_geometry_gate = nn.Linear(scalar_width, 2 * num_heads)

        self.local_scalar_query = nn.Linear(scalar_width, local_rank)
        self.local_scalar_key = nn.Linear(scalar_width, local_rank)
        self.local_polar_query = _ChannelMix(num_heads, local_rank)
        self.local_polar_key = _ChannelMix(num_heads, local_rank)
        self.local_axial_query = _ChannelMix(num_heads, local_rank)
        self.local_axial_key = _ChannelMix(num_heads, local_rank)
        self.local_query_key_gates = nn.Linear(scalar_width, 4 * local_rank)
        self.local_radial_score = nn.Linear(num_rbf, local_rank, bias=False)
        self.local_score_bias = nn.Parameter(torch.zeros(local_rank))
        self.local_polar_mix = nn.Parameter(torch.full((2, local_rank), 0.1))
        self.local_axial_mix = nn.Parameter(torch.full((2, local_rank), 0.1))
        self.relation_score_bias = (
            nn.Parameter(torch.zeros(num_edge_relations, local_rank))
            if num_edge_relations
            else None
        )

        self.local_scalar_value = nn.Linear(
            scalar_width,
            local_rank * self.head_dim,
        )
        self.local_odd_value = _ChannelMix(num_heads, local_rank)
        self.local_polar_value = _ChannelMix(num_heads, local_rank)
        self.local_axial_value = _ChannelMix(num_heads, local_rank)
        self.local_even_tensor_value = _ChannelMix(num_heads, local_rank)
        self.local_odd_tensor_value = _ChannelMix(num_heads, local_rank)
        self.local_direction_gate = nn.Linear(scalar_width, 3 * local_rank)
        self.local_radial_value = nn.Linear(
            num_rbf,
            9 * local_rank,
            bias=False,
        )
        nn.init.zeros_(self.local_radial_value.weight)

        self.local_scalar_out = _ChannelMix(
            local_rank,
            num_heads,
            zero_init=True,
        )
        self.local_odd_out = _ChannelMix(
            local_rank,
            num_heads,
            zero_init=True,
        )
        self.local_polar_out = _ChannelMix(
            local_rank,
            num_heads,
            zero_init=True,
        )
        self.local_axial_out = _ChannelMix(
            local_rank,
            num_heads,
            zero_init=True,
        )
        self.local_even_tensor_out = _ChannelMix(
            local_rank,
            num_heads,
            zero_init=True,
        )
        self.local_odd_tensor_out = _ChannelMix(
            local_rank,
            num_heads,
            zero_init=True,
        )
        self.local_chiral_scalar_out = _ChannelMix(
            local_rank,
            num_heads,
            zero_init=True,
        )
        self.local_chiral_axial_out = _ChannelMix(
            local_rank,
            num_heads,
            zero_init=True,
        )
        self.local_chiral_tensor_out = _ChannelMix(
            local_rank,
            num_heads,
            zero_init=True,
        )
        self.local_mass_out = nn.Linear(
            2 * local_rank,
            scalar_width,
            bias=False,
        )
        nn.init.zeros_(self.local_mass_out.weight)

        invariant_width = scalar_width + 6 * num_heads
        self.even_update_norm = nn.LayerNorm(invariant_width)
        self.even_update = nn.Sequential(
            nn.Linear(invariant_width, 2 * scalar_width),
            nn.SiLU(),
            nn.Linear(2 * scalar_width, scalar_width),
        )
        self.odd_coefficients = nn.Linear(scalar_width, 4 * num_heads)
        self.odd_update = _ChannelMix(num_heads, num_heads)
        self.vector_coefficients = nn.Linear(scalar_width, 8 * num_heads)
        self.polar_update = _ChannelMix(num_heads, num_heads)
        self.axial_update = _ChannelMix(num_heads, num_heads)
        self.tensor_coefficients = nn.Linear(scalar_width, 8 * num_heads)
        self.even_tensor_update = _ChannelMix(num_heads, num_heads)
        self.odd_tensor_update = _ChannelMix(num_heads, num_heads)

        ffn_width = scalar_width + 5 * num_heads
        self.ffn_norm = nn.LayerNorm(scalar_width)
        self.ffn_even = nn.Sequential(
            nn.Linear(ffn_width, 2 * scalar_width),
            nn.SiLU(),
            nn.Linear(2 * scalar_width, scalar_width),
        )
        self.ffn_gates = nn.Linear(scalar_width, 5 * num_heads)
        self.ffn_odd = _ChannelMix(num_heads, num_heads)
        self.ffn_polar = _ChannelMix(num_heads, num_heads)
        self.ffn_axial = _ChannelMix(num_heads, num_heads)
        self.ffn_even_tensor = _ChannelMix(num_heads, num_heads)
        self.ffn_odd_tensor = _ChannelMix(num_heads, num_heads)
        self.ffn_polar_cross = _ChannelMix(num_heads, num_heads)
        self.ffn_axial_cross = _ChannelMix(num_heads, num_heads)
        self.ffn_even_tensor_cross = _ChannelMix(num_heads, num_heads)
        self.ffn_odd_tensor_cross = _ChannelMix(num_heads, num_heads)

        scale = float(residual_scale_init)
        self.scalar_scale = nn.Parameter(torch.tensor(scale))
        self.odd_scale = nn.Parameter(torch.tensor(scale))
        self.polar_scale = nn.Parameter(torch.tensor(scale))
        self.axial_scale = nn.Parameter(torch.tensor(scale))
        self.even_tensor_scale = nn.Parameter(torch.tensor(scale))
        self.odd_tensor_scale = nn.Parameter(torch.tensor(scale))
        self.ffn_scale = nn.Parameter(torch.tensor(scale))

    def forward(
        self,
        state: _ParityState,
        normalized_pos: torch.Tensor,
        geometry: _StaticGeometry,
        batch: torch.Tensor,
        graph_layout: PackedGraphLayout,
    ) -> _ParityState:
        global_message = self._global_message(
            state,
            normalized_pos,
            batch,
            graph_layout,
        )
        local_message = self._local_message(state, geometry)
        updated = self._update_state(state, global_message, local_message)
        return self._ffn(updated)

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
        alignment = F.softplus(self.raw_alignment_scale)
        polar_scale = alignment[0]
        axial_scale = alignment[1]
        constant = torch.sqrt(
            1.0 + polar_scale + axial_scale
        )[None, :, None]
        query_feature = torch.cat(
            [
                query_scalar,
                constant.expand(num_nodes, -1, -1),
                torch.sqrt(polar_scale)[None, :, None] * query_polar,
                torch.sqrt(axial_scale)[None, :, None] * query_axial,
            ],
            dim=-1,
        )
        key_feature = torch.cat(
            [
                key_scalar,
                constant.expand(num_nodes, -1, -1),
                torch.sqrt(polar_scale)[None, :, None] * key_polar,
                torch.sqrt(axial_scale)[None, :, None] * key_axial,
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
        relative_vector = (
            vector_first - vector_gate_mass * position
        )
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
        receiver = geometry.receiver
        sender = geometry.sender
        direction = geometry.direction
        rbf = geometry.rbf.to(dtype=state.even_scalar.dtype)
        dtype = _compute_dtype(
            state.even_scalar,
            state.polar_vector,
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
        receiver_polar = polar_query[receiver]
        sender_polar = polar_key[sender]
        receiver_axial = axial_query[receiver]
        sender_axial = axial_key[sender]
        polar_dot = (receiver_polar * sender_polar).sum(dim=-1)
        axial_dot = (receiver_axial * sender_axial).sum(dim=-1)
        receiver_polar_axis = (receiver_polar * edge_direction).sum(dim=-1)
        sender_polar_axis = (sender_polar * edge_direction).sum(dim=-1)
        receiver_axial_axis = (receiver_axial * edge_direction).sum(dim=-1)
        sender_axial_axis = (sender_axial * edge_direction).sum(dim=-1)
        score = (
            scalar_query[receiver] * scalar_key[sender]
            + self.local_score_bias.to(dtype=dtype)[None, :]
            + self.local_radial_score(rbf).to(dtype=dtype)
            + self.local_polar_mix[0].to(dtype=dtype)[None, :] * polar_dot
            + self.local_polar_mix[1].to(dtype=dtype)[None, :]
            * receiver_polar_axis
            * sender_polar_axis
            + self.local_axial_mix[0].to(dtype=dtype)[None, :] * axial_dot
            + self.local_axial_mix[1].to(dtype=dtype)[None, :]
            * receiver_axial_axis
            * sender_axial_axis
        )
        if self.relation_score_bias is not None:
            if geometry.relation_id is None:
                raise ValueError(
                    "relation-aware unified model requires relation metadata"
                )
            score = score + self.relation_score_bias.to(dtype=dtype)[
                geometry.relation_id
            ]
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
        chiral_scalar = (chiral_axial * third_direction).sum(dim=-1)
        chiral_tensor = _st_cross(third_direction, chiral_axial)

        scalar_message = self.local_scalar_out(scalar_rank)
        scalar_message = scalar_message + self.local_mass_out(
            torch.cat([torch.log1p(mass), torch.log1p(mass_square)], dim=-1)
            .to(dtype=state.even_scalar.dtype)
        ).reshape(num_nodes, self.num_heads, self.head_dim).to(dtype=dtype)
        return (
            scalar_message,
            self.local_odd_out(odd_rank)
            + self.local_chiral_scalar_out(chiral_scalar),
            self.local_polar_out(polar_rank),
            self.local_axial_out(axial_rank)
            + self.local_chiral_axial_out(chiral_axial),
            self.local_even_tensor_out(even_tensor_rank),
            self.local_odd_tensor_out(odd_tensor_rank)
            + self.local_chiral_tensor_out(chiral_tensor),
            self.local_chiral_scalar_out(chiral_scalar),
        )

    def _update_state(
        self,
        state: _ParityState,
        global_message: tuple[torch.Tensor, ...],
        local_message: tuple[torch.Tensor, ...],
    ) -> _ParityState:
        (
            global_scalar,
            global_odd,
            global_polar,
            global_axial,
            global_even_tensor,
            global_odd_tensor,
        ) = global_message
        (
            local_scalar,
            local_odd,
            local_polar,
            local_axial,
            local_even_tensor,
            local_odd_tensor,
            chiral_scalar,
        ) = local_message
        dtype = _compute_dtype(
            global_scalar,
            local_scalar,
            state.polar_vector,
        )
        scalar_message = (global_scalar + local_scalar).reshape(
            state.even_scalar.shape[0],
            self.scalar_width,
        )
        odd_message = global_odd + local_odd
        polar_message = global_polar + local_polar
        axial_message = global_axial + local_axial
        even_tensor_message = global_even_tensor + local_even_tensor
        odd_tensor_message = global_odd_tensor + local_odd_tensor

        even_invariants = torch.cat(
            [
                scalar_message.to(dtype=dtype),
                odd_message.square(),
                polar_message.square().sum(dim=-1),
                axial_message.square().sum(dim=-1),
                _st_square(even_tensor_message),
                _st_square(odd_tensor_message),
                chiral_scalar.square(),
            ],
            dim=-1,
        )
        even_delta = self.even_update(
            self.even_update_norm(
                even_invariants.to(dtype=state.even_scalar.dtype)
            )
        ).to(dtype=dtype)

        odd_basis = torch.stack(
            [
                odd_message,
                (polar_message * axial_message).sum(dim=-1),
                _st_inner(even_tensor_message, odd_tensor_message),
                chiral_scalar,
            ],
            dim=-1,
        )
        odd_coefficients = torch.tanh(
            self.odd_coefficients(self.scalar_norm(state.even_scalar))
        ).reshape(
            state.even_scalar.shape[0],
            self.num_heads,
            4,
        ).to(dtype=dtype)
        odd_delta = self.odd_update(
            (odd_coefficients * odd_basis).sum(dim=-1)
        )

        vector_coefficients = torch.tanh(
            self.vector_coefficients(self.scalar_norm(state.even_scalar))
        ).reshape(
            state.even_scalar.shape[0],
            self.num_heads,
            8,
        ).to(dtype=dtype)
        polar_basis = torch.stack(
            [
                polar_message,
                odd_message.unsqueeze(-1) * axial_message,
                _st_matvec(even_tensor_message, polar_message),
                _st_matvec(odd_tensor_message, axial_message),
            ],
            dim=-2,
        )
        axial_basis = torch.stack(
            [
                axial_message,
                odd_message.unsqueeze(-1) * polar_message,
                _st_matvec(even_tensor_message, axial_message),
                _st_matvec(odd_tensor_message, polar_message),
            ],
            dim=-2,
        )
        polar_delta = self.polar_update(
            (
                vector_coefficients[..., :4].unsqueeze(-1)
                * polar_basis
            ).sum(dim=-2)
        )
        axial_delta = self.axial_update(
            (
                vector_coefficients[..., 4:].unsqueeze(-1)
                * axial_basis
            ).sum(dim=-2)
        )

        tensor_coefficients = torch.tanh(
            self.tensor_coefficients(self.scalar_norm(state.even_scalar))
        ).reshape(
            state.even_scalar.shape[0],
            self.num_heads,
            8,
        ).to(dtype=dtype)
        even_tensor_basis = torch.stack(
            [
                even_tensor_message,
                _st_from_vector(polar_message),
                _st_from_vector(axial_message),
                odd_message.unsqueeze(-1) * odd_tensor_message,
            ],
            dim=-2,
        )
        odd_tensor_basis = torch.stack(
            [
                odd_tensor_message,
                _st_cross(polar_message, axial_message),
                odd_message.unsqueeze(-1) * even_tensor_message,
                _st_cross(
                    _st_matvec(even_tensor_message, polar_message),
                    axial_message,
                ),
            ],
            dim=-2,
        )
        even_tensor_delta = self.even_tensor_update(
            (
                tensor_coefficients[..., :4].unsqueeze(-1)
                * even_tensor_basis
            ).sum(dim=-2)
        )
        odd_tensor_delta = self.odd_tensor_update(
            (
                tensor_coefficients[..., 4:].unsqueeze(-1)
                * odd_tensor_basis
            ).sum(dim=-2)
        )

        return _ParityState(
            even_scalar=state.even_scalar
            + (
                self.scalar_scale.to(dtype=dtype) * even_delta
            ).to(dtype=state.even_scalar.dtype),
            odd_scalar=state.odd_scalar
            + (
                self.odd_scale.to(dtype=dtype)
                * _bounded_scalar(odd_delta, self.eps)
            ).to(dtype=state.odd_scalar.dtype),
            polar_vector=state.polar_vector
            + (
                self.polar_scale.to(dtype=dtype)
                * _unit_ball(polar_delta, self.eps)
            ).to(dtype=state.polar_vector.dtype),
            axial_vector=state.axial_vector
            + (
                self.axial_scale.to(dtype=dtype)
                * _unit_ball(axial_delta, self.eps)
            ).to(dtype=state.axial_vector.dtype),
            even_tensor=state.even_tensor
            + (
                self.even_tensor_scale.to(dtype=dtype)
                * _bounded_st(even_tensor_delta, self.eps)
            ).to(dtype=state.even_tensor.dtype),
            odd_tensor=state.odd_tensor
            + (
                self.odd_tensor_scale.to(dtype=dtype)
                * _bounded_st(odd_tensor_delta, self.eps)
            ).to(dtype=state.odd_tensor.dtype),
        )

    def _ffn(self, state: _ParityState) -> _ParityState:
        scalar_state = self.ffn_norm(state.even_scalar)
        invariants = torch.cat(
            [
                scalar_state,
                state.odd_scalar.square(),
                state.polar_vector.square().sum(dim=-1),
                state.axial_vector.square().sum(dim=-1),
                _st_square(state.even_tensor),
                _st_square(state.odd_tensor),
            ],
            dim=-1,
        )
        even_delta = self.ffn_even(invariants)
        gates = torch.tanh(self.ffn_gates(scalar_state)).reshape(
            scalar_state.shape[0],
            self.num_heads,
            5,
        )
        odd_delta = gates[..., 0] * self.ffn_odd(state.odd_scalar)
        polar_delta = gates[..., 1, None] * (
            self.ffn_polar(state.polar_vector)
            + state.odd_scalar.unsqueeze(-1)
            * self.ffn_polar_cross(state.axial_vector)
        )
        axial_delta = gates[..., 2, None] * (
            self.ffn_axial(state.axial_vector)
            + state.odd_scalar.unsqueeze(-1)
            * self.ffn_axial_cross(state.polar_vector)
        )
        even_tensor_delta = gates[..., 3, None] * (
            self.ffn_even_tensor(state.even_tensor)
            + state.odd_scalar.unsqueeze(-1)
            * self.ffn_even_tensor_cross(state.odd_tensor)
        )
        odd_tensor_delta = gates[..., 4, None] * (
            self.ffn_odd_tensor(state.odd_tensor)
            + state.odd_scalar.unsqueeze(-1)
            * self.ffn_odd_tensor_cross(state.even_tensor)
        )
        scale = self.ffn_scale
        return _ParityState(
            even_scalar=state.even_scalar
            + (scale * even_delta).to(dtype=state.even_scalar.dtype),
            odd_scalar=state.odd_scalar
            + (
                scale * _bounded_scalar(odd_delta, self.eps)
            ).to(dtype=state.odd_scalar.dtype),
            polar_vector=state.polar_vector
            + (
                scale * _unit_ball(polar_delta, self.eps)
            ).to(dtype=state.polar_vector.dtype),
            axial_vector=state.axial_vector
            + (
                scale * _unit_ball(axial_delta, self.eps)
            ).to(dtype=state.axial_vector.dtype),
            even_tensor=state.even_tensor
            + (
                scale * _bounded_st(even_tensor_delta, self.eps)
            ).to(dtype=state.even_tensor.dtype),
            odd_tensor=state.odd_tensor
            + (
                scale * _bounded_st(odd_tensor_delta, self.eps)
            ).to(dtype=state.odd_tensor.dtype),
        )


class _OutputProjection(nn.Module):
    def __init__(
        self,
        output_irreps: str | IrrepLayout,
        *,
        scalar_width: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.layout = IrrepLayout.parse(output_irreps)
        if not self.layout.blocks:
            raise ValueError("output_irreps must not be empty")
        unsupported = [
            block.irrep
            for block in self.layout.blocks
            if block.irrep.degree > 2
        ]
        if unsupported:
            raise ValueError(
                "unified parity-complete core supports output degrees l<=2; "
                f"unsupported={unsupported}"
            )
        self.scalar_paths = nn.ModuleDict()
        self.tensor_paths = nn.ModuleDict()
        for block in self.layout.blocks:
            key = str(block.irrep)
            source_channels = (
                scalar_width if block.irrep.degree == 0 and block.irrep.parity == "e"
                else num_heads
            )
            if block.irrep.degree == 0:
                self.scalar_paths[key] = nn.Linear(
                    source_channels,
                    block.multiplicity,
                    bias=block.irrep.parity == "e",
                )
            else:
                self.tensor_paths[key] = _ChannelMix(
                    source_channels,
                    block.multiplicity,
                )

    def forward(self, state: _ParityState) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        for block in self.layout.blocks:
            irrep = block.irrep
            key = str(irrep)
            if irrep == Irrep(0, "e"):
                value = self.scalar_paths[key](state.even_scalar).unsqueeze(-1)
            elif irrep == Irrep(0, "o"):
                value = self.scalar_paths[key](state.odd_scalar).unsqueeze(-1)
            elif irrep == Irrep(1, "o"):
                value = self.tensor_paths[key](state.polar_vector)
            elif irrep == Irrep(1, "e"):
                value = self.tensor_paths[key](state.axial_vector)
            elif irrep == Irrep(2, "e"):
                value = self.tensor_paths[key](state.even_tensor)
            elif irrep == Irrep(2, "o"):
                value = self.tensor_paths[key](state.odd_tensor)
            else:
                raise RuntimeError(f"unhandled output irrep {irrep}")
            outputs.append(value.flatten(start_dim=-2))
        return torch.cat(outputs, dim=-1)


class ParityCompleteSE3Core(nn.Module):
    """Parity-complete, chirality-capable linear-complexity 3D core.

    The persistent internal carrier is fixed automatically to
    ``0e + 0o + 1o + 1e + 2e + 2o``.  Every block evaluates the same exact
    graph-global finite-feature transport and the same positive receiver-CSR
    local residual.  Users choose only the final output irreps; internal parity,
    angular degree, and cross-sector couplings are construction-time constants.
    """

    symmetry = "SE3"
    internal_symmetry = "O3_parity_complete"
    attention_kind = "parity_complete_factorized_moment"

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
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if relation_cutoffs and len(relation_cutoffs) != num_edge_relations:
            raise ValueError(
                "relation_cutoffs length must equal num_edge_relations"
            )
        if any(
            value <= 0.0 or value > float(local_cutoff)
            for value in relation_cutoffs
        ):
            raise ValueError(
                "relation cutoffs must be positive and no larger than local_cutoff"
            )
        self.input_irreps = IrrepLayout.parse(input_irreps)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.local_rank = local_rank
        self.local_cutoff = float(local_cutoff)
        self.num_rbf = num_rbf
        self.num_node_roles = num_node_roles
        self.num_edge_relations = num_edge_relations
        self.relation_cutoffs = tuple(float(value) for value in relation_cutoffs)
        self.eps = float(eps)

        self.internal_irreps = IrrepLayout.parse(
            f"{hidden_dim}x0e + {num_heads}x0o + "
            f"{num_heads}x1o + {num_heads}x1e + "
            f"{num_heads}x2e + {num_heads}x2o"
        )
        self.output_projection = _OutputProjection(
            output_irreps,
            scalar_width=hidden_dim,
            num_heads=num_heads,
        )
        self.output_irreps = self.output_projection.layout

        self.input_projection = _InputProjection(
            self.input_irreps,
            scalar_width=hidden_dim,
            num_heads=num_heads,
        )
        self.role_embedding = (
            nn.Embedding(num_node_roles, hidden_dim)
            if num_node_roles
            else None
        )
        self.initial_vector_gate = nn.Linear(hidden_dim, num_heads)
        self.initial_tensor_gate = nn.Linear(hidden_dim, num_heads)

        block_scale = residual_scale_init / sqrt(max(1, num_layers))
        self.blocks = nn.ModuleList(
            [
                _ParityCompleteBlock(
                    scalar_width=hidden_dim,
                    num_heads=num_heads,
                    local_rank=local_rank,
                    num_rbf=num_rbf,
                    num_edge_relations=num_edge_relations,
                    residual_scale_init=block_scale,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )
        centers = torch.linspace(0.0, 1.0, num_rbf)
        spacing = 1.0 / max(1, num_rbf - 1)
        width = torch.tensor(1.0 / max(spacing * spacing, 1e-6))
        self.register_buffer("_rbf_centers", centers, persistent=False)
        self.register_buffer("_rbf_width", width, persistent=False)

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
        normalized_pos = self._normalize_positions(pos, batch, graph_layout)
        geometry = self._build_geometry(pos, neighbors)

        external = self.input_projection(node_irreps)
        even_scalar = external.even_scalar
        if self.role_embedding is not None:
            if node_role_id is None:
                raise RuntimeError("validated role IDs are missing")
            even_scalar = even_scalar + self.role_embedding(
                node_role_id.to(dtype=torch.long)
            )
        polar = torch.tanh(
            self.initial_vector_gate(even_scalar)
        ).unsqueeze(-1) * normalized_pos[:, None, :].to(
            dtype=even_scalar.dtype
        )
        polar = polar + external.polar_vector.to(dtype=polar.dtype)
        even_tensor = torch.tanh(
            self.initial_tensor_gate(even_scalar)
        ).unsqueeze(-1) * _st_from_vector(
            normalized_pos[:, None, :].expand(-1, self.num_heads, -1)
        ).to(dtype=even_scalar.dtype)
        even_tensor = even_tensor + external.even_tensor.to(
            dtype=even_tensor.dtype
        )

        state = _ParityState(
            even_scalar=even_scalar,
            odd_scalar=external.odd_scalar.to(dtype=even_scalar.dtype),
            polar_vector=polar,
            axial_vector=external.axial_vector.to(dtype=polar.dtype),
            even_tensor=even_tensor,
            odd_tensor=external.odd_tensor.to(dtype=even_tensor.dtype),
        )
        for block in self.blocks:
            state = block(
                state,
                normalized_pos,
                geometry,
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

    def split_output(
        self,
        value: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if value.shape[-1] != self.output_irreps.dim:
            raise ValueError(
                f"value final dimension must be {self.output_irreps.dim}"
            )
        return {
            str(block.irrep): value[..., self.output_irreps.slice_for(block.irrep)]
            .reshape(
                *value.shape[:-1],
                block.multiplicity,
                block.irrep.dim,
            )
            for block in self.output_irreps.blocks
        }

    def _normalize_positions(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor,
        layout: PackedGraphLayout,
    ) -> torch.Tensor:
        dtype = _compute_dtype(pos)
        position = pos.to(dtype=dtype)
        counts = layout.graph_counts.to(dtype=dtype).clamp_min(1.0)
        center = _segment_sum(position, batch, layout.num_graphs) / counts[:, None]
        centered = position - center[batch]
        square = centered.square().sum(dim=-1)
        radius = torch.sqrt(
            _segment_sum(square, batch, layout.num_graphs) / counts
            + self.eps
        )
        return centered / radius[batch, None].clamp_min(self.eps)

    def _build_geometry(
        self,
        pos: torch.Tensor,
        neighbors: PackedNeighborGraph,
    ) -> _StaticGeometry:
        receiver = neighbors.receiver_index().to(dtype=torch.long)
        sender = neighbors.sender.to(dtype=torch.long)
        dtype = _compute_dtype(pos)
        cutoff = pos.new_tensor(self.local_cutoff, dtype=dtype)
        direction = (
            pos.to(dtype=dtype)[sender] - pos.to(dtype=dtype)[receiver]
        ) / cutoff
        squared_distance = direction.square().sum(dim=-1)
        cutoff_argument = squared_distance
        relation_id = None
        if self.num_edge_relations:
            if neighbors.relation_id is None:
                raise ValueError(
                    "relation-aware unified model requires relation IDs"
                )
            relation_id = neighbors.relation_id.to(dtype=torch.long)
            relation_cutoff = torch.tensor(
                self.relation_cutoffs,
                dtype=dtype,
                device=pos.device,
            )[relation_id]
            cutoff_argument = squared_distance * (
                cutoff / relation_cutoff
            ).square()
        smooth = _smooth_cutoff(cutoff_argument)
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

    def _validate_inputs(
        self,
        node_irreps: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        graph_layout: PackedGraphLayout,
        neighbors: PackedNeighborGraph,
        *,
        node_role_id: torch.Tensor | None,
    ) -> None:
        if (
            node_irreps.ndim != 2
            or node_irreps.shape[1] != self.input_irreps.dim
        ):
            raise ValueError(
                f"node_irreps must have shape (N, {self.input_irreps.dim})"
            )
        if pos.shape != (node_irreps.shape[0], 3):
            raise ValueError("pos must have shape (N, 3)")
        if batch.shape != (node_irreps.shape[0],):
            raise ValueError("batch must have shape (N,)")
        if batch.dtype != torch.long:
            raise TypeError("batch must use torch.long")
        if any(
            value.device != node_irreps.device
            for value in (pos, batch, graph_layout.batch, neighbors.row_ptr)
        ):
            raise ValueError("all inputs and prepared metadata must share one device")
        graph_layout.validate_batch(batch)
        if neighbors.num_nodes != node_irreps.shape[0]:
            raise ValueError("neighbor node count must match input nodes")
        self._assert_finite("node_irreps", node_irreps)
        self._assert_finite("pos", pos)
        if self.num_edge_relations and neighbors.relation_id is not None:
            relation_id = neighbors.relation_id
            valid_relations = (
                (relation_id >= 0)
                & (relation_id < self.num_edge_relations)
            ).all()
            async_assert = getattr(torch, "_assert_async", None)
            if relation_id.device.type == "cuda" and async_assert is not None:
                async_assert(
                    valid_relations,
                    "edge relation IDs are outside the configured range",
                )
            elif not bool(valid_relations):
                raise ValueError(
                    "edge relation IDs are outside the configured range"
                )

        if self.num_node_roles:
            if node_role_id is None or node_role_id.shape != (node_irreps.shape[0],):
                raise ValueError(
                    "node_role_id with shape (N,) is required"
                )
        elif node_role_id is not None:
            raise ValueError("node_role_id requires positive num_node_roles")

    @staticmethod
    def _assert_finite(name: str, value: torch.Tensor) -> None:
        finite = torch.isfinite(value).all()
        async_assert = getattr(torch, "_assert_async", None)
        if value.device.type == "cuda" and async_assert is not None:
            async_assert(finite, f"{name} must be finite")
        elif not bool(finite):
            raise ValueError(f"{name} must be finite")


__all__ = ["ParityCompleteSE3Core"]
