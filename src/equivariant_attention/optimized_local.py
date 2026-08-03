from __future__ import annotations

import torch

from .canonical_se3 import _safe_unit_direction
from .multipole_ops import _normalize_st
from .parity_se3 import (
    _ParityState,
    _StaticGeometry,
    _compute_dtype,
    _st_cross,
    _st_from_vector,
    _st_inner,
    _st_matvec,
)
from .triton_ops import (
    _trusted_csr_sum_many,
    _trusted_weighted_gather_reduce_pair,
    active_backend,
)


def triton_local_message(
    self: object,
    state: _ParityState,
    geometry: _StaticGeometry,
) -> tuple[torch.Tensor, ...]:
    """Canonical local operator with memory-bounded grouped CSR reductions.

    Projection and edge-score equations remain ordinary PyTorch autograd. The
    receiver statistics are reduced in five semantic groups instead of packing
    the full local operator into one giant ``[E,F]`` tensor. This trades a few
    launches for a lower peak edge-payload lifetime while preserving the
    canonical FP32/FP64 work-precision equations.
    """

    original = getattr(type(self), "_ela_torch_local_message", None)
    if original is None:
        raise RuntimeError("canonical PyTorch local reference was not installed")
    work_dtype = _compute_dtype(
        state.even_scalar,
        state.polar_vector,
        state.even_tensor,
        geometry.direction,
    )
    if (
        geometry.direction.shape[0] == 0
        or active_backend(
            geometry.direction,
            geometry.row_ptr,
            payload_dtype=work_dtype,
        )
        != "triton"
    ):
        return original(self, state, geometry)

    receiver = geometry.receiver
    sender = geometry.sender
    direction = geometry.direction
    rbf = geometry.rbf.to(dtype=state.even_scalar.dtype)

    scalar_state = self.scalar_norm(state.even_scalar)
    scalar_query = self.local_scalar_query(scalar_state).to(dtype=work_dtype)
    scalar_key = self.local_scalar_key(scalar_state).to(dtype=work_dtype)
    gates = torch.tanh(self.local_query_key_gates(scalar_state)).reshape(
        scalar_state.shape[0],
        4,
        self.local_rank,
        1,
    )
    polar_query = (self.local_polar_query(state.polar_vector) * gates[:, 0]).to(
        dtype=work_dtype
    )
    polar_key = (self.local_polar_key(state.polar_vector) * gates[:, 1]).to(
        dtype=work_dtype
    )
    axial_query = (self.local_axial_query(state.axial_vector) * gates[:, 2]).to(
        dtype=work_dtype
    )
    axial_key = (self.local_axial_key(state.axial_vector) * gates[:, 3]).to(
        dtype=work_dtype
    )

    edge_direction = direction[:, None, :].to(dtype=work_dtype)
    unit_direction = _safe_unit_direction(
        direction.to(dtype=work_dtype),
        geometry.squared_distance.to(dtype=work_dtype),
        self.eps,
    )[:, None, :]
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

    odd_query = self.local_odd_scalar_query(state.odd_scalar).to(dtype=work_dtype)
    odd_key = self.local_odd_scalar_key(state.odd_scalar).to(dtype=work_dtype)
    even_tensor_query = _normalize_st(
        self.local_even_tensor_query(state.even_tensor).to(dtype=work_dtype),
        self.eps,
    )
    even_tensor_key = _normalize_st(
        self.local_even_tensor_key(state.even_tensor).to(dtype=work_dtype),
        self.eps,
    )
    odd_tensor_query = _normalize_st(
        self.local_odd_tensor_query(state.odd_tensor).to(dtype=work_dtype),
        self.eps,
    )
    odd_tensor_key = _normalize_st(
        self.local_odd_tensor_key(state.odd_tensor).to(dtype=work_dtype),
        self.eps,
    )
    even_query_axis = (
        _st_matvec(even_tensor_query[receiver], unit_direction) * unit_direction
    ).sum(dim=-1)
    even_key_axis = (
        _st_matvec(even_tensor_key[sender], unit_direction) * unit_direction
    ).sum(dim=-1)
    odd_query_axis = (
        _st_matvec(odd_tensor_query[receiver], unit_direction) * unit_direction
    ).sum(dim=-1)
    odd_key_axis = (
        _st_matvec(odd_tensor_key[sender], unit_direction) * unit_direction
    ).sum(dim=-1)
    tensor_mix = self.local_tensor_mix.to(dtype=work_dtype)
    tensor_score = (
        self.local_tensor_radial_score(rbf).to(dtype=work_dtype)
        + tensor_mix[0][None, :] * odd_query[receiver] * odd_key[sender]
        + tensor_mix[1][None, :]
        * _st_inner(even_tensor_query[receiver], even_tensor_key[sender])
        + tensor_mix[2][None, :]
        * _st_inner(odd_tensor_query[receiver], odd_tensor_key[sender])
        + tensor_mix[3][None, :] * even_query_axis * even_key_axis
        + tensor_mix[4][None, :] * odd_query_axis * odd_key_axis
    )

    score = (
        scalar_query[receiver] * scalar_key[sender]
        + self.local_score_bias.to(dtype=work_dtype)[None, :]
        + self.local_radial_score(rbf).to(dtype=work_dtype)
        + self.local_polar_mix[0].to(dtype=work_dtype)[None, :] * polar_dot
        + self.local_polar_mix[1].to(dtype=work_dtype)[None, :]
        * receiver_polar_axis
        * sender_polar_axis
        + self.local_axial_mix[0].to(dtype=work_dtype)[None, :] * axial_dot
        + self.local_axial_mix[1].to(dtype=work_dtype)[None, :]
        * receiver_axial_axis
        * sender_axial_axis
        + tensor_score
    )
    if self.relation_score_bias is not None:
        if geometry.relation_id is None:
            raise ValueError("relation-aware ELA requires relation metadata")
        score = (
            score + self.relation_score_bias.to(dtype=work_dtype)[geometry.relation_id]
        )
    positive_gate = torch.exp(3.0 * torch.tanh(score / 3.0))
    raw_weight = geometry.cutoff.to(dtype=work_dtype)[:, None] * positive_gate

    # Group 1: normalization statistics stay FP32/FP64.
    mass, mass_square = _trusted_csr_sum_many(
        (raw_weight, raw_weight.square()),
        geometry.row_ptr,
        policy="triton",
        receiver=receiver,
    )
    denominator = 1.0 + mass

    num_nodes = state.even_scalar.shape[0]
    scalar_value = (
        self.local_scalar_value(state.even_scalar)
        .reshape(
            num_nodes,
            self.local_rank,
            self.head_dim,
        )
        .to(dtype=work_dtype)
    )
    odd_value = self.local_odd_value(state.odd_scalar).to(dtype=work_dtype)
    polar_value = self.local_polar_value(state.polar_vector).to(dtype=work_dtype)
    axial_value = self.local_axial_value(state.axial_vector).to(dtype=work_dtype)
    even_tensor_value = self.local_even_tensor_value(state.even_tensor).to(
        dtype=work_dtype
    )
    odd_tensor_value = self.local_odd_tensor_value(state.odd_tensor).to(
        dtype=work_dtype
    )
    direction_gate = (
        torch.tanh(self.local_direction_gate(scalar_state))
        .reshape(num_nodes, 3, self.local_rank)
        .to(dtype=work_dtype)
    )
    radial_gate = 2.0 * torch.sigmoid(
        self.local_radial_value(rbf).reshape(
            geometry.rbf.shape[0],
            self.local_rank,
            9,
        )
    ).to(dtype=work_dtype)
    weight = raw_weight
    direction_payload = edge_direction

    if torch.is_grad_enabled():
        # Training keeps the differentiable reference payload equations.
        scalar_rank, odd_rank = _trusted_csr_sum_many(
            (
                weight.unsqueeze(-1)
                * radial_gate[..., 0, None]
                * scalar_value[sender],
                weight * radial_gate[..., 1] * odd_value[sender],
            ),
            geometry.row_ptr,
            policy="triton",
            receiver=receiver,
        )
        polar_rank, axial_rank = _trusted_csr_sum_many(
            (
                weight.unsqueeze(-1)
                * radial_gate[..., 2, None]
                * polar_value[sender],
                weight.unsqueeze(-1)
                * radial_gate[..., 3, None]
                * axial_value[sender],
            ),
            geometry.row_ptr,
            policy="triton",
            receiver=receiver,
        )
    else:
        # Inference fuses sender gather, scalar weighting, and receiver sum so
        # the four largest edge payloads are never materialized.
        scalar_rank, odd_rank = _trusted_weighted_gather_reduce_pair(
            weight,
            radial_gate,
            scalar_value,
            odd_value,
            sender,
            geometry.row_ptr,
            gate_lanes=(0, 1),
            policy="triton",
        )
        polar_rank, axial_rank = _trusted_weighted_gather_reduce_pair(
            weight,
            radial_gate,
            polar_value,
            axial_value,
            sender,
            geometry.row_ptr,
            gate_lanes=(2, 3),
            policy="triton",
        )

    # Group 4: symmetric-traceless tensor values.
    direction_tensor = _st_from_vector(direction_payload)
    even_tensor_rank, odd_tensor_rank = _trusted_csr_sum_many(
        (
            weight.unsqueeze(-1)
            * radial_gate[..., 4, None]
            * (even_tensor_value[sender] + direction_tensor),
            weight.unsqueeze(-1) * radial_gate[..., 5, None] * odd_tensor_value[sender],
        ),
        geometry.row_ptr,
        policy="triton",
        receiver=receiver,
    )

    # Group 5: three directional moments used for chirality.
    first_direction, second_direction, third_direction = _trusted_csr_sum_many(
        tuple(
            weight.unsqueeze(-1)
            * radial_gate[..., 6 + index, None]
            * direction_gate[sender, index].unsqueeze(-1)
            * direction_payload
            for index in range(3)
        ),
        geometry.row_ptr,
        policy="triton",
        receiver=receiver,
    )

    # Normalize in work precision after the compact reductions.
    scalar_rank = scalar_rank.to(dtype=work_dtype) / denominator.unsqueeze(-1)
    odd_rank = odd_rank.to(dtype=work_dtype) / denominator
    polar_rank = polar_rank.to(dtype=work_dtype) / denominator.unsqueeze(-1)
    axial_rank = axial_rank.to(dtype=work_dtype) / denominator.unsqueeze(-1)
    even_tensor_rank = even_tensor_rank.to(dtype=work_dtype) / denominator.unsqueeze(-1)
    odd_tensor_rank = odd_tensor_rank.to(dtype=work_dtype) / denominator.unsqueeze(-1)
    first_direction = first_direction.to(dtype=work_dtype) / denominator.unsqueeze(-1)
    second_direction = second_direction.to(dtype=work_dtype) / denominator.unsqueeze(-1)
    third_direction = third_direction.to(dtype=work_dtype) / denominator.unsqueeze(-1)

    chiral_axial = torch.cross(first_direction, second_direction, dim=-1)
    chiral_scalar = (chiral_axial * third_direction).sum(dim=-1)
    chiral_tensor = _st_cross(third_direction, chiral_axial)

    scalar_message = self.local_scalar_out(scalar_rank)
    scalar_message = scalar_message + self.local_mass_out(
        torch.cat([torch.log1p(mass), torch.log1p(mass_square)], dim=-1).to(
            dtype=state.even_scalar.dtype
        )
    ).reshape(num_nodes, self.num_heads, self.head_dim).to(dtype=work_dtype)
    return (
        scalar_message,
        self.local_odd_out(odd_rank) + self.local_chiral_scalar_out(chiral_scalar),
        self.local_polar_out(polar_rank),
        self.local_axial_out(axial_rank) + self.local_chiral_axial_out(chiral_axial),
        self.local_even_tensor_out(even_tensor_rank),
        self.local_odd_tensor_out(odd_tensor_rank)
        + self.local_chiral_tensor_out(chiral_tensor),
        self.local_chiral_scalar_out(chiral_scalar),
    )


__all__ = ["triton_local_message"]
