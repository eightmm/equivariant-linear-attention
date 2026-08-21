"""Parity-complete retained closure with compact transient l=3/l=4 tensors."""

from __future__ import annotations

import torch
from torch import nn

from .geometry import MomentFeatures
from .ops import (
    bounded_scalar,
    bounded_st,
    matrix_to_st,
    st_cross,
    st_inner,
    st_to_matrix,
    stf3_contract_st,
    stf3_contract_vector,
    stf4_contract_st,
    stf4_contract_st_vector,
    unit_ball,
)
from .relation import RelationMessage
from .state import ChannelMix, ParityState


def _commutator_vector(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    commutator = left @ right - right @ left
    return torch.stack(
        (commutator[..., 2, 1], commutator[..., 0, 2], commutator[..., 1, 0]),
        dim=-1,
    )


def _jordan_st(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return matrix_to_st(0.5 * (left @ right + right @ left))


def _vector_tensor_from_matrix(
    vector: torch.Tensor,
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    l1 = torch.einsum("...ab,...b->...a", tensor, vector)
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    cross = torch.stack(
        (
            torch.stack((zero, -z, y), dim=-1),
            torch.stack((z, zero, -x), dim=-1),
            torch.stack((-y, x, zero), dim=-1),
        ),
        dim=-2,
    )
    return l1, matrix_to_st(cross @ tensor - tensor @ cross)


class EquivariantClosure(nn.Module):
    """All retained l<=2 couplings plus compact transient l=3/l=4 paths."""

    def __init__(
        self,
        *,
        scalar_width: int,
        num_heads: int,
        head_dim: int,
        moment_rank: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.scalar_width = scalar_width
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.moment_rank = moment_rank
        self.eps = float(eps)

        self.message_scalar = nn.Linear(num_heads * head_dim, scalar_width)
        self.message_odd = ChannelMix(num_heads, num_heads)
        self.message_polar = ChannelMix(num_heads, num_heads)
        self.message_axial = ChannelMix(num_heads, num_heads)
        self.message_even_tensor = ChannelMix(num_heads, num_heads)
        self.message_odd_tensor = ChannelMix(num_heads, num_heads)

        self.moment_scalar = nn.Linear(3 * moment_rank, scalar_width)
        self.moment_odd = ChannelMix(moment_rank, num_heads)
        self.moment_polar = ChannelMix(moment_rank, num_heads)
        # Preserve the previous initialized function while allowing the missing
        # Sym^3 trace lane to open immediately through its weight gradient.
        self.moment_third_polar = ChannelMix(moment_rank, num_heads, zero_init=True)
        self.moment_axial = ChannelMix(moment_rank, num_heads)
        self.moment_even_tensor = ChannelMix(moment_rank, num_heads)
        self.moment_odd_tensor = ChannelMix(moment_rank, num_heads)
        self.moment_fourth_tensor = ChannelMix(moment_rank, num_heads)

        self.even_invariant = nn.Linear(4 * num_heads, scalar_width)
        self.odd_invariant = nn.Linear(2 * num_heads, num_heads, bias=False)
        self.polar_mix = ChannelMix(num_heads, num_heads)
        self.axial_mix = ChannelMix(num_heads, num_heads)
        self.even_tensor_mix = ChannelMix(num_heads, num_heads)
        self.odd_tensor_mix = ChannelMix(num_heads, num_heads)

        transient_rank = max(1, min(4, num_heads))
        self.third_mix = ChannelMix(moment_rank, transient_rank)
        self.third_polar = ChannelMix(num_heads, transient_rank)
        self.third_axial = ChannelMix(num_heads, transient_rank)
        self.third_even_tensor = ChannelMix(num_heads, transient_rank)
        self.third_odd_tensor = ChannelMix(num_heads, transient_rank)
        self.third_polar_out = ChannelMix(transient_rank, num_heads)
        self.third_axial_out = ChannelMix(transient_rank, num_heads)
        self.third_even_out = ChannelMix(transient_rank, num_heads)
        self.third_odd_out = ChannelMix(transient_rank, num_heads)

        fourth_rank = max(1, min(3, num_heads))
        self.fourth_mix = ChannelMix(moment_rank, fourth_rank)
        self.fourth_polar = ChannelMix(num_heads, fourth_rank)
        self.fourth_axial = ChannelMix(num_heads, fourth_rank)
        self.fourth_even_tensor = ChannelMix(num_heads, fourth_rank)
        self.fourth_odd_tensor = ChannelMix(num_heads, fourth_rank)
        self.fourth_polar_out = ChannelMix(fourth_rank, num_heads)
        self.fourth_axial_out = ChannelMix(fourth_rank, num_heads)
        self.fourth_even_out = ChannelMix(fourth_rank, num_heads)
        self.fourth_odd_out = ChannelMix(fourth_rank, num_heads)

    def forward(
        self,
        state: ParityState,
        message: RelationMessage,
        moments: MomentFeatures,
    ) -> ParityState:
        # Moment reductions intentionally accumulate in FP32 under mixed
        # precision. They rejoin the learned equivariant carrier only here.
        moments = moments.to_dtype(state.even_scalar.dtype)
        scalar_message = message.scalar.flatten(1)
        pp = (state.polar_vector * message.polar_vector).sum(dim=-1)
        aa = (state.axial_vector * message.axial_vector).sum(dim=-1)
        ee = st_inner(state.even_tensor, message.even_tensor)
        oo = st_inner(state.odd_tensor, message.odd_tensor)
        pa = (state.polar_vector * message.axial_vector).sum(dim=-1)
        eo = st_inner(state.even_tensor, message.odd_tensor)

        even_scalar = (
            self.message_scalar(scalar_message)
            + self.moment_scalar(
                torch.cat(
                    (moments.mass, moments.second_scalar, moments.fourth_scalar),
                    dim=-1,
                )
            )
            + self.even_invariant(torch.cat((pp, aa, ee / 5.0, oo / 5.0), dim=-1))
        )
        odd_scalar = (
            self.message_odd(message.odd_scalar)
            + self.moment_odd(moments.odd_scalar)
            + self.odd_invariant(torch.cat((pa, eo / 5.0), dim=-1))
        )

        # Convert each ST carrier to a matrix once, then reuse it across every
        # retained Clebsch-Gordan path in this closure.
        state_even_matrix = st_to_matrix(state.even_tensor)
        state_odd_matrix = st_to_matrix(state.odd_tensor)
        message_even_matrix = st_to_matrix(message.even_tensor)
        message_odd_matrix = st_to_matrix(message.odd_tensor)

        polar_l1_a, odd_l2_a = _vector_tensor_from_matrix(
            state.polar_vector,
            message_even_matrix,
        )
        polar_l1_b, odd_l2_b = _vector_tensor_from_matrix(
            state.axial_vector,
            message_odd_matrix,
        )
        axial_l1_a, even_l2_a = _vector_tensor_from_matrix(
            state.axial_vector,
            message_even_matrix,
        )
        axial_l1_b, even_l2_b = _vector_tensor_from_matrix(
            state.polar_vector,
            message_odd_matrix,
        )

        polar = (
            self.message_polar(message.polar_vector)
            + self.moment_polar(moments.polar)
            + self.moment_third_polar(moments.third_trace)
            + self.polar_mix(
                torch.cross(state.polar_vector, message.axial_vector, dim=-1)
                + _commutator_vector(state_even_matrix, message_odd_matrix)
                + polar_l1_a
                + polar_l1_b
            )
        )
        axial = (
            self.message_axial(message.axial_vector)
            + self.moment_axial(moments.axial)
            + self.axial_mix(
                torch.cross(state.polar_vector, message.polar_vector, dim=-1)
                + torch.cross(state.axial_vector, message.axial_vector, dim=-1)
                + _commutator_vector(state_even_matrix, message_even_matrix)
                + _commutator_vector(state_odd_matrix, message_odd_matrix)
                + axial_l1_a
                + axial_l1_b
            )
        )
        even_tensor = (
            self.message_even_tensor(message.even_tensor)
            + self.moment_even_tensor(moments.even_tensor)
            + self.moment_fourth_tensor(moments.fourth_tensor)
            + self.even_tensor_mix(
                st_cross(state.polar_vector, message.polar_vector)
                + st_cross(state.axial_vector, message.axial_vector)
                + _jordan_st(state_even_matrix, message_even_matrix)
                + _jordan_st(state_odd_matrix, message_odd_matrix)
                + even_l2_a
                + even_l2_b
            )
        )
        odd_tensor = (
            self.message_odd_tensor(message.odd_tensor)
            + self.moment_odd_tensor(moments.odd_tensor)
            + self.odd_tensor_mix(
                st_cross(state.polar_vector, message.axial_vector)
                + st_cross(state.axial_vector, message.polar_vector)
                + _jordan_st(state_even_matrix, message_odd_matrix)
                + _jordan_st(state_odd_matrix, message_even_matrix)
                + odd_l2_a
                + odd_l2_b
            )
        )

        third = self.third_mix(moments.third_tensor)
        third_polar = self.third_polar(state.polar_vector)
        third_axial = self.third_axial(state.axial_vector)
        third_even_tensor = self.third_even_tensor(state.even_tensor)
        third_odd_tensor = self.third_odd_tensor(state.odd_tensor)
        polar = polar + self.third_polar_out(stf3_contract_st(third, third_even_tensor))
        axial = axial + self.third_axial_out(stf3_contract_st(third, third_odd_tensor))
        even_tensor = even_tensor + self.third_even_out(
            stf3_contract_vector(third, third_polar)
        )
        odd_tensor = odd_tensor + self.third_odd_out(
            stf3_contract_vector(third, third_axial)
        )

        fourth = self.fourth_mix(moments.fourth_rank4)
        fourth_polar = self.fourth_polar(state.polar_vector)
        fourth_axial = self.fourth_axial(state.axial_vector)
        fourth_even_tensor = self.fourth_even_tensor(state.even_tensor)
        fourth_odd_tensor = self.fourth_odd_tensor(state.odd_tensor)
        even_fourth = stf4_contract_st(fourth, fourth_even_tensor)
        odd_fourth = stf4_contract_st(fourth, fourth_odd_tensor)
        even_tensor = even_tensor + self.fourth_even_out(even_fourth)
        odd_tensor = odd_tensor + self.fourth_odd_out(odd_fourth)
        polar = polar + self.fourth_polar_out(
            stf4_contract_st_vector(fourth, fourth_even_tensor, fourth_polar)
            + stf4_contract_st_vector(fourth, fourth_odd_tensor, fourth_axial)
        )
        axial = axial + self.fourth_axial_out(
            stf4_contract_st_vector(fourth, fourth_even_tensor, fourth_axial)
            + stf4_contract_st_vector(fourth, fourth_odd_tensor, fourth_polar)
        )

        return ParityState(
            bounded_scalar(even_scalar, self.eps),
            bounded_scalar(odd_scalar, self.eps),
            unit_ball(polar, self.eps),
            unit_ball(axial, self.eps),
            bounded_st(even_tensor, self.eps),
            bounded_st(odd_tensor, self.eps),
        )


__all__ = ["EquivariantClosure"]
