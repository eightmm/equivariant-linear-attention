from __future__ import annotations

import torch
from torch import nn

from .local_geometry import PointwiseLocalFeatures
from .local_projection import LocalFeatureProjection
from .ops import (
    bounded_scalar,
    bounded_st,
    st_commutator_vector,
    st_cross,
    st_inner,
    st_jordan_product,
    st_matvec,
    stf3_contract_st,
    stf3_contract_vector,
    stf4_contract_st,
    stf4_contract_st_vector,
    unit_ball,
)
from .state import ChannelMix, ParityState, state_invariants


class LocalEquivariantClosure(nn.Module):
    """Close pointwise jets and low-rank body products into persistent irreps."""

    def __init__(
        self,
        *,
        scalar_width: int,
        num_heads: int,
        moment_rank: int,
        probe_rank: int,
        num_scales: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        body_rank = max(2, min(6, num_heads))
        self.project = LocalFeatureProjection(
            scalar_width=scalar_width,
            num_heads=num_heads,
            moment_rank=moment_rank,
            probe_rank=probe_rank,
            num_scales=num_scales,
            body_rank=body_rank,
        )
        self.even_invariant = nn.Linear(4 * num_heads, scalar_width, bias=False)
        self.odd_invariant = nn.Linear(2 * num_heads, num_heads, bias=False)
        self.polar_coupling = ChannelMix(num_heads, num_heads)
        self.axial_coupling = ChannelMix(num_heads, num_heads)
        self.even_coupling = ChannelMix(num_heads, num_heads)
        self.odd_coupling = ChannelMix(num_heads, num_heads)

        transient_rank = max(1, min(4, num_heads))
        self.third = ChannelMix(moment_rank, transient_rank)
        self.third_polar = ChannelMix(num_heads, transient_rank)
        self.third_axial = ChannelMix(num_heads, transient_rank)
        self.third_even = ChannelMix(num_heads, transient_rank)
        self.third_odd = ChannelMix(num_heads, transient_rank)
        self.third_polar_out = ChannelMix(transient_rank, num_heads)
        self.third_axial_out = ChannelMix(transient_rank, num_heads)
        self.third_even_out = ChannelMix(transient_rank, num_heads)
        self.third_odd_out = ChannelMix(transient_rank, num_heads)

        fourth_rank = max(1, min(3, num_heads))
        self.fourth = ChannelMix(moment_rank, fourth_rank)
        self.fourth_polar = ChannelMix(num_heads, fourth_rank)
        self.fourth_axial = ChannelMix(num_heads, fourth_rank)
        self.fourth_even = ChannelMix(num_heads, fourth_rank)
        self.fourth_odd = ChannelMix(num_heads, fourth_rank)
        self.fourth_polar_out = ChannelMix(fourth_rank, num_heads)
        self.fourth_axial_out = ChannelMix(fourth_rank, num_heads)
        self.fourth_even_out = ChannelMix(fourth_rank, num_heads)
        self.fourth_odd_out = ChannelMix(fourth_rank, num_heads)

        invariant_width = scalar_width + 5 * num_heads
        self.gates = nn.Linear(invariant_width, 5 * num_heads)
        nn.init.zeros_(self.gates.weight)
        nn.init.zeros_(self.gates.bias)
        self.num_heads = int(num_heads)

    def forward(
        self,
        state: ParityState,
        local: PointwiseLocalFeatures,
    ) -> ParityState:
        # Cumulants and local polynomial solves accumulate in FP32. Cast once
        # at the learned closure boundary, not inside their numerical cores.
        local = local.to_dtype(state.even_scalar.dtype)
        projected = self.project(local)
        # Autocast may return BF16 linear projections while the persistent
        # equivariant carrier remains FP32.  Geometric bilinear operations
        # (cross products and tensor products) require matching dtypes, so the
        # local projection rejoins the carrier dtype at this explicit boundary.
        base = ParityState(
            *(value.to(dtype=state.even_scalar.dtype) for value in projected.as_tuple())
        )
        pp = (state.polar_vector * base.polar_vector).sum(dim=-1)
        aa = (state.axial_vector * base.axial_vector).sum(dim=-1)
        ee = st_inner(state.even_tensor, base.even_tensor) / 5.0
        oo = st_inner(state.odd_tensor, base.odd_tensor) / 5.0
        pa = (state.polar_vector * base.axial_vector).sum(dim=-1)
        eo = st_inner(state.even_tensor, base.odd_tensor) / 5.0
        even_scalar = base.even_scalar + self.even_invariant(
            torch.cat((pp, aa, ee, oo), dim=-1)
        )
        odd_scalar = base.odd_scalar + self.odd_invariant(torch.cat((pa, eo), dim=-1))

        polar = base.polar_vector + self.polar_coupling(
            st_matvec(base.even_tensor, state.polar_vector)
            + st_matvec(base.odd_tensor, state.axial_vector)
        )
        axial = base.axial_vector + self.axial_coupling(
            torch.cross(state.polar_vector, base.polar_vector, dim=-1)
            + torch.cross(state.axial_vector, base.axial_vector, dim=-1)
            + st_commutator_vector(state.even_tensor, base.even_tensor)
            + st_commutator_vector(state.odd_tensor, base.odd_tensor)
        )
        even_tensor = base.even_tensor + self.even_coupling(
            st_cross(state.polar_vector, base.polar_vector)
            + st_cross(state.axial_vector, base.axial_vector)
            + st_jordan_product(state.even_tensor, base.even_tensor)
            + st_jordan_product(state.odd_tensor, base.odd_tensor)
        )
        odd_tensor = base.odd_tensor + self.odd_coupling(
            st_cross(state.polar_vector, base.axial_vector)
            + st_cross(state.axial_vector, base.polar_vector)
            + st_jordan_product(state.even_tensor, base.odd_tensor)
            + st_jordan_product(state.odd_tensor, base.even_tensor)
        )

        moments = local.moments
        third = self.third(moments.third_tensor)
        third_polar = self.third_polar(state.polar_vector)
        third_axial = self.third_axial(state.axial_vector)
        third_even = self.third_even(state.even_tensor)
        third_odd = self.third_odd(state.odd_tensor)
        polar = polar + self.third_polar_out(stf3_contract_st(third, third_even))
        axial = axial + self.third_axial_out(stf3_contract_st(third, third_odd))
        even_tensor = even_tensor + self.third_even_out(
            stf3_contract_vector(third, third_polar)
        )
        odd_tensor = odd_tensor + self.third_odd_out(
            stf3_contract_vector(third, third_axial)
        )

        fourth = self.fourth(moments.fourth_rank4)
        fourth_polar = self.fourth_polar(state.polar_vector)
        fourth_axial = self.fourth_axial(state.axial_vector)
        fourth_even = self.fourth_even(state.even_tensor)
        fourth_odd = self.fourth_odd(state.odd_tensor)
        even_fourth = stf4_contract_st(fourth, fourth_even)
        odd_fourth = stf4_contract_st(fourth, fourth_odd)
        even_tensor = even_tensor + self.fourth_even_out(even_fourth)
        odd_tensor = odd_tensor + self.fourth_odd_out(odd_fourth)
        polar = polar + self.fourth_polar_out(
            stf4_contract_st_vector(fourth, fourth_even, fourth_polar)
            + stf4_contract_st_vector(fourth, fourth_odd, fourth_axial)
        )
        axial = axial + self.fourth_axial_out(
            stf4_contract_st_vector(fourth, fourth_even, fourth_axial)
            + stf4_contract_st_vector(fourth, fourth_odd, fourth_polar)
        )

        gates = 2.0 * torch.sigmoid(
            self.gates(state_invariants(state, self.eps))
        ).reshape(state.num_nodes, 5, self.num_heads)
        return ParityState(
            bounded_scalar(even_scalar, self.eps),
            bounded_scalar(gates[:, 0] * odd_scalar, self.eps),
            unit_ball(gates[:, 1, :, None] * polar, self.eps),
            unit_ball(gates[:, 2, :, None] * axial, self.eps),
            bounded_st(gates[:, 3, :, None] * even_tensor, self.eps),
            bounded_st(gates[:, 4, :, None] * odd_tensor, self.eps),
        )


__all__ = ["LocalEquivariantClosure"]
