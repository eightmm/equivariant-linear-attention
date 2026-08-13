"""Correctly dimensioned projection of local features into parity sectors."""

from __future__ import annotations

import torch
from torch import nn

from .local_body import LocalBodyAlgebra
from .local_geometry import PointwiseLocalFeatures
from .ops import bounded_scalar, bounded_st, unit_ball
from .state import ChannelMix, ParityState


def _flatten_invariant(value: torch.Tensor) -> torch.Tensor:
    return value.flatten(1)


def _merge_geometric(value: torch.Tensor) -> torch.Tensor:
    return value.flatten(1, value.ndim - 2)


class LocalFeatureProjection(nn.Module):
    """Map cumulants, local jets, wavelets, and body products to irreps."""

    def __init__(
        self,
        *,
        scalar_width: int,
        num_heads: int,
        moment_rank: int,
        num_scales: int,
        probe_rank: int,
        body_rank: int,
        eps: float,
    ) -> None:
        super().__init__()
        jet_channels = num_scales * probe_rank
        wave_channels = (num_scales - 1) * probe_rank
        scalar_dim = (
            3 * moment_rank
            + 2 * jet_channels
            + 2 * wave_channels
            + 2 * num_scales
            + 1
        )
        polar_channels = (
            moment_rank + jet_channels + wave_channels + body_rank
        )
        even_tensor_channels = (
            2 * moment_rank + jet_channels + wave_channels + body_rank
        )
        parity_body_channels = moment_rank + body_rank
        self.body = LocalBodyAlgebra(moment_rank=moment_rank, body_rank=body_rank)
        self.scalar = nn.Linear(scalar_dim, scalar_width)
        self.body_scalar = nn.Linear(2 * body_rank, scalar_width, bias=False)
        self.odd = ChannelMix(moment_rank, num_heads)
        self.body_odd = ChannelMix(body_rank, num_heads)
        self.polar = ChannelMix(polar_channels, num_heads)
        self.axial = ChannelMix(parity_body_channels, num_heads)
        self.even_tensor = ChannelMix(even_tensor_channels, num_heads)
        self.odd_tensor = ChannelMix(parity_body_channels, num_heads)
        self.eps = float(eps)

    def forward(self, local: PointwiseLocalFeatures) -> ParityState:
        moments = local.moments
        jet = local.jet
        body = self.body(moments)
        invariant = torch.cat(
            (
                moments.mass,
                moments.second_scalar,
                moments.fourth_scalar,
                _flatten_invariant(jet.value),
                _flatten_invariant(jet.laplacian),
                _flatten_invariant(jet.wavelet_value),
                _flatten_invariant(jet.wavelet_laplacian),
                jet.confidence,
                torch.log1p(jet.scale),
                torch.log1p(local.support_scale).unsqueeze(-1),
            ),
            dim=-1,
        )
        polar_input = torch.cat(
            (
                moments.polar,
                _merge_geometric(jet.gradient),
                _merge_geometric(jet.wavelet_gradient),
                body.polar,
            ),
            dim=1,
        )
        even_tensor_input = torch.cat(
            (
                moments.even_tensor,
                moments.fourth_tensor,
                _merge_geometric(jet.hessian),
                _merge_geometric(jet.wavelet_hessian),
                body.even_tensor,
            ),
            dim=1,
        )
        return ParityState(
            bounded_scalar(
                self.scalar(invariant)
                + self.body_scalar(
                    torch.cat((body.even_scalar, body.fourth_scalar), dim=-1)
                ),
                self.eps,
            ),
            bounded_scalar(
                self.odd(moments.odd_scalar) + self.body_odd(body.odd_scalar),
                self.eps,
            ),
            unit_ball(self.polar(polar_input), self.eps),
            unit_ball(
                self.axial(torch.cat((moments.axial, body.axial), dim=1)),
                self.eps,
            ),
            bounded_st(self.even_tensor(even_tensor_input), self.eps),
            bounded_st(
                self.odd_tensor(
                    torch.cat((moments.odd_tensor, body.odd_tensor), dim=1)
                ),
                self.eps,
            ),
        )


__all__ = ["LocalFeatureProjection"]
