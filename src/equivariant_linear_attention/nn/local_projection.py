from __future__ import annotations

import torch
from torch import nn

from .local_body import LocalBodyAlgebra
from .local_geometry import PointwiseLocalFeatures
from .state import ChannelMix, ParityState


def _merge(value: torch.Tensor) -> torch.Tensor:
    return value.flatten(1, value.ndim - 2)


class LocalFeatureProjection(nn.Module):
    """Project local cumulants, jets, wavelets, and body products to irreps."""

    def __init__(
        self,
        *,
        scalar_width: int,
        num_heads: int,
        moment_rank: int,
        probe_rank: int,
        num_scales: int,
        body_rank: int,
    ) -> None:
        super().__init__()
        jet_channels = probe_rank * num_scales
        wave_channels = probe_rank * (num_scales - 1)
        scalar_dim = (
            3 * moment_rank + 2 * jet_channels + 2 * wave_channels + 2 * num_scales + 1
        )
        hidden = max(64, 2 * scalar_width)
        self.scalar = nn.Sequential(
            nn.Linear(scalar_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, scalar_width),
        )
        self.body_scalar = nn.Linear(2 * body_rank, scalar_width, bias=False)
        self.odd = ChannelMix(moment_rank, num_heads)
        self.polar = ChannelMix(moment_rank, num_heads)
        self.axial = ChannelMix(moment_rank, num_heads)
        self.even_tensor = ChannelMix(moment_rank, num_heads)
        self.fourth_tensor = ChannelMix(moment_rank, num_heads)
        self.odd_tensor = ChannelMix(moment_rank, num_heads)
        self.jet_gradient = ChannelMix(jet_channels, num_heads)
        self.jet_hessian = ChannelMix(jet_channels, num_heads)
        self.wave_gradient = ChannelMix(wave_channels, num_heads)
        self.wave_hessian = ChannelMix(wave_channels, num_heads)
        self.body_odd = ChannelMix(body_rank, num_heads)
        self.body_polar = ChannelMix(body_rank, num_heads)
        self.body_axial = ChannelMix(body_rank, num_heads)
        self.body_even = ChannelMix(body_rank, num_heads)
        self.body_odd_tensor = ChannelMix(body_rank, num_heads)
        self.body = LocalBodyAlgebra(moment_rank=moment_rank, body_rank=body_rank)

    def forward(self, local: PointwiseLocalFeatures) -> ParityState:
        moments = local.moments
        jet = local.jet
        body = self.body(moments)
        scalar = torch.cat(
            (
                moments.mass,
                moments.second_scalar,
                moments.fourth_scalar,
                jet.value.flatten(1),
                jet.laplacian.flatten(1),
                jet.wavelet_value.flatten(1),
                jet.wavelet_laplacian.flatten(1),
                jet.confidence,
                torch.log1p(jet.scale),
                torch.log1p(local.support_scale).unsqueeze(-1),
            ),
            dim=-1,
        )
        return ParityState(
            self.scalar(scalar) + self.body_scalar(body.even_scalar),
            self.odd(moments.odd_scalar) + self.body_odd(body.odd_scalar),
            self.polar(moments.polar)
            + self.jet_gradient(_merge(jet.gradient))
            + self.wave_gradient(_merge(jet.wavelet_gradient))
            + self.body_polar(body.polar),
            self.axial(moments.axial) + self.body_axial(body.axial),
            self.even_tensor(moments.even_tensor)
            + self.fourth_tensor(moments.fourth_tensor)
            + self.jet_hessian(_merge(jet.hessian))
            + self.wave_hessian(_merge(jet.wavelet_hessian))
            + self.body_even(body.even_tensor),
            self.odd_tensor(moments.odd_tensor) + self.body_odd_tensor(body.odd_tensor),
        )


__all__ = ["LocalFeatureProjection"]
