"""Pointwise local geometry orchestrator."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .geometry import GeometryContext, MomentFeatures
from .local_jet import ReproducingLocalJet
from .local_jet_types import LocalFeatureJet
from .local_moments import LocalCumulantBank
from .local_support import LocalSupport, build_local_support


@dataclass(frozen=True)
class PointwiseLocalFeatures:
    moments: MomentFeatures
    jet: LocalFeatureJet
    support_scale: torch.Tensor

    def to_dtype(self, dtype: torch.dtype) -> PointwiseLocalFeatures:
        """Cast FP32 local statistics when they rejoin the carrier."""

        return PointwiseLocalFeatures(
            moments=self.moments.to_dtype(dtype),
            jet=self.jet.to_dtype(dtype),
            support_scale=self.support_scale.to(dtype=dtype),
        )


class PointwiseLocalGeometry(nn.Module):
    """Compute compact cumulants and latent differential jets per point."""

    def __init__(
        self,
        *,
        scalar_width: int,
        moment_rank: int,
        probe_rank: int,
        num_scales: int,
        max_points: int,
        chunk_size: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.max_points = int(max_points)
        self.chunk_size = int(chunk_size)
        self.eps = float(eps)
        self.cumulants = LocalCumulantBank(
            scalar_width=scalar_width,
            rank=moment_rank,
            eps=eps,
        )
        self.jet = ReproducingLocalJet(
            scalar_width=scalar_width,
            probe_rank=probe_rank,
            num_scales=num_scales,
            eps=eps,
        )

    def forward(
        self,
        scalar: torch.Tensor,
        geometry: GeometryContext,
        support: LocalSupport | None = None,
    ) -> PointwiseLocalFeatures:
        if support is None:
            support = build_local_support(
                geometry,
                max_points=self.max_points,
                chunk_size=self.chunk_size,
                eps=self.eps,
            )
        return PointwiseLocalFeatures(
            moments=self.cumulants(scalar, support),
            jet=self.jet(scalar, support),
            support_scale=support.scale,
        )


__all__ = ["PointwiseLocalFeatures", "PointwiseLocalGeometry"]
