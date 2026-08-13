from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .geometry import MomentFeatures
from .ops import (
    st_commutator_vector,
    st_cross,
    st_inner,
    st_jordan_product,
    st_matvec,
)
from .state import ChannelMix


@dataclass(frozen=True)
class LocalBodyFeatures:
    even_scalar: torch.Tensor
    odd_scalar: torch.Tensor
    polar: torch.Tensor
    axial: torch.Tensor
    even_tensor: torch.Tensor
    odd_tensor: torch.Tensor


class LocalBodyAlgebra(nn.Module):
    """Low-rank density-product algebra through local four-body order."""

    def __init__(self, *, moment_rank: int, body_rank: int) -> None:
        super().__init__()
        self.polar_u = ChannelMix(moment_rank, body_rank)
        self.polar_v = ChannelMix(moment_rank, body_rank)
        self.polar_w = ChannelMix(moment_rank, body_rank)
        self.tensor_u = ChannelMix(moment_rank, body_rank)
        self.tensor_v = ChannelMix(moment_rank, body_rank)

    def forward(self, moments: MomentFeatures) -> LocalBodyFeatures:
        u = self.polar_u(moments.polar)
        v = self.polar_v(moments.polar)
        w = self.polar_w(moments.polar)
        tensor_u = self.tensor_u(moments.even_tensor)
        tensor_v = self.tensor_v(moments.even_tensor)
        axial = torch.cross(u, v, dim=-1) + st_commutator_vector(
            tensor_u, tensor_v
        )
        even_tensor = st_cross(u, v) + st_jordan_product(tensor_u, tensor_v)
        polar = st_matvec(tensor_u, w) + st_matvec(tensor_v, u)
        odd_scalar = (axial * w).sum(dim=-1)
        odd_tensor = st_cross(w, axial)
        even_scalar = torch.cat(
            ((u * v).sum(dim=-1), st_inner(tensor_u, tensor_v) / 5.0),
            dim=-1,
        )
        return LocalBodyFeatures(
            even_scalar=even_scalar,
            odd_scalar=odd_scalar,
            polar=polar,
            axial=axial,
            even_tensor=even_tensor,
            odd_tensor=odd_tensor,
        )


__all__ = ["LocalBodyAlgebra", "LocalBodyFeatures"]
