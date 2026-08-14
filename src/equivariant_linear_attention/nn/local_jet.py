from __future__ import annotations

import torch
from torch import nn

from .local_jet_types import LocalFeatureJet, decode_local_jet
from .local_support import LocalSupport, wendland_c2
from .ops import segment_sum, symmetric_monomials, work_dtype

_BASIS_SIZE = 10


def _logit(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(1e-5, 1.0 - 1e-5)
    return torch.log(value) - torch.log1p(-value)


class ReproducingLocalJet(nn.Module):
    """Degree-two local polynomial jet in an orthonormal O(3) basis."""

    def __init__(
        self,
        *,
        scalar_width: int,
        probe_rank: int,
        num_scales: int,
        eps: float,
    ) -> None:
        super().__init__()
        if probe_rank <= 0 or num_scales < 2:
            raise ValueError("invalid local jet dimensions")
        self.probe_rank = int(probe_rank)
        self.num_scales = int(num_scales)
        self.eps = float(eps)
        self.probe = nn.Linear(scalar_width, probe_rank, bias=False)
        desired = torch.linspace(0.35, 0.95, num_scales)
        self.raw_scale = nn.Parameter(_logit((desired - 0.15) / 0.85))
        self.raw_ridge = nn.Parameter(torch.full((num_scales,), -2.0))

    def forward(
        self,
        scalar: torch.Tensor,
        support: LocalSupport,
    ) -> LocalFeatureJet:
        dtype = work_dtype(scalar, support.displacement, self.raw_scale)
        fraction = 0.15 + 0.85 * torch.sigmoid(self.raw_scale.to(dtype=dtype))
        scale = (support.scale.to(dtype=dtype)[:, None] * fraction[None, :]).clamp_min(
            self.eps
        )
        local_scale = scale[support.receiver]
        coordinate = support.displacement[:, None, :] / local_scale[..., None]
        weight = wendland_c2(support.distance[:, None] / local_scale)
        basis = symmetric_monomials(
            coordinate.reshape(-1, 3), orthonormal=True
        )[:, :_BASIS_SIZE].reshape(coordinate.shape[0], self.num_scales, _BASIS_SIZE)
        receiver = support.receiver
        num_nodes = scalar.shape[0]
        gram = segment_sum(
            weight[..., None, None] * basis[..., :, None] * basis[..., None, :],
            receiver,
            num_nodes,
        )
        probe = self.probe(scalar).to(dtype=dtype)
        right = segment_sum(
            weight[..., None, None]
            * basis[..., :, None]
            * probe[support.source, None, None, :],
            receiver,
            num_nodes,
        )
        trace = gram.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
        ridge = (
            torch.nn.functional.softplus(self.raw_ridge.to(dtype=dtype))[None, :]
            * trace.clamp_min(self.eps)
            + self.eps
        )
        identity = torch.eye(_BASIS_SIZE, device=gram.device, dtype=gram.dtype)
        system = gram + ridge[..., None, None] * identity
        factor = torch.linalg.cholesky(system)
        coefficient = torch.cholesky_solve(right, factor)
        return decode_local_jet(coefficient, system, scale, eps=self.eps)


__all__ = ["ReproducingLocalJet"]
