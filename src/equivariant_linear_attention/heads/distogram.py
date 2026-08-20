"""Symmetric auxiliary distance head over an ordered latent pair state."""

from __future__ import annotations

import torch
from torch import nn

from ..nn.pair_state import DensePairState


class DistogramHead(nn.Module):
    """Predict distance bins without symmetrizing the persistent pair latent."""

    def __init__(
        self,
        *,
        pair_width: int,
        num_bins: int,
        max_distance: float,
    ) -> None:
        super().__init__()
        if num_bins <= 1:
            raise ValueError("num_bins must be greater than one")
        if max_distance <= 0.0:
            raise ValueError("max_distance must be positive")
        self.norm = nn.LayerNorm(pair_width)
        self.projection = nn.Linear(pair_width, num_bins)
        self.register_buffer(
            "boundaries",
            torch.linspace(0.0, float(max_distance), num_bins + 1)[1:-1],
            persistent=True,
        )

    @property
    def num_bins(self) -> int:
        return int(self.projection.out_features)

    def forward(self, pair: DensePairState) -> torch.Tensor:
        symmetric = 0.5 * (pair.z + pair.z.transpose(1, 2))
        logits = self.projection(self.norm(symmetric))
        return logits * pair.pair_mask[..., None].to(dtype=logits.dtype)

    def targets(
        self,
        pair: DensePairState,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        dense = pair.unpack_node_tensor(positions)
        distance = torch.cdist(dense.float(), dense.float())
        target = torch.bucketize(
            distance,
            self.boundaries.float(),
        )
        return target.masked_fill(~pair.pair_mask, -1)


__all__ = ["DistogramHead"]
