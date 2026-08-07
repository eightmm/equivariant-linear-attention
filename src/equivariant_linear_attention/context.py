"""Invariant context and semantic-order encoding."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .graph import ELAGraph


@dataclass(frozen=True, slots=True)
class ELAFeatures:
    condition_dim: int = 0
    order_dim: int = 0

    def __post_init__(self) -> None:
        for name, value in (("condition_dim", self.condition_dim), ("order_dim", self.order_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")

    def order_bands(self, width: int) -> int:
        return 0 if self.order_dim == 0 else max(2, min(16, width // max(8, 2 * self.order_dim)))

    def encoded_dim(self, width: int) -> int:
        return self.condition_dim + 2 * self.order_dim * self.order_bands(width)


class FourierOrderEncoder(nn.Module):
    def __init__(self, *, order_dim: int, num_bands: int) -> None:
        super().__init__()
        self.order_dim = order_dim
        self.num_bands = num_bands
        self.register_buffer("frequencies", 2.0 ** torch.arange(num_bands, dtype=torch.float32))

    @property
    def output_dim(self) -> int:
        return 2 * self.order_dim * self.num_bands

    def forward(self, order: torch.Tensor) -> torch.Tensor:
        if order.ndim != 2 or order.shape[-1] != self.order_dim:
            raise ValueError(f"order must have shape (N,{self.order_dim})")
        phase = order[..., None] * self.frequencies.to(device=order.device, dtype=order.dtype)
        return torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1).flatten(1)


class InvariantContextEncoder(nn.Module):
    def __init__(self, *, features: ELAFeatures, width: int) -> None:
        super().__init__()
        self.features = features
        self.order_encoder: FourierOrderEncoder | None = None
        if features.order_dim:
            self.order_encoder = FourierOrderEncoder(
                order_dim=features.order_dim,
                num_bands=features.order_bands(width),
            )
        encoded = features.encoded_dim(width)
        self.projection: nn.Module | None = None
        if encoded:
            self.projection = nn.Sequential(nn.Linear(encoded, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, graph: ELAGraph) -> torch.Tensor | None:
        if self.projection is None:
            if graph.condition is not None or graph.order is not None:
                raise ValueError("graph context was supplied but the model declares none")
            return None
        parts: list[torch.Tensor] = []
        if self.features.condition_dim:
            condition = graph.condition_matrix
            if condition is None:
                raise ValueError("condition is required")
            if condition.shape[-1] != self.features.condition_dim:
                raise ValueError("condition dimension does not match the model")
            parts.append(condition[graph.batch_index].to(dtype=graph.x.dtype))
        elif graph.condition is not None:
            raise ValueError("condition was supplied but condition_dim=0")
        if self.features.order_dim:
            if graph.order is None or self.order_encoder is None:
                raise ValueError("order is required")
            parts.append(self.order_encoder(graph.order).to(dtype=graph.x.dtype))
        elif graph.order is not None:
            raise ValueError("order was supplied but order_dim=0")
        return self.projection(torch.cat(parts, dim=-1))


__all__ = ["ELAFeatures", "FourierOrderEncoder", "InvariantContextEncoder"]
