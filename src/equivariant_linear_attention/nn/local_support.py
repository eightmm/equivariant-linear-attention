"""Transient pointwise support for local ELA operators."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .geometry import GeometryContext
from .ops import work_dtype


def wendland_c2(value: torch.Tensor) -> torch.Tensor:
    """Compact C2 radial window with support on [0, 1]."""
    remainder = (1.0 - value).clamp_min(0.0)
    return remainder.pow(4) * (1.0 + 4.0 * value.clamp_min(0.0))


@dataclass(frozen=True)
class LocalSupport:
    source: torch.Tensor
    receiver: torch.Tensor
    displacement: torch.Tensor
    distance: torch.Tensor
    scale: torch.Tensor


def build_local_support(
    geometry: GeometryContext,
    *,
    max_points: int,
    chunk_size: int,
    eps: float,
) -> LocalSupport:
    """Build bounded local candidates without materializing an N x N tensor."""
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    coordinate = geometry.positions.to(dtype=work_dtype(geometry.positions))
    source_parts: list[torch.Tensor] = []
    receiver_parts: list[torch.Tensor] = []
    scale = coordinate.new_zeros(coordinate.shape[0])
    for segment in range(geometry.num_segments):
        nodes = torch.nonzero(geometry.index == segment, as_tuple=False).flatten()
        count = int(nodes.numel())
        if count == 0:
            continue
        active = min(max_points, count)
        candidates = min(active + 1, count)
        local_coordinate = coordinate[nodes]
        fallback = geometry.radius[segment].to(dtype=coordinate.dtype).clamp_min(eps)
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            receiver_nodes = nodes[start:stop]
            distance = torch.cdist(local_coordinate[start:stop], local_coordinate)
            ordered_distance, ordered_index = torch.topk(
                distance, k=candidates, dim=-1, largest=False, sorted=True
            )
            if count > active:
                local_scale = 0.5 * (
                    ordered_distance[:, active - 1] + ordered_distance[:, active]
                )
            else:
                local_scale = 1.25 * ordered_distance[:, -1]
            local_scale = torch.maximum(
                local_scale, fallback.expand_as(local_scale) * 0.25
            ).clamp_min(eps)
            scale[receiver_nodes] = local_scale.detach()
            source_parts.append(nodes[ordered_index.reshape(-1)])
            receiver_parts.append(
                receiver_nodes[:, None].expand(-1, candidates).reshape(-1)
            )
    source = torch.cat(source_parts, dim=0)
    receiver = torch.cat(receiver_parts, dim=0)
    displacement = coordinate[source] - coordinate[receiver]
    distance = torch.linalg.vector_norm(displacement, dim=-1)
    return LocalSupport(source, receiver, displacement, distance, scale)


__all__ = ["LocalSupport", "build_local_support", "wendland_c2"]
