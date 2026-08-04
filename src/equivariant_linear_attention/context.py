from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi

import torch
from torch import nn

from .geometry.prepared import Prepared3DGraph


_INTEGER_DTYPES = frozenset(
    {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
)


@dataclass(frozen=True, slots=True)
class ELAFeatures:
    """Optional invariant inputs of the one canonical ELA architecture."""

    condition_dim: int = 0
    order_dim: int = 0

    def __post_init__(self) -> None:
        for name in ("condition_dim", "order_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")

    @staticmethod
    def order_bands(width: int) -> int:
        return max(2, min(8, width // 32))

    def order_encoding_dim(self, width: int) -> int:
        if self.order_dim == 0:
            return 0
        bands = self.order_bands(width)
        return self.order_dim * (3 + 2 * bands) + 2

    def total_condition_dim(self, width: int) -> int:
        return self.condition_dim + self.order_encoding_dim(width)

    def contract(self, width: int) -> dict[str, object]:
        return {
            "invariant_condition_dim": self.condition_dim,
            "semantic_order_dim": self.order_dim,
            "order_fourier_bands": self.order_bands(width) if self.order_dim else 0,
        }


@dataclass(frozen=True)
class OrderContext:
    """Semantic sequence/grid coordinates attached to nodes."""

    coordinates: torch.Tensor
    group_index: torch.Tensor | None = None
    periods: torch.Tensor | None = None
    enabled: torch.Tensor | None = None

    @classmethod
    def sequence(
        cls,
        rank: torch.Tensor,
        *,
        segment_id: torch.Tensor | None = None,
        period: float | None = None,
        enabled: torch.Tensor | None = None,
    ) -> "OrderContext":
        if rank.ndim != 1:
            raise ValueError("sequence rank must have shape (N,)")
        periods = None
        if period is not None:
            if not isfinite(float(period)) or float(period) <= 0.0:
                raise ValueError("period must be finite and positive")
            dtype = rank.dtype if rank.is_floating_point() else torch.get_default_dtype()
            periods = rank.new_tensor([period], dtype=dtype)
        return cls(
            coordinates=rank.unsqueeze(-1),
            group_index=segment_id,
            periods=periods,
            enabled=enabled,
        )

    @classmethod
    def grid(
        cls,
        coordinates: torch.Tensor,
        *,
        segment_id: torch.Tensor | None = None,
        periods: torch.Tensor | None = None,
        enabled: torch.Tensor | None = None,
    ) -> "OrderContext":
        return cls(
            coordinates=coordinates,
            group_index=segment_id,
            periods=periods,
            enabled=enabled,
        )

    @classmethod
    def permutation_rank(
        cls,
        rank: torch.Tensor,
        *,
        segment_id: torch.Tensor | None = None,
        enabled: torch.Tensor | None = None,
    ) -> "OrderContext":
        return cls.sequence(rank, segment_id=segment_id, enabled=enabled)

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "OrderContext":
        target = torch.device(device)
        return OrderContext(
            coordinates=self.coordinates.to(device=target, non_blocking=non_blocking),
            group_index=None
            if self.group_index is None
            else self.group_index.to(device=target, non_blocking=non_blocking),
            periods=None
            if self.periods is None
            else self.periods.to(device=target, non_blocking=non_blocking),
            enabled=None
            if self.enabled is None
            else self.enabled.to(device=target, non_blocking=non_blocking),
        )


@dataclass(frozen=True)
class ELAContext:
    """Optional invariant runtime inputs carried by :class:`ELAGraph`."""

    condition: torch.Tensor | None = None
    order: OrderContext | None = None


class FourierOrderEncoder(nn.Module):
    """Permutation-consistent invariant PE for semantic order coordinates."""

    def __init__(
        self,
        *,
        coordinate_dim: int,
        num_bands: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if coordinate_dim <= 0 or num_bands <= 0:
            raise ValueError("coordinate_dim and num_bands must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.coordinate_dim = int(coordinate_dim)
        self.num_bands = int(num_bands)
        self.eps = float(eps)
        self.register_buffer(
            "frequencies",
            torch.pow(2.0, torch.arange(num_bands, dtype=torch.float32)),
            persistent=True,
        )

    @property
    def output_dim(self) -> int:
        return self.coordinate_dim * (3 + 2 * self.num_bands) + 2

    @staticmethod
    def _work_dtype(value: torch.Tensor) -> torch.dtype:
        return torch.float64 if value.dtype == torch.float64 else torch.float32

    @staticmethod
    def _segment_sum(
        value: torch.Tensor,
        group: torch.Tensor,
        num_groups: int,
    ) -> torch.Tensor:
        output = value.new_zeros((num_groups, *value.shape[1:]))
        output.index_add_(0, group, value)
        return output

    def _validate(self, order: OrderContext, graph: Prepared3DGraph) -> None:
        if not isinstance(order, OrderContext):
            raise TypeError("order must be an OrderContext")
        coordinates = order.coordinates
        if coordinates.shape != (graph.num_nodes, self.coordinate_dim):
            raise ValueError(
                "order coordinates must have shape "
                f"({graph.num_nodes}, {self.coordinate_dim})"
            )
        if coordinates.device != graph.device:
            raise ValueError("order coordinates and graph must share one device")
        if not coordinates.is_floating_point() and coordinates.dtype not in _INTEGER_DTYPES:
            raise TypeError("order coordinates must use a numeric dtype")
        if not bool(torch.isfinite(coordinates.to(self._work_dtype(coordinates))).all()):
            raise ValueError("order coordinates must be finite")
        if order.group_index is not None:
            group = order.group_index
            if group.shape != (graph.num_nodes,):
                raise ValueError("group_index must have shape (N,)")
            if group.dtype not in _INTEGER_DTYPES:
                raise TypeError("group_index must use an integer dtype")
            if group.device != graph.device:
                raise ValueError("group_index and graph must share one device")
            if group.numel() and int(group.min().item()) < 0:
                raise ValueError("group_index must be nonnegative")
        if order.periods is not None:
            periods = order.periods
            if periods.shape != (self.coordinate_dim,):
                raise ValueError(
                    f"periods must have shape ({self.coordinate_dim},)"
                )
            if periods.device != graph.device:
                raise ValueError("periods and graph must share one device")
            work = periods.to(self._work_dtype(periods))
            if not bool(torch.isfinite(work).all()) or bool((work < 0).any()):
                raise ValueError("periods must be finite and nonnegative")
        if order.enabled is not None:
            if order.enabled.shape != (graph.num_nodes,):
                raise ValueError("enabled must have shape (N,)")
            if order.enabled.dtype != torch.bool:
                raise TypeError("enabled must use torch.bool")
            if order.enabled.device != graph.device:
                raise ValueError("enabled and graph must share one device")

    def _combined_groups(
        self,
        order: OrderContext,
        graph: Prepared3DGraph,
    ) -> torch.Tensor:
        if order.group_index is None:
            return graph.batch
        pair = torch.stack(
            [graph.batch, order.group_index.to(dtype=torch.long)],
            dim=-1,
        )
        _, inverse = torch.unique(
            pair,
            dim=0,
            sorted=True,
            return_inverse=True,
        )
        return inverse

    def forward(self, order: OrderContext, graph: Prepared3DGraph) -> torch.Tensor:
        self._validate(order, graph)
        coordinates = order.coordinates
        if graph.num_nodes == 0:
            return coordinates.new_zeros((0, self.output_dim), dtype=torch.float32)

        dtype = self._work_dtype(coordinates)
        coordinate = coordinates.to(dtype=dtype)
        group = self._combined_groups(order, graph)
        num_groups = int(group.max().item()) + 1
        enabled = (
            coordinate.new_ones((graph.num_nodes, 1))
            if order.enabled is None
            else order.enabled.to(dtype=dtype).unsqueeze(-1)
        )

        raw_count = self._segment_sum(enabled, group, num_groups)
        count = raw_count.clamp_min(1.0)
        mean = self._segment_sum(coordinate * enabled, group, num_groups) / count
        centered = coordinate - mean[group]
        variance = self._segment_sum(
            centered.square() * enabled,
            group,
            num_groups,
        ) / count
        rms = torch.sqrt(variance + self.eps)
        normalized = centered / rms[group]

        periods = (
            coordinate.new_zeros((self.coordinate_dim,))
            if order.periods is None
            else order.periods.to(device=coordinate.device, dtype=dtype)
        )
        periodic = periods > 0.0
        safe_period = torch.where(periodic, periods, torch.ones_like(periods))
        phase = torch.where(
            periodic.unsqueeze(0),
            2.0 * pi * coordinate / safe_period.unsqueeze(0),
            pi * normalized,
        )
        frequency = self.frequencies.to(device=coordinate.device, dtype=dtype)
        angle = phase.unsqueeze(-1) * frequency.reshape(1, 1, -1)

        base_coordinate = torch.where(
            periodic.unsqueeze(0),
            torch.zeros_like(normalized),
            normalized,
        )
        scale = torch.where(
            periodic.unsqueeze(0),
            torch.log1p(periods).unsqueeze(0).expand_as(normalized),
            torch.log1p(rms[group]),
        )
        periodic_flag = periodic.to(dtype=dtype).unsqueeze(0).expand_as(normalized)
        per_coordinate = torch.cat(
            [
                base_coordinate.unsqueeze(-1),
                scale.unsqueeze(-1),
                periodic_flag.unsqueeze(-1),
                torch.sin(angle),
                torch.cos(angle),
            ],
            dim=-1,
        )
        per_coordinate = (per_coordinate * enabled.unsqueeze(-1)).reshape(
            graph.num_nodes,
            -1,
        )
        group_size = torch.log1p(raw_count[group]) * enabled
        return torch.cat([per_coordinate, group_size, enabled], dim=-1)


__all__ = [
    "ELAContext",
    "ELAFeatures",
    "FourierOrderEncoder",
    "OrderContext",
]
