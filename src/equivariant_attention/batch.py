from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .data import collate_graphs as _collate_graphs


_FLOAT_VALUE_KEYS = frozenset(
    {
        "node_irreps",
        "x",
        "node_features",
        "target",
        "condition",
    }
)
_GEOMETRY_KEYS = frozenset({"pos", "positions"})


def _move_value(
    key: str,
    value: Any,
    *,
    device: torch.device,
    dtype: torch.dtype | None,
    geometry_dtype: torch.dtype | None,
    non_blocking: bool,
) -> Any:
    if isinstance(value, torch.Tensor):
        target_dtype = None
        if value.is_floating_point():
            if key in _GEOMETRY_KEYS:
                target_dtype = geometry_dtype
            elif key in _FLOAT_VALUE_KEYS or key.startswith("order"):
                target_dtype = dtype
        return value.to(
            device=device,
            dtype=target_dtype,
            non_blocking=non_blocking,
        )
    if isinstance(value, tuple):
        return tuple(
            _move_value(
                key,
                item,
                device=device,
                dtype=dtype,
                geometry_dtype=geometry_dtype,
                non_blocking=non_blocking,
            )
            for item in value
        )
    if isinstance(value, list):
        return [
            _move_value(
                key,
                item,
                device=device,
                dtype=dtype,
                geometry_dtype=geometry_dtype,
                non_blocking=non_blocking,
            )
            for item in value
        ]
    return value


def _pin_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.pin_memory()
    if isinstance(value, tuple):
        return tuple(_pin_value(item) for item in value)
    if isinstance(value, list):
        return [_pin_value(item) for item in value]
    return value


class ELABatch(dict[str, Any]):
    """Plain mapping batch with lightweight device-transfer convenience."""

    @property
    def num_nodes(self) -> int:
        value = self.get("node_irreps", self.get("x", self.get("node_features")))
        if not isinstance(value, torch.Tensor):
            raise KeyError("batch has no node tensor")
        if value.ndim == 2:
            return int(value.shape[0])
        mask = self.get("mask", self.get("node_mask"))
        if isinstance(mask, torch.Tensor):
            return int(mask.sum().item())
        return int(value.shape[0] * value.shape[1])

    @property
    def num_graphs(self) -> int:
        batch = self.get("batch")
        if isinstance(batch, torch.Tensor):
            return 0 if batch.numel() == 0 else int(batch.max().item()) + 1
        node = self.get("node_irreps", self.get("x", self.get("node_features")))
        if isinstance(node, torch.Tensor) and node.ndim == 3:
            return int(node.shape[0])
        return 1 if self.num_nodes else 0

    def to(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
        *,
        geometry_dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> ELABatch:
        target = torch.device(device)
        if geometry_dtype is None:
            geometry_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        if geometry_dtype not in {torch.float32, torch.float64}:
            raise TypeError("geometry_dtype must be float32 or float64")
        return ELABatch(
            {
                key: _move_value(
                    key,
                    value,
                    device=target,
                    dtype=dtype,
                    geometry_dtype=geometry_dtype,
                    non_blocking=non_blocking,
                )
                for key, value in self.items()
            }
        )

    def pin_memory(self) -> ELABatch:
        return ELABatch({key: _pin_value(value) for key, value in self.items()})



def collate_graphs(samples: Sequence[Mapping[str, Any]]) -> ELABatch:
    """Collate graph mappings and return an :class:`ELABatch`."""

    return ELABatch(_collate_graphs(samples))


__all__ = ["ELABatch", "collate_graphs"]
