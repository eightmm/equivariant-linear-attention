"""Generic immutable scalar annotations for sparse 3D graphs.

This module deliberately contains no domain vocabulary.  Adapters may attach
their own meanings to relation names, while the core only sees integer IDs.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch


_INTEGER_DTYPES = frozenset(
    {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
)


@dataclass(frozen=True)
class RelationTable:
    """Declare relation names and the explicit ID of each reverse relation.

    ``reverse_id`` must be an involution: applying it twice returns the
    original relation.  A reverse-CSR reduction view does *not* apply this
    mapping; adapters apply :meth:`reverse` only when they create an actual
    oppositely directed edge.
    """

    names: tuple[str, ...]
    reverse_id: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.names:
            raise ValueError("RelationTable requires at least one relation")
        if any(not isinstance(name, str) or not name for name in self.names):
            raise ValueError("relation names must be nonempty strings")
        if len(set(self.names)) != len(self.names):
            raise ValueError("relation names must be unique")
        if len(self.reverse_id) != len(self.names):
            raise ValueError("reverse_id must have the same length as names")
        if any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in self.reverse_id
        ):
            raise TypeError("reverse relation IDs must be integers")
        if any(not 0 <= index < len(self.names) for index in self.reverse_id):
            raise ValueError("reverse relation IDs are out of range")
        if any(
            self.reverse_id[self.reverse_id[index]] != index
            for index in range(len(self.names))
        ):
            raise ValueError("reverse relation mapping must be an involution")

    @property
    def num_relations(self) -> int:
        return len(self.names)

    def reverse(self, relation_id: torch.Tensor) -> torch.Tensor:
        """Map IDs for newly created opposite-direction edges."""

        if not isinstance(relation_id, torch.Tensor):
            raise TypeError("relation_id must be a tensor")
        if relation_id.dtype not in _INTEGER_DTYPES:
            raise TypeError("relation_id must use an integer dtype")
        if relation_id.numel():
            relation_long = relation_id.to(dtype=torch.long)
            if bool((relation_long < 0).any().item()) or bool(
                (relation_long >= self.num_relations).any().item()
            ):
                raise ValueError("relation_id contains an out-of-range value")
        lookup = torch.tensor(
            self.reverse_id,
            dtype=relation_id.dtype,
            device=relation_id.device,
        )
        return lookup[relation_id.to(dtype=torch.long)]


@dataclass(frozen=True)
class DistanceBandSpec:
    """Overlapping smooth radial gates for optional additive corrections.

    Each column is an independent compact cosine gate.  Columns are neither
    normalized nor subtracted from one another, so they are intentionally not
    a partition of unity and do not replace the exact global transport.
    """

    cutoffs: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.cutoffs:
            raise ValueError("distance bands require at least one cutoff")
        numeric: list[float] = []
        for cutoff in self.cutoffs:
            if isinstance(cutoff, bool) or not isinstance(
                cutoff, (int, float)
            ):
                raise TypeError("distance-band cutoffs must be real numbers")
            value = float(cutoff)
            if not isfinite(value):
                raise ValueError("distance-band cutoffs must be finite")
            if value <= 0.0:
                raise ValueError("distance-band cutoffs must be positive")
            numeric.append(value)
        if any(
            right <= left
            for left, right in zip(numeric, numeric[1:])
        ):
            raise ValueError(
                "distance-band cutoffs must be strictly increasing"
            )
        object.__setattr__(self, "cutoffs", tuple(numeric))

    @property
    def additive_not_partition(self) -> bool:
        return True

    def gates(self, physical_squared_distance: torch.Tensor) -> torch.Tensor:
        if not isinstance(physical_squared_distance, torch.Tensor):
            raise TypeError("physical_squared_distance must be a tensor")
        if not torch.is_floating_point(physical_squared_distance):
            raise TypeError(
                "physical_squared_distance must be floating point"
            )
        if physical_squared_distance.numel() and bool(
            (physical_squared_distance < 0).any().item()
        ):
            raise ValueError(
                "physical_squared_distance must be nonnegative"
            )
        cutoffs = torch.tensor(
            self.cutoffs,
            dtype=physical_squared_distance.dtype,
            device=physical_squared_distance.device,
        )
        scaled_square = (
            physical_squared_distance.unsqueeze(-1)
            / cutoffs.square()
        )
        return torch.where(
            scaled_square < 1.0,
            0.5 * (1.0 + torch.cos(torch.pi * scaled_square)),
            torch.zeros_like(scaled_square),
        )
