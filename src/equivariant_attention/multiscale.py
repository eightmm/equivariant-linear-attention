"""Validated fine/coarse hierarchy operations for generic 3D systems."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch


HierarchyReduction = Literal["mean", "sum"]

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
class HierarchyAssignment:
    """A graph-isolated, globally indexed map from fine to coarse nodes.

    Coarse IDs must be contiguous and every coarse node must own at least one
    fine node.  A coarse ID may never span multiple packed graphs.  These
    constraints make pooling and coarse-to-fine broadcast unambiguous without
    an ``N x N`` assignment matrix.
    """

    fine_to_coarse: torch.Tensor
    fine_batch: torch.Tensor
    num_coarse: int | None = None
    coarse_batch: torch.Tensor = field(init=False)
    coarse_counts: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_index_vector(self.fine_to_coarse, "fine_to_coarse")
        _validate_index_vector(self.fine_batch, "fine_batch")
        if self.fine_to_coarse.shape != self.fine_batch.shape:
            raise ValueError(
                "fine_to_coarse and fine_batch must have the same shape"
            )
        if self.fine_to_coarse.device != self.fine_batch.device:
            raise ValueError(
                "fine_to_coarse and fine_batch must be on the same device"
            )

        assignment = self.fine_to_coarse.to(dtype=torch.long)
        batch = self.fine_batch.to(dtype=torch.long)
        if assignment.numel() and bool((assignment < 0).any().item()):
            raise ValueError("fine_to_coarse IDs must be nonnegative")
        if batch.numel() and bool((batch < 0).any().item()):
            raise ValueError("fine_batch IDs must be nonnegative")
        _validate_contiguous_ids(batch, "fine_batch")

        if self.num_coarse is None:
            resolved_coarse = (
                int(assignment.max().item()) + 1
                if assignment.numel()
                else 0
            )
        else:
            if (
                isinstance(self.num_coarse, bool)
                or not isinstance(self.num_coarse, int)
                or self.num_coarse < 0
            ):
                raise ValueError(
                    "num_coarse must be a nonnegative integer"
                )
            resolved_coarse = self.num_coarse
        if assignment.numel() and int(assignment.max().item()) >= resolved_coarse:
            raise ValueError("fine_to_coarse contains an out-of-range ID")

        if assignment.numel() == 0 and resolved_coarse != 0:
            raise ValueError(
                "an empty fine set cannot assign nonempty coarse nodes"
            )
        counts = torch.bincount(assignment, minlength=resolved_coarse)
        if counts.numel() and bool((counts == 0).any().item()):
            raise ValueError(
                "fine_to_coarse IDs must be contiguous and every coarse "
                "node must be assigned"
            )

        coarse_batch_long = torch.empty(
            resolved_coarse,
            dtype=torch.long,
            device=batch.device,
        )
        for coarse_id in range(resolved_coarse):
            member_batch = batch[assignment == coarse_id]
            first_graph = member_batch[0]
            if bool((member_batch != first_graph).any().item()):
                raise ValueError(
                    "one coarse node cannot contain fine nodes from "
                    "multiple graphs"
                )
            coarse_batch_long[coarse_id] = first_graph

        object.__setattr__(self, "num_coarse", resolved_coarse)
        object.__setattr__(
            self,
            "coarse_batch",
            coarse_batch_long.to(dtype=self.fine_batch.dtype),
        )
        object.__setattr__(self, "coarse_counts", counts)

    @property
    def num_fine(self) -> int:
        return self.fine_to_coarse.shape[0]

    @property
    def num_graphs(self) -> int:
        if self.fine_batch.numel() == 0:
            return 0
        return int(self.fine_batch.max().item()) + 1

    def pool_scalars(
        self,
        scalars: torch.Tensor,
        *,
        reduction: HierarchyReduction = "mean",
    ) -> torch.Tensor:
        """Pool invariant scalar channels from fine to coarse nodes."""

        _validate_fine_value(scalars, self, name="scalars")
        if scalars.ndim < 2:
            raise ValueError("scalars must have shape (N, channels, ...)")
        return self._pool(scalars, reduction=reduction)

    def pool_vectors(
        self,
        vectors: torch.Tensor,
        *,
        reduction: HierarchyReduction = "mean",
    ) -> torch.Tensor:
        """Linearly pool polar or axial vector channels."""

        _validate_fine_value(vectors, self, name="vectors")
        if vectors.ndim < 2 or vectors.shape[-1] != 3:
            raise ValueError(
                "vectors must have final dimension 3 and leading dimension N"
            )
        return self._pool(vectors, reduction=reduction)

    def pool_tensors(
        self,
        tensors: torch.Tensor,
        *,
        reduction: HierarchyReduction = "mean",
    ) -> torch.Tensor:
        """Linearly pool Cartesian rank-two tensor channels."""

        _validate_fine_value(tensors, self, name="tensors")
        if tensors.ndim < 3 or tensors.shape[-2:] != (3, 3):
            raise ValueError(
                "rank-two tensors must end in shape (3, 3)"
            )
        return self._pool(tensors, reduction=reduction)

    def centroids(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Return fine-coordinate centroids for every coarse node."""

        _validate_fine_value(coordinates, self, name="coordinates")
        if coordinates.ndim != 2 or coordinates.shape[-1] != 3:
            raise ValueError("coordinates must have shape (N, 3)")
        return self._pool(coordinates, reduction="mean")

    def broadcast(self, coarse_values: torch.Tensor) -> torch.Tensor:
        """Broadcast coarse values back to their assigned fine nodes."""

        if not isinstance(coarse_values, torch.Tensor):
            raise TypeError("coarse_values must be a tensor")
        if coarse_values.ndim == 0 or coarse_values.shape[0] != self.num_coarse:
            raise ValueError(
                "coarse_values leading dimension must equal num_coarse"
            )
        if coarse_values.device != self.fine_to_coarse.device:
            raise ValueError(
                "coarse_values and assignment must be on the same device"
            )
        return coarse_values[self.fine_to_coarse.to(dtype=torch.long)]

    def _pool(
        self,
        value: torch.Tensor,
        *,
        reduction: HierarchyReduction,
    ) -> torch.Tensor:
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        output = value.new_zeros((self.num_coarse, *value.shape[1:]))
        output = output.index_add(
            0,
            self.fine_to_coarse.to(dtype=torch.long),
            value,
        )
        if reduction == "sum":
            return output
        count_shape = (
            self.num_coarse,
            *((1,) * (value.ndim - 1)),
        )
        return output / self.coarse_counts.reshape(count_shape).to(
            dtype=output.dtype
        )


def _validate_index_vector(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.ndim != 1:
        raise ValueError(f"{name} must have shape (N,)")
    if value.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"{name} must use an integer dtype")


def _validate_contiguous_ids(value: torch.Tensor, name: str) -> None:
    if value.numel() == 0:
        return
    counts = torch.bincount(value)
    if bool((counts == 0).any().item()):
        raise ValueError(f"{name} IDs must be contiguous and start at zero")


def _validate_fine_value(
    value: torch.Tensor,
    hierarchy: HierarchyAssignment,
    *,
    name: str,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must use a floating-point dtype")
    if value.ndim == 0 or value.shape[0] != hierarchy.num_fine:
        raise ValueError(
            f"{name} leading dimension must equal the number of fine nodes"
        )
    if value.device != hierarchy.fine_to_coarse.device:
        raise ValueError(
            f"{name} and hierarchy assignment must be on the same device"
        )
