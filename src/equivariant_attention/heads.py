"""Generic O(3)-equivariant vector and coordinate heads."""

from __future__ import annotations

from math import sqrt
from typing import Literal

import torch
from torch import nn


CenteringMode = Literal["none", "graph", "selected"]

_INTEGER_DTYPES = frozenset(
    {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
)


class EquivariantVectorHead(nn.Module):
    """Produce vectors using only invariant coefficients and vector carriers.

    The learned coefficient for every output/input-channel pair is a scalar
    function of invariant node features.  Consequently the head commutes with
    every orthogonal transform, including reflections, and introduces no
    preferred Cartesian direction.
    """

    def __init__(
        self,
        scalar_channels: int,
        vector_channels: int,
        *,
        output_channels: int = 1,
        hidden_channels: int | None = None,
    ) -> None:
        super().__init__()
        _validate_channel_count(
            scalar_channels,
            "scalar_channels",
            allow_zero=True,
        )
        _validate_channel_count(vector_channels, "vector_channels")
        _validate_channel_count(output_channels, "output_channels")
        if hidden_channels is None:
            hidden_channels = max(
                8,
                scalar_channels,
                vector_channels * output_channels,
            )
        _validate_channel_count(hidden_channels, "hidden_channels")

        self.scalar_channels = scalar_channels
        self.vector_channels = vector_channels
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels
        self.base_weight = nn.Parameter(
            torch.empty(output_channels, vector_channels)
        )
        nn.init.normal_(
            self.base_weight,
            std=1.0 / sqrt(vector_channels),
        )
        self.scalar_mixer = nn.Sequential(
            nn.Linear(scalar_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(
                hidden_channels,
                output_channels * vector_channels,
            ),
        )

    def forward(
        self,
        scalars: torch.Tensor,
        vectors: torch.Tensor,
    ) -> torch.Tensor:
        _validate_scalar_vector_inputs(
            scalars,
            vectors,
            scalar_channels=self.scalar_channels,
            vector_channels=self.vector_channels,
        )
        conditioned = self.scalar_mixer(scalars).reshape(
            scalars.shape[0],
            self.output_channels,
            self.vector_channels,
        )
        coefficients = conditioned + self.base_weight.unsqueeze(0)
        return torch.einsum("nov,nvc->noc", coefficients, vectors)


class CoordinateUpdateHead(nn.Module):
    """Update selected coordinates with an equivariant predicted displacement.

    Centering is explicit:

    - ``none`` leaves the predicted displacement unchanged.
    - ``graph`` subtracts the mean prediction over every node in its graph,
      then applies the update mask.
    - ``selected`` first restricts to selected nodes and subtracts their mean.
      This preserves both the selected-set and whole-graph centroid.

    In every mode, unselected coordinates are returned by an exact
    ``torch.where`` identity branch.
    """

    def __init__(
        self,
        scalar_channels: int,
        vector_channels: int,
        *,
        centering: CenteringMode = "graph",
        hidden_channels: int | None = None,
    ) -> None:
        super().__init__()
        if centering not in {"none", "graph", "selected"}:
            raise ValueError(
                "centering must be 'none', 'graph', or 'selected'"
            )
        self.centering = centering
        self.vector_head = EquivariantVectorHead(
            scalar_channels,
            vector_channels,
            output_channels=1,
            hidden_channels=hidden_channels,
        )

    def displacement(
        self,
        scalars: torch.Tensor,
        vectors: torch.Tensor,
        batch: torch.Tensor,
        *,
        update_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the exactly masked, optionally centered displacement."""

        raw = self.vector_head(scalars, vectors).squeeze(1)
        batch_index, num_graphs, graph_counts = _validate_batch(
            batch,
            num_nodes=raw.shape[0],
            device=raw.device,
        )
        selected = _resolve_update_mask(
            update_mask,
            batch,
            device=raw.device,
        )

        if self.centering == "none":
            centered = raw
        elif self.centering == "graph":
            graph_sum = raw.new_zeros((num_graphs, 3)).index_add(
                0,
                batch_index,
                raw,
            )
            graph_mean = graph_sum / graph_counts[:, None].to(
                dtype=raw.dtype
            )
            centered = raw - graph_mean[batch_index]
        else:
            selected_raw = torch.where(
                selected[:, None],
                raw,
                torch.zeros_like(raw),
            )
            selected_sum = raw.new_zeros((num_graphs, 3)).index_add(
                0,
                batch_index,
                selected_raw,
            )
            selected_count = torch.bincount(
                batch_index[selected],
                minlength=num_graphs,
            )
            selected_mean = selected_sum / selected_count.clamp_min(1)[
                :, None
            ].to(dtype=raw.dtype)
            centered = raw - selected_mean[batch_index]

        return torch.where(
            selected[:, None],
            centered,
            torch.zeros_like(centered),
        )

    def forward(
        self,
        scalars: torch.Tensor,
        vectors: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        *,
        update_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not isinstance(positions, torch.Tensor):
            raise TypeError("positions must be a tensor")
        if (
            positions.ndim != 2
            or positions.shape[1] != 3
            or positions.shape[0] != scalars.shape[0]
        ):
            raise ValueError("positions must have shape (N, 3)")
        if not torch.is_floating_point(positions):
            raise TypeError("positions must use a floating-point dtype")
        if positions.device != scalars.device:
            raise ValueError(
                "positions and node features must be on the same device"
            )
        selected = _resolve_update_mask(
            update_mask,
            batch,
            device=positions.device,
        )
        step = self.displacement(
            scalars,
            vectors,
            batch,
            update_mask=selected,
        ).to(dtype=positions.dtype)
        candidate = positions + step
        return torch.where(selected[:, None], candidate, positions)


def _validate_channel_count(
    value: int,
    name: str,
    *,
    allow_zero: bool = False,
) -> None:
    minimum = 0 if allow_zero else 1
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")


def _validate_scalar_vector_inputs(
    scalars: torch.Tensor,
    vectors: torch.Tensor,
    *,
    scalar_channels: int,
    vector_channels: int,
) -> None:
    if not isinstance(scalars, torch.Tensor):
        raise TypeError("scalars must be a tensor")
    if not isinstance(vectors, torch.Tensor):
        raise TypeError("vectors must be a tensor")
    if not torch.is_floating_point(scalars) or not torch.is_floating_point(
        vectors
    ):
        raise TypeError("scalar and vector features must be floating point")
    if scalars.ndim != 2 or scalars.shape[1] != scalar_channels:
        raise ValueError(
            f"scalars must have shape (N, {scalar_channels})"
        )
    if (
        vectors.ndim != 3
        or vectors.shape[1] != vector_channels
        or vectors.shape[2] != 3
    ):
        raise ValueError(
            "vectors must have shape "
            f"(N, {vector_channels}, 3); final dimension must be 3"
        )
    if vectors.shape[0] != scalars.shape[0]:
        raise ValueError("scalar and vector features must have the same length")
    if vectors.device != scalars.device:
        raise ValueError(
            "scalar and vector features must be on the same device"
        )
    if vectors.dtype != scalars.dtype:
        raise TypeError(
            "scalar and vector features must use the same dtype"
        )


def _validate_batch(
    batch: torch.Tensor,
    *,
    num_nodes: int,
    device: torch.device,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    if not isinstance(batch, torch.Tensor):
        raise TypeError("batch must be a tensor")
    if batch.ndim != 1 or batch.shape[0] != num_nodes:
        raise ValueError("batch must have shape (N,)")
    if batch.dtype not in _INTEGER_DTYPES:
        raise TypeError("batch must use an integer dtype")
    if batch.device != device:
        raise ValueError("batch and node features must be on the same device")
    if num_nodes == 0:
        raise ValueError("coordinate updates require at least one node")
    index = batch.to(dtype=torch.long)
    if bool((index < 0).any().item()):
        raise ValueError("batch IDs must be nonnegative")
    num_graphs = int(index.max().item()) + 1
    counts = torch.bincount(index, minlength=num_graphs)
    if bool((counts == 0).any().item()):
        raise ValueError("batch IDs must be contiguous and start at zero")
    return index, num_graphs, counts


def _resolve_update_mask(
    update_mask: torch.Tensor | None,
    batch: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    if update_mask is None:
        return torch.ones(
            batch.shape,
            dtype=torch.bool,
            device=device,
        )
    if not isinstance(update_mask, torch.Tensor):
        raise TypeError("update_mask must be a tensor")
    if update_mask.dtype != torch.bool:
        raise TypeError("update_mask must use boolean dtype")
    if update_mask.shape != batch.shape:
        raise ValueError("update_mask must have shape (N,)")
    if update_mask.device != device:
        raise ValueError(
            "update_mask and node features must be on the same device"
        )
    return update_mask
