"""Domain-neutral pooling for invariant features on packed 3D graphs."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn


Reduction = Literal["mean", "sum"]
EmptyPolicy = Literal["error", "zero"]

_INTEGER_DTYPES = frozenset(
    {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
)


class MaskedInvariantPooling(nn.Module):
    """Pool invariant node features without mixing graph or selection scopes.

    ``global_pool`` reduces every node in each graph. ``interface_pool`` is a
    separate, explicitly masked reduction.  The latter name denotes a generic
    selected interface and carries no domain-specific role semantics.

    Empty selections are never silently converted into a mean over another
    scope. ``empty_policy="error"`` rejects them; ``"zero"`` emits an exact
    zero for that graph.
    """

    reduction: Reduction
    empty_policy: EmptyPolicy

    def __init__(
        self,
        *,
        reduction: Reduction = "mean",
        empty_policy: EmptyPolicy = "error",
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        if empty_policy not in {"error", "zero"}:
            raise ValueError("empty_policy must be 'error' or 'zero'")
        self.reduction = reduction
        self.empty_policy = empty_policy

    def forward(
        self,
        invariants: torch.Tensor,
        batch: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        num_graphs: int | None = None,
    ) -> torch.Tensor:
        """Reduce a named scope; ``mask=None`` means the global scope."""

        return self._pool(
            invariants,
            batch,
            mask=mask,
            num_graphs=num_graphs,
        )

    def global_pool(
        self,
        invariants: torch.Tensor,
        batch: torch.Tensor,
        *,
        num_graphs: int | None = None,
    ) -> torch.Tensor:
        """Pool all nodes independently in every graph."""

        return self._pool(
            invariants,
            batch,
            mask=None,
            num_graphs=num_graphs,
        )

    def interface_pool(
        self,
        invariants: torch.Tensor,
        batch: torch.Tensor,
        interface_mask: torch.Tensor,
        *,
        num_graphs: int | None = None,
    ) -> torch.Tensor:
        """Pool only explicitly selected interface nodes in every graph."""

        return self._pool(
            invariants,
            batch,
            mask=interface_mask,
            num_graphs=num_graphs,
        )

    def _pool(
        self,
        invariants: torch.Tensor,
        batch: torch.Tensor,
        *,
        mask: torch.Tensor | None,
        num_graphs: int | None,
    ) -> torch.Tensor:
        resolved_graphs, batch_index = _validate_pool_inputs(
            invariants,
            batch,
            num_graphs=num_graphs,
        )
        if mask is None:
            selected = torch.ones(
                batch.shape,
                dtype=torch.bool,
                device=batch.device,
            )
        else:
            _validate_mask(mask, batch)
            selected = mask

        selected_index = batch_index[selected]
        selected_value = invariants[selected]
        counts = torch.bincount(
            selected_index,
            minlength=resolved_graphs,
        )
        empty = counts == 0
        if self.empty_policy == "error" and bool(empty.any().item()):
            graph_ids = empty.nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(
                "pooling scope is empty for graph IDs "
                f"{graph_ids}; use empty_policy='zero' to emit zeros"
            )

        output = invariants.new_zeros(
            (resolved_graphs, *invariants.shape[1:])
        )
        output = output.index_add(0, selected_index, selected_value)
        if self.reduction == "sum":
            return output

        count_shape = (resolved_graphs, *((1,) * (invariants.ndim - 1)))
        safe_counts = counts.clamp_min(1).reshape(count_shape)
        mean = output / safe_counts.to(dtype=output.dtype)
        if self.empty_policy == "zero" and bool(empty.any().item()):
            empty_shape = empty.reshape(count_shape)
            mean = torch.where(empty_shape, torch.zeros_like(mean), mean)
        return mean


def _validate_pool_inputs(
    invariants: torch.Tensor,
    batch: torch.Tensor,
    *,
    num_graphs: int | None,
) -> tuple[int, torch.Tensor]:
    if not isinstance(invariants, torch.Tensor):
        raise TypeError("invariants must be a tensor")
    if not torch.is_floating_point(invariants):
        raise TypeError("invariants must use a floating-point dtype")
    if invariants.ndim == 0:
        raise ValueError("invariants must have a leading node dimension")
    if not isinstance(batch, torch.Tensor):
        raise TypeError("batch must be a tensor")
    if batch.ndim != 1:
        raise ValueError("batch must have shape (N,)")
    if batch.shape[0] != invariants.shape[0]:
        raise ValueError("invariants and batch must have the same length")
    if batch.dtype not in _INTEGER_DTYPES:
        raise TypeError("batch must use an integer dtype")
    if batch.device != invariants.device:
        raise ValueError("invariants and batch must be on the same device")

    batch_index = batch.to(dtype=torch.long)
    if batch_index.numel() and bool((batch_index < 0).any().item()):
        raise ValueError("batch indices must be nonnegative")
    if num_graphs is not None and (
        isinstance(num_graphs, bool)
        or not isinstance(num_graphs, int)
        or num_graphs < 0
    ):
        raise ValueError("num_graphs must be a nonnegative integer")
    if num_graphs is None:
        if batch_index.numel() == 0:
            raise ValueError(
                "num_graphs is required when the node set is empty"
            )
        resolved_graphs = int(batch_index.max().item()) + 1
    else:
        resolved_graphs = num_graphs
    if batch_index.numel() and int(batch_index.max().item()) >= resolved_graphs:
        raise ValueError("batch contains an index outside num_graphs")
    return resolved_graphs, batch_index


def _validate_mask(mask: torch.Tensor, batch: torch.Tensor) -> None:
    if not isinstance(mask, torch.Tensor):
        raise TypeError("interface mask must be a tensor")
    if mask.dtype != torch.bool:
        raise TypeError("interface mask must use boolean dtype")
    if mask.shape != batch.shape:
        raise ValueError("interface mask must have shape (N,)")
    if mask.device != batch.device:
        raise ValueError("interface mask and batch must be on the same device")
