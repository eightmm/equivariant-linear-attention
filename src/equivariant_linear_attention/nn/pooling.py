"""Invariant node-to-graph pooling."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from .ops import INTEGER_DTYPES, segment_count, segment_sum


class MaskedInvariantPooling(nn.Module):
    def __init__(
        self,
        *,
        reduction: Literal["sum", "mean"] = "sum",
        empty_policy: Literal["zero", "error"] = "zero",
    ) -> None:
        super().__init__()
        if reduction not in {"sum", "mean"}:
            raise ValueError("reduction must be 'sum' or 'mean'")
        if empty_policy not in {"zero", "error"}:
            raise ValueError("empty_policy must be 'zero' or 'error'")
        self.reduction = reduction
        self.empty_policy = empty_policy

    def forward(
        self,
        value: torch.Tensor,
        batch: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        num_graphs: int | None = None,
    ) -> torch.Tensor:
        if batch.dtype not in INTEGER_DTYPES or batch.shape != (value.shape[0],):
            raise ValueError("batch must be integer with shape (N,)")
        index = batch.to(dtype=torch.long)
        inferred = 0 if index.numel() == 0 else int(index.max().item()) + 1
        num_graphs = inferred if num_graphs is None else num_graphs
        selected = (
            torch.ones(value.shape[0], dtype=torch.bool, device=value.device)
            if mask is None
            else mask
        )
        if selected.dtype != torch.bool or selected.shape != (value.shape[0],):
            raise ValueError("mask must be boolean with shape (N,)")
        selected_index = index[selected]
        output = segment_sum(value[selected], selected_index, num_graphs)
        count = segment_count(selected_index, num_graphs, dtype=value.dtype)
        if self.empty_policy == "error" and bool((count == 0).any().item()):
            raise ValueError("empty graph selection")
        if self.reduction == "mean":
            output = output / count.clamp_min(1.0).reshape(
                num_graphs, *((1,) * (value.ndim - 1))
            )
        return output


__all__ = ["MaskedInvariantPooling"]
