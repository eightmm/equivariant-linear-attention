"""Inference helper for the edge-free model."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .model import ELA


def prepare_for_inference(
    model: ELA,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    compile_model: bool = False,
    compile_kwargs: dict[str, Any] | None = None,
) -> nn.Module:
    if not isinstance(model, ELA):
        raise TypeError("model must be ELA")
    if device is not None or dtype is not None:
        model = model.to(device=device, dtype=dtype)
    model.eval()
    return torch.compile(model, **({} if compile_kwargs is None else compile_kwargs)) if compile_model else model


__all__ = ["prepare_for_inference"]
