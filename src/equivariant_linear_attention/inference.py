"""Inference helper for the tensor-fused edge-free model."""

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
    if not compile_model:
        return model
    kwargs = {"dynamic": True}
    if compile_kwargs is not None:
        kwargs.update(compile_kwargs)
    return torch.compile(model, **kwargs)


__all__ = ["prepare_for_inference"]
