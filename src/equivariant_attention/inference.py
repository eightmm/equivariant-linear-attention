from __future__ import annotations

import torch
from torch import nn


def prepare_for_inference(
    model: nn.Module,
    device: str | torch.device | None = None,
    dtype: torch.dtype | str | None = "auto",
    compile_model: bool = True,
    compile_mode: str = "reduce-overhead",
    allow_tf32: bool = True,
) -> nn.Module:
    """Prepare a model for inference with FP32 parameters by default.

    ``dtype="auto"`` preserves FP32 parameters and uses CUDA autocast when a
    lower-precision CUDA lane is supported. Explicit ``bf16``/``fp16`` keeps
    the previous whole-model conversion available for controlled experiments.
    """

    if allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    target_device = torch.device(device) if device is not None else next(model.parameters()).device
    automatic = dtype == "auto"
    target_dtype = torch.float32 if automatic else _resolve_dtype(dtype, target_device)
    model = model.to(device=target_device, dtype=target_dtype).eval()
    for param in model.parameters():
        param.requires_grad_(False)

    if automatic and target_device.type == "cuda":
        model = _AutocastInferenceModule(model, autocast_dtype(target_device))
    if compile_model:
        model = torch.compile(model, mode=compile_mode)
    return model


class _AutocastInferenceModule(nn.Module):
    def __init__(self, model: nn.Module, dtype: torch.dtype) -> None:
        super().__init__()
        self.model = model
        self.dtype = dtype

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> object:
        device = next(self.model.parameters()).device
        with torch.autocast(device_type=device.type, dtype=self.dtype):
            return self.model(*args, **kwargs)


def autocast_dtype(device: str | torch.device = "cuda") -> torch.dtype:
    target_device = torch.device(device)
    if target_device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if target_device.type == "cuda":
        return torch.float16
    return torch.float32


def _resolve_dtype(dtype: torch.dtype | str | None, device: torch.device) -> torch.dtype:
    if dtype is None:
        return torch.float32
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype == "auto":
        return autocast_dtype(device)
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype in {"fp32", "float32"}:
        return torch.float32
    msg = f"unknown dtype: {dtype}"
    raise ValueError(msg)
