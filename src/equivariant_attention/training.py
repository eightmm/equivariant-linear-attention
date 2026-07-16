from __future__ import annotations

from contextlib import nullcontext
from collections.abc import Iterable
from collections.abc import Iterator
import torch
import torch.nn.functional as F
from torch import nn

from .benchmarking import GraphBatch, GraphSample
from .moment import EquivariantAttention, EquivariantAttentionConfig


def build_regression_model(
    node_dim: int,
    hidden_dim: int = 64,
    num_layers: int = 3,
    num_heads: int = 4,
    linear_kernel_init: float = 0.05,
    use_linear_kernel: bool = True,
    use_key_balancing: bool = True,
) -> nn.Module:
    model = EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=node_dim,
            hidden_irreps=f"{hidden_dim}x0e + {max(1, hidden_dim // 16)}x1o",
            output_irreps="1x0e",
            num_layers=num_layers,
            num_heads=num_heads,
            linear_kernel_init=linear_kernel_init,
            use_linear_kernel=use_linear_kernel,
            use_key_balancing=use_key_balancing,
        )
    )
    _zero_init_linear(model.scalar_out)
    return model


def predict_graph_scalar(model: nn.Module, batch: GraphBatch) -> torch.Tensor:
    out = model(batch.node_feats, batch.pos, batch=batch.batch)
    return out["graph_scalars"].reshape(batch.target.shape[0], -1)


def train_regression_step(
    model: nn.Module,
    batch: GraphBatch,
    optimizer: torch.optim.Optimizer,
    grad_clip: float | None = 1.0,
    target_normalizer: TargetNormalizer | None = None,
    amp_dtype: torch.dtype | None = None,
) -> float:
    device, dtype = _model_device_dtype(model)
    batch = batch.to(device=device, dtype=dtype)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with _autocast_context(device, amp_dtype):
        pred = predict_graph_scalar(model, batch)
    target = batch.target.reshape_as(pred)
    if target_normalizer is not None:
        target = target_normalizer.transform(target)
    loss = F.mse_loss(pred.float(), target.float())
    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return float(loss.detach().cpu().item())


@torch.no_grad()
def evaluate_regression(
    model: nn.Module,
    batches: Iterable[GraphBatch],
    target_normalizer: TargetNormalizer | None = None,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, float]:
    device, dtype = _model_device_dtype(model)
    model.eval()
    abs_error = 0.0
    square_error = 0.0
    count = 0
    for batch in batches:
        batch = batch.to(device=device, dtype=dtype)
        with _autocast_context(device, amp_dtype):
            pred = predict_graph_scalar(model, batch)
        pred = pred.float()
        target = batch.target.reshape_as(pred).float()
        if target_normalizer is not None:
            pred = target_normalizer.inverse(pred)
        diff = pred - target
        abs_error += diff.abs().sum().item()
        square_error += diff.square().sum().item()
        count += diff.numel()
    if count == 0:
        raise ValueError("at least one prediction is required for evaluation")
    return {"mae": abs_error / count, "rmse": (square_error / count) ** 0.5}


def _model_device_dtype(model: nn.Module) -> tuple[torch.device, torch.dtype]:
    param = next(model.parameters())
    return param.device, param.dtype


class TargetNormalizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean = mean
        self.std = std.clamp_min(1e-12)

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> TargetNormalizer:
        return TargetNormalizer(
            mean=self.mean.to(device=device, dtype=dtype),
            std=self.std.to(device=device, dtype=dtype),
        )

    def transform(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.mean.to(device=value.device, dtype=value.dtype)) / self.std.to(device=value.device, dtype=value.dtype)

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.std.to(device=value.device, dtype=value.dtype) + self.mean.to(device=value.device, dtype=value.dtype)

    def as_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.reshape(-1).tolist(), "std": self.std.reshape(-1).tolist()}


def fit_target_normalizer(samples: Iterable[GraphSample]) -> TargetNormalizer:
    targets = [sample.target.reshape(1, -1).float() for sample in samples]
    if not targets:
        raise ValueError("at least one sample is required to fit target normalizer")
    stacked = torch.cat(targets, dim=0)
    return TargetNormalizer(mean=stacked.mean(dim=0, keepdim=True), std=stacked.std(dim=0, keepdim=True, unbiased=False))


def _autocast_context(device: torch.device, amp_dtype: torch.dtype | None) -> Iterator[None]:
    if amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def _zero_init_linear(layer: nn.Linear) -> None:
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
