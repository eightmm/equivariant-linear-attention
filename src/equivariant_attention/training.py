from __future__ import annotations

from contextlib import nullcontext
from collections.abc import Iterable
from collections.abc import Iterator
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .baselines import EGNNBaseline, EGNNBaselineConfig
from .benchmarking import GraphBatch, GraphSample
from .moment import EquivariantMomentAttention, EquivariantMomentAttentionConfig
from .rich import RichEquivariantAttention, RichEquivariantAttentionConfig


RegressionModelName = Literal["egnn", "rich_local", "rich_linear", "rich_linear_light", "moment_linear"]


def build_regression_model(
    name: RegressionModelName,
    node_dim: int,
    hidden_dim: int = 64,
    num_layers: int = 3,
    num_heads: int = 4,
    *,
    moment_radial_trace: bool = False,
    moment_full_gram_invariants: bool = False,
    moment_shifted_angular_kernel: bool = False,
    moment_radial_distance_kernel: bool = False,
    moment_dynamic_moment_routing: bool = False,
    moment_sinkhorn_iterations: int = 1,
    moment_learnable_balance_exponent: bool = False,
    moment_equivariant_ffn: bool = True,
    moment_ffn_hidden_ratio: float = 2.0,
    moment_radial_distance_shift_init: float = 1.1,
    moment_routing_hidden_dim: int = 16,
    moment_routing_delta_scale: float = 0.25,
) -> nn.Module:
    if name == "egnn":
        model = EGNNBaseline(EGNNBaselineConfig(node_dim=node_dim, hidden_dim=hidden_dim, num_layers=num_layers))
        _zero_init_linear(model.graph_scalar)
        return model
    if name == "rich_local":
        return _build_rich_regression_model(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            attention_mode="local",
            vector_edge_bias=True,
            vector_channels=max(1, hidden_dim // 8),
            tensor_channels=max(1, hidden_dim // 16),
        )
    if name == "rich_linear":
        return _build_rich_regression_model(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            attention_mode="linear",
            vector_edge_bias=False,
            vector_channels=max(1, hidden_dim // 8),
            tensor_channels=max(1, hidden_dim // 16),
        )
    if name == "rich_linear_light":
        return _build_rich_regression_model(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            attention_mode="linear",
            vector_edge_bias=False,
            vector_channels=max(1, hidden_dim // 16),
            tensor_channels=0,
        )
    if name == "moment_linear":
        model = EquivariantMomentAttention(
            EquivariantMomentAttentionConfig(
                node_dim=node_dim,
                hidden_irreps=f"{hidden_dim}x0e + {max(1, hidden_dim // 16)}x1o",
                output_irreps="1x0e",
                num_layers=num_layers,
                num_heads=num_heads,
                radial_trace=moment_radial_trace,
                full_gram_invariants=moment_full_gram_invariants,
                shifted_angular_kernel=moment_shifted_angular_kernel,
                radial_distance_kernel=moment_radial_distance_kernel,
                dynamic_moment_routing=moment_dynamic_moment_routing,
                sinkhorn_iterations=moment_sinkhorn_iterations,
                learnable_balance_exponent=moment_learnable_balance_exponent,
                equivariant_ffn=moment_equivariant_ffn,
                ffn_hidden_ratio=moment_ffn_hidden_ratio,
                radial_distance_shift_init=moment_radial_distance_shift_init,
                routing_hidden_dim=moment_routing_hidden_dim,
                routing_delta_scale=moment_routing_delta_scale,
            )
        )
        _zero_init_linear(model.scalar_out)
        return model
    raise ValueError(f"unknown model name: {name}")


def predict_graph_scalar(model: nn.Module, batch: GraphBatch) -> torch.Tensor:
    kwargs = {"batch": batch.batch}
    if _uses_neighbor_index(model):
        kwargs["neighbor_index"] = batch.neighbor_index
        kwargs["neighbor_mask"] = batch.neighbor_mask
    out = model(batch.node_feats, batch.pos, **kwargs)
    graph_scalar = out["graph_scalar"] if "graph_scalar" in out else out["graph_scalars"]
    return graph_scalar.reshape(batch.target.shape[0], -1)


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


def _build_rich_regression_model(
    node_dim: int,
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    attention_mode: Literal["local", "linear"],
    vector_edge_bias: bool,
    vector_channels: int,
    tensor_channels: int,
) -> RichEquivariantAttention:
    model = RichEquivariantAttention(
        RichEquivariantAttentionConfig(
            node_dim=node_dim,
            hidden_irreps=_format_rich_hidden_irreps(hidden_dim, vector_channels, tensor_channels),
            output_irreps="1x0e",
            num_layers=num_layers,
            num_heads=num_heads,
            attention_mode=attention_mode,
            vector_edge_bias=vector_edge_bias,
        )
    )
    _zero_init_linear(model.scalar_out)
    return model


def _format_rich_hidden_irreps(hidden_dim: int, vector_channels: int, tensor_channels: int) -> str:
    terms = [f"{hidden_dim}x0e"]
    if vector_channels > 0:
        terms.append(f"{vector_channels}x1o")
    if tensor_channels > 0:
        terms.append(f"{tensor_channels}x2e")
    return " + ".join(terms)


def _uses_neighbor_index(model: nn.Module) -> bool:
    config = getattr(model, "config", None)
    if isinstance(model, EGNNBaseline):
        return True
    return getattr(config, "attention_mode", None) == "local"


def _autocast_context(device: torch.device, amp_dtype: torch.dtype | None) -> Iterator[None]:
    if amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def _zero_init_linear(layer: nn.Linear) -> None:
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
