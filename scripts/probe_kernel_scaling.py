#!/usr/bin/env python3
"""Bounded synthetic size probe for the registered row-normalized kernel.

The probe compares the fixed baseline with the corrected inverse-size baseline
without key balancing.  Exact row statistics are evaluated in query blocks;
only a small fixed set of query rows participates in the gradient probe.  No
effective rank or full-matrix decomposition is computed.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch


DEFAULT_SIZES = (16, 32, 64, 128, 512, 2048)
FORMULA = "a_dot_b + s_N*(c + beta*(1 + delta*t)) + gamma*t^2"
MODES = ("fixed", "inverse_graph_size")


def _baseline_scale(mode: str, graph_size: int) -> float:
    if mode == "fixed":
        return 1.0
    if mode == "inverse_graph_size":
        return 1.0 / graph_size
    raise ValueError(f"unknown kernel baseline mode: {mode}")


def _kernel_block(
    query_scalar: torch.Tensor,
    key_scalar: torch.Tensor,
    query_vector: torch.Tensor,
    key_vector: torch.Tensor,
    *,
    kernel_floor: float,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    alignment_linear_term: bool,
    mode: str,
    graph_size: int,
) -> torch.Tensor:
    """Return one exact query block of the registered positive kernel."""

    baseline_scale = _baseline_scale(mode, graph_size)
    content = query_scalar @ key_scalar.T
    angular = query_vector @ key_vector.T
    shifted_alignment = beta * (
        1.0 + angular if alignment_linear_term else torch.ones_like(angular)
    )
    return (
        content
        + baseline_scale * (kernel_floor + shifted_alignment)
        + gamma * angular.square()
    )


def _unit_rows(values: torch.Tensor) -> torch.Tensor:
    return values / torch.linalg.vector_norm(values, dim=-1, keepdim=True)


def _synthetic_inputs(
    graph_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build deterministic, analytic positive-content and polar-vector inputs."""

    phase = (torch.arange(graph_size, device=device, dtype=dtype) + 0.5) / graph_size
    tau = 2.0 * math.pi
    query_scalar = _unit_rows(
        torch.stack(
            (
                1.20 + 0.35 * torch.sin(tau * phase),
                1.10 + 0.30 * torch.cos(2.0 * tau * phase + 0.2),
                0.95 + 0.25 * torch.sin(3.0 * tau * phase + 0.4),
                1.05 + 0.20 * torch.cos(5.0 * tau * phase + 0.1),
            ),
            dim=-1,
        )
    )
    key_scalar = _unit_rows(
        torch.stack(
            (
                1.05 + 0.30 * torch.cos(tau * phase + 0.3),
                1.15 + 0.25 * torch.sin(2.0 * tau * phase + 0.5),
                1.00 + 0.20 * torch.cos(4.0 * tau * phase + 0.7),
                0.90 + 0.15 * torch.sin(7.0 * tau * phase + 0.2),
            ),
            dim=-1,
        )
    )
    query_vector = _unit_rows(
        torch.stack(
            (
                torch.cos(tau * phase),
                torch.sin(tau * phase),
                0.65 * torch.sin(3.0 * tau * phase + 0.1),
            ),
            dim=-1,
        )
    )
    key_vector = _unit_rows(
        torch.stack(
            (
                torch.cos(2.0 * tau * phase + 0.4),
                torch.sin(2.0 * tau * phase + 0.4),
                0.55 * torch.cos(5.0 * tau * phase + 0.2),
            ),
            dim=-1,
        )
    )
    values = torch.stack(
        (
            torch.sin(tau * phase + 0.15),
            torch.cos(3.0 * tau * phase + 0.25),
            torch.sin(5.0 * tau * phase + 0.35),
            torch.cos(7.0 * tau * phase + 0.45),
        ),
        dim=-1,
    )
    return query_scalar, key_scalar, query_vector, key_vector, values


@torch.no_grad()
def _stream_attention_stats(
    query_scalar: torch.Tensor,
    key_scalar: torch.Tensor,
    query_vector: torch.Tensor,
    key_vector: torch.Tensor,
    *,
    kernel_floor: float,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    alignment_linear_term: bool,
    mode: str,
    block_rows: int,
) -> dict[str, float]:
    """Compute exact mean row max and entropy without retaining all rows."""

    graph_size = query_scalar.shape[0]
    max_weight_sum = torch.zeros((), dtype=torch.float64, device=query_scalar.device)
    entropy_sum = torch.zeros((), dtype=torch.float64, device=query_scalar.device)
    for start in range(0, graph_size, block_rows):
        stop = min(start + block_rows, graph_size)
        kernel = _kernel_block(
            query_scalar[start:stop],
            key_scalar,
            query_vector[start:stop],
            key_vector,
            kernel_floor=kernel_floor,
            beta=beta,
            gamma=gamma,
            alignment_linear_term=alignment_linear_term,
            mode=mode,
            graph_size=graph_size,
        )
        if not bool(torch.isfinite(kernel).all().item()) or bool(
            (kernel <= 0.0).any().item()
        ):
            raise RuntimeError("synthetic kernel must remain finite and positive")
        weights = kernel / kernel.sum(dim=-1, keepdim=True)
        max_weight_sum += weights.amax(dim=-1).to(dtype=torch.float64).sum()
        entropy_sum -= (
            (weights * weights.log()).sum(dim=-1).to(dtype=torch.float64).sum()
        )

    max_weight = max_weight_sum / graph_size
    normalized_entropy = entropy_sum / (graph_size * math.log(graph_size))
    return {
        "max_weight": float(max_weight.item()),
        "entropy_over_log_n": float(normalized_entropy.item()),
    }


def _gradient_probe(
    query_scalar: torch.Tensor,
    key_scalar: torch.Tensor,
    query_vector: torch.Tensor,
    key_vector: torch.Tensor,
    values: torch.Tensor,
    *,
    kernel_floor: float,
    beta_value: float,
    gamma_value: float,
    alignment_linear_term: bool,
    mode: str,
    probe_rows: int,
) -> tuple[dict[str, float], float]:
    """Differentiate a small deterministic output probe, not all query rows."""

    graph_size = query_scalar.shape[0]
    row_count = min(probe_rows, graph_size)
    row_indices = (
        torch.linspace(
            0,
            graph_size - 1,
            steps=row_count,
            dtype=query_scalar.dtype,
            device=query_scalar.device,
        )
        .round()
        .to(dtype=torch.long)
    )
    beta = torch.tensor(
        beta_value,
        dtype=query_scalar.dtype,
        device=query_scalar.device,
        requires_grad=True,
    )
    gamma = torch.tensor(
        gamma_value,
        dtype=query_scalar.dtype,
        device=query_scalar.device,
        requires_grad=True,
    )
    differentiable_values = values.detach().clone().requires_grad_(True)
    kernel = _kernel_block(
        query_scalar[row_indices],
        key_scalar,
        query_vector[row_indices],
        key_vector,
        kernel_floor=kernel_floor,
        beta=beta,
        gamma=gamma,
        alignment_linear_term=alignment_linear_term,
        mode=mode,
        graph_size=graph_size,
    )
    weights = kernel / kernel.sum(dim=-1, keepdim=True)
    output = weights @ differentiable_values
    probe = torch.linspace(
        0.35,
        1.15,
        steps=output.numel(),
        dtype=output.dtype,
        device=output.device,
    ).reshape_as(output)
    loss = (output * probe).sum() / math.sqrt(output.numel())
    beta_gradient, gamma_gradient, value_gradient = torch.autograd.grad(
        loss, (beta, gamma, differentiable_values)
    )
    gradient_norms = {
        "beta": float(beta_gradient.detach().abs().item()),
        "gamma": float(gamma_gradient.detach().abs().item()),
        "values": float(torch.linalg.vector_norm(value_gradient.detach()).item()),
    }
    output_norm = float(torch.linalg.vector_norm(output.detach()).item())
    if not all(
        math.isfinite(value) for value in (*gradient_norms.values(), output_norm)
    ):
        raise RuntimeError("gradient probe produced a non-finite value")
    return gradient_norms, output_norm


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed_attention_stats(
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    kernel_floor: float,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    alignment_linear_term: bool,
    mode: str,
    block_rows: int,
    warmup: int,
    repeats: int,
) -> tuple[dict[str, float], float]:
    kwargs = {
        "kernel_floor": kernel_floor,
        "beta": beta,
        "gamma": gamma,
        "alignment_linear_term": alignment_linear_term,
        "mode": mode,
        "block_rows": block_rows,
    }
    for _ in range(warmup):
        _stream_attention_stats(*inputs[:4], **kwargs)

    timings: list[float] = []
    statistics_result: dict[str, float] | None = None
    for _ in range(repeats):
        _synchronize(inputs[0].device)
        started = time.perf_counter()
        statistics_result = _stream_attention_stats(*inputs[:4], **kwargs)
        _synchronize(inputs[0].device)
        timings.append((time.perf_counter() - started) * 1000.0)
    if statistics_result is None:
        raise RuntimeError("at least one timed repeat is required")
    return statistics_result, float(statistics.median(timings))


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if resolved.type not in {"cpu", "cuda"}:
        raise ValueError("device must resolve to CPU or CUDA")
    return resolved


def _resolve_dtype(dtype: str) -> torch.dtype:
    if dtype == "float32":
        return torch.float32
    if dtype == "float64":
        return torch.float64
    raise ValueError("dtype must be float32 or float64")


def _positive_finite(value: float, *, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return converted


def run_probe(
    *,
    sizes: Sequence[int] = DEFAULT_SIZES,
    device: str = "auto",
    dtype: str = "float64",
    block_rows: int = 128,
    probe_rows: int = 8,
    warmup: int = 1,
    repeats: int = 3,
    kernel_floor: float = 0.5,
    beta: float = 0.25,
    gamma: float = 0.5,
    alignment_linear_term: bool = True,
) -> dict[str, Any]:
    """Run fixed and inverse-baseline probes for each requested graph size."""

    validated_sizes = list(sizes)
    if not validated_sizes or any(
        isinstance(size, bool) or not isinstance(size, int) or size < 2
        for size in validated_sizes
    ):
        raise ValueError("sizes must be a nonempty sequence of integers at least two")
    if len(set(validated_sizes)) != len(validated_sizes):
        raise ValueError("sizes must be unique")
    if (
        isinstance(block_rows, bool)
        or not isinstance(block_rows, int)
        or block_rows <= 0
    ):
        raise ValueError("block_rows must be a positive integer")
    if (
        isinstance(probe_rows, bool)
        or not isinstance(probe_rows, int)
        or probe_rows <= 0
    ):
        raise ValueError("probe_rows must be a positive integer")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("repeats must be a positive integer")
    if not isinstance(alignment_linear_term, bool):
        raise TypeError("alignment_linear_term must be bool")
    kernel_floor = _positive_finite(kernel_floor, name="kernel_floor")
    beta = _positive_finite(beta, name="beta")
    gamma = _positive_finite(gamma, name="gamma")
    resolved_device = _resolve_device(device)
    resolved_dtype = _resolve_dtype(dtype)
    element_bytes = torch.empty((), dtype=resolved_dtype).element_size()

    runs: list[dict[str, Any]] = []
    for graph_size in validated_sizes:
        inputs = _synthetic_inputs(
            graph_size, device=resolved_device, dtype=resolved_dtype
        )
        for mode in MODES:
            beta_tensor = torch.tensor(
                beta, dtype=resolved_dtype, device=resolved_device
            )
            gamma_tensor = torch.tensor(
                gamma, dtype=resolved_dtype, device=resolved_device
            )
            attention_statistics, runtime_ms = _timed_attention_stats(
                inputs,
                kernel_floor=kernel_floor,
                beta=beta_tensor,
                gamma=gamma_tensor,
                alignment_linear_term=alignment_linear_term,
                mode=mode,
                block_rows=block_rows,
                warmup=warmup,
                repeats=repeats,
            )
            gradient_norms, output_probe_norm = _gradient_probe(
                *inputs,
                kernel_floor=kernel_floor,
                beta_value=beta,
                gamma_value=gamma,
                alignment_linear_term=alignment_linear_term,
                mode=mode,
                probe_rows=probe_rows,
            )
            statistics_pair_elements = min(block_rows, graph_size) * graph_size
            gradient_pair_elements = min(probe_rows, graph_size) * graph_size
            runs.append(
                {
                    "size": graph_size,
                    "mode": mode,
                    "formula": FORMULA,
                    "config": {
                        "kernel_floor": kernel_floor,
                        "beta": beta,
                        "gamma": gamma,
                        "alignment_linear_term": alignment_linear_term,
                        "baseline_scale": _baseline_scale(mode, graph_size),
                        "key_balancing": False,
                    },
                    "statistics": attention_statistics,
                    "gradient_norms": gradient_norms,
                    "output_probe_norm": output_probe_norm,
                    "runtime_ms": runtime_ms,
                    "runtime_scope": "synchronized_exact_statistics_pass",
                    "resource_bound": {
                        "block_rows": min(block_rows, graph_size),
                        "probe_rows": min(probe_rows, graph_size),
                        "statistics_pair_elements": statistics_pair_elements,
                        "gradient_pair_elements": gradient_pair_elements,
                        "largest_explicit_pair_tensor_bytes": max(
                            statistics_pair_elements, gradient_pair_elements
                        )
                        * element_bytes,
                        "full_attention_matrix_persisted": False,
                        "statistics_block_covers_all_rows": min(block_rows, graph_size)
                        == graph_size,
                    },
                }
            )

    result: dict[str, Any] = {
        "schema_version": 1,
        "probe": "deterministic_analytic_kernel_scaling_v1",
        "device": str(resolved_device),
        "dtype": dtype,
        "sizes": validated_sizes,
        "key_balancing": False,
        "effective_rank_computed": False,
        "runtime_repeats": repeats,
        "runtime_warmup": warmup,
        "runs": runs,
    }
    json.dumps(result, allow_nan=False)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=list(DEFAULT_SIZES))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--block-rows", type=int, default=128)
    parser.add_argument("--probe-rows", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--kernel-floor", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument(
        "--alignment-linear-term",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--indent", type=int, default=2)
    parser.add_argument("--metrics-out", type=Path, default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_probe(
        sizes=args.sizes,
        device=args.device,
        dtype=args.dtype,
        block_rows=args.block_rows,
        probe_rows=args.probe_rows,
        warmup=args.warmup,
        repeats=args.repeats,
        kernel_floor=args.kernel_floor,
        beta=args.beta,
        gamma=args.gamma,
        alignment_linear_term=args.alignment_linear_term,
    )
    text = json.dumps(result, indent=args.indent, sort_keys=True, allow_nan=False)
    print(text)
    if args.metrics_out is not None:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(text + "\n")


if __name__ == "__main__":
    main()
