#!/usr/bin/env python3
"""Controlled scaling benchmark for exact global and sparse local execution.

The same-kernel section compares two implementations of one finite positive
kernel.  The model section separately compares public EC-LGL execution with the
private EGNN control on explicit edge sets.  Edge construction is never timed.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any

import torch

from equivariant_attention._egnn_baseline import _StaticEGNNBaseline
from equivariant_attention.training import build_regression_model


DEFAULT_SIZES = (32, 64, 128, 256, 512, 1024, 2048, 4096)


def bounded_ring_edge_index(
    num_nodes: int,
    *,
    degree: int,
    device: torch.device,
) -> torch.Tensor:
    """Return deterministic receiver/sender edges with one self edge per node."""
    if isinstance(num_nodes, bool) or not isinstance(num_nodes, int) or num_nodes <= 0:
        raise ValueError("num_nodes must be a positive integer")
    if isinstance(degree, bool) or not isinstance(degree, int) or degree <= 0:
        raise ValueError("degree must be a positive integer")
    degree = min(degree, num_nodes)
    receiver = torch.arange(num_nodes, device=device).repeat_interleave(degree)
    offsets = torch.arange(degree, device=device).repeat(num_nodes)
    sender = (receiver + offsets) % num_nodes
    return torch.stack([receiver, sender])


def density_degrees(num_nodes: int) -> list[int]:
    if isinstance(num_nodes, bool) or not isinstance(num_nodes, int) or num_nodes <= 0:
        raise ValueError("num_nodes must be a positive integer")
    return sorted({1, min(4, num_nodes), min(16, num_nodes), max(1, num_nodes // 4), num_nodes})


def same_kernel_inputs(
    num_nodes: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    scalar_dim: int = 16,
    value_dim: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create deterministic normalized features for the registered finite kernel."""
    phase = (torch.arange(num_nodes, device=device, dtype=dtype) + 0.5) / num_nodes
    scalar_frequency = torch.arange(1, scalar_dim + 1, device=device, dtype=dtype)
    query_scalar = 1.2 + 0.25 * torch.sin(
        2.0 * math.pi * phase[:, None] * scalar_frequency[None, :]
    )
    key_scalar = 1.1 + 0.20 * torch.cos(
        2.0 * math.pi * phase[:, None] * (scalar_frequency[None, :] + 0.5)
    )
    query_scalar = query_scalar / torch.linalg.vector_norm(
        query_scalar, dim=-1, keepdim=True
    )
    key_scalar = key_scalar / torch.linalg.vector_norm(
        key_scalar, dim=-1, keepdim=True
    )
    query_vector = torch.stack(
        [
            torch.cos(2.0 * math.pi * phase),
            torch.sin(2.0 * math.pi * phase),
            0.7 * torch.sin(6.0 * math.pi * phase + 0.1),
        ],
        dim=-1,
    )
    key_vector = torch.stack(
        [
            torch.cos(4.0 * math.pi * phase + 0.2),
            torch.sin(4.0 * math.pi * phase + 0.2),
            0.6 * torch.cos(10.0 * math.pi * phase + 0.3),
        ],
        dim=-1,
    )
    query_vector = query_vector / torch.linalg.vector_norm(
        query_vector, dim=-1, keepdim=True
    )
    key_vector = key_vector / torch.linalg.vector_norm(
        key_vector, dim=-1, keepdim=True
    )
    value_frequency = torch.arange(1, value_dim + 1, device=device, dtype=dtype)
    values = torch.sin(
        2.0 * math.pi * phase[:, None] * value_frequency[None, :] + 0.17
    )
    return query_scalar, key_scalar, query_vector, key_vector, values


def dense_same_kernel(
    query_scalar: torch.Tensor,
    key_scalar: torch.Tensor,
    query_vector: torch.Tensor,
    key_vector: torch.Tensor,
    values: torch.Tensor,
    *,
    kernel_floor: float = 0.5,
    beta: float = 0.25,
    gamma: float = 0.5,
) -> torch.Tensor:
    """Materialize and row-normalize the exact finite positive kernel."""
    content = query_scalar @ key_scalar.T
    angular = query_vector @ key_vector.T
    kernel = content + kernel_floor + beta * (1.0 + angular) + gamma * angular.square()
    return kernel @ values / kernel.sum(dim=-1, keepdim=True)


def factorized_same_kernel(
    query_scalar: torch.Tensor,
    key_scalar: torch.Tensor,
    query_vector: torch.Tensor,
    key_vector: torch.Tensor,
    values: torch.Tensor,
    *,
    kernel_floor: float = 0.5,
    beta: float = 0.25,
    gamma: float = 0.5,
) -> torch.Tensor:
    """Evaluate the same normalized kernel through sufficient statistics."""
    ones = values.new_ones((values.shape[0], 1))
    transported = torch.cat([values, ones], dim=-1)
    scalar_summary = key_scalar.T @ transported
    constant_summary = transported.sum(dim=0)
    linear_summary = torch.einsum("na,nv->av", key_vector, transported)
    quadratic_summary = torch.einsum(
        "na,nb,nv->abv", key_vector, key_vector, transported
    )
    result = query_scalar @ scalar_summary
    result = result + (kernel_floor + beta) * constant_summary.unsqueeze(0)
    result = result + beta * torch.einsum(
        "na,av->nv", query_vector, linear_summary
    )
    result = result + gamma * torch.einsum(
        "na,abv,nb->nv", query_vector, quadratic_summary, query_vector
    )
    return result[:, :-1] / result[:, -1:]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def _measure(
    function: Callable[[], Any],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[float, int]:
    for _ in range(warmup):
        output = function()
        del output
    _sync(device)
    if device.type == "cuda":
        baseline_memory = int(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
    else:
        baseline_memory = 0
    timings = []
    for _ in range(repeats):
        _sync(device)
        started = time.perf_counter()
        output = function()
        _sync(device)
        timings.append((time.perf_counter() - started) * 1000.0)
        del output
    _sync(device)
    peak_delta = (
        max(0, int(torch.cuda.max_memory_allocated(device)) - baseline_memory)
        if device.type == "cuda"
        else 0
    )
    return float(statistics.median(timings)), peak_delta


def _slope(rows: list[dict[str, Any]]) -> float:
    completed = [row for row in rows if row.get("status") == "completed"][-3:]
    if len(completed) < 2:
        return 0.0
    x = [math.log(float(row["nodes"])) for row in completed]
    y = [math.log(max(float(row["median_ms"]), 1e-12)) for row in completed]
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    if denominator == 0.0:
        return 0.0
    return sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x, y, strict=True)
    ) / denominator


def _same_kernel_section(
    *,
    sizes: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
    dense_max: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    errors: list[float] = []
    bytes_per_element = torch.empty((), dtype=dtype).element_size()
    for num_nodes in sizes:
        inputs = same_kernel_inputs(num_nodes, device=device, dtype=dtype)
        factorized_time, factorized_peak = _measure(
            lambda inputs=inputs: factorized_same_kernel(*inputs),
            device=device,
            warmup=warmup,
            repeats=repeats,
        )
        value_dim = inputs[-1].shape[-1]
        factorized_working = max(
            num_nodes * (value_dim + 1),
            inputs[0].shape[-1] * (value_dim + 1),
            9 * (value_dim + 1),
        )
        rows.append(
            {
                "status": "completed",
                "method": "factorized",
                "nodes": num_nodes,
                "median_ms": factorized_time,
                "peak_cuda_bytes_delta": factorized_peak,
                "materialized_pair_elements": 0,
                "declared_largest_intermediate_elements": factorized_working,
                "declared_largest_intermediate_bytes": (
                    factorized_working * bytes_per_element
                ),
            }
        )
        if num_nodes > dense_max:
            continue
        dense_time, dense_peak = _measure(
            lambda inputs=inputs: dense_same_kernel(*inputs),
            device=device,
            warmup=warmup,
            repeats=repeats,
        )
        dense_output = dense_same_kernel(*inputs)
        factorized_output = factorized_same_kernel(*inputs)
        errors.append(float((dense_output - factorized_output).abs().max().item()))
        del dense_output, factorized_output
        rows.append(
            {
                "status": "completed",
                "method": "materialized_dense",
                "nodes": num_nodes,
                "median_ms": dense_time,
                "peak_cuda_bytes_delta": dense_peak,
                "materialized_pair_elements": num_nodes * num_nodes,
                "declared_largest_intermediate_elements": num_nodes * num_nodes,
                "declared_largest_intermediate_bytes": (
                    num_nodes * num_nodes * bytes_per_element
                ),
            }
        )
    factorized_rows = [row for row in rows if row["method"] == "factorized"]
    dense_rows = [row for row in rows if row["method"] == "materialized_dense"]
    dense_by_size = {row["nodes"]: row for row in dense_rows}
    crossover = next(
        (
            row["nodes"]
            for row in factorized_rows
            if row["nodes"] in dense_by_size
            and row["median_ms"] < dense_by_size[row["nodes"]]["median_ms"]
        ),
        None,
    )
    factorized_slope = _slope(factorized_rows)
    dense_slope = _slope(dense_rows)
    max_error = max(errors, default=0.0)
    return {
        "kernel_formula": (
            "q0_dot_k0 + floor + beta*(1 + q1_dot_k1) "
            "+ gamma*(q1_dot_k1)^2"
        ),
        "normalization": "exact_row_sum",
        "max_abs_error": max_error,
        "rows": sorted(rows, key=lambda row: (row["nodes"], row["method"])),
        "runtime_loglog_slope": {
            "factorized": factorized_slope,
            "materialized_dense": dense_slope,
        },
        "first_measured_runtime_crossover_nodes": crossover,
        "correctness_gate_passed": max_error < 1e-10,
        "quadratic_pair_tensor_in_factorized_path": False,
    }


def _model_inputs(
    num_nodes: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    phase = (
        torch.arange(num_nodes, device=device, dtype=torch.float32) + 0.5
    ) / num_nodes
    frequencies = torch.arange(1, 12, device=device, dtype=torch.float32)
    node_feats = torch.sin(
        2.0 * math.pi * phase[:, None] * frequencies[None, :] + 0.13
    )
    pos = 0.1 * torch.stack(
        [
            torch.cos(2.0 * math.pi * phase),
            torch.sin(2.0 * math.pi * phase),
            torch.sin(6.0 * math.pi * phase + 0.2),
        ],
        dim=-1,
    )
    batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
    return node_feats, pos, batch


def _model_row(
    *,
    model: torch.nn.Module,
    model_name: str,
    node_feats: torch.Tensor,
    pos: torch.Tensor,
    batch: torch.Tensor,
    edge_index: torch.Tensor,
    comparison: str,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    device = node_feats.device
    elapsed, peak = _measure(
        lambda: model(
            node_feats,
            pos,
            batch=batch,
            edge_index=edge_index,
            edge_index_is_validated=True,
        ),
        device=device,
        warmup=warmup,
        repeats=repeats,
    )
    self_edges = node_feats.shape[0]
    return {
        "status": "completed",
        "model": model_name,
        "comparison": comparison,
        "nodes": node_feats.shape[0],
        "candidate_edges_including_self": edge_index.shape[1],
        "effective_nonself_edges": edge_index.shape[1] - self_edges,
        "candidate_degree": edge_index.shape[1] // node_feats.shape[0],
        "median_ms": elapsed,
        "peak_cuda_bytes_delta": peak,
    }


def _model_scaling_section(
    *,
    sizes: Sequence[int],
    device: torch.device,
    degree: int,
    dense_egnn_max: int,
    density_size: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    torch.manual_seed(20260722)
    candidate = build_regression_model(
        node_dim=11,
        hidden_dim=64,
        num_layers=3,
        num_heads=4,
        local_head_counts=(4, 0, 4),
        local_cutoff=2.5,
        num_rbf=16,
        use_edge_conditioned_local_transport=True,
    ).to(device=device, dtype=torch.float32).eval()
    egnn = _StaticEGNNBaseline(
        node_dim=11,
        hidden_dim=91,
        num_layers=3,
    ).to(device=device, dtype=torch.float32).eval()
    fixed_rows: list[dict[str, Any]] = []
    dense_rows: list[dict[str, Any]] = []
    for num_nodes in sizes:
        node_feats, pos, batch = _model_inputs(num_nodes, device=device)
        bounded_edges = bounded_ring_edge_index(
            num_nodes,
            degree=degree,
            device=device,
        )
        for model, name in ((candidate, "ec_lgl"), (egnn, "static_egnn")):
            fixed_rows.append(
                _model_row(
                    model=model,
                    model_name=name,
                    node_feats=node_feats,
                    pos=pos,
                    batch=batch,
                    edge_index=bounded_edges,
                    comparison="same_fixed_degree_edge_set",
                    warmup=warmup,
                    repeats=repeats,
                )
            )
        if num_nodes <= dense_egnn_max:
            dense_edges = bounded_ring_edge_index(
                num_nodes,
                degree=num_nodes,
                device=device,
            )
            dense_rows.append(
                _model_row(
                    model=egnn,
                    model_name="static_egnn",
                    node_feats=node_feats,
                    pos=pos,
                    batch=batch,
                    edge_index=dense_edges,
                    comparison="explicit_complete_edge_set",
                    warmup=warmup,
                    repeats=repeats,
                )
            )

    node_feats, pos, batch = _model_inputs(density_size, device=device)
    density_rows: list[dict[str, Any]] = []
    for density_degree in density_degrees(density_size):
        edges = bounded_ring_edge_index(
            density_size,
            degree=density_degree,
            device=device,
        )
        for model, name in ((candidate, "ec_lgl"), (egnn, "static_egnn")):
            density_rows.append(
                _model_row(
                    model=model,
                    model_name=name,
                    node_feats=node_feats,
                    pos=pos,
                    batch=batch,
                    edge_index=edges,
                    comparison="same_density_control_edge_set",
                    warmup=warmup,
                    repeats=repeats,
                )
            )

    fixed_candidate = [row for row in fixed_rows if row["model"] == "ec_lgl"]
    fixed_egnn = [row for row in fixed_rows if row["model"] == "static_egnn"]
    dense_by_size = {row["nodes"]: row for row in dense_rows}
    dense_crossover = next(
        (
            row["nodes"]
            for row in fixed_candidate
            if row["nodes"] in dense_by_size
            and row["median_ms"] < dense_by_size[row["nodes"]]["median_ms"]
        ),
        None,
    )
    density_egnn = {
        row["candidate_degree"]: row
        for row in density_rows
        if row["model"] == "static_egnn"
    }
    same_edge_crossover = next(
        (
            row["candidate_degree"]
            for row in density_rows
            if row["model"] == "ec_lgl"
            and row["median_ms"]
            < density_egnn[row["candidate_degree"]]["median_ms"]
        ),
        None,
    )
    return {
        "status": "completed",
        "candidate_parameters": sum(
            parameter.numel()
            for parameter in candidate.parameters()
            if parameter.requires_grad
        ),
        "egnn_parameters": sum(
            parameter.numel()
            for parameter in egnn.parameters()
            if parameter.requires_grad
        ),
        "fixed_degree_rows": fixed_rows,
        "dense_egnn_rows": dense_rows,
        "density_rows": density_rows,
        "runtime_loglog_slope_fixed_degree": {
            "ec_lgl": _slope(fixed_candidate),
            "static_egnn": _slope(fixed_egnn),
        },
        "runtime_loglog_slope_dense_egnn": _slope(dense_rows),
        "first_descriptive_dense_egnn_crossover_nodes": dense_crossover,
        "first_same_edge_density_crossover_degree": same_edge_crossover,
    }


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device must resolve to cpu or cuda")
    return device


def _resolve_dtype(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError("dtype must be float32 or float64")


def run_scaling_benchmark(
    *,
    sizes: Sequence[int] = DEFAULT_SIZES,
    device: str = "auto",
    dtype: str = "float64",
    degree: int = 16,
    same_kernel_dense_max: int = 2048,
    dense_egnn_max: int = 512,
    density_size: int = 512,
    warmup: int = 2,
    repeats: int = 5,
    include_model_benchmarks: bool = True,
) -> dict[str, Any]:
    validated_sizes = list(sizes)
    if not validated_sizes or any(
        isinstance(size, bool) or not isinstance(size, int) or size < 2
        for size in validated_sizes
    ):
        raise ValueError("sizes must be a nonempty sequence of integers at least two")
    if len(set(validated_sizes)) != len(validated_sizes):
        raise ValueError("sizes must be unique")
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be nonnegative and repeats must be positive")
    resolved_device = _resolve_device(device)
    resolved_dtype = _resolve_dtype(dtype)
    same_kernel = _same_kernel_section(
        sizes=validated_sizes,
        device=resolved_device,
        dtype=resolved_dtype,
        dense_max=same_kernel_dense_max,
        warmup=warmup,
        repeats=repeats,
    )
    if include_model_benchmarks:
        model_scaling = _model_scaling_section(
            sizes=validated_sizes,
            device=resolved_device,
            degree=degree,
            dense_egnn_max=dense_egnn_max,
            density_size=density_size,
            warmup=warmup,
            repeats=repeats,
        )
    else:
        model_scaling = {
            "status": "skipped",
            "reason": "include_model_benchmarks=False",
        }
    return {
        "schema_version": 1,
        "device": str(resolved_device),
        "same_kernel_dtype": str(resolved_dtype).removeprefix("torch."),
        "model_dtype": "float32",
        "sizes": validated_sizes,
        "degree": degree,
        "warmup": warmup,
        "repeats": repeats,
        "same_kernel": same_kernel,
        "model_scaling": model_scaling,
        "inference_boundary": {
            "neighbor_builder_included": False,
            "fixed_degree_model_complexity": "O(E_local + N) at fixed width",
            "dense_egnn_computation_identical_to_candidate": False,
            "same_edge_density_computation_identical": False,
            "same_kernel_computation_identical": True,
            "qm9_accuracy_inferred": False,
            "domain_generalization_inferred": False,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=list(DEFAULT_SIZES))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--degree", type=int, default=16)
    parser.add_argument("--same-kernel-dense-max", type=int, default=2048)
    parser.add_argument("--dense-egnn-max", type=int, default=512)
    parser.add_argument("--density-size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--skip-model-benchmarks", action="store_true")
    parser.add_argument("--metrics-out", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    result = run_scaling_benchmark(
        sizes=args.sizes,
        device=args.device,
        dtype=args.dtype,
        degree=args.degree,
        same_kernel_dense_max=args.same_kernel_dense_max,
        dense_egnn_max=args.dense_egnn_max,
        density_size=args.density_size,
        warmup=args.warmup,
        repeats=args.repeats,
        include_model_benchmarks=not args.skip_model_benchmarks,
    )
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
