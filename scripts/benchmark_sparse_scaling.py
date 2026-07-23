#!/usr/bin/env python3
"""Controlled scaling benchmark for exact global and sparse local execution.

The same-kernel section compares two implementations of one finite positive
kernel.  The model section separately compares public EC-LGL execution with the
private EGNN control on explicit edge sets.  Edge construction is never timed.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Any

import torch

from equivariant_attention._egnn_baseline import _StaticEGNNBaseline
from equivariant_attention.training import build_regression_model


DEFAULT_SIZES = (32, 64, 128, 256, 512, 1024, 2048, 4096)
DEFAULT_EDGE_MULTIPLIERS = (4, 8, 16, 32, 64, 128)


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


def seeded_exact_edge_index(
    num_nodes: int,
    *,
    edge_multiplier: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Return exact seeded pseudo-random edges without replacement.

    One self edge is emitted for every node. For each receiver, the remaining
    senders follow an independently seeded affine permutation of that node's
    nonself sender universe. This gives every receiver exactly
    ``edge_multiplier`` candidate edges while keeping construction O(E),
    deterministic, duplicate-free, and outside timed model execution.
    """
    if isinstance(num_nodes, bool) or not isinstance(num_nodes, int) or num_nodes <= 0:
        raise ValueError("num_nodes must be a positive integer")
    if (
        isinstance(edge_multiplier, bool)
        or not isinstance(edge_multiplier, int)
        or edge_multiplier <= 0
    ):
        raise ValueError("edge_multiplier must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    admitted_multiplier = min(edge_multiplier, num_nodes)
    self_nodes = torch.arange(num_nodes, device=device, dtype=torch.long)
    if admitted_multiplier == 1 or num_nodes == 1:
        return torch.stack([self_nodes, self_nodes])

    nonself_universe = num_nodes - 1
    nonself_per_receiver = admitted_multiplier - 1
    generator = random.Random(seed)
    if nonself_universe == 1:
        offsets = torch.zeros(num_nodes, device=device, dtype=torch.long)
        steps = torch.ones(num_nodes, device=device, dtype=torch.long)
    else:
        receiver_offsets: list[int] = []
        receiver_steps: list[int] = []
        for _ in range(num_nodes):
            receiver_offsets.append(generator.randrange(nonself_universe))
            step = generator.randrange(1, nonself_universe + 1)
            while math.gcd(step, nonself_universe) != 1:
                step = step % nonself_universe + 1
            receiver_steps.append(step)
        offsets = torch.tensor(receiver_offsets, device=device, dtype=torch.long)
        steps = torch.tensor(receiver_steps, device=device, dtype=torch.long)

    receiver = self_nodes[:, None].expand(-1, nonself_per_receiver).reshape(-1)
    traversal = torch.arange(
        nonself_per_receiver,
        device=device,
        dtype=torch.long,
    )
    compressed_sender = torch.remainder(
        offsets[:, None] + steps[:, None] * traversal[None, :],
        nonself_universe,
    ).reshape(-1)
    sender = compressed_sender + (compressed_sender >= receiver).to(torch.long)
    return torch.stack(
        [
            torch.cat([self_nodes, receiver]),
            torch.cat([self_nodes, sender]),
        ]
    )


def _edge_index_sha256(edge_index: torch.Tensor) -> str:
    materialized = (
        edge_index.detach().to(device="cpu", dtype=torch.long).contiguous().numpy()
    )
    return hashlib.sha256(materialized.tobytes()).hexdigest()


def _module_state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        materialized = tensor.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(materialized.dtype).encode("ascii"))
        digest.update(json.dumps(list(materialized.shape)).encode("ascii"))
        digest.update(materialized.numpy().tobytes())
    return digest.hexdigest()


def _receiver_degree_summary(
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
) -> dict[str, float | int]:
    degree = torch.bincount(edge_index[0], minlength=num_nodes).to(torch.float64)
    return {
        "minimum": int(degree.min().item()),
        "maximum": int(degree.max().item()),
        "mean": float(degree.mean().item()),
        "population_std": float(degree.std(unbiased=False).item()),
    }


def density_degrees(num_nodes: int) -> list[int]:
    if isinstance(num_nodes, bool) or not isinstance(num_nodes, int) or num_nodes <= 0:
        raise ValueError("num_nodes must be a positive integer")
    return sorted(
        {1, min(4, num_nodes), min(16, num_nodes), max(1, num_nodes // 4), num_nodes}
    )


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
    key_scalar = key_scalar / torch.linalg.vector_norm(key_scalar, dim=-1, keepdim=True)
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
    key_vector = key_vector / torch.linalg.vector_norm(key_vector, dim=-1, keepdim=True)
    value_frequency = torch.arange(1, value_dim + 1, device=device, dtype=dtype)
    values = torch.sin(2.0 * math.pi * phase[:, None] * value_frequency[None, :] + 0.17)
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
    result = result + beta * torch.einsum("na,av->nv", query_vector, linear_summary)
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
    return (
        sum(
            (x_value - x_mean) * (y_value - y_mean)
            for x_value, y_value in zip(x, y, strict=True)
        )
        / denominator
    )


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
            "q0_dot_k0 + floor + beta*(1 + q1_dot_k1) + gamma*(q1_dot_k1)^2"
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
    node_feats = torch.sin(2.0 * math.pi * phase[:, None] * frequencies[None, :] + 0.13)
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
    candidate = (
        build_regression_model(
            node_dim=11,
            hidden_dim=64,
            num_layers=3,
            num_heads=4,
            local_head_counts=(4, 0, 4),
            local_cutoff=2.5,
            num_rbf=16,
            use_edge_conditioned_local_transport=True,
        )
        .to(device=device, dtype=torch.float32)
        .eval()
    )
    egnn = (
        _StaticEGNNBaseline(
            node_dim=11,
            hidden_dim=91,
            num_layers=3,
        )
        .to(device=device, dtype=torch.float32)
        .eval()
    )
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
            and row["median_ms"] < density_egnn[row["candidate_degree"]]["median_ms"]
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


def _latency_edge_fit(
    cells: Sequence[dict[str, Any]],
    *,
    model_name: str,
) -> dict[str, Any]:
    points = [
        (
            float(cell["candidate_edges_including_self"]) / 1_000_000.0,
            float(cell["models"][model_name]["median_ms"]),
        )
        for cell in cells
        if cell.get("status") == "completed"
        and cell["models"][model_name].get("status") == "completed"
    ]
    if len(points) < 2:
        return {
            "status": "insufficient_points",
            "point_count": len(points),
        }
    x_mean = sum(point[0] for point in points) / len(points)
    y_mean = sum(point[1] for point in points) / len(points)
    denominator = sum((point[0] - x_mean) ** 2 for point in points)
    if denominator == 0.0:
        return {
            "status": "insufficient_edge_variation",
            "point_count": len(points),
        }
    slope = (
        sum((x_value - x_mean) * (y_value - y_mean) for x_value, y_value in points)
        / denominator
    )
    intercept = y_mean - slope * x_mean
    residual_sum = sum(
        (y_value - (intercept + slope * x_value)) ** 2 for x_value, y_value in points
    )
    total_sum = sum((y_value - y_mean) ** 2 for _, y_value in points)
    r_squared = 1.0 if total_sum == 0.0 and residual_sum == 0.0 else 0.0
    if total_sum > 0.0:
        r_squared = 1.0 - residual_sum / total_sum
    return {
        "status": "completed",
        "point_count": len(points),
        "intercept_ms": float(intercept),
        "slope_ms_per_million_candidate_edges": float(slope),
        "r_squared": float(r_squared),
    }


def _timed_model_metrics(
    *,
    model: torch.nn.Module,
    node_feats: torch.Tensor,
    pos: torch.Tensor,
    batch: torch.Tensor,
    edge_index: torch.Tensor,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    try:
        with torch.inference_mode():
            output = model(
                node_feats,
                pos,
                batch=batch,
                edge_index=edge_index,
                edge_index_is_validated=True,
            )
        if not _model_output_is_finite(output):
            return {
                "status": "failed",
                "failure_class": "nonfinite_output",
            }
        del output
        row = _model_row(
            model=model,
            model_name="model",
            node_feats=node_feats,
            pos=pos,
            batch=batch,
            edge_index=edge_index,
            comparison="same_seeded_exact_edge_set",
            warmup=warmup,
            repeats=repeats,
        )
    except torch.OutOfMemoryError as error:
        if node_feats.device.type == "cuda":
            torch.cuda.empty_cache()
        return {
            "status": "failed",
            "failure_class": "out_of_memory",
            "message": str(error).splitlines()[0],
        }
    return {
        "status": "completed",
        "median_ms": row["median_ms"],
        "peak_cuda_bytes_delta": row["peak_cuda_bytes_delta"],
    }


def _model_output_is_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(_model_output_is_finite(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(_model_output_is_finite(child) for child in value)
    return True


def _timed_edge_free_metrics(
    *,
    model: torch.nn.Module,
    node_feats: torch.Tensor,
    pos: torch.Tensor,
    batch: torch.Tensor,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    try:
        with torch.inference_mode():
            output = model(node_feats, pos, batch=batch)
        if not _model_output_is_finite(output):
            return {
                "status": "failed",
                "failure_class": "nonfinite_output",
            }
        del output
        elapsed, peak = _measure(
            lambda: model(node_feats, pos, batch=batch),
            device=node_feats.device,
            warmup=warmup,
            repeats=repeats,
        )
    except torch.OutOfMemoryError as error:
        if node_feats.device.type == "cuda":
            torch.cuda.empty_cache()
        return {
            "status": "failed",
            "failure_class": "out_of_memory",
            "message": str(error).splitlines()[0],
        }
    return {
        "status": "completed",
        "median_ms": elapsed,
        "peak_cuda_bytes_delta": peak,
        "edge_index_bytes": 0,
        "peak_cuda_bytes_delta_plus_edge_index": peak,
    }


def _module_parameter_bytes(module: torch.nn.Module) -> int:
    return sum(
        parameter.numel() * parameter.element_size()
        for parameter in module.parameters()
    )


def _build_train_step_model(
    name: str,
    *,
    device: torch.device,
    model_seed: int,
) -> torch.nn.Module:
    torch.manual_seed(model_seed)
    if name == "spatial_static":
        model = build_regression_model(
            node_dim=11,
            hidden_dim=64,
            num_layers=3,
            num_heads=4,
            local_head_counts=(0, 0, 0),
            coordinate_updates=False,
            use_multiscale_spatial_kernel=True,
        )
    elif name == "spatial_dynamic":
        model = build_regression_model(
            node_dim=11,
            hidden_dim=64,
            num_layers=3,
            num_heads=4,
            local_head_counts=(0, 0, 0),
            coordinate_updates=True,
            use_multiscale_spatial_kernel=True,
        )
    elif name == "static_egnn":
        model = _StaticEGNNBaseline(
            node_dim=11,
            hidden_dim=91,
            num_layers=3,
        )
    else:
        raise ValueError(f"unknown train-step model: {name}")
    return model.to(device=device, dtype=torch.float32).train()


def _train_step_loss(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    node_feats: torch.Tensor,
    pos: torch.Tensor,
    batch: torch.Tensor,
    edge_index: torch.Tensor | None,
    update: bool,
) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    if edge_index is None:
        output = model(node_feats, pos, batch=batch)
    else:
        output = model(
            node_feats,
            pos,
            batch=batch,
            edge_index=edge_index,
            edge_index_is_validated=True,
        )
    prediction = output["graph_scalars"]
    target = torch.full_like(prediction, 0.75)
    loss = torch.nn.functional.mse_loss(prediction, target)
    loss.backward()
    if update:
        optimizer.step()
    return loss.detach()


def _timed_train_step_metrics(
    *,
    model: torch.nn.Module,
    node_feats: torch.Tensor,
    pos: torch.Tensor,
    batch: torch.Tensor,
    edge_index: torch.Tensor | None,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    device = node_feats.device
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    initial_state_sha256 = _module_state_sha256(model)
    try:
        for _ in range(warmup):
            warmup_loss = _train_step_loss(
                model=model,
                optimizer=optimizer,
                node_feats=node_feats,
                pos=pos,
                batch=batch,
                edge_index=edge_index,
                update=True,
            )
            if not bool(torch.isfinite(warmup_loss).item()):
                return {
                    "status": "failed",
                    "failure_class": "nonfinite_warmup_loss",
                }
        _sync(device)
        if device.type == "cuda":
            baseline_memory = int(torch.cuda.memory_allocated(device))
            torch.cuda.reset_peak_memory_stats(device)
        else:
            baseline_memory = None

        timings: list[float] = []
        final_loss = 0.0
        for _ in range(repeats):
            _sync(device)
            started = time.perf_counter()
            loss = _train_step_loss(
                model=model,
                optimizer=optimizer,
                node_feats=node_feats,
                pos=pos,
                batch=batch,
                edge_index=edge_index,
                update=True,
            )
            _sync(device)
            timings.append((time.perf_counter() - started) * 1000.0)
            final_loss = float(loss.item())

        if device.type == "cuda":
            peak_memory = int(torch.cuda.max_memory_allocated(device))
            peak_delta = max(0, peak_memory - int(baseline_memory))
        else:
            peak_memory = None
            peak_delta = None

        validation_loss = _train_step_loss(
            model=model,
            optimizer=optimizer,
            node_feats=node_feats,
            pos=pos,
            batch=batch,
            edge_index=edge_index,
            update=False,
        )
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        all_finite = bool(
            gradients
            and all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients)
            and bool(torch.isfinite(validation_loss).item())
        )
        nonzero_elements = sum(
            int(torch.count_nonzero(gradient).item()) for gradient in gradients
        )
    except torch.OutOfMemoryError as error:
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {
            "status": "failed",
            "failure_class": "out_of_memory",
            "message": str(error).splitlines()[0],
            "initial_state_sha256": initial_state_sha256,
        }
    return {
        "status": "completed",
        "median_train_step_ms": float(statistics.median(timings)),
        "final_loss": final_loss,
        "gradient_validation": {
            "all_finite": all_finite,
            "nonzero_elements": nonzero_elements,
        },
        "optimizer": "AdamW",
        "optimizer_lr": 1e-4,
        "optimizer_weight_decay": 0.0,
        "initial_state_sha256": initial_state_sha256,
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "parameter_bytes": _module_parameter_bytes(model),
        "edge_index_supplied": edge_index is not None,
        "baseline_cuda_bytes": baseline_memory,
        "peak_cuda_bytes": peak_memory,
        "peak_cuda_bytes_delta": peak_delta,
    }


def run_edge_free_train_step_benchmark(
    *,
    sizes: Sequence[int],
    edge_multipliers: Sequence[int] = DEFAULT_EDGE_MULTIPLIERS,
    device: str = "auto",
    seed: int = 20260723,
    model_seed: int = 20260723,
    warmup: int = 5,
    repeats: int = 20,
    max_wall_seconds: float = 1200.0,
) -> dict[str, Any]:
    """Measure full eager training steps for edge-free attention and EGNN."""
    validated_sizes = list(sizes)
    validated_multipliers = list(edge_multipliers)
    if not validated_sizes or any(
        isinstance(size, bool) or not isinstance(size, int) or size < 2
        for size in validated_sizes
    ):
        raise ValueError("sizes must contain unique integers at least two")
    if len(set(validated_sizes)) != len(validated_sizes):
        raise ValueError("sizes must be unique")
    if not validated_multipliers or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in validated_multipliers
    ):
        raise ValueError("edge_multipliers must contain positive integers")
    if len(set(validated_multipliers)) != len(validated_multipliers):
        raise ValueError("edge_multipliers must be unique")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (
        isinstance(model_seed, bool)
        or not isinstance(model_seed, int)
        or model_seed < 0
    ):
        raise ValueError("model_seed must be a nonnegative integer")
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be nonnegative and repeats must be positive")
    if (
        isinstance(max_wall_seconds, bool)
        or not isinstance(max_wall_seconds, (int, float))
        or not math.isfinite(float(max_wall_seconds))
        or max_wall_seconds <= 0
    ):
        raise ValueError("max_wall_seconds must be finite and positive")

    resolved_device = _resolve_device(device)
    cells: list[dict[str, Any]] = []
    benchmark_started = time.perf_counter()
    model_names = ("spatial_static", "spatial_dynamic", "static_egnn")
    for size_index, num_nodes in enumerate(validated_sizes):
        for multiplier_index, edge_multiplier in enumerate(validated_multipliers):
            if edge_multiplier > num_nodes:
                cells.append(
                    {
                        "status": "skipped",
                        "reason": "edge_multiplier_exceeds_num_nodes",
                        "nodes": num_nodes,
                        "edge_multiplier": edge_multiplier,
                    }
                )
                continue
            if time.perf_counter() - benchmark_started >= max_wall_seconds:
                cells.append(
                    {
                        "status": "skipped",
                        "reason": "wall_time_ceiling_reached",
                        "nodes": num_nodes,
                        "edge_multiplier": edge_multiplier,
                    }
                )
                continue

            cell_seed = seed + num_nodes * 1_000_003 + edge_multiplier * 97
            graph_started = time.perf_counter()
            cpu_edge_index = seeded_exact_edge_index(
                num_nodes,
                edge_multiplier=edge_multiplier,
                seed=cell_seed,
                device=torch.device("cpu"),
            )
            graph_cpu_ms = (time.perf_counter() - graph_started) * 1000.0
            node_feats, pos, batch = _model_inputs(
                num_nodes,
                device=resolved_device,
            )
            execution_order = list(model_names)
            rotation = (size_index + multiplier_index) % len(execution_order)
            execution_order = execution_order[rotation:] + execution_order[:rotation]
            model_metrics: dict[str, dict[str, Any]] = {}
            device_transfer_ms = 0.0
            for model_name in execution_order:
                if time.perf_counter() - benchmark_started >= max_wall_seconds:
                    model_metrics[model_name] = {
                        "status": "skipped",
                        "reason": "wall_time_ceiling_reached",
                    }
                    continue
                edge_index: torch.Tensor | None = None
                if model_name == "static_egnn":
                    if resolved_device.type == "cuda":
                        transfer_started = time.perf_counter()
                        edge_index = cpu_edge_index.to(device=resolved_device)
                        _sync(resolved_device)
                        device_transfer_ms = (
                            time.perf_counter() - transfer_started
                        ) * 1000.0
                    else:
                        edge_index = cpu_edge_index
                model = _build_train_step_model(
                    model_name,
                    device=resolved_device,
                    model_seed=model_seed,
                )
                metrics = _timed_train_step_metrics(
                    model=model,
                    node_feats=node_feats,
                    pos=pos,
                    batch=batch,
                    edge_index=edge_index,
                    warmup=warmup,
                    repeats=repeats,
                )
                model_metrics[model_name] = metrics
                del model
                if edge_index is not None:
                    del edge_index
                if resolved_device.type == "cuda":
                    torch.cuda.empty_cache()

            completed = all(
                model_metrics.get(name, {}).get("status") == "completed"
                for name in model_names
            )
            cell: dict[str, Any] = {
                "status": "completed" if completed else "partial",
                "nodes": num_nodes,
                "edge_multiplier": edge_multiplier,
                "cell_seed": cell_seed,
                "candidate_edges_including_self": int(cpu_edge_index.shape[1]),
                "effective_nonself_edges": int(cpu_edge_index.shape[1] - num_nodes),
                "edge_index_sha256": _edge_index_sha256(cpu_edge_index),
                "edge_index_bytes": (
                    cpu_edge_index.numel() * cpu_edge_index.element_size()
                ),
                "receiver_candidate_degree": _receiver_degree_summary(
                    cpu_edge_index,
                    num_nodes=num_nodes,
                ),
                "graph_construction": {
                    "cpu_build_ms": float(graph_cpu_ms),
                    "device_transfer_ms": float(device_transfer_ms),
                    "total_once_ms": float(graph_cpu_ms + device_transfer_ms),
                },
                "model_execution_order": execution_order,
                "models": model_metrics,
            }
            if completed:
                egnn_ms = model_metrics["static_egnn"]["median_train_step_ms"]
                for candidate_name in ("spatial_static", "spatial_dynamic"):
                    candidate_ms = model_metrics[candidate_name][
                        "median_train_step_ms"
                    ]
                    cell[f"{candidate_name}_to_egnn_train_step_ratio"] = (
                        candidate_ms / egnn_ms
                    )
                    candidate_peak = model_metrics[candidate_name]["peak_cuda_bytes"]
                    egnn_peak = model_metrics["static_egnn"]["peak_cuda_bytes"]
                    if candidate_peak is not None and egnn_peak:
                        cell[f"{candidate_name}_to_egnn_peak_memory_ratio"] = (
                            candidate_peak / egnn_peak
                        )
                graph_once_ms = cell["graph_construction"]["total_once_ms"]
                cell["spatial_static_to_egnn_first_step_system_ratio"] = (
                    model_metrics["spatial_static"]["median_train_step_ms"]
                    / (egnn_ms + graph_once_ms)
                )
                cell["spatial_dynamic_to_egnn_first_step_system_ratio"] = (
                    model_metrics["spatial_dynamic"]["median_train_step_ms"]
                    / (egnn_ms + graph_once_ms)
                )
            cells.append(cell)
            del node_feats, pos, batch, cpu_edge_index
            if resolved_device.type == "cuda":
                torch.cuda.empty_cache()

    elapsed = time.perf_counter() - benchmark_started
    return {
        "schema_version": 1,
        "benchmark": "edge_free_train_step_vs_edge_scaled_egnn",
        "device": str(resolved_device),
        "model_dtype": "float32",
        "seed": seed,
        "graph_seed": seed,
        "model_seed": model_seed,
        "sizes": validated_sizes,
        "edge_multipliers": validated_multipliers,
        "warmup": warmup,
        "repeats": repeats,
        "max_wall_seconds": float(max_wall_seconds),
        "elapsed_wall_seconds": float(elapsed),
        "timed_step": [
            "optimizer.zero_grad(set_to_none=True)",
            "forward",
            "mse_loss",
            "backward",
            "adamw.step",
        ],
        "cells": cells,
        "completed_cell_count": sum(cell["status"] == "completed" for cell in cells),
        "failed_or_skipped_cell_count": sum(
            cell["status"] != "completed" for cell in cells
        ),
        "memory_accounting": {
            "peak_cuda_bytes": (
                "absolute max allocated bytes with one model, its AdamW state, "
                "inputs, activations, gradients, and any supplied edge index resident"
            ),
            "peak_cuda_bytes_delta": (
                "peak allocated bytes minus the post-warmup allocated baseline"
            ),
            "cpu_memory": "not measured; CUDA memory fields are null on CPU",
        },
        "inference_boundary": {
            "graph_construction_timed_separately": True,
            "model_step_excludes_graph_construction": True,
            "model_and_optimizer_construction_timed": False,
            "candidate_edge_index": None,
            "egnn_receives_prebuilt_edges": True,
            "loss": "synthetic single-graph MSE to constant 0.75",
            "task_accuracy_inferred": False,
            "domain_generalization_inferred": False,
        },
    }


def run_edge_free_spatial_benchmark(
    *,
    sizes: Sequence[int],
    edge_multipliers: Sequence[int] = DEFAULT_EDGE_MULTIPLIERS,
    device: str = "auto",
    seed: int = 20260723,
    model_seed: int = 20260723,
    warmup: int = 3,
    repeats: int = 7,
    max_wall_seconds: float = 120.0,
) -> dict[str, Any]:
    """Compare edge-free global variants with EGNN over exact E=kN graphs."""
    validated_sizes = list(sizes)
    validated_multipliers = list(edge_multipliers)
    if not validated_sizes or any(
        isinstance(size, bool) or not isinstance(size, int) or size < 2
        for size in validated_sizes
    ):
        raise ValueError("sizes must contain positive integers at least two")
    if len(set(validated_sizes)) != len(validated_sizes):
        raise ValueError("sizes must be unique")
    if not validated_multipliers or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in validated_multipliers
    ):
        raise ValueError("edge_multipliers must contain positive integers")
    if len(set(validated_multipliers)) != len(validated_multipliers):
        raise ValueError("edge_multipliers must be unique")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (
        isinstance(model_seed, bool)
        or not isinstance(model_seed, int)
        or model_seed < 0
    ):
        raise ValueError("model_seed must be a nonnegative integer")
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be nonnegative and repeats must be positive")
    if (
        isinstance(max_wall_seconds, bool)
        or not isinstance(max_wall_seconds, (int, float))
        or not math.isfinite(float(max_wall_seconds))
        or max_wall_seconds <= 0
    ):
        raise ValueError("max_wall_seconds must be finite and positive")

    resolved_device = _resolve_device(device)

    def build_edge_free(
        *,
        spatial: bool,
        coordinate_updates: bool,
    ) -> torch.nn.Module:
        torch.manual_seed(model_seed)
        return (
            build_regression_model(
                node_dim=11,
                hidden_dim=64,
                num_layers=3,
                num_heads=4,
                local_head_counts=(0, 0, 0),
                coordinate_updates=coordinate_updates,
                use_multiscale_spatial_kernel=spatial,
            )
            .to(device=resolved_device, dtype=torch.float32)
            .eval()
        )

    ggg = build_edge_free(spatial=False, coordinate_updates=False)
    spatial_static = build_edge_free(spatial=True, coordinate_updates=False)
    spatial_dynamic = build_edge_free(spatial=True, coordinate_updates=True)
    torch.manual_seed(model_seed)
    egnn = (
        _StaticEGNNBaseline(
            node_dim=11,
            hidden_dim=91,
            num_layers=3,
        )
        .to(device=resolved_device, dtype=torch.float32)
        .eval()
    )
    models = {
        "ggg": ggg,
        "spatial_static": spatial_static,
        "spatial_dynamic": spatial_dynamic,
        "static_egnn": egnn,
    }

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for size_index, num_nodes in enumerate(validated_sizes):
        if time.perf_counter() - started >= max_wall_seconds:
            rows.append(
                {
                    "status": "skipped",
                    "reason": "wall_time_ceiling_reached",
                    "nodes": num_nodes,
                    "edge_free_models": {},
                    "egnn": [],
                }
            )
            continue
        node_feats, pos, batch = _model_inputs(
            num_nodes,
            device=resolved_device,
        )
        edge_free_order = [
            ("ggg", ggg),
            ("spatial_static", spatial_static),
            ("spatial_dynamic", spatial_dynamic),
        ]
        if size_index % 2:
            edge_free_order.reverse()
        edge_free_metrics = {
            name: _timed_edge_free_metrics(
                model=model,
                node_feats=node_feats,
                pos=pos,
                batch=batch,
                warmup=warmup,
                repeats=repeats,
            )
            for name, model in edge_free_order
        }

        multiplier_order = list(validated_multipliers)
        if size_index % 2:
            multiplier_order.reverse()
        egnn_cells: list[dict[str, Any]] = []
        for edge_multiplier in multiplier_order:
            if edge_multiplier > num_nodes:
                egnn_cells.append(
                    {
                        "status": "skipped",
                        "reason": "edge_multiplier_exceeds_num_nodes",
                        "nodes": num_nodes,
                        "edge_multiplier": edge_multiplier,
                    }
                )
                continue
            if time.perf_counter() - started >= max_wall_seconds:
                egnn_cells.append(
                    {
                        "status": "skipped",
                        "reason": "wall_time_ceiling_reached",
                        "nodes": num_nodes,
                        "edge_multiplier": edge_multiplier,
                    }
                )
                continue
            cell_seed = seed + num_nodes * 1_000_003 + edge_multiplier * 97
            edge_index = seeded_exact_edge_index(
                num_nodes,
                edge_multiplier=edge_multiplier,
                seed=cell_seed,
                device=resolved_device,
            )
            metrics = _timed_model_metrics(
                model=egnn,
                node_feats=node_feats,
                pos=pos,
                batch=batch,
                edge_index=edge_index,
                warmup=warmup,
                repeats=repeats,
            )
            edge_bytes = edge_index.numel() * edge_index.element_size()
            cell: dict[str, Any] = {
                **metrics,
                "nodes": num_nodes,
                "edge_multiplier": edge_multiplier,
                "cell_seed": cell_seed,
                "candidate_edges_including_self": edge_index.shape[1],
                "effective_nonself_edges": edge_index.shape[1] - num_nodes,
                "edge_index_sha256": _edge_index_sha256(edge_index),
                "edge_index_bytes": edge_bytes,
                "receiver_candidate_degree": _receiver_degree_summary(
                    edge_index,
                    num_nodes=num_nodes,
                ),
            }
            if metrics["status"] == "completed":
                cell["peak_cuda_bytes_delta_plus_edge_index"] = (
                    metrics["peak_cuda_bytes_delta"] + edge_bytes
                )
                for candidate_name, candidate_metrics in edge_free_metrics.items():
                    if candidate_metrics["status"] != "completed":
                        continue
                    cell[f"{candidate_name}_to_egnn_latency_ratio"] = (
                        candidate_metrics["median_ms"] / metrics["median_ms"]
                    )
                    cell[f"{candidate_name}_to_egnn_memory_ratio_including_edges"] = (
                        candidate_metrics[
                            "peak_cuda_bytes_delta_plus_edge_index"
                        ]
                        / cell["peak_cuda_bytes_delta_plus_edge_index"]
                    )
            egnn_cells.append(cell)
            del edge_index
            if resolved_device.type == "cuda":
                torch.cuda.empty_cache()

        egnn_cells.sort(key=lambda cell: cell["edge_multiplier"])
        all_completed = all(
            metrics["status"] == "completed"
            for metrics in edge_free_metrics.values()
        ) and all(cell["status"] == "completed" for cell in egnn_cells)
        rows.append(
            {
                "status": "completed" if all_completed else "partial",
                "nodes": num_nodes,
                "edge_free_execution_order": [
                    name for name, _model in edge_free_order
                ],
                "egnn_execution_order": multiplier_order,
                "edge_free_models": edge_free_metrics,
                "egnn": egnn_cells,
            }
        )
        del node_feats, pos, batch
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()

    elapsed = time.perf_counter() - started
    return {
        "schema_version": 1,
        "benchmark": "edge_free_spatial_vs_edge_scaled_egnn",
        "graph_generator": "seeded_exact_receiver_regular_directed",
        "device": str(resolved_device),
        "model_dtype": "float32",
        "seed": seed,
        "graph_seed": seed,
        "model_seed": model_seed,
        "sizes": validated_sizes,
        "edge_multipliers": validated_multipliers,
        "warmup": warmup,
        "repeats": repeats,
        "max_wall_seconds": float(max_wall_seconds),
        "elapsed_wall_seconds": float(elapsed),
        "model_parameters_semantics": "trainable_parameters",
        "model_parameters": {
            name: sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
            for name, model in models.items()
        },
        "model_total_parameters": {
            name: sum(parameter.numel() for parameter in model.parameters())
            for name, model in models.items()
        },
        "model_parameter_bytes": {
            name: _module_parameter_bytes(model)
            for name, model in models.items()
        },
        "model_state_sha256": {
            name: _module_state_sha256(model)
            for name, model in models.items()
        },
        "rows": rows,
        "kernel_contract": {
            "spatial_feature_dimension_per_head": 10,
            "head_scales": [0.125, 0.25, 0.5, 1.0],
            "node_pair_tensor_materialized": False,
            "candidate_complexity_at_fixed_width": "O(N)",
            "egnn_complexity_at_fixed_width": "O(E)",
        },
        "memory_accounting": {
            "peak_cuda_bytes_delta": (
                "maximum allocated bytes during synchronized forwards minus "
                "the allocated baseline after inputs, models, and edges exist"
            ),
            "edge_index_bytes": "added separately because it exists at baseline",
            "model_parameter_bytes": (
                "all parameter tensors, trainable and frozen, reported "
                "separately from forward delta"
            ),
        },
        "inference_boundary": {
            "edge_construction_timed": False,
            "coordinate_updates_timed": True,
            "candidate_edge_index": None,
            "egnn_receives_prebuilt_edges": True,
            "topology_matched": False,
        },
        "limitations": {
            "forward_only": True,
            "backward_or_training_inferred": False,
            "task_accuracy_inferred": False,
            "same_computation_inferred": False,
        },
    }


def run_edge_multiplier_benchmark(
    *,
    sizes: Sequence[int],
    edge_multipliers: Sequence[int] = DEFAULT_EDGE_MULTIPLIERS,
    device: str = "auto",
    seed: int = 20260723,
    model_seed: int = 20260723,
    warmup: int = 3,
    repeats: int = 7,
    max_wall_seconds: float = 120.0,
) -> dict[str, Any]:
    """Benchmark both models on identical exact E=kN pseudo-random graphs."""
    validated_sizes = list(sizes)
    validated_multipliers = list(edge_multipliers)
    if not validated_sizes or any(
        isinstance(size, bool) or not isinstance(size, int) or size < 2
        for size in validated_sizes
    ):
        raise ValueError("sizes must contain positive integers at least two")
    if len(set(validated_sizes)) != len(validated_sizes):
        raise ValueError("sizes must be unique")
    if not validated_multipliers or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in validated_multipliers
    ):
        raise ValueError("edge_multipliers must contain positive integers")
    if len(set(validated_multipliers)) != len(validated_multipliers):
        raise ValueError("edge_multipliers must be unique")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (
        isinstance(model_seed, bool)
        or not isinstance(model_seed, int)
        or model_seed < 0
    ):
        raise ValueError("model_seed must be a nonnegative integer")
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be nonnegative and repeats must be positive")
    if (
        isinstance(max_wall_seconds, bool)
        or not isinstance(max_wall_seconds, (int, float))
        or not math.isfinite(float(max_wall_seconds))
        or max_wall_seconds <= 0
    ):
        raise ValueError("max_wall_seconds must be finite and positive")

    resolved_device = _resolve_device(device)
    torch.manual_seed(model_seed)
    candidate = (
        build_regression_model(
            node_dim=11,
            hidden_dim=64,
            num_layers=3,
            num_heads=4,
            local_head_counts=(4, 0, 4),
            local_cutoff=2.5,
            num_rbf=16,
            use_edge_conditioned_local_transport=True,
        )
        .to(device=resolved_device, dtype=torch.float32)
        .eval()
    )
    egnn = (
        _StaticEGNNBaseline(
            node_dim=11,
            hidden_dim=91,
            num_layers=3,
        )
        .to(device=resolved_device, dtype=torch.float32)
        .eval()
    )

    cells: list[dict[str, Any]] = []
    started = time.perf_counter()
    for size_index, num_nodes in enumerate(validated_sizes):
        for multiplier_index, edge_multiplier in enumerate(validated_multipliers):
            if edge_multiplier > num_nodes:
                cells.append(
                    {
                        "status": "skipped",
                        "reason": "edge_multiplier_exceeds_num_nodes",
                        "nodes": num_nodes,
                        "edge_multiplier": edge_multiplier,
                    }
                )
                continue
            if time.perf_counter() - started >= max_wall_seconds:
                cells.append(
                    {
                        "status": "skipped",
                        "reason": "wall_time_ceiling_reached",
                        "nodes": num_nodes,
                        "edge_multiplier": edge_multiplier,
                    }
                )
                continue

            cell_seed = seed + num_nodes * 1_000_003 + edge_multiplier * 97
            node_feats, pos, batch = _model_inputs(
                num_nodes,
                device=resolved_device,
            )
            edge_index = seeded_exact_edge_index(
                num_nodes,
                edge_multiplier=edge_multiplier,
                seed=cell_seed,
                device=resolved_device,
            )
            model_order = [
                ("ec_lgl", candidate),
                ("static_egnn", egnn),
            ]
            if (size_index + multiplier_index) % 2:
                model_order.reverse()
            model_metrics = {
                name: _timed_model_metrics(
                    model=model,
                    node_feats=node_feats,
                    pos=pos,
                    batch=batch,
                    edge_index=edge_index,
                    warmup=warmup,
                    repeats=repeats,
                )
                for name, model in model_order
            }
            completed = all(
                metrics["status"] == "completed" for metrics in model_metrics.values()
            )
            cell: dict[str, Any] = {
                "status": "completed" if completed else "failed",
                "nodes": num_nodes,
                "edge_multiplier": edge_multiplier,
                "cell_seed": cell_seed,
                "candidate_edges_including_self": edge_index.shape[1],
                "effective_nonself_edges": edge_index.shape[1] - num_nodes,
                "edge_index_sha256": _edge_index_sha256(edge_index),
                "receiver_candidate_degree": _receiver_degree_summary(
                    edge_index,
                    num_nodes=num_nodes,
                ),
                "model_execution_order": [name for name, _ in model_order],
                "models": model_metrics,
            }
            if completed:
                cell["ec_lgl_to_egnn_latency_ratio"] = (
                    model_metrics["ec_lgl"]["median_ms"]
                    / model_metrics["static_egnn"]["median_ms"]
                )
            cells.append(cell)
            del node_feats, pos, batch, edge_index
            if resolved_device.type == "cuda":
                torch.cuda.empty_cache()

    fits_by_nodes: dict[str, Any] = {}
    crossover_cells: list[dict[str, int]] = []
    for num_nodes in validated_sizes:
        node_cells = [
            cell
            for cell in cells
            if cell.get("nodes") == num_nodes and cell.get("status") == "completed"
        ]
        node_cells.sort(key=lambda cell: cell["edge_multiplier"])
        crossover_multiplier = next(
            (
                cell["edge_multiplier"]
                for cell in node_cells
                if cell["ec_lgl_to_egnn_latency_ratio"] < 1.0
            ),
            None,
        )
        if crossover_multiplier is not None:
            crossover_cells.append(
                {
                    "nodes": num_nodes,
                    "edge_multiplier": crossover_multiplier,
                }
            )
        ratio_trend: dict[str, Any] = {"status": "insufficient_points"}
        if len(node_cells) >= 2:
            lowest = float(node_cells[0]["ec_lgl_to_egnn_latency_ratio"])
            highest = float(node_cells[-1]["ec_lgl_to_egnn_latency_ratio"])
            ratio_trend = {
                "status": "completed",
                "lowest_multiplier": node_cells[0]["edge_multiplier"],
                "lowest_multiplier_ratio": lowest,
                "highest_multiplier": node_cells[-1]["edge_multiplier"],
                "highest_multiplier_ratio": highest,
                "ratio_decreased": highest < lowest,
            }
        fits_by_nodes[str(num_nodes)] = {
            "ec_lgl": _latency_edge_fit(node_cells, model_name="ec_lgl"),
            "static_egnn": _latency_edge_fit(
                node_cells,
                model_name="static_egnn",
            ),
            "first_same_edge_crossover_multiplier": crossover_multiplier,
            "latency_ratio_trend": ratio_trend,
        }

    elapsed = time.perf_counter() - started
    completed_cells = sum(cell["status"] == "completed" for cell in cells)
    return {
        "schema_version": 1,
        "benchmark": "same_edge_exact_multiplier_grid",
        "graph_generator": "seeded_exact_receiver_regular_directed",
        "device": str(resolved_device),
        "model_dtype": "float32",
        "seed": seed,
        "graph_seed": seed,
        "model_seed": model_seed,
        "sizes": validated_sizes,
        "edge_multipliers": validated_multipliers,
        "warmup": warmup,
        "repeats": repeats,
        "max_wall_seconds": float(max_wall_seconds),
        "elapsed_wall_seconds": float(elapsed),
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
        "model_state_sha256": {
            "ec_lgl": _module_state_sha256(candidate),
            "static_egnn": _module_state_sha256(egnn),
        },
        "cells": cells,
        "completed_cell_count": completed_cells,
        "failed_or_skipped_cell_count": len(cells) - completed_cells,
        "fits_by_nodes": fits_by_nodes,
        "same_edge_crossover_cells": crossover_cells,
        "prediction": {
            "no_same_edge_crossover_through_registered_grid": not crossover_cells,
            "ratio_decreased_for_every_node_size": all(
                row["latency_ratio_trend"].get("ratio_decreased", False)
                for row in fits_by_nodes.values()
            ),
        },
        "edge_generation": {
            "method": "seeded_per_receiver_affine_sender_permutation_without_replacement",
            "construction_complexity": "O(E)",
            "timed": False,
            "self_edge_per_node": True,
        },
        "inference_boundary": {
            "edge_construction_timed": False,
            "same_edge_tensor_for_models": True,
            "model_computations_identical": False,
            "forward_only": True,
            "backward_or_training_inferred": False,
            "task_accuracy_inferred": False,
            "domain_generalization_inferred": False,
        },
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
    parser.add_argument("--edge-multiplier-grid", action="store_true")
    parser.add_argument("--edge-free-spatial-grid", action="store_true")
    parser.add_argument("--edge-free-train-step-grid", action="store_true")
    parser.add_argument(
        "--edge-multipliers",
        nargs="+",
        type=int,
        default=list(DEFAULT_EDGE_MULTIPLIERS),
    )
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--model-seed", type=int, default=20260723)
    parser.add_argument("--max-wall-seconds", type=float, default=120.0)
    parser.add_argument("--metrics-out", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    selected_modes = sum(
        (
            args.edge_multiplier_grid,
            args.edge_free_spatial_grid,
            args.edge_free_train_step_grid,
        )
    )
    if selected_modes > 1:
        raise ValueError(
            "edge multiplier, edge-free spatial, and train-step grids are exclusive"
        )
    if args.edge_free_train_step_grid:
        result = run_edge_free_train_step_benchmark(
            sizes=args.sizes,
            edge_multipliers=args.edge_multipliers,
            device=args.device,
            seed=args.seed,
            model_seed=args.model_seed,
            warmup=args.warmup,
            repeats=args.repeats,
            max_wall_seconds=args.max_wall_seconds,
        )
    elif args.edge_free_spatial_grid:
        result = run_edge_free_spatial_benchmark(
            sizes=args.sizes,
            edge_multipliers=args.edge_multipliers,
            device=args.device,
            seed=args.seed,
            model_seed=args.model_seed,
            warmup=args.warmup,
            repeats=args.repeats,
            max_wall_seconds=args.max_wall_seconds,
        )
    elif args.edge_multiplier_grid:
        result = run_edge_multiplier_benchmark(
            sizes=args.sizes,
            edge_multipliers=args.edge_multipliers,
            device=args.device,
            seed=args.seed,
            model_seed=args.model_seed,
            warmup=args.warmup,
            repeats=args.repeats,
            max_wall_seconds=args.max_wall_seconds,
        )
    else:
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
