#!/usr/bin/env python3
"""Profile named stages and operators in one full synthetic training step."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import math
from pathlib import Path
import runpy
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile, record_function


SCALING = runpy.run_path(
    str(Path(__file__).with_name("benchmark_sparse_scaling.py"))
)


def _event_value(event: Any, primary: str, fallback: str) -> float:
    value = getattr(event, primary, None)
    if value is None:
        value = getattr(event, fallback, 0.0)
    return float(value)


def _event_record(event: Any) -> dict[str, Any]:
    return {
        "name": str(event.key),
        "count": int(event.count),
        "cpu_time_total_us": float(event.cpu_time_total),
        "self_cpu_time_total_us": float(event.self_cpu_time_total),
        "device_time_total_us": _event_value(
            event,
            "device_time_total",
            "cuda_time_total",
        ),
        "self_device_time_total_us": _event_value(
            event,
            "self_device_time_total",
            "self_cuda_time_total",
        ),
        "cpu_memory_usage_bytes": int(event.cpu_memory_usage),
        "device_memory_usage_bytes": int(
            getattr(event, "device_memory_usage", 0)
        ),
    }


def _profiled_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    node_feats: torch.Tensor,
    pos: torch.Tensor,
    batch: torch.Tensor,
    edge_index: torch.Tensor | None,
) -> torch.Tensor:
    with record_function("stage.zero_grad"):
        optimizer.zero_grad(set_to_none=True)
    with record_function("stage.forward"):
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
    with record_function("stage.loss"):
        prediction = output["graph_scalars"]
        loss = torch.nn.functional.mse_loss(
            prediction,
            torch.full_like(prediction, 0.75),
        )
    with record_function("stage.backward"):
        loss.backward()
    with record_function("stage.optimizer_step"):
        optimizer.step()
    return loss.detach()


def run_train_step_profile(
    *,
    model_name: str,
    num_nodes: int,
    edge_multiplier: int,
    device: str,
    model_seed: int,
    graph_seed: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    if model_name not in {"spatial_static", "spatial_dynamic", "static_egnn"}:
        raise ValueError("unsupported model_name")
    if isinstance(num_nodes, bool) or not isinstance(num_nodes, int) or num_nodes < 2:
        raise ValueError("num_nodes must be an integer at least two")
    if (
        isinstance(edge_multiplier, bool)
        or not isinstance(edge_multiplier, int)
        or edge_multiplier <= 0
        or edge_multiplier > num_nodes
    ):
        raise ValueError("edge_multiplier must be in [1, num_nodes]")
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be nonnegative and repeats must be positive")

    resolved_device = SCALING["_resolve_device"](device)
    node_feats, pos, batch = SCALING["_model_inputs"](
        num_nodes,
        device=resolved_device,
    )
    edge_index = None
    edge_hash = None
    if model_name == "static_egnn":
        edge_index = SCALING["seeded_exact_edge_index"](
            num_nodes,
            edge_multiplier=edge_multiplier,
            seed=graph_seed + num_nodes * 1_000_003 + edge_multiplier * 97,
            device=resolved_device,
        )
        edge_hash = SCALING["_edge_index_sha256"](edge_index)
    model = SCALING["_build_train_step_model"](
        model_name,
        device=resolved_device,
        model_seed=model_seed,
    )
    initial_state_sha256 = SCALING["_module_state_sha256"](model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    for _ in range(warmup):
        SCALING["_train_step_loss"](
            model=model,
            optimizer=optimizer,
            node_feats=node_feats,
            pos=pos,
            batch=batch,
            edge_index=edge_index,
            update=True,
        )
    SCALING["_sync"](resolved_device)
    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)

    activities = [ProfilerActivity.CPU]
    if resolved_device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        final_loss = None
        for _ in range(repeats):
            final_loss = _profiled_step(
                model=model,
                optimizer=optimizer,
                node_feats=node_feats,
                pos=pos,
                batch=batch,
                edge_index=edge_index,
            )
    SCALING["_sync"](resolved_device)

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    all_finite = bool(
        final_loss is not None
        and bool(torch.isfinite(final_loss).item())
        and gradients
        and all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients)
    )
    nonzero_elements = sum(
        int(torch.count_nonzero(gradient).item()) for gradient in gradients
    )
    records = [_event_record(event) for event in profiler.key_averages()]
    sort_key = (
        "device_time_total_us"
        if resolved_device.type == "cuda"
        else "cpu_time_total_us"
    )
    records.sort(key=lambda record: float(record[sort_key]), reverse=True)
    stages = {
        record["name"]: record
        for record in records
        if str(record["name"]).startswith("stage.")
    }
    operators = [
        record
        for record in records
        if not str(record["name"]).startswith("stage.")
    ][:50]
    result = {
        "schema_version": 1,
        "status": "completed" if all_finite and nonzero_elements > 0 else "failed",
        "model": model_name,
        "device": str(resolved_device),
        "dtype": "float32",
        "nodes": num_nodes,
        "edge_multiplier": edge_multiplier,
        "edge_index_supplied": edge_index is not None,
        "edge_index_sha256": edge_hash,
        "model_seed": model_seed,
        "graph_seed": graph_seed,
        "warmup": warmup,
        "profiled_steps": repeats,
        "initial_state_sha256": initial_state_sha256,
        "final_loss": float(final_loss.item()) if final_loss is not None else None,
        "gradient_validation": {
            "all_finite": all_finite,
            "nonzero_elements": nonzero_elements,
        },
        "peak_cuda_bytes": (
            int(torch.cuda.max_memory_allocated(resolved_device))
            if resolved_device.type == "cuda"
            else None
        ),
        "stages": stages,
        "operators": operators,
        "limitations": {
            "profiler_overhead_included": True,
            "operator_times_are_diagnostic_not_benchmark_latency": True,
            "post_registered_outcome_diagnostic": True,
        },
    }
    if not math.isfinite(float(result["final_loss"])):
        result["status"] = "failed"
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["spatial_static", "spatial_dynamic", "static_egnn"],
        required=True,
    )
    parser.add_argument("--nodes", type=int, default=8192)
    parser.add_argument("--edge-multiplier", type=int, default=128)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--model-seed", type=int, default=20260723)
    parser.add_argument("--graph-seed", type=int, default=20260723)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    result = run_train_step_profile(
        model_name=args.model,
        num_nodes=args.nodes,
        edge_multiplier=args.edge_multiplier,
        device=args.device,
        model_seed=args.model_seed,
        graph_seed=args.graph_seed,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
