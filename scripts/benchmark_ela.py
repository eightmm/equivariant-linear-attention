#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from equivariant_attention import ELA, ELABatch
from equivariant_attention.triton_ops import triton_available


def _dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
    }[name]


def _fixed_degree_edges(
    nodes: int,
    degree: int,
    device: torch.device,
) -> torch.Tensor:
    if not 0 < degree < nodes:
        raise ValueError("degree must satisfy 0 < degree < nodes")
    receiver = torch.arange(nodes, device=device).repeat_interleave(degree)
    offset = torch.arange(1, degree + 1, device=device).repeat(nodes)
    sender = (receiver + offset) % nodes
    return torch.stack([receiver, sender])


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(
    function: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[dict[str, Any], int | None]:
    for _ in range(warmup):
        function()
    _sync(device)
    samples: list[float] = []
    peak = 0 if device.type == "cuda" else None
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        function()
        _sync(device)
        samples.append((time.perf_counter() - start) * 1000.0)
        if peak is not None:
            peak = max(peak, torch.cuda.max_memory_allocated(device))
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }, peak


def _activate_local_outputs(model: ELA) -> None:
    with torch.no_grad():
        for layer in model.layers:
            for name in (
                "local_scalar_out",
                "local_odd_out",
                "local_polar_out",
                "local_axial_out",
                "local_even_tensor_out",
                "local_odd_tensor_out",
                "local_mass_out",
            ):
                module = getattr(layer, name)
                if hasattr(module, "weight"):
                    module.weight.normal_(mean=0.0, std=0.02)


def _functional_probe(
    model: ELA,
    prepared: ELABatch,
    *,
    backend: str,
) -> dict[str, torch.Tensor]:
    os.environ["ELA_KERNEL_BACKEND"] = backend
    features = prepared.node_irreps.detach().clone().requires_grad_(True)
    positions = prepared.positions.detach().clone().requires_grad_(True)
    batch = ELABatch(
        node_irreps=features,
        positions=positions,
        ptr=prepared.ptr,
        edge_index=prepared.edge_index,
        edge_relation_id=prepared.edge_relation_id,
        _prepared_graph=prepared._prepared_graph,
    )
    model.zero_grad(set_to_none=True)
    output = model.forward_prepared(batch)["node_irreps"]
    output.float().square().mean().backward()
    parameter = model.layers[0].local_scalar_out.weight
    if parameter.grad is None or features.grad is None or positions.grad is None:
        raise RuntimeError("functional probe gradients are missing")
    return {
        "output": output.detach().float().cpu(),
        "feature_gradient": features.grad.detach().float().cpu(),
        "position_gradient": positions.grad.detach().float().cpu(),
        "local_parameter_gradient": parameter.grad.detach().float().cpu(),
    }


def _profile_backend(
    model: ELA,
    prepared: ELABatch,
    *,
    backend: str,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    os.environ["ELA_KERNEL_BACKEND"] = backend
    model.eval()

    def inference() -> torch.Tensor:
        with torch.inference_mode():
            return model.forward_prepared(prepared)["node_irreps"]

    inference_timing, inference_peak = _measure(
        inference,
        device=device,
        warmup=warmup,
        repeats=repeats,
    )

    features = prepared.node_irreps.detach().clone().requires_grad_(True)
    positions = prepared.positions.detach().clone().requires_grad_(True)
    training_batch = ELABatch(
        node_irreps=features,
        positions=positions,
        ptr=prepared.ptr,
        edge_index=prepared.edge_index,
        edge_relation_id=prepared.edge_relation_id,
        _prepared_graph=prepared._prepared_graph,
    )
    model.train()

    def forward_backward() -> torch.Tensor:
        model.zero_grad(set_to_none=True)
        features.grad = None
        positions.grad = None
        output = model.forward_prepared(training_batch)["node_irreps"]
        output.float().square().mean().backward()
        return output

    training_timing, training_peak = _measure(
        forward_backward,
        device=device,
        warmup=warmup,
        repeats=repeats,
    )
    return {
        "inference": inference_timing,
        "inference_peak_allocated_bytes": inference_peak,
        "forward_backward": training_timing,
        "forward_backward_peak_allocated_bytes": training_peak,
    }


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max().item()) if left.numel() else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark canonical ELA PyTorch and Triton kernels"
    )
    parser.add_argument("--nodes", type=int, default=4096)
    parser.add_argument("--degree", type=int, default=32)
    parser.add_argument("--input-irreps", default="32x0e")
    parser.add_argument("--output-irreps", default="1x0e")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64", "bfloat16"],
        default="bfloat16" if torch.cuda.is_available() else "float64",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    model_dtype = (
        torch.float32
        if dtype == torch.bfloat16 and device.type == "cpu"
        else dtype
    )
    torch.manual_seed(0)
    model = ELA(
        input_irreps=args.input_irreps,
        output_irreps=args.output_irreps,
        width=args.width,
        depth=args.depth,
        cutoff=10.0,
    ).to(device=device, dtype=model_dtype)
    _activate_local_outputs(model)
    input_dim = model.config.input_layout.dim
    features = torch.randn(
        args.nodes,
        input_dim,
        device=device,
        dtype=model_dtype,
    )
    positions = torch.randn(args.nodes, 3, device=device, dtype=torch.float32)
    edges = _fixed_degree_edges(args.nodes, args.degree, device)
    prepared = model.prepare(
        ELABatch(
            node_irreps=features,
            positions=positions,
            edge_index=edges,
        )
    )

    original_policy = os.environ.get("ELA_KERNEL_BACKEND")
    try:
        torch_probe = _functional_probe(model, prepared, backend="torch")
        profiles = {
            "torch": _profile_backend(
                model,
                prepared,
                backend="torch",
                device=device,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        }
        equivalence = None
        triton_enabled = (
            device.type == "cuda"
            and triton_available()
            and dtype != torch.float64
        )
        if triton_enabled:
            triton_probe = _functional_probe(model, prepared, backend="triton")
            profiles["triton"] = _profile_backend(
                model,
                prepared,
                backend="triton",
                device=device,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            equivalence = {
                name: _max_abs(triton_probe[name], torch_probe[name])
                for name in torch_probe
            }
    finally:
        if original_policy is None:
            os.environ.pop("ELA_KERNEL_BACKEND", None)
        else:
            os.environ["ELA_KERNEL_BACKEND"] = original_policy

    payload = {
        "schema_version": 2,
        "experiment": "ela_kernel_backend",
        "device": str(device),
        "dtype": args.dtype,
        "input_irreps": str(model.config.input_layout),
        "output_irreps": str(model.config.output_layout),
        "input_dim": input_dim,
        "nodes": args.nodes,
        "edges": int(edges.shape[1]),
        "degree": args.degree,
        "width": args.width,
        "depth": args.depth,
        "prepared_graph": True,
        "neighbor_discovery_included": False,
        "triton_runtime_available": triton_available(),
        "triton_profiled": "triton" in profiles,
        "equivalence_max_abs": equivalence,
        "profiles": profiles,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
