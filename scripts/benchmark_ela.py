#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from equivariant_linear_attention import ELA, ELABatch
from equivariant_linear_attention.kernels import kernel_backend, triton_available


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


def _measure_backends(
    functions: dict[str, Callable[[], torch.Tensor]],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, tuple[dict[str, Any], int | None]]:
    for function in functions.values():
        for _ in range(warmup):
            function()
    _sync(device)
    wall_samples = {name: [] for name in functions}
    event_samples = {name: [] for name in functions}
    peaks = {name: 0 if device.type == "cuda" else None for name in functions}
    names = tuple(functions)
    for repeat in range(repeats):
        order = names if repeat % 2 == 0 else tuple(reversed(names))
        for name in order:
            _sync(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
            start = time.perf_counter()
            if device.type == "cuda":
                start_event.record()
            functions[name]()
            if device.type == "cuda":
                end_event.record()
            _sync(device)
            wall_samples[name].append((time.perf_counter() - start) * 1000.0)
            if device.type == "cuda":
                event_samples[name].append(start_event.elapsed_time(end_event))
                peak = peaks[name]
                assert peak is not None
                peaks[name] = max(peak, torch.cuda.max_memory_allocated(device))

    results: dict[str, tuple[dict[str, Any], int | None]] = {}
    for name, samples in wall_samples.items():
        timing: dict[str, Any] = {
            "median_ms": statistics.median(samples),
            "min_ms": min(samples),
            "max_ms": max(samples),
            "samples_ms": samples,
        }
        cuda_samples = event_samples[name]
        if cuda_samples:
            timing["cuda_event_median_ms"] = statistics.median(cuda_samples)
            timing["cuda_event_min_ms"] = min(cuda_samples)
            timing["cuda_event_max_ms"] = max(cuda_samples)
            timing["cuda_event_samples_ms"] = cuda_samples
        results[name] = timing, peaks[name]
    return results


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
    with kernel_backend(backend):
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


def _profile_backends(
    model: ELA,
    prepared: ELABatch,
    *,
    backends: tuple[str, ...],
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, dict[str, Any]]:
    model.eval()

    def inference(backend: str) -> torch.Tensor:
        with torch.inference_mode(), kernel_backend(backend):
            return model.forward_prepared(prepared)["node_irreps"]

    inference_profiles = _measure_backends(
        {
            backend: lambda backend=backend: inference(backend)
            for backend in backends
        },
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

    def forward_backward(backend: str) -> torch.Tensor:
        model.zero_grad(set_to_none=True)
        features.grad = None
        positions.grad = None
        with kernel_backend(backend):
            output = model.forward_prepared(training_batch)["node_irreps"]
            output.float().square().mean().backward()
        return output

    training_profiles = _measure_backends(
        {
            backend: lambda backend=backend: forward_backward(backend)
            for backend in backends
        },
        device=device,
        warmup=warmup,
        repeats=repeats,
    )
    return {
        backend: {
            "inference": inference_profiles[backend][0],
            "inference_peak_allocated_bytes": inference_profiles[backend][1],
            "forward_backward": training_profiles[backend][0],
            "forward_backward_peak_allocated_bytes": training_profiles[backend][1],
        }
        for backend in backends
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
        torch.float32 if dtype == torch.bfloat16 and device.type == "cpu" else dtype
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

    torch_probe = _functional_probe(model, prepared, backend="torch")
    equivalence = None
    triton_enabled = (
        device.type == "cuda" and triton_available() and dtype != torch.float64
    )
    backends = ("torch", "triton") if triton_enabled else ("torch",)
    if triton_enabled:
        triton_probe = _functional_probe(model, prepared, backend="triton")
        equivalence = {
            name: _max_abs(triton_probe[name], torch_probe[name])
            for name in torch_probe
        }
    profiles = _profile_backends(
        model,
        prepared,
        backends=backends,
        device=device,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    payload = {
        "schema_version": 3,
        "experiment": "ela_kernel_backend",
        "device": str(device),
        "dtype": args.dtype,
        "model_dtype": str(model_dtype).removeprefix("torch."),
        "csr_payload_dtype": (
            "float64" if model_dtype == torch.float64 else "float32"
        ),
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
        "warmup": args.warmup,
        "repeats": args.repeats,
        "backend_ordering": "alternating",
        "cold_compile_included": False,
        "timing_modes": ["wall", "cuda_event"]
        if device.type == "cuda"
        else ["wall"],
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
