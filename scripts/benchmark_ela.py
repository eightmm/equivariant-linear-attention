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

from equivariant_linear_attention import ELA, ELAGraph
from equivariant_linear_attention.batch import ELABatch
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


def _percentile(samples: list[float], quantile: float) -> float:
    """Return a linearly interpolated percentile without a large dependency."""

    if not samples:
        raise ValueError("at least one timing sample is required")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(samples)
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _timing_summary(samples: list[float]) -> dict[str, Any]:
    return {
        "median_ms": statistics.median(samples),
        "p50_ms": _percentile(samples, 0.50),
        "p90_ms": _percentile(samples, 0.90),
        "p95_ms": _percentile(samples, 0.95),
        "p99_ms": _percentile(samples, 0.99),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


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
        timing = _timing_summary(samples)
        cuda_samples = event_samples[name]
        if cuda_samples:
            cuda_timing = _timing_summary(cuda_samples)
            timing.update(
                {
                    f"cuda_event_{key}": value
                    for key, value in cuda_timing.items()
                }
            )
        results[name] = timing, peaks[name]
    return results


def _grid_positions(
    nodes: int,
    *,
    cutoff: float,
    device: torch.device,
) -> torch.Tensor:
    """Build deterministic, bounded-degree geometry for ingestion profiling."""

    side = max(1, int(round(nodes ** (1.0 / 3.0))))
    while side**3 < nodes:
        side += 1
    index = torch.arange(nodes, device=device, dtype=torch.long)
    coordinates = torch.stack(
        (
            index % side,
            (index // side) % side,
            index // (side * side),
        ),
        dim=-1,
    )
    return coordinates.to(dtype=torch.float32) * (0.75 * cutoff)


def _profile_end_to_end(
    *,
    input_irreps: str,
    output_irreps: str,
    width: int,
    depth: int,
    nodes: int,
    degree: int,
    cutoff: float,
    device: torch.device,
    model_dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    """Profile complete public ``ELAGraph -> ELA -> ELAGraph`` execution."""

    if nodes < 2:
        raise ValueError("end-to-end profiling requires at least two nodes")
    if not 0 < degree < nodes:
        raise ValueError("e2e degree must satisfy 0 < degree < e2e nodes")

    model = ELA(
        input_irreps=input_irreps,
        output_irreps=output_irreps,
        width=width,
        depth=depth,
        cutoff=cutoff,
    ).to(device=device, dtype=model_dtype)
    moving_model = ELA(
        input_irreps=input_irreps,
        output_irreps=output_irreps,
        width=width,
        depth=depth,
        cutoff=cutoff,
        update_positions=True,
    ).to(device=device, dtype=model_dtype)
    _activate_local_outputs(model)
    _activate_local_outputs(moving_model)
    _activate_coordinate_outputs(moving_model)
    model.eval()
    moving_model.eval()

    input_dim = model.config.input_layout.dim
    features = torch.randn(
        nodes,
        input_dim,
        device=device,
        dtype=model_dtype,
    )
    positions = _grid_positions(nodes, cutoff=cutoff, device=device)
    receiver_sender = _fixed_degree_edges(nodes, degree, device)
    # The private numerical benchmark is receiver-major. Public ELAGraph edges
    # are sender-to-receiver, so reverse the rows at the API boundary.
    public_edges = receiver_sender.flip(0).contiguous()

    cached_explicit_graph = ELAGraph(
        features,
        positions,
        edge_index=public_edges,
    ).assume_immutable()
    with torch.inference_mode():
        model(cached_explicit_graph)
    if (
        cached_explicit_graph._prepared_graph is None
        or cached_explicit_graph._packed_template is None
    ):
        raise RuntimeError("explicit cache priming did not attach a packed template")
    with torch.inference_mode():
        moving_probe = moving_model(ELAGraph(features, positions))
    if moving_probe.delta is None or not bool(moving_probe.delta.abs().max().item()):
        raise RuntimeError("moving-coordinate benchmark produced no displacement")

    def cold_automatic_radius() -> torch.Tensor:
        # Construction is intentionally inside the timed region: validation,
        # topology discovery, CSR preparation, numerical execution, and output
        # wrapping all belong to the public end-to-end cost.
        graph = ELAGraph(features, positions)
        with torch.inference_mode():
            return model(graph).x

    def prepared_cache_reuse() -> torch.Tensor:
        with torch.inference_mode():
            return model(cached_explicit_graph).x

    def cold_explicit_topology() -> torch.Tensor:
        graph = ELAGraph(features, positions, edge_index=public_edges)
        with torch.inference_mode():
            return model(graph).x

    def moving_coordinate_execution() -> torch.Tensor:
        graph = ELAGraph(features, positions)
        with torch.inference_mode():
            return moving_model(graph).x

    measurements = _measure_backends(
        {
            "cold_automatic_radius": cold_automatic_radius,
            "prepared_cache_reuse": prepared_cache_reuse,
            "cold_explicit_topology": cold_explicit_topology,
            "moving_coordinate_execution": moving_coordinate_execution,
        },
        device=device,
        warmup=warmup,
        repeats=repeats,
    )
    lane_contracts: dict[str, dict[str, Any]] = {
        "cold_automatic_radius": {
            "topology_source": "automatic_radius",
            "prepared_cache_reused": False,
            "neighbor_discovery_included": True,
            "coordinate_rebuild_included": False,
            "elagraph_construction_included": True,
        },
        "prepared_cache_reuse": {
            "topology_source": "explicit_edge_index_cache",
            "prepared_cache_reused": True,
            "packed_template_reused": True,
            "immutable_storage_assumed": True,
            "content_revalidation_included": False,
            "neighbor_discovery_included": False,
            "coordinate_rebuild_included": False,
            "elagraph_construction_included": False,
            "excluded_setup": (
                "one untimed explicit COO-to-CSR preparation and immutable seal"
            ),
        },
        "cold_explicit_topology": {
            "topology_source": "explicit_edge_index",
            "prepared_cache_reused": False,
            "neighbor_discovery_included": False,
            "coordinate_rebuild_included": False,
            "elagraph_construction_included": True,
            "coo_to_csr_included": True,
        },
        "moving_coordinate_execution": {
            "topology_source": "automatic_radius",
            "prepared_cache_reused": False,
            "neighbor_discovery_included": True,
            # The final coordinate update needs no topology for another layer.
            # Therefore a depth-one model moves coordinates but performs no
            # interstage radius refresh; deeper stacks refresh depth-1 times.
            "coordinate_rebuild_included": depth > 1,
            "coordinate_rebuilds_per_call": max(0, depth - 1),
            "elagraph_construction_included": True,
            "update_positions": True,
        },
    }
    profiles = {
        name: {
            **lane_contracts[name],
            "graph_ingestion_included": True,
            "public_execution_path": "ELAGraph -> ELA -> ELAGraph",
            "timing": timing,
            "peak_allocated_bytes": peak,
        }
        for name, (timing, peak) in measurements.items()
    }
    return {
        "nodes": nodes,
        "explicit_edges": int(public_edges.shape[1]),
        "explicit_degree": degree,
        "cutoff": cutoff,
        "graph_ingestion_included": True,
        "neighbor_discovery_included": True,
        "neighbor_discovery_lanes": [
            name
            for name, profile in profiles.items()
            if profile["neighbor_discovery_included"]
        ],
        "execution": "forward_inference",
        "peak_memory_metric": (
            "torch.cuda.max_memory_allocated"
            if device.type == "cuda"
            else "unavailable_on_cpu"
        ),
        "profiles": profiles,
    }


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


def _activate_coordinate_outputs(model: ELA) -> None:
    """Make the benchmark's coordinate-update lane exercise real movement."""

    if model.coordinate_head is None:
        raise ValueError("coordinate outputs require update_positions=True")
    with torch.no_grad():
        model.coordinate_head.base_weight.normal_(mean=0.0, std=0.02)


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
        output = model._forward_prepared(batch)["node_irreps"]
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
            return model._forward_prepared(prepared)["node_irreps"]

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
            output = model._forward_prepared(training_batch)["node_irreps"]
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
        "--include-end-to-end",
        action="store_true",
        help=(
            "also profile public model(graph) ingestion, radius discovery, "
            "prepared-cache reuse, explicit topology, and stagewise coordinate execution"
        ),
    )
    parser.add_argument(
        "--e2e-nodes",
        type=int,
        default=512,
        help="node count for the bounded end-to-end lanes",
    )
    parser.add_argument(
        "--e2e-degree",
        type=int,
        default=16,
        help="explicit-topology degree for the end-to-end lane",
    )
    parser.add_argument(
        "--e2e-cutoff",
        type=float,
        default=2.0,
        help="automatic-radius cutoff for the end-to-end lanes",
    )
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
    prepared = model._prepare_packed(
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
    end_to_end = None
    if args.include_end_to_end:
        end_to_end = _profile_end_to_end(
            input_irreps=args.input_irreps,
            output_irreps=args.output_irreps,
            width=args.width,
            depth=args.depth,
            nodes=args.e2e_nodes,
            degree=args.e2e_degree,
            cutoff=args.e2e_cutoff,
            device=device,
            model_dtype=model_dtype,
            warmup=args.warmup,
            repeats=args.repeats,
        )

    payload = {
        "schema_version": 4,
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
        "end_to_end": end_to_end,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
