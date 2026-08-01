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

from equivariant_attention import ELA, collate_graphs


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(
    function: Callable[[], object],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, float | int]:
    for _ in range(warmup):
        function()
    _synchronize(device)

    samples: list[float] = []
    peak = 0
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        function()
        _synchronize(device)
        samples.append(1000.0 * (time.perf_counter() - start))
        if device.type == "cuda":
            peak = max(peak, torch.cuda.max_memory_allocated(device))
    return {
        "median_ms": statistics.median(samples),
        "p90_ms": statistics.quantiles(
            samples,
            n=10,
            method="inclusive",
        )[8]
        if len(samples) > 1
        else samples[0],
        "peak_allocated_bytes": peak,
    }


def _ring_edges(nodes: int, degree: int, *, device: torch.device) -> torch.Tensor:
    if degree <= 0 or degree > nodes:
        raise ValueError("degree must be in [1, nodes]")
    receiver = torch.arange(nodes, device=device).repeat_interleave(degree)
    offset = torch.arange(degree, device=device).repeat(nodes)
    sender = (receiver + offset) % nodes
    return torch.stack([receiver, sender])


def _batch_edges(
    graphs: int,
    nodes_per_graph: int,
    degree: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    local = _ring_edges(nodes_per_graph, degree, device=device)
    offsets = (
        torch.arange(graphs, device=device, dtype=torch.long)
        * nodes_per_graph
    )
    flat = torch.cat([local + offset for offset in offsets], dim=1)
    padded = local.unsqueeze(0).expand(graphs, -1, -1).clone()
    edge_mask = torch.ones(
        (graphs, local.shape[1]),
        device=device,
        dtype=torch.bool,
    )
    return flat, padded, edge_mask


def _dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
    }[name]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark ELA flat, padded, mapping, and graph-build paths"
    )
    parser.add_argument("--graphs", type=int, default=8)
    parser.add_argument("--nodes-per-graph", type=int, default=64)
    parser.add_argument("--node-dim", type=int, default=16)
    parser.add_argument("--degree", type=int, default=16)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64", "bfloat16"],
        default="float32",
    )
    parser.add_argument("--compile-prepared", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for name in (
        "graphs",
        "nodes_per_graph",
        "node_dim",
        "degree",
        "width",
        "depth",
        "warmup",
        "repeats",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.degree > args.nodes_per_graph:
        raise ValueError("degree must not exceed nodes_per_graph")

    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    geometry_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    torch.manual_seed(31)

    model = ELA.scalar(
        args.node_dim,
        width=args.width,
        depth=args.depth,
        cutoff=args.cutoff,
    ).to(device=device, dtype=dtype)
    model.eval()

    total_nodes = args.graphs * args.nodes_per_graph
    node_irreps = torch.randn(
        total_nodes,
        args.node_dim,
        device=device,
        dtype=dtype,
    )
    positions = torch.randn(
        total_nodes,
        3,
        device=device,
        dtype=geometry_dtype,
    )
    batch = torch.arange(
        args.graphs,
        device=device,
        dtype=torch.long,
    ).repeat_interleave(args.nodes_per_graph)
    edge_index, padded_edges, padded_edge_mask = _batch_edges(
        args.graphs,
        args.nodes_per_graph,
        args.degree,
        device=device,
    )
    padded_nodes = node_irreps.reshape(
        args.graphs,
        args.nodes_per_graph,
        args.node_dim,
    )
    padded_positions = positions.reshape(
        args.graphs,
        args.nodes_per_graph,
        3,
    )
    node_mask = torch.ones(
        (args.graphs, args.nodes_per_graph),
        device=device,
        dtype=torch.bool,
    )
    samples = [
        {
            "node_irreps": padded_nodes[index],
            "pos": padded_positions[index],
            "edge_index": padded_edges[index],
        }
        for index in range(args.graphs)
    ]
    mapping_batch = collate_graphs(samples)
    prepared = model.prepare_graph(
        positions,
        batch=batch,
        edge_index=edge_index,
    )

    results: dict[str, dict[str, float | int]] = {}
    with torch.inference_mode():
        results["build_radius_graph"] = _measure(
            lambda: model.prepare_graph(positions, batch=batch),
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        results["pack_supplied_edges"] = _measure(
            lambda: model.prepare_graph(
                positions,
                batch=batch,
                edge_index=edge_index,
            ),
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        results["prepared_hot_forward"] = _measure(
            lambda: model.forward_prepared(node_irreps, positions, prepared),
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        results["prepared_validated_forward"] = _measure(
            lambda: model(node_irreps, positions, prepared),
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        results["flat_supplied_edges_forward"] = _measure(
            lambda: model(
                node_irreps,
                positions,
                batch=batch,
                edge_index=edge_index,
            ),
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        results["flat_auto_radius_forward"] = _measure(
            lambda: model(node_irreps, positions, batch=batch),
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        results["padded_supplied_edges_forward"] = _measure(
            lambda: model(
                padded_nodes,
                padded_positions,
                mask=node_mask,
                edge_index=padded_edges,
                edge_mask=padded_edge_mask,
            ),
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        results["mapping_forward"] = _measure(
            lambda: model(mapping_batch),
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        )

    compiled_result: dict[str, float | int] | None = None
    if args.compile_prepared:
        compiled_forward = torch.compile(
            model.forward_prepared,
            mode="reduce-overhead",
        )
        with torch.inference_mode():
            compiled_result = _measure(
                lambda: compiled_forward(node_irreps, positions, prepared),
                device=device,
                warmup=max(args.warmup, 3),
                repeats=args.repeats,
            )

    prepared_ms = float(results["prepared_hot_forward"]["median_ms"])
    payload: dict[str, Any] = {
        "schema_version": 2,
        "device": str(device),
        "dtype": args.dtype,
        "graphs": args.graphs,
        "nodes_per_graph": args.nodes_per_graph,
        "total_nodes": total_nodes,
        "edges": int(edge_index.shape[1]),
        "degree": args.degree,
        "width": args.width,
        "depth": args.depth,
        "neighbor_discovery_in_model": {
            "prepared_hot_forward": False,
            "prepared_validated_forward": False,
            "flat_supplied_edges_forward": False,
            "flat_auto_radius_forward": True,
            "padded_supplied_edges_forward": False,
            "mapping_forward": False,
        },
        "results": results,
        "compiled_prepared_forward": compiled_result,
        "relative_to_prepared_hot": {
            name: float(result["median_ms"]) / prepared_ms
            for name, result in results.items()
            if name.endswith("forward")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
