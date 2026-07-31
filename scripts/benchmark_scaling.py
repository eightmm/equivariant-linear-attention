#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

from equivariant_attention import (
    EquivariantAttentionResidualConfig,
    EquivariantAttentionResiduals,
    EquivariantLinearAttention,
    EquivariantLinearAttentionConfig,
    prepare_3d_graph,
)
from equivariant_attention.implicit_spatial import (
    ImplicitGaussianSpatialKernel,
    ImplicitSpatialKernelConfig,
)
from equivariant_attention.scaling_contract import (
    estimate_attention_residuals,
    estimate_base_linear_attention,
    estimate_implicit_spatial_kernel,
    fit_log_log_slope,
)


def _csv_ints(value: str) -> list[int]:
    result = [int(item) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError(
            "expected comma-separated positive integers"
        )
    return result


def _csv_floats(value: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value.split(",") if item.strip())
    if not result or any(item <= 0.0 for item in result):
        raise argparse.ArgumentTypeError(
            "expected comma-separated positive floats"
        )
    return result


def _dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float64": torch.float64,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype {name!r}") from exc


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _batch(nodes: int, graphs: int, device: torch.device) -> torch.Tensor:
    if graphs > nodes or nodes % graphs:
        raise ValueError("nodes must be divisible by graphs")
    return torch.arange(graphs, device=device).repeat_interleave(nodes // graphs)


def _fixed_degree_edges(
    nodes: int,
    degree: int,
    graphs: int,
    device: torch.device,
) -> torch.Tensor:
    if graphs > nodes or nodes % graphs:
        raise ValueError("nodes must be divisible by graphs")
    nodes_per_graph = nodes // graphs
    if degree >= nodes_per_graph:
        raise ValueError("degree must be smaller than nodes per graph")

    local_receiver = torch.arange(nodes_per_graph, device=device).repeat_interleave(
        degree
    )
    offsets = torch.arange(1, degree + 1, device=device).repeat(nodes_per_graph)
    local_sender = (local_receiver + offsets) % nodes_per_graph
    graph_offset = torch.arange(graphs, device=device).repeat_interleave(
        local_receiver.numel()
    )
    receiver = local_receiver.repeat(graphs) + graph_offset * nodes_per_graph
    sender = local_sender.repeat(graphs) + graph_offset * nodes_per_graph
    return torch.stack([receiver, sender])


def _clear_gradients(tensors: Sequence[torch.Tensor]) -> None:
    for tensor in tensors:
        tensor.grad = None


def _measure(
    function: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
    backward: bool,
    gradient_tensors: Sequence[torch.Tensor] = (),
) -> tuple[float, int]:
    def invoke() -> torch.Tensor:
        if backward:
            return function()
        with torch.inference_mode():
            return function()

    for _ in range(warmup):
        result = invoke()
        if backward:
            result.float().square().mean().backward()
            _clear_gradients(gradient_tensors)
    _synchronize(device)

    samples: list[float] = []
    peak = 0
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        result = invoke()
        if backward:
            result.float().square().mean().backward()
        _synchronize(device)
        samples.append((time.perf_counter() - start) * 1000.0)
        if device.type == "cuda":
            peak = max(peak, torch.cuda.max_memory_allocated(device))
        if backward:
            _clear_gradients(gradient_tensors)
    return statistics.median(samples), peak


def _base_row(
    *,
    nodes: int,
    graphs: int,
    degree: int,
    layers: int,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
    backward: bool,
    include_graph_pack: bool,
) -> dict[str, Any]:
    config = EquivariantLinearAttentionConfig(
        input_irreps="16x0e",
        output_irreps="1x0e",
        hidden_dim=64,
        num_layers=layers,
        num_heads=4,
        local_rank=4,
        local_cutoff=6.0,
        num_rbf=16,
    )
    model = EquivariantLinearAttention(config).to(device=device, dtype=dtype)
    model.train(backward)
    features = torch.randn(
        nodes,
        16,
        device=device,
        dtype=dtype,
        requires_grad=backward,
    )
    positions = torch.randn(
        nodes,
        3,
        device=device,
        dtype=dtype,
        requires_grad=backward,
    )
    batch = _batch(nodes, graphs, device)
    edge_index = _fixed_degree_edges(nodes, degree, graphs, device)
    graph = prepare_3d_graph(batch, edge_index)
    gradient_tensors = (*model.parameters(), features, positions)

    def layer_only() -> torch.Tensor:
        return model(features, positions, graph)["node_irreps"]

    layer_ms, peak = _measure(
        layer_only,
        device=device,
        warmup=warmup,
        repeats=repeats,
        backward=backward,
        gradient_tensors=gradient_tensors,
    )
    graph_pack_ms = None
    if include_graph_pack:

        def with_pack() -> torch.Tensor:
            prepared = prepare_3d_graph(batch, edge_index)
            return model(features, positions, prepared)["node_irreps"]

        graph_pack_ms, peak_with_pack = _measure(
            with_pack,
            device=device,
            warmup=warmup,
            repeats=repeats,
            backward=backward,
            gradient_tensors=gradient_tensors,
        )
        peak = max(peak, peak_with_pack)

    estimate = estimate_base_linear_attention(
        nodes=nodes,
        edges=nodes * degree,
        layers=layers,
        channel_factor=64,
        edge_scaling="linear",
    )
    return {
        "mode": "base",
        "nodes": nodes,
        "graphs": graphs,
        "edges": nodes * degree,
        "degree": degree,
        "layers": layers,
        "blocks": None,
        "blocks_policy": None,
        "layer_only_ms": layer_ms,
        "with_graph_pack_ms": graph_pack_ms,
        "peak_allocated_bytes": peak,
        "arithmetic_proxy": estimate.arithmetic_proxy,
        "formula": estimate.formula,
    }


def _attnres_row(
    *,
    nodes: int,
    graphs: int,
    degree: int,
    layers: int,
    blocks: int,
    blocks_policy: str,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
    backward: bool,
    include_graph_pack: bool,
) -> dict[str, Any]:
    if blocks > layers:
        raise ValueError("AttnRes blocks must not exceed layers")
    if blocks_policy not in {"fixed", "equal_depth"}:
        raise ValueError("blocks_policy must be fixed or equal_depth")
    config = EquivariantAttentionResidualConfig(
        input_irreps="16x0e",
        output_irreps="1x0e",
        hidden_dim=64,
        num_layers=layers,
        num_heads=4,
        local_rank=4,
        local_cutoff=6.0,
        num_rbf=16,
        attention_residual_blocks=blocks,
    )
    model = EquivariantAttentionResiduals(config).to(device=device, dtype=dtype)
    model.train(backward)
    features = torch.randn(
        nodes,
        16,
        device=device,
        dtype=dtype,
        requires_grad=backward,
    )
    positions = torch.randn(
        nodes,
        3,
        device=device,
        dtype=dtype,
        requires_grad=backward,
    )
    batch = _batch(nodes, graphs, device)
    edge_index = _fixed_degree_edges(nodes, degree, graphs, device)
    graph = prepare_3d_graph(batch, edge_index)
    gradient_tensors = (*model.parameters(), features, positions)

    def layer_only() -> torch.Tensor:
        return model(features, positions, graph)["node_irreps"]

    layer_ms, peak = _measure(
        layer_only,
        device=device,
        warmup=warmup,
        repeats=repeats,
        backward=backward,
        gradient_tensors=gradient_tensors,
    )
    graph_pack_ms = None
    if include_graph_pack:

        def with_pack() -> torch.Tensor:
            prepared = prepare_3d_graph(batch, edge_index)
            return model(features, positions, prepared)["node_irreps"]

        graph_pack_ms, peak_with_pack = _measure(
            with_pack,
            device=device,
            warmup=warmup,
            repeats=repeats,
            backward=backward,
            gradient_tensors=gradient_tensors,
        )
        peak = max(peak, peak_with_pack)

    estimate = estimate_attention_residuals(
        nodes=nodes,
        edges=nodes * degree,
        layers=layers,
        blocks=blocks,
        channel_factor=64,
        edge_scaling="linear",
        blocks_fixed_with_depth=blocks_policy == "fixed",
    )
    return {
        "mode": "attnres",
        "nodes": nodes,
        "graphs": graphs,
        "edges": nodes * degree,
        "degree": degree,
        "layers": layers,
        "blocks": blocks,
        "blocks_policy": blocks_policy,
        "layer_only_ms": layer_ms,
        "with_graph_pack_ms": graph_pack_ms,
        "peak_allocated_bytes": peak,
        "arithmetic_proxy": estimate.arithmetic_proxy,
        "formula": estimate.formula,
        "depth_linear_contract": estimate.depth_linear,
    }


def _implicit_row(
    *,
    nodes: int,
    graphs: int,
    scales: tuple[float, ...],
    value_width: int,
    applications: int,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
    backward: bool,
) -> dict[str, Any]:
    kernel = ImplicitGaussianSpatialKernel(
        ImplicitSpatialKernelConfig(
            scales=scales,
            order=2,
            exclude_self=True,
            normalization="one_plus_mass",
        )
    ).to(device=device, dtype=dtype)
    values = torch.randn(
        nodes,
        value_width,
        device=device,
        dtype=dtype,
        requires_grad=backward,
    )
    positions = torch.randn(
        nodes,
        3,
        device=device,
        dtype=dtype,
        requires_grad=backward,
    )
    batch = _batch(nodes, graphs, device)
    gradient_tensors = (*kernel.parameters(), values, positions)

    def run() -> torch.Tensor:
        value = values
        for _ in range(applications):
            value = kernel(value, positions, batch).output
        return value

    milliseconds, peak = _measure(
        run,
        device=device,
        warmup=warmup,
        repeats=repeats,
        backward=backward,
        gradient_tensors=gradient_tensors,
    )
    estimate = estimate_implicit_spatial_kernel(
        nodes=nodes,
        feature_rank=kernel.feature_rank,
        value_width=value_width,
        applications=applications,
        graphs=graphs,
        chunk_size=kernel.config.chunk_size,
    )
    return {
        "mode": "implicit",
        "nodes": nodes,
        "graphs": graphs,
        "edges": 0,
        "degree": 0,
        "layers": applications,
        "blocks": None,
        "blocks_policy": None,
        "layer_only_ms": milliseconds,
        "with_graph_pack_ms": None,
        "peak_allocated_bytes": peak,
        "feature_rank": kernel.feature_rank,
        "arithmetic_proxy": estimate.arithmetic_proxy,
        "formula": estimate.formula,
    }


def _fit_group(
    rows: list[dict[str, Any]],
    *,
    axis: str,
    size_key: str,
    grouping: Callable[[dict[str, Any]], tuple[Any, ...]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(grouping(row), []).append(row)
    output = []
    for key, group in groups.items():
        unique = {row[size_key] for row in group}
        if len(unique) < 2:
            continue
        group.sort(key=lambda item: item[size_key])
        fit = fit_log_log_slope(
            [row[size_key] for row in group],
            [row["layer_only_ms"] for row in group],
        )
        output.append(
            {
                "axis": axis,
                "group": list(key),
                "slope": fit.slope,
                "r_squared": fit.r_squared,
            }
        )
    return output


def _fits(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    node_fits = _fit_group(
        rows,
        axis="nodes",
        size_key="nodes",
        grouping=lambda row: (
            row["mode"],
            row["graphs"],
            row["degree"],
            row["layers"],
            row["blocks_policy"],
            row["blocks"],
        ),
    )

    def depth_group(row: dict[str, Any]) -> tuple[Any, ...]:
        block_key = row["blocks"] if row["blocks_policy"] == "fixed" else None
        return (
            row["mode"],
            row["graphs"],
            row["nodes"],
            row["degree"],
            row["blocks_policy"],
            block_key,
        )

    depth_fits = _fit_group(
        rows,
        axis="layers",
        size_key="layers",
        grouping=depth_group,
    )
    degree_fits = _fit_group(
        [row for row in rows if row["mode"] != "implicit"],
        axis="edges",
        size_key="edges",
        grouping=lambda row: (
            row["mode"],
            row["graphs"],
            row["nodes"],
            row["layers"],
            row["blocks_policy"],
            row["blocks"],
        ),
    )
    return [*node_fits, *depth_fits, *degree_fits]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure conditional scaling of equivariant linear attention"
    )
    parser.add_argument("--modes", default="base,attnres,implicit")
    parser.add_argument(
        "--nodes",
        type=_csv_ints,
        default=[256, 512, 1024, 2048],
    )
    parser.add_argument("--graphs", type=_csv_ints, default=[1])
    parser.add_argument("--depths", type=_csv_ints, default=[4, 8, 16])
    parser.add_argument("--blocks", type=_csv_ints, default=[4, 8])
    parser.add_argument("--degrees", type=_csv_ints, default=[32])
    parser.add_argument(
        "--scales",
        type=_csv_floats,
        default=(1.0, 2.0, 4.0),
    )
    parser.add_argument("--value-width", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float64"],
        default="float32",
    )
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--include-graph-pack", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    modes = {item.strip() for item in args.modes.split(",") if item.strip()}
    unknown = modes - {"base", "attnres", "implicit"}
    if unknown:
        raise ValueError(f"unknown modes: {sorted(unknown)}")
    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    rows: list[dict[str, Any]] = []

    for nodes in args.nodes:
        for graphs in args.graphs:
            if graphs > nodes or nodes % graphs:
                continue
            nodes_per_graph = nodes // graphs
            for depth in args.depths:
                if "implicit" in modes:
                    rows.append(
                        _implicit_row(
                            nodes=nodes,
                            graphs=graphs,
                            scales=args.scales,
                            value_width=args.value_width,
                            applications=depth,
                            device=device,
                            dtype=dtype,
                            warmup=args.warmup,
                            repeats=args.repeats,
                            backward=args.backward,
                        )
                    )
                for degree in args.degrees:
                    if degree >= nodes_per_graph:
                        continue
                    if "base" in modes:
                        rows.append(
                            _base_row(
                                nodes=nodes,
                                graphs=graphs,
                                degree=degree,
                                layers=depth,
                                device=device,
                                dtype=dtype,
                                warmup=args.warmup,
                                repeats=args.repeats,
                                backward=args.backward,
                                include_graph_pack=args.include_graph_pack,
                            )
                        )
                    if "attnres" in modes:
                        for blocks in args.blocks:
                            if blocks > depth:
                                continue
                            rows.append(
                                _attnres_row(
                                    nodes=nodes,
                                    graphs=graphs,
                                    degree=degree,
                                    layers=depth,
                                    blocks=blocks,
                                    blocks_policy="fixed",
                                    device=device,
                                    dtype=dtype,
                                    warmup=args.warmup,
                                    repeats=args.repeats,
                                    backward=args.backward,
                                    include_graph_pack=args.include_graph_pack,
                                )
                            )
                        rows.append(
                            _attnres_row(
                                nodes=nodes,
                                graphs=graphs,
                                degree=degree,
                                layers=depth,
                                blocks=depth,
                                blocks_policy="equal_depth",
                                device=device,
                                dtype=dtype,
                                warmup=args.warmup,
                                repeats=args.repeats,
                                backward=args.backward,
                                include_graph_pack=args.include_graph_pack,
                            )
                        )

    payload = {
        "schema_version": 3,
        "device": str(device),
        "dtype": args.dtype,
        "backward": args.backward,
        "neighbor_discovery_included": False,
        "graph_pack_included": args.include_graph_pack,
        "rows": rows,
        "fits": _fits(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
