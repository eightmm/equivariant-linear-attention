"""Microbenchmark the single factorized-moment attention implementation."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

import torch

from equivariant_attention import (
    EquivariantAttention,
    EquivariantAttentionConfig,
    prepare_for_inference,
)


BENCHMARK_COLUMNS = (
    "graphs",
    "nodes_per_graph",
    "total_nodes",
    "pass",
    "ms",
    "peak_mem_mib",
    "implementation",
    "routing",
    "local_head_counts",
    "local_cutoff",
    "num_rbf",
    "memory_count",
    "memory_interaction",
    "memory_assignment_temperature",
    "memory_assignment_scale",
    "memory_interaction_cutoff",
    "radial_trace",
    "dtype",
    "compiled",
    "compile_mode",
)


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype)
    config = benchmark_config(args)
    print(",".join(BENCHMARK_COLUMNS))
    for graphs in args.graphs:
        for nodes_per_graph in args.nodes_per_graph:
            for backward in (False, True):
                ms, memory = benchmark(
                    config=config,
                    graphs=graphs,
                    nodes_per_graph=nodes_per_graph,
                    device=device,
                    dtype=dtype,
                    compile_model=args.compile,
                    compile_mode=args.compile_mode,
                    backward=backward,
                    iters=args.iters,
                    warmup=args.warmup,
                )
                row = benchmark_row(
                    args=args,
                    config=config,
                    graphs=graphs,
                    nodes_per_graph=nodes_per_graph,
                    backward=backward,
                    elapsed_ms=ms,
                    memory_mib=memory,
                )
                print(",".join(str(value) for value in row))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--graphs", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--nodes-per-graph", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument(
        "--dtype", default="float32", choices=["float32", "bf16", "float64"]
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--routing", choices=["ggg", "lgl", "lll"], default="ggg")
    parser.add_argument("--local-cutoff", type=float, default=2.5)
    parser.add_argument("--num-rbf", type=int, default=16)
    parser.add_argument("--memory-count", type=int, choices=[1, 4, 8], default=1)
    parser.add_argument("--memory-interaction", action="store_true")
    parser.add_argument("--memory-assignment-temperature", type=float, default=1.0)
    parser.add_argument("--memory-assignment-scale", type=float, default=2.5)
    parser.add_argument("--memory-interaction-cutoff", type=float, default=2.5)
    parser.add_argument("--radial-trace", action="store_true")
    return parser.parse_args(argv)


def benchmark_config(args: argparse.Namespace) -> EquivariantAttentionConfig:
    return EquivariantAttentionConfig(
        node_dim=32,
        hidden_irreps="64x0e + 4x1o",
        output_irreps="1x0e + 1x1o + 1x2e",
        num_layers=3,
        num_heads=4,
        local_head_counts=_routing_head_counts(args.routing),
        local_cutoff=args.local_cutoff,
        num_rbf=args.num_rbf,
        global_memory_count=args.memory_count,
        use_memory_interaction=args.memory_interaction,
        memory_assignment_temperature=args.memory_assignment_temperature,
        memory_assignment_scale=args.memory_assignment_scale,
        memory_interaction_cutoff=args.memory_interaction_cutoff,
        use_radial_trace=args.radial_trace,
    )


def benchmark_row(
    *,
    args: argparse.Namespace,
    config: EquivariantAttentionConfig,
    graphs: int,
    nodes_per_graph: int,
    backward: bool,
    elapsed_ms: float,
    memory_mib: float,
) -> tuple[object, ...]:
    local_head_counts = config.local_head_counts
    if local_head_counts is None:
        local_head_counts = (0,) * config.num_layers
    return (
        graphs,
        nodes_per_graph,
        graphs * nodes_per_graph,
        "forward_backward" if backward else "forward",
        f"{elapsed_ms:.3f}",
        f"{memory_mib:.1f}",
        "factorized_moment",
        args.routing,
        "|".join(str(value) for value in local_head_counts),
        config.local_cutoff,
        config.num_rbf,
        config.global_memory_count,
        config.use_memory_interaction,
        config.memory_assignment_temperature,
        config.memory_assignment_scale,
        config.memory_interaction_cutoff,
        config.use_radial_trace,
        args.dtype,
        args.compile and not backward,
        args.compile_mode,
    )


def _routing_head_counts(routing: str) -> tuple[int, int, int]:
    if routing == "ggg":
        return (0, 0, 0)
    if routing == "lgl":
        return (4, 0, 4)
    if routing == "lll":
        return (4, 4, 4)
    raise ValueError(f"unknown routing preset: {routing}")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but unavailable")
    return torch.device(name)


def benchmark(
    config: EquivariantAttentionConfig,
    graphs: int,
    nodes_per_graph: int,
    device: torch.device,
    dtype: torch.dtype,
    compile_model: bool,
    compile_mode: str,
    backward: bool,
    iters: int,
    warmup: int,
) -> tuple[float, float]:
    if graphs <= 0 or nodes_per_graph <= 0:
        raise ValueError("graphs and nodes_per_graph must be positive")
    torch.manual_seed(123)
    model = EquivariantAttention(config).to(device=device, dtype=dtype)
    if compile_model and not backward:
        model = prepare_for_inference(
            model,
            device=device,
            dtype=dtype,
            compile_model=True,
            compile_mode=compile_mode,
        )
    total_nodes = graphs * nodes_per_graph
    node_feats = torch.randn(total_nodes, 32, device=device, dtype=dtype)
    geometry_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    pos = torch.randn(total_nodes, 3, device=device, dtype=geometry_dtype)
    batch = torch.arange(graphs, device=device).repeat_interleave(nodes_per_graph)

    for _ in range(warmup):
        run_step(model, node_feats, pos, batch, backward)
    sync(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(iters):
        run_step(model, node_feats, pos, batch, backward)
    sync(device)

    elapsed_ms = (time.perf_counter() - started) * 1000.0 / iters
    memory_mib = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda"
        else 0.0
    )
    return elapsed_ms, memory_mib


def run_step(
    model: torch.nn.Module,
    node_feats: torch.Tensor,
    pos: torch.Tensor,
    batch: torch.Tensor,
    backward: bool,
) -> None:
    if backward:
        model.zero_grad(set_to_none=True)
        outputs = model(node_feats, pos, batch=batch)
        loss = sum(value.float().square().mean() for value in outputs.values())
        loss.backward()
    else:
        with torch.no_grad():
            model(node_feats, pos, batch=batch)


def parse_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "float64":
        return torch.float64
    return torch.float32


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


if __name__ == "__main__":
    raise SystemExit(main())
