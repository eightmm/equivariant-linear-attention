"""Microbenchmark the single factorized-moment attention implementation."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

import torch

from equivariant_attention import EquivariantAttention, EquivariantAttentionConfig, prepare_for_inference


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype)
    print("graphs,nodes_per_graph,total_nodes,pass,ms,peak_mem_mib,implementation")
    for graphs in args.graphs:
        for nodes_per_graph in args.nodes_per_graph:
            for backward in (False, True):
                ms, memory = benchmark(
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
                pass_name = "forward_backward" if backward else "forward"
                total_nodes = graphs * nodes_per_graph
                print(
                    f"{graphs},{nodes_per_graph},{total_nodes},{pass_name},"
                    f"{ms:.3f},{memory:.1f},factorized_moment"
                )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--graphs", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--nodes-per-graph", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--dtype", default="float32", choices=["float32", "bf16", "float64"])
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    return parser.parse_args(argv)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but unavailable")
    return torch.device(name)


def benchmark(
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
    model = EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=32,
            hidden_irreps="64x0e + 4x1o",
            output_irreps="1x0e + 1x1o + 1x2e",
            num_layers=3,
            num_heads=4,
        )
    ).to(device=device, dtype=dtype)
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
    memory_mib = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
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
