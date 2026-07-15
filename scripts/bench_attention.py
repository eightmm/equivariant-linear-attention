"""Microbenchmark dense vs linear equivariant attention."""

from __future__ import annotations

import argparse
import time

import torch

from equivariant_attention import (
    EquivariantAttention,
    EquivariantAttentionConfig,
    EquivariantMomentAttention,
    EquivariantMomentAttentionConfig,
    RichEquivariantAttention,
    RichEquivariantAttentionConfig,
    prepare_for_inference,
)


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    print("mode,nodes,pass,ms,peak_mem_mib,backend")
    for mode in args.modes:
        for n_nodes in args.nodes:
            for backward in (False, True):
                ms, mem, backend = benchmark(
                    mode=mode,
                    n_nodes=n_nodes,
                    device=device,
                    dtype=parse_dtype(args.dtype),
                    compile_model=args.compile,
                    compile_mode=args.compile_mode,
                    backward=backward,
                    iters=args.iters,
                    warmup=args.warmup,
                )
                pass_name = "forward_backward" if backward else "forward"
                print(f"{mode},{n_nodes},{pass_name},{ms:.3f},{mem:.1f},{backend}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["linear", "linear_sh", "local_indexed", "local", "dense", "rich_linear", "rich_local", "moment_linear"],
        choices=["linear", "linear_sh", "local_indexed", "local", "dense", "rich_linear", "rich_local", "moment_linear"],
    )
    parser.add_argument("--nodes", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--dtype", default="float32", choices=["float32", "bf16", "float64"])
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but unavailable")
    return torch.device(name)


def benchmark(
    mode: str,
    n_nodes: int,
    device: torch.device,
    dtype: torch.dtype,
    compile_model: bool,
    compile_mode: str,
    backward: bool,
    iters: int,
    warmup: int,
) -> tuple[float, float, str]:
    torch.manual_seed(123)
    if mode == "moment_linear":
        return benchmark_moment(
            n_nodes=n_nodes,
            device=device,
            dtype=dtype,
            compile_model=compile_model,
            compile_mode=compile_mode,
            backward=backward,
            iters=iters,
            warmup=warmup,
        )
    if mode.startswith("rich_"):
        return benchmark_rich(
            mode=mode.removeprefix("rich_"),
            n_nodes=n_nodes,
            device=device,
            dtype=dtype,
            compile_model=compile_model,
            compile_mode=compile_mode,
            backward=backward,
            iters=iters,
            warmup=warmup,
        )
    attention_mode = "local" if mode == "local_indexed" else mode
    edge_dim = 4 if attention_mode == "dense" else 0
    model = EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=32,
            edge_dim=edge_dim,
            hidden_dim=64,
            num_layers=3,
            num_heads=4,
            attention_mode=attention_mode,  # type: ignore[arg-type]
            local_radius=4.0,
            max_neighbors=32,
        )
    ).to(device=device, dtype=dtype)
    if compile_model and not backward:
        model = prepare_for_inference(model, device=device, dtype=dtype, compile_model=True, compile_mode=compile_mode)
    node_feats = torch.randn(n_nodes, 32, device=device, dtype=dtype)
    pos = torch.randn(n_nodes, 3, device=device, dtype=dtype)
    batch = None
    edge_feats = torch.randn(n_nodes, n_nodes, edge_dim, device=device, dtype=dtype) if edge_dim > 0 else None
    neighbor_index = make_ring_neighbors(n_nodes, 32, device) if mode == "local_indexed" else None
    neighbor_mask = torch.ones_like(neighbor_index, dtype=torch.bool) if neighbor_index is not None else None

    for _ in range(warmup):
        run_step(model, node_feats, pos, edge_feats, batch, neighbor_index, neighbor_mask, backward)
    sync(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        run_step(model, node_feats, pos, edge_feats, batch, neighbor_index, neighbor_mask, backward)
    sync(device)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / iters
    mem_mib = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
    return elapsed_ms, mem_mib, model.geometry.active


def benchmark_rich(
    mode: str,
    n_nodes: int,
    device: torch.device,
    dtype: torch.dtype,
    compile_model: bool,
    compile_mode: str,
    backward: bool,
    iters: int,
    warmup: int,
) -> tuple[float, float, str]:
    model = RichEquivariantAttention(
        RichEquivariantAttentionConfig(
            node_dim=32,
            hidden_irreps="64x0e + 8x1o + 4x2e",
            output_irreps="1x0e + 1x1o + 1x2e",
            num_layers=3,
            num_heads=4,
            attention_mode=mode,  # type: ignore[arg-type]
        )
    ).to(device=device, dtype=dtype)
    if compile_model and not backward and mode == "linear":
        model = prepare_for_inference(model, device=device, dtype=dtype, compile_model=True, compile_mode=compile_mode)
    node_feats = torch.randn(n_nodes, 32, device=device, dtype=dtype)
    pos = torch.randn(n_nodes, 3, device=device, dtype=dtype)
    batch = None
    neighbor_index = make_ring_neighbors(n_nodes, 32, device) if mode == "local" else None
    neighbor_mask = torch.ones_like(neighbor_index, dtype=torch.bool) if neighbor_index is not None else None

    for _ in range(warmup):
        run_step(model, node_feats, pos, None, batch, neighbor_index, neighbor_mask, backward)
    sync(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        run_step(model, node_feats, pos, None, batch, neighbor_index, neighbor_mask, backward)
    sync(device)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / iters
    mem_mib = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
    return elapsed_ms, mem_mib, "cartesian_irreps"


def benchmark_moment(
    n_nodes: int,
    device: torch.device,
    dtype: torch.dtype,
    compile_model: bool,
    compile_mode: str,
    backward: bool,
    iters: int,
    warmup: int,
) -> tuple[float, float, str]:
    model = EquivariantMomentAttention(
        EquivariantMomentAttentionConfig(
            node_dim=32,
            hidden_irreps="64x0e + 4x1o",
            output_irreps="1x0e + 1x1o + 1x2e",
            num_layers=3,
            num_heads=4,
        )
    ).to(device=device, dtype=dtype)
    if compile_model and not backward:
        model = prepare_for_inference(model, device=device, dtype=dtype, compile_model=True, compile_mode=compile_mode)
    node_feats = torch.randn(n_nodes, 32, device=device, dtype=dtype)
    pos = torch.randn(n_nodes, 3, device=device, dtype=dtype)

    for _ in range(warmup):
        run_step(model, node_feats, pos, None, None, None, None, backward)
    sync(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        run_step(model, node_feats, pos, None, None, None, None, backward)
    sync(device)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / iters
    mem_mib = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
    return elapsed_ms, mem_mib, "moment_irreps"


def make_ring_neighbors(n_nodes: int, width: int, device: torch.device) -> torch.Tensor:
    offsets = torch.arange(width, device=device)
    base = torch.arange(n_nodes, device=device).unsqueeze(1)
    return (base + offsets.unsqueeze(0)) % n_nodes


def parse_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "float64":
        return torch.float64
    return torch.float32


def run_step(
    model: EquivariantAttention | RichEquivariantAttention | EquivariantMomentAttention,
    node_feats: torch.Tensor,
    pos: torch.Tensor,
    edge_feats: torch.Tensor | None,
    batch: torch.Tensor | None,
    neighbor_index: torch.Tensor | None,
    neighbor_mask: torch.Tensor | None,
    backward: bool,
) -> None:
    if backward:
        model.zero_grad(set_to_none=True)
        if isinstance(model, EquivariantMomentAttention):
            out = model(node_feats, pos, batch=batch)
        elif isinstance(model, RichEquivariantAttention):
            out = model(node_feats, pos, batch=batch, neighbor_index=neighbor_index, neighbor_mask=neighbor_mask)
        else:
            out = model(node_feats, pos, edge_feats=edge_feats, batch=batch, neighbor_index=neighbor_index, neighbor_mask=neighbor_mask)
        loss = sum(v.square().mean() for v in out.values() if torch.is_tensor(v))
        loss.backward()
    else:
        with torch.inference_mode():
            if isinstance(model, EquivariantMomentAttention):
                model(node_feats, pos, batch=batch)
            elif isinstance(model, RichEquivariantAttention):
                model(node_feats, pos, batch=batch, neighbor_index=neighbor_index, neighbor_mask=neighbor_mask)
            else:
                model(node_feats, pos, edge_feats=edge_feats, batch=batch, neighbor_index=neighbor_index, neighbor_mask=neighbor_mask)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


if __name__ == "__main__":
    raise SystemExit(main())
