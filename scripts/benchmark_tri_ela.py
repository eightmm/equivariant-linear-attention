"""Benchmark the public TriELA graph-to-graph execution path.

Each measured iteration constructs an :class:`ELAGraph` and runs the canonical
model, so validation/ingestion cost is included.  Forward and forward/backward
are reported separately after warmup.  CPU eager execution is the default;
CUDA and ``torch.compile`` are opt-in and never fall back silently.

A small CPU smoke run is::

    uv run python scripts/benchmark_tri_ela.py \
        --nodes 8 --width 16 --pair-width 8 --triangle-hidden 8 \
        --warmup 1 --repeats 1
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any

import torch
from benchmark_triangle import (
    emit_json,
    measure_operation,
    parse_choice_grid,
    parse_integer_grid,
    resolve_device,
    resolve_dtype,
)
from torch import nn


def _compile(module: nn.Module, backend: str) -> nn.Module:
    if backend == "eager":
        return module
    if backend != "compile":
        raise ValueError(f"unsupported benchmark backend: {backend}")
    try:
        return torch.compile(module, dynamic=False, fullgraph=False)
    except Exception as error:
        raise RuntimeError(
            "torch.compile was requested and failed; no eager fallback used"
        ) from error


def _row(
    measured: dict[str, float | int | None],
    *,
    phase: str,
    batch_size: int,
    nodes_per_graph: int,
) -> dict[str, float | int | None | str]:
    seconds = float(measured["seconds_per_step"])
    total_nodes = batch_size * nodes_per_graph
    return {
        "phase": phase,
        **measured,
        "graphs_per_second": batch_size / seconds,
        "nodes_per_second": total_nodes / seconds,
        "dense_pair_cells_per_second": batch_size * nodes_per_graph**2 / seconds,
    }


def _graph_factory(
    graph_type: type,
    x: torch.Tensor,
    pos: torch.Tensor,
    batch: torch.Tensor,
) -> Callable[[], object]:
    def ingest() -> object:
        return graph_type(x=x, pos=pos, batch=batch)

    return ingest


def _forward_operation(
    runner: nn.Module,
    ingest_graph: Callable[[], object],
) -> Callable[[], torch.Tensor]:
    def operation() -> torch.Tensor:
        with torch.no_grad():
            result = runner(ingest_graph())
        return result.x

    return operation


def _forward_backward_operation(
    runner: nn.Module,
    ingest_graph: Callable[[], object],
) -> Callable[[], torch.Tensor]:
    def operation() -> torch.Tensor:
        runner.zero_grad(set_to_none=True)
        result = runner(ingest_graph())
        loss = result.x.float().square().mean()
        loss.backward()
        return loss

    return operation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", default="32,64,128")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--output-dim", type=int, default=1)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--pair-width", type=int, default=16)
    parser.add_argument("--triangle-hidden", type=int, default=16)
    parser.add_argument("--num-stages", type=int, default=1)
    parser.add_argument("--pair-blocks-per-stage", type=int, default=1)
    parser.add_argument("--local-blocks-per-stage", type=int, default=1)
    parser.add_argument("--max-pair-tokens", type=int, default=512)
    parser.add_argument(
        "--backends",
        default="eager",
        help="comma-separated subset of eager,compile",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="CPU intra-op threads; 0 preserves the current PyTorch setting",
    )
    parser.add_argument("--output", help="optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nodes_grid = parse_integer_grid(args.nodes, name="nodes")
    backends = parse_choice_grid(
        args.backends,
        name="backends",
        choices=("eager", "compile"),
    )
    if "compile" in backends and args.warmup < 1:
        raise ValueError("compile benchmarks require at least one untimed warmup")
    positive = {
        "batch-size": args.batch_size,
        "input-dim": args.input_dim,
        "output-dim": args.output_dim,
        "width": args.width,
        "pair-width": args.pair_width,
        "triangle-hidden": args.triangle_hidden,
        "num-stages": args.num_stages,
        "pair-blocks-per-stage": args.pair_blocks_per_stage,
        "local-blocks-per-stage": args.local_blocks_per_stage,
        "max-pair-tokens": args.max_pair_tokens,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"these arguments must be positive: {', '.join(invalid)}")
    if max(nodes_grid) > args.max_pair_tokens:
        raise ValueError(
            "nodes exceed max-pair-tokens; raise the guard explicitly if intended"
        )
    if args.threads < 0:
        raise ValueError("threads must be nonnegative")
    if args.threads:
        torch.set_num_threads(args.threads)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device=device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # Imports are deliberately lazy: CLI discovery does not depend on an
    # installed editable package, while execution requires the exact public
    # surface and fails immediately if it is absent.
    from equivariant_linear_attention import ELAGraph, TriELA, TriELAConfig

    rows: list[dict[str, Any]] = []
    for nodes_per_graph in nodes_grid:
        total_nodes = args.batch_size * nodes_per_graph
        for backend in backends:
            torch.manual_seed(args.seed)
            model = TriELA(
                f"{args.input_dim}x0e",
                f"{args.output_dim}x0e",
                width=args.width,
                pair_width=args.pair_width,
                triangle_hidden=args.triangle_hidden,
                num_stages=args.num_stages,
                pair_blocks_per_stage=args.pair_blocks_per_stage,
                local_blocks_per_stage=args.local_blocks_per_stage,
                pair_dropout=0.0,
                max_pair_tokens=args.max_pair_tokens,
            ).to(device=device, dtype=dtype)
            if not isinstance(model.config, TriELAConfig):
                raise TypeError("TriELA.config must expose the public TriELAConfig")
            runner = _compile(model, backend)
            x = torch.randn(
                total_nodes,
                args.input_dim,
                device=device,
                dtype=dtype,
            )
            pos = torch.randn(total_nodes, 3, device=device, dtype=dtype)
            batch = torch.arange(args.batch_size, device=device).repeat_interleave(
                nodes_per_graph
            )

            ingest_graph = _graph_factory(ELAGraph, x, pos, batch)

            runner.eval()
            forward = measure_operation(
                _forward_operation(runner, ingest_graph),
                device=device,
                warmup=args.warmup,
                repeats=args.repeats,
            )

            runner.train()
            forward_backward = measure_operation(
                _forward_backward_operation(runner, ingest_graph),
                device=device,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            common: dict[str, Any] = {
                "nodes_per_graph": nodes_per_graph,
                "batch_size": args.batch_size,
                "total_nodes": total_nodes,
                "backend": backend,
                "parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "pair_state_lower_bound_bytes": (
                    args.batch_size
                    * nodes_per_graph**2
                    * args.pair_width
                    * torch.tensor([], dtype=dtype).element_size()
                ),
            }
            rows.append(
                {
                    **common,
                    **_row(
                        forward,
                        phase="forward",
                        batch_size=args.batch_size,
                        nodes_per_graph=nodes_per_graph,
                    ),
                }
            )
            rows.append(
                {
                    **common,
                    **_row(
                        forward_backward,
                        phase="forward_backward",
                        batch_size=args.batch_size,
                        nodes_per_graph=nodes_per_graph,
                    ),
                }
            )

    emit_json(
        {
            "benchmark": "tri_ela_end_to_end",
            "includes_graph_ingestion": True,
            "scaling_contract": {
                "pair_memory": "O(B*N_max^2*Cz)",
                "triangle_time": "O(B*N_max^3*Ch)",
                "linear_scaling_claim": False,
            },
            "model": {
                "input_irreps": f"{args.input_dim}x0e",
                "output_irreps": f"{args.output_dim}x0e",
                "width": args.width,
                "pair_width": args.pair_width,
                "triangle_hidden": args.triangle_hidden,
                "num_stages": args.num_stages,
                "pair_blocks_per_stage": args.pair_blocks_per_stage,
                "local_blocks_per_stage": args.local_blocks_per_stage,
                "max_pair_tokens": args.max_pair_tokens,
            },
            "environment": {
                "torch_version": torch.__version__,
                "device": str(device),
                "dtype": args.dtype,
                "cpu_threads": torch.get_num_threads(),
                "seed": args.seed,
                "warmup": args.warmup,
                "repeats": args.repeats,
            },
            "rows": rows,
        },
        args.output,
    )


if __name__ == "__main__":
    main()
