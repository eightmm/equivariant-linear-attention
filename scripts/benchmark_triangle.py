"""Benchmark the exact dense gated triangle primitive.

The default grid follows the dense-reference validation contract.  It is
deliberately expensive: exact triangle multiplication stores ``O(N^2 Cz)``
pair state and performs ``O(N^3 Ch)`` work.  Use explicit smaller grids for a
smoke run, for example::

    uv run python scripts/benchmark_triangle.py \
        --nodes 8 --pair-widths 8 --warmup 1 --repeats 1

CPU eager execution is the default.  CUDA and ``torch.compile`` are used only
when explicitly requested; an unavailable device or failed compilation is an
error, never an implicit fallback.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def parse_integer_grid(value: str, *, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{name} must be comma-separated integers"
        ) from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(f"{name} values must be positive")
    return values


def parse_choice_grid(
    value: str,
    *,
    name: str,
    choices: tuple[str, ...],
) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(values) - set(choices))
    if not values or unknown:
        expected = ", ".join(choices)
        raise argparse.ArgumentTypeError(
            f"{name} must contain only {expected}; got {unknown or 'nothing'}"
        )
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"{name} must not contain duplicates")
    return values


def resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or an explicit cuda device")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested explicitly but is unavailable")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index {device.index} is unavailable")
    return device


def resolve_dtype(value: str, *, device: torch.device) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
    }
    dtype = mapping[value]
    if (
        dtype == torch.bfloat16
        and device.type == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("bfloat16 was requested but this CUDA device lacks support")
    return dtype


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_operation(
    operation: Callable[[], object],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, float | int | None]:
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be nonnegative and repeats must be positive")
    for _ in range(warmup):
        operation()
    synchronize(device)

    baseline_allocated: int | None = None
    baseline_reserved: int | None = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated = torch.cuda.memory_allocated(device)
        baseline_reserved = torch.cuda.memory_reserved(device)

    started = time.perf_counter()
    for _ in range(repeats):
        operation()
    synchronize(device)
    elapsed = time.perf_counter() - started

    peak_allocated: int | None = None
    peak_reserved: int | None = None
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
    return {
        "repeats": repeats,
        "total_seconds": elapsed,
        "seconds_per_step": elapsed / repeats,
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
    }


def emit_json(payload: dict[str, Any], output: str | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output is not None:
        Path(output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")


class _TriangleRunner(nn.Module):
    def __init__(self, module: nn.Module, *, checkpoint_enabled: bool) -> None:
        super().__init__()
        self.module = module
        self.checkpoint_enabled = checkpoint_enabled

    def forward(self, z: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        if self.checkpoint_enabled and torch.is_grad_enabled():
            return checkpoint(
                self.module,
                z,
                pair_mask,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return self.module(z, pair_mask)


def _compile(module: nn.Module, backend: str) -> nn.Module:
    if backend == "eager":
        return module
    if backend != "compile":
        raise ValueError(f"unsupported benchmark backend: {backend}")
    try:
        return torch.compile(module, dynamic=False, fullgraph=True)
    except Exception as error:
        raise RuntimeError(
            "torch.compile was requested and failed; no fallback used"
        ) from error


def _phase_row(
    measured: dict[str, float | int | None],
    *,
    phase: str,
    batch_size: int,
    nodes: int,
) -> dict[str, float | int | None | str]:
    seconds = float(measured["seconds_per_step"])
    return {
        "phase": phase,
        **measured,
        "pairs_per_second": batch_size * nodes * nodes / seconds,
        "contracted_triplets_per_second": batch_size * nodes**3 / seconds,
    }


def _forward_operation(
    runner: nn.Module,
    z: torch.Tensor,
    pair_mask: torch.Tensor,
) -> Callable[[], torch.Tensor]:
    def operation() -> torch.Tensor:
        with torch.no_grad():
            return runner(z, pair_mask)

    return operation


def _forward_backward_operation(
    runner: nn.Module,
    z: torch.Tensor,
    pair_mask: torch.Tensor,
) -> Callable[[], torch.Tensor]:
    def operation() -> torch.Tensor:
        runner.zero_grad(set_to_none=True)
        z.grad = None
        result = runner(z, pair_mask)
        result.float().square().mean().backward()
        return result

    return operation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", default="64,128,256,384,512")
    parser.add_argument("--pair-widths", default="32,64,128")
    parser.add_argument(
        "--hidden-width",
        type=int,
        default=0,
        help="triangle hidden width; 0 uses the current pair width",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--directions",
        default="outgoing,incoming",
        help="comma-separated subset of outgoing,incoming",
    )
    parser.add_argument(
        "--backends",
        default="eager",
        help="comma-separated subset of eager,compile",
    )
    parser.add_argument(
        "--checkpoints",
        default="off",
        help="comma-separated subset of off,on",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
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
    width_grid = parse_integer_grid(args.pair_widths, name="pair-widths")
    directions = parse_choice_grid(
        args.directions,
        name="directions",
        choices=("outgoing", "incoming"),
    )
    backends = parse_choice_grid(
        args.backends,
        name="backends",
        choices=("eager", "compile"),
    )
    checkpoints = parse_choice_grid(
        args.checkpoints,
        name="checkpoints",
        choices=("off", "on"),
    )
    if args.hidden_width < 0 or args.batch_size <= 0:
        raise ValueError("hidden-width must be nonnegative and batch-size positive")
    if "compile" in backends and args.warmup < 1:
        raise ValueError("compile benchmarks require at least one untimed warmup")
    if args.threads < 0:
        raise ValueError("threads must be nonnegative")
    if args.threads:
        torch.set_num_threads(args.threads)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device=device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # Lazy by design: `--help` remains useful while the new core is being
    # assembled, while an actual benchmark fails loudly if the public symbol
    # is missing.
    from equivariant_linear_attention.nn.triangle import (
        GatedTriangleMultiplication,
    )

    rows: list[dict[str, Any]] = []
    for nodes in nodes_grid:
        for pair_width in width_grid:
            hidden_width = args.hidden_width or pair_width
            for direction in directions:
                for backend in backends:
                    for checkpoint_mode in checkpoints:
                        torch.manual_seed(args.seed)
                        module = GatedTriangleMultiplication(
                            pair_width=pair_width,
                            hidden_width=hidden_width,
                            direction=direction,
                        ).to(device=device, dtype=dtype)
                        runner = _compile(
                            _TriangleRunner(
                                module,
                                checkpoint_enabled=checkpoint_mode == "on",
                            ),
                            backend,
                        )
                        z = torch.randn(
                            args.batch_size,
                            nodes,
                            nodes,
                            pair_width,
                            device=device,
                            dtype=dtype,
                        )
                        pair_mask = torch.ones(
                            args.batch_size,
                            nodes,
                            nodes,
                            device=device,
                            dtype=torch.bool,
                        )

                        runner.eval()

                        forward = measure_operation(
                            _forward_operation(runner, z, pair_mask),
                            device=device,
                            warmup=args.warmup,
                            repeats=args.repeats,
                        )

                        runner.train()
                        train_z = z.detach().requires_grad_(True)

                        forward_backward = measure_operation(
                            _forward_backward_operation(runner, train_z, pair_mask),
                            device=device,
                            warmup=args.warmup,
                            repeats=args.repeats,
                        )
                        common = {
                            "nodes": nodes,
                            "pair_width": pair_width,
                            "hidden_width": hidden_width,
                            "batch_size": args.batch_size,
                            "direction": direction,
                            "backend": backend,
                            "activation_checkpoint": checkpoint_mode == "on",
                        }
                        rows.append(
                            {
                                **common,
                                **_phase_row(
                                    forward,
                                    phase="forward",
                                    batch_size=args.batch_size,
                                    nodes=nodes,
                                ),
                            }
                        )
                        rows.append(
                            {
                                **common,
                                **_phase_row(
                                    forward_backward,
                                    phase="forward_backward",
                                    batch_size=args.batch_size,
                                    nodes=nodes,
                                ),
                            }
                        )

    emit_json(
        {
            "benchmark": "exact_dense_gated_triangle",
            "scaling_contract": {
                "pair_memory": "O(B*N^2*Cz)",
                "triangle_time": "O(B*N^3*Ch)",
                "linear_scaling_claim": False,
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
