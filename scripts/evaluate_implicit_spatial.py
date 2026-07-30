#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from equivariant_attention import (
    ImplicitGaussianSpatialKernel,
    ImplicitSpatialKernelConfig,
)


def _csv_floats(value: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value.split(",") if item.strip())
    if not result or any(item <= 0.0 for item in result):
        raise argparse.ArgumentTypeError(
            "expected comma-separated positive scales"
        )
    return result


def _topk_overlap(
    reference: torch.Tensor,
    approximation: torch.Tensor,
    k: int,
) -> float:
    nodes = reference.shape[0]
    if nodes <= 1:
        return 1.0
    k = min(k, nodes - 1)
    reference_index = reference.topk(k, dim=-1).indices
    approximate_index = approximation.topk(k, dim=-1).indices
    matches = (
        reference_index.unsqueeze(-1)
        == approximate_index.unsqueeze(-2)
    ).any(dim=-1)
    return float(matches.float().mean().item())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the edge-free feature kernel with dense references"
    )
    parser.add_argument("--nodes", type=int, default=256)
    parser.add_argument("--value-width", type=int, default=32)
    parser.add_argument("--scales", type=_csv_floats, default=(1.0, 2.0, 4.0))
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.nodes <= 1:
        raise ValueError("nodes must exceed one")
    if args.value_width <= 0:
        raise ValueError("value-width must be positive")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    positions = torch.randn(args.nodes, 3, device=device, dtype=torch.float64)
    values = torch.randn(
        args.nodes,
        args.value_width,
        device=device,
        dtype=torch.float64,
    )
    batch = torch.zeros(args.nodes, device=device, dtype=torch.long)
    kernel = ImplicitGaussianSpatialKernel(
        ImplicitSpatialKernelConfig(
            scales=args.scales,
            order=2,
            exclude_self=True,
            normalization="none",
        )
    ).to(device=device, dtype=torch.float64)

    context = kernel.prepare(positions, batch)
    approximate = context.features @ context.features.T
    approximate.fill_diagonal_(0.0)

    difference = positions[:, None, :] - positions[None, :, :]
    squared_distance = difference.square().sum(dim=-1)
    exact = torch.stack(
        [
            torch.exp(-squared_distance / (2.0 * scale**2))
            for scale in args.scales
        ],
        dim=0,
    ).mean(dim=0)
    exact.fill_diagonal_(0.0)

    exact_message = exact @ values
    approximate_message = kernel(values, positions, batch).output
    kernel_error = torch.linalg.vector_norm(approximate - exact) / torch.linalg.vector_norm(
        exact
    ).clamp_min(1e-12)
    message_error = torch.linalg.vector_norm(
        approximate_message - exact_message
    ) / torch.linalg.vector_norm(exact_message).clamp_min(1e-12)

    payload = {
        "schema_version": 1,
        "nodes": args.nodes,
        "value_width": args.value_width,
        "scales": list(args.scales),
        "feature_rank": kernel.feature_rank,
        "dense_reference_only": True,
        "kernel_relative_frobenius_error": float(kernel_error.item()),
        "message_relative_l2_error": float(message_error.item()),
        "topk_overlap": _topk_overlap(exact, approximate, args.topk),
        "topk": min(args.topk, args.nodes - 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
