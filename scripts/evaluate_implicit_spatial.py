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


def _relative_error(actual: torch.Tensor, reference: torch.Tensor) -> float:
    value = torch.linalg.vector_norm(actual - reference) / torch.linalg.vector_norm(
        reference
    ).clamp_min(1e-12)
    return float(value.item())


def _c2_cutoff(squared_distance: torch.Tensor, cutoff: float) -> torch.Tensor:
    ratio = squared_distance / cutoff**2
    inside = ratio < 1.0
    u = ratio.clamp(min=0.0, max=1.0)
    value = 1.0 - 10.0 * u.pow(3) + 15.0 * u.pow(4) - 6.0 * u.pow(5)
    return torch.where(inside, value, torch.zeros_like(value))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the edge-free feature kernel with dense references"
    )
    parser.add_argument("--nodes", type=int, default=256)
    parser.add_argument("--value-width", type=int, default=32)
    parser.add_argument("--scales", type=_csv_floats, default=(1.0, 2.0, 4.0))
    parser.add_argument("--cutoff", type=float, default=3.0)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.nodes <= 1:
        raise ValueError("nodes must exceed one")
    if args.value_width <= 0:
        raise ValueError("value-width must be positive")
    if args.cutoff <= 0.0:
        raise ValueError("cutoff must be positive")

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
    exact_gaussian = torch.stack(
        [
            torch.exp(-squared_distance / (2.0 * scale**2))
            for scale in args.scales
        ],
        dim=0,
    ).mean(dim=0)
    exact_gaussian.fill_diagonal_(0.0)

    cutoff_kernel = _c2_cutoff(squared_distance, args.cutoff)
    cutoff_kernel.fill_diagonal_(0.0)

    approximate_message = kernel(values, positions, batch).output
    gaussian_message = exact_gaussian @ values
    cutoff_message = cutoff_kernel @ values

    payload = {
        "schema_version": 2,
        "nodes": args.nodes,
        "value_width": args.value_width,
        "scales": list(args.scales),
        "cutoff": args.cutoff,
        "feature_rank": kernel.feature_rank,
        "dense_reference_only": True,
        "gaussian_kernel_relative_frobenius_error": _relative_error(
            approximate,
            exact_gaussian,
        ),
        "gaussian_message_relative_l2_error": _relative_error(
            approximate_message,
            gaussian_message,
        ),
        "gaussian_topk_overlap": _topk_overlap(
            exact_gaussian,
            approximate,
            args.topk,
        ),
        "cutoff_kernel_relative_frobenius_error": _relative_error(
            approximate,
            cutoff_kernel,
        ),
        "cutoff_message_relative_l2_error": _relative_error(
            approximate_message,
            cutoff_message,
        ),
        "cutoff_topk_overlap": _topk_overlap(
            cutoff_kernel,
            approximate,
            args.topk,
        ),
        "topk": min(args.topk, args.nodes - 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
