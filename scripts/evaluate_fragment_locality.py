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


def _c2_cutoff_matrix(positions: torch.Tensor, cutoff: float) -> torch.Tensor:
    displacement = positions[:, None, :] - positions[None, :, :]
    ratio = displacement.square().sum(dim=-1) / cutoff**2
    inside = ratio < 1.0
    u = ratio.clamp(min=0.0, max=1.0)
    result = 1.0 - 10.0 * u.pow(3) + 15.0 * u.pow(4) - 6.0 * u.pow(5)
    result = torch.where(inside, result, torch.zeros_like(result))
    result.fill_diagonal_(0.0)
    return result


def _relative(actual: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        (
            torch.linalg.vector_norm(actual - reference)
            / torch.linalg.vector_norm(reference).clamp_min(1e-12)
        ).item()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure fragment locality and centroid-induced kernel drift"
    )
    parser.add_argument("--base-nodes", type=int, default=64)
    parser.add_argument("--fragment-nodes", type=int, default=16)
    parser.add_argument("--fragment-distance", type=float, default=20.0)
    parser.add_argument("--value-width", type=int, default=16)
    parser.add_argument("--cutoff", type=float, default=3.0)
    parser.add_argument("--scales", type=_csv_floats, default=(2.0, 4.0, 8.0))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for name in ("base_nodes", "fragment_nodes", "value_width"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.fragment_distance <= 0.0 or args.cutoff <= 0.0:
        raise ValueError("distances must be positive")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    base_positions = torch.randn(
        args.base_nodes,
        3,
        device=device,
        dtype=torch.float64,
    )
    base_values = torch.randn(
        args.base_nodes,
        args.value_width,
        device=device,
        dtype=torch.float64,
    )
    fragment_positions = torch.randn(
        args.fragment_nodes,
        3,
        device=device,
        dtype=torch.float64,
    )
    fragment_positions[:, 0] += args.fragment_distance
    zero_fragment_values = torch.zeros(
        args.fragment_nodes,
        args.value_width,
        device=device,
        dtype=torch.float64,
    )
    random_fragment_values = torch.randn_like(zero_fragment_values)

    kernel = ImplicitGaussianSpatialKernel(
        ImplicitSpatialKernelConfig(
            scales=args.scales,
            order=2,
            exclude_self=True,
            normalization="none",
        )
    ).to(device=device, dtype=torch.float64)

    base_batch = torch.zeros(args.base_nodes, device=device, dtype=torch.long)
    base_context = kernel.prepare(base_positions, base_batch)
    base_kernel = base_context.features @ base_context.features.T
    base_kernel.fill_diagonal_(0.0)
    base_message = kernel(base_values, base_positions, base_batch).output

    combined_positions = torch.cat([base_positions, fragment_positions], dim=0)
    combined_batch = torch.zeros(
        combined_positions.shape[0],
        device=device,
        dtype=torch.long,
    )
    combined_context = kernel.prepare(combined_positions, combined_batch)
    combined_kernel = combined_context.features @ combined_context.features.T
    combined_kernel.fill_diagonal_(0.0)
    original_pair_kernel_after_fragment = combined_kernel[
        : args.base_nodes,
        : args.base_nodes,
    ]

    zero_values = torch.cat([base_values, zero_fragment_values], dim=0)
    random_values = torch.cat([base_values, random_fragment_values], dim=0)
    zero_fragment_message = kernel(
        zero_values,
        combined_positions,
        combined_batch,
    ).output[: args.base_nodes]
    random_fragment_message = kernel(
        random_values,
        combined_positions,
        combined_batch,
    ).output[: args.base_nodes]

    explicit_base = _c2_cutoff_matrix(base_positions, args.cutoff)
    explicit_combined = _c2_cutoff_matrix(combined_positions, args.cutoff)
    explicit_original_after_fragment = explicit_combined[
        : args.base_nodes,
        : args.base_nodes,
    ]
    explicit_message = explicit_base @ base_values
    explicit_zero_fragment_message = (
        explicit_combined @ zero_values
    )[: args.base_nodes]

    payload = {
        "schema_version": 1,
        "base_nodes": args.base_nodes,
        "fragment_nodes": args.fragment_nodes,
        "fragment_distance": args.fragment_distance,
        "cutoff": args.cutoff,
        "scales": list(args.scales),
        "implicit_original_pair_kernel_relative_drift": _relative(
            original_pair_kernel_after_fragment,
            base_kernel,
        ),
        "implicit_zero_fragment_message_relative_drift": _relative(
            zero_fragment_message,
            base_message,
        ),
        "implicit_random_fragment_message_relative_drift": _relative(
            random_fragment_message,
            base_message,
        ),
        "explicit_original_pair_kernel_max_abs_drift": float(
            (explicit_original_after_fragment - explicit_base).abs().max().item()
        ),
        "explicit_zero_fragment_message_max_abs_drift": float(
            (explicit_zero_fragment_message - explicit_message).abs().max().item()
        ),
        "interpretation": (
            "Implicit drift includes truncation-origin/centroid dependence and "
            "soft long-range coupling; explicit compact-cutoff drift should be zero."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
