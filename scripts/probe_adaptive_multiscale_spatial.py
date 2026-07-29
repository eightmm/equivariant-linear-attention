#!/usr/bin/env python3
"""Bounded CPU witness and resource probe for adaptive multiscale LGL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
from statistics import median
from time import perf_counter
from typing import Any

import torch

from equivariant_attention import EquivariantAttention, EquivariantAttentionConfig
import equivariant_attention.moment as moment


def _collision_witness() -> dict[str, float]:
    dtype = torch.float64
    root_five = torch.tensor(5.0, dtype=dtype).sqrt()
    cloud_a = torch.zeros(4, 3, dtype=dtype)
    cloud_b = torch.zeros(4, 3, dtype=dtype)
    cloud_a[:, 0] = (
        torch.tensor([-3.0, -1.0, 1.0, 3.0], dtype=dtype) / root_five
    )
    root_eight_fifths = torch.tensor(8.0 / 5.0, dtype=dtype).sqrt()
    root_two_fifths = torch.tensor(2.0 / 5.0, dtype=dtype).sqrt()
    cloud_b[:, 0] = torch.stack(
        [
            -root_eight_fifths,
            -root_two_fifths,
            root_two_fifths,
            root_eight_fifths,
        ]
    )
    scales = torch.tensor(moment._ADAPTIVE_SPATIAL_SCALES, dtype=dtype)
    zero_logits = torch.zeros(4, 1, scales.numel(), dtype=dtype)
    summary_a = moment._adaptive_multiscale_spatial_features(
        cloud_a,
        scales,
        zero_logits,
    ).sum(dim=0)
    summary_b = moment._adaptive_multiscale_spatial_features(
        cloud_b,
        scales,
        zero_logits,
    ).sum(dim=0)
    second_a = torch.einsum("ni,nj->ij", cloud_a, cloud_a)
    second_b = torch.einsum("ni,nj->ij", cloud_b, cloud_b)
    fourth_a = cloud_a[:, 0].pow(4).mean()
    fourth_b = cloud_b[:, 0].pow(4).mean()
    return {
        "centroid_max_error": float(
            (cloud_a.mean(dim=0) - cloud_b.mean(dim=0)).abs().max()
        ),
        "second_moment_max_error": float((second_a - second_b).abs().max()),
        "fourth_moment_a": float(fourth_a),
        "fourth_moment_b": float(fourth_b),
        "spatial_summary_distance": float(
            torch.linalg.vector_norm(summary_a - summary_b)
        ),
    }


def _receiver_regular_edges(nodes: int, edge_multiplier: int) -> torch.Tensor:
    if nodes <= 0:
        raise ValueError("nodes must be positive")
    if not 1 <= edge_multiplier <= nodes:
        raise ValueError("edge_multiplier must lie in [1, nodes]")
    receiver = torch.arange(nodes).repeat_interleave(edge_multiplier)
    offsets = torch.arange(edge_multiplier).repeat(nodes)
    sender = (receiver + offsets) % nodes
    return torch.stack([receiver, sender])


def _build_model(
    *,
    node_dim: int,
    hidden_dim: int,
    heads: int,
    adaptive: bool,
    seed: int,
) -> EquivariantAttention:
    if hidden_dim % heads:
        raise ValueError("hidden_dim must be divisible by heads")
    torch.manual_seed(seed)
    return EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=node_dim,
            hidden_irreps=f"{hidden_dim}x0e + {heads}x1o",
            output_irreps="1x0e",
            num_layers=3,
            num_heads=heads,
            local_head_counts=(heads, 0, heads),
            local_cutoff=5.0,
            use_gated_local_transport=True,
            use_grouped_invariant_normalization=True,
            use_adaptive_multiscale_spatial_kernel=adaptive,
        )
    ).float()


def _state_bytes(model: torch.nn.Module) -> int:
    return sum(value.numel() * value.element_size() for value in model.state_dict().values())


def _counterbalanced_forward_ms(
    baseline: EquivariantAttention,
    candidate: EquivariantAttention,
    node_feats: torch.Tensor,
    pos: torch.Tensor,
    batch: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    warmup: int,
    repeats: int,
) -> tuple[
    dict[str, float],
    dict[str, list[float]],
    dict[str, dict[str, torch.Tensor]],
    list[list[str]],
]:
    models = {"baseline": baseline, "candidate": candidate}
    for model in models.values():
        model.eval()
    samples = {name: [] for name in models}
    outputs: dict[str, dict[str, torch.Tensor]] = {}
    timing_order: list[list[str]] = []

    def order_for(index: int) -> tuple[str, str]:
        return (
            ("baseline", "candidate")
            if index % 2 == 0
            else ("candidate", "baseline")
        )

    with torch.no_grad():
        for index in range(warmup):
            for name in order_for(index):
                outputs[name] = models[name](
                    node_feats,
                    pos,
                    batch=batch,
                    edge_index=edge_index,
                    edge_index_is_validated=True,
                )
        for index in range(repeats):
            order = order_for(index)
            timing_order.append(list(order))
            for name in order:
                started = perf_counter()
                outputs[name] = models[name](
                    node_feats,
                    pos,
                    batch=batch,
                    edge_index=edge_index,
                    edge_index_is_validated=True,
                )
                samples[name].append(1000.0 * (perf_counter() - started))

    medians = {name: median(values) for name, values in samples.items()}
    return medians, samples, outputs, timing_order


def run_probe(
    *,
    nodes: int,
    edge_multiplier: int,
    hidden_dim: int,
    heads: int,
    warmup: int,
    repeats: int,
    threads: int,
    seed: int,
) -> dict[str, Any]:
    for name, value in (
        ("warmup", warmup),
        ("repeats", repeats),
        ("threads", threads),
    ):
        if value < (0 if name == "warmup" else 1):
            raise ValueError(f"{name} is out of range")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        node_dim = 8
        generator = torch.Generator().manual_seed(seed)
        node_feats = torch.randn(nodes, node_dim, generator=generator)
        pos = 0.2 * torch.randn(nodes, 3, generator=generator)
        batch = torch.zeros(nodes, dtype=torch.long)
        edge_index = _receiver_regular_edges(nodes, edge_multiplier)
        baseline = _build_model(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            heads=heads,
            adaptive=False,
            seed=seed,
        )
        candidate = _build_model(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            heads=heads,
            adaptive=True,
            seed=seed,
        )
        baseline_state = baseline.state_dict()
        candidate_state = candidate.state_dict()
        common_names = baseline_state.keys() & candidate_state.keys()
        common_state_equal = bool(common_names) and all(
            torch.equal(baseline_state[name], candidate_state[name])
            for name in common_names
        )
        medians, timing_samples, outputs, timing_order = (
            _counterbalanced_forward_ms(
                baseline,
                candidate,
                node_feats,
                pos,
                batch,
                edge_index,
                warmup=warmup,
                repeats=repeats,
            )
        )
    finally:
        torch.set_num_threads(previous_threads)

    baseline_ms = medians["baseline"]
    candidate_ms = medians["candidate"]
    baseline_output = outputs["baseline"]
    candidate_output = outputs["candidate"]
    baseline_bytes = _state_bytes(baseline)
    candidate_bytes = _state_bytes(candidate)
    latency_ratio = candidate_ms / baseline_ms
    state_byte_ratio = candidate_bytes / baseline_bytes
    adaptive_layers = [
        index
        for index, layer in enumerate(candidate.layers)
        if layer.adaptive_spatial_gate is not None
    ]
    finite_output = all(
        torch.isfinite(value).all().item() for value in candidate_output.values()
    )
    output_delta = torch.linalg.vector_norm(
        candidate_output["node_scalars"] - baseline_output["node_scalars"]
    )
    source_path = Path(moment.__file__)
    witness = _collision_witness()
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed",
        "device": "cpu",
        "seed": seed,
        "config": {
            "nodes": nodes,
            "edge_multiplier": edge_multiplier,
            "edges": int(edge_index.shape[1]),
            "hidden_dim": hidden_dim,
            "heads": heads,
            "warmup": warmup,
            "repeats": repeats,
            "threads": threads,
            "scales": list(moment._ADAPTIVE_SPATIAL_SCALES),
        },
        "candidate": {
            "adaptive_middle_layer": (
                adaptive_layers[0] if len(adaptive_layers) == 1 else None
            ),
            "adaptive_layer_count": len(adaptive_layers),
            "common_state_equal": common_state_equal,
            "finite_output": bool(finite_output),
            "node_scalar_delta_l2": float(output_delta),
        },
        "witness": witness,
        "resource": {
            "baseline_median_ms": baseline_ms,
            "candidate_median_ms": candidate_ms,
            "latency_ratio": latency_ratio,
            "baseline_samples_ms": timing_samples["baseline"],
            "candidate_samples_ms": timing_samples["candidate"],
            "timing_order": timing_order,
            "baseline_state_bytes": baseline_bytes,
            "candidate_state_bytes": candidate_bytes,
            "state_byte_ratio": state_byte_ratio,
        },
        "host": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_affinity": (
                sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None
            ),
        },
        "acceptance": {
            "witness_separated": witness["spatial_summary_distance"] > 1e-6,
            "latency_ratio_at_most_1_25": latency_ratio <= 1.25,
            "state_byte_ratio_at_most_1_25": state_byte_ratio <= 1.25,
        },
        "provenance": {
            "torch_version": torch.__version__,
            "moment_source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        },
    }
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=512)
    parser.add_argument("--edge-multiplier", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2807)
    parser.add_argument("--metrics-out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_probe(
        nodes=args.nodes,
        edge_multiplier=args.edge_multiplier,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        warmup=args.warmup,
        repeats=args.repeats,
        threads=args.threads,
        seed=args.seed,
    )
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
