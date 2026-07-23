#!/usr/bin/env python3
"""Diagnostic profiler for the registered high-density same-edge cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile

from equivariant_attention._egnn_baseline import _StaticEGNNBaseline
from equivariant_attention.training import build_regression_model


SCALING = runpy.run_path(
    str(Path(__file__).parents[2] / "scripts" / "benchmark_sparse_scaling.py")
)
_model_inputs = SCALING["_model_inputs"]
_edge_index_sha256 = SCALING["_edge_index_sha256"]
_module_state_sha256 = SCALING["_module_state_sha256"]
seeded_exact_edge_index = SCALING["seeded_exact_edge_index"]


def _event_value(event: Any, name: str) -> float:
    return float(getattr(event, name, 0.0) or 0.0)


def _profile_model(
    model: torch.nn.Module,
    *,
    node_feats: torch.Tensor,
    pos: torch.Tensor,
    batch: torch.Tensor,
    edge_index: torch.Tensor,
    warmup: int,
    forwards: int,
) -> dict[str, Any]:
    with torch.inference_mode():
        for _ in range(warmup):
            model(
                node_feats,
                pos,
                batch=batch,
                edge_index=edge_index,
                edge_index_is_validated=True,
            )
        torch.cuda.synchronize()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=True,
        ) as profiler:
            for _ in range(forwards):
                model(
                    node_feats,
                    pos,
                    batch=batch,
                    edge_index=edge_index,
                    edge_index_is_validated=True,
                )
        torch.cuda.synchronize()

    events = list(profiler.key_averages())
    events.sort(
        key=lambda event: _event_value(event, "device_time_total"),
        reverse=True,
    )
    return {
        "profiled_forwards": forwards,
        "warmup_forwards": warmup,
        "aggregate_self_cpu_time_ms": sum(
            _event_value(event, "self_cpu_time_total") for event in events
        )
        / 1000.0,
        "aggregate_self_device_time_ms": sum(
            _event_value(event, "self_device_time_total") for event in events
        )
        / 1000.0,
        "unique_operator_observed": any("unique" in event.key for event in events),
        "top_device_events": [
            {
                "name": event.key,
                "calls": int(event.count),
                "device_time_total_ms": _event_value(
                    event,
                    "device_time_total",
                )
                / 1000.0,
                "self_device_time_total_ms": _event_value(
                    event,
                    "self_device_time_total",
                )
                / 1000.0,
                "self_cpu_time_total_ms": _event_value(
                    event,
                    "self_cpu_time_total",
                )
                / 1000.0,
            }
            for event in events[:25]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=8192)
    parser.add_argument("--edge-multiplier", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--model-seed", type=int, default=20260723)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--forwards", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the registered profiler")
    device = torch.device("cuda")
    torch.manual_seed(args.model_seed)
    candidate = (
        build_regression_model(
            node_dim=11,
            hidden_dim=64,
            num_layers=3,
            num_heads=4,
            local_head_counts=(4, 0, 4),
            local_cutoff=2.5,
            num_rbf=16,
            use_edge_conditioned_local_transport=True,
        )
        .to(device=device, dtype=torch.float32)
        .eval()
    )
    egnn = (
        _StaticEGNNBaseline(
            node_dim=11,
            hidden_dim=91,
            num_layers=3,
        )
        .to(device=device, dtype=torch.float32)
        .eval()
    )
    node_feats, pos, batch = _model_inputs(args.nodes, device=device)
    cell_seed = args.seed + args.nodes * 1_000_003 + args.edge_multiplier * 97
    edge_index = seeded_exact_edge_index(
        args.nodes,
        edge_multiplier=args.edge_multiplier,
        seed=cell_seed,
        device=device,
    )
    result = {
        "schema_version": 1,
        "device": torch.cuda.get_device_name(device),
        "nodes": args.nodes,
        "edge_multiplier": args.edge_multiplier,
        "candidate_edges_including_self": edge_index.shape[1],
        "cell_seed": cell_seed,
        "graph_seed": args.seed,
        "model_seed": args.model_seed,
        "graph_generator": "seeded_exact_receiver_regular_directed",
        "edge_index_sha256": _edge_index_sha256(edge_index),
        "model_state_sha256": {
            "ec_lgl": _module_state_sha256(candidate),
            "static_egnn": _module_state_sha256(egnn),
        },
        "edge_index_is_validated": True,
        "instrumentation_boundary": (
            "Profiler timings are diagnostic and do not replace synchronized "
            "benchmark medians."
        ),
        "models": {
            "ec_lgl": _profile_model(
                candidate,
                node_feats=node_feats,
                pos=pos,
                batch=batch,
                edge_index=edge_index,
                warmup=args.warmup,
                forwards=args.forwards,
            ),
            "static_egnn": _profile_model(
                egnn,
                node_feats=node_feats,
                pos=pos,
                batch=batch,
                edge_index=edge_index,
                warmup=args.warmup,
                forwards=args.forwards,
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
