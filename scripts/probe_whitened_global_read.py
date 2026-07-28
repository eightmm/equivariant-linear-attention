#!/usr/bin/env python3
"""Measure whether the whitened global read actually de-uniformizes the kernel.

The registered global diagnosis is that the exact factorized kernel is
numerically uniform, so the useful content of the global path collapses to the
graph mean. The whitened lane claims to fix the *metric* of the read rather than
the weights. That claim has a falsifiable prediction independent of accuracy:
at identical features, the equivalent attention rows must become materially more
dispersed than the incumbent kernel's rows.

This probe captures the real query/key tensors of an actual global stage by
intercepting the global message call during one forward pass, then reconstructs
both dense row distributions for the captured graph. It is bounded and CPU only.

    uv run python scripts/probe_whitened_global_read.py \
      artifacts/whitened-global-read/probe.json --dataset lba --complexes 2
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import equivariant_attention.moment as moment  # noqa: E402
from equivariant_attention.benchmarking import GraphSample, collate_graphs  # noqa: E402
from equivariant_attention.pdbbind import (  # noqa: E402
    ATOM3D_LBA_NODE_DIM,
    ATOM3D_LBA_REVISION,
    load_atom3d_lba_split_samples,
    segment_balanced_knn_edge_index,
)
from equivariant_attention.training import build_regression_model  # noqa: E402

LOCAL_CUTOFF_ANGSTROM = 6.0
INTRA_K = 16
CROSS_K = 16
RIDGE_GRID = (1.0, 0.5, 0.1, 0.05, 0.01, 0.005)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--dataset", choices=("lba", "synthetic"), default="lba")
    parser.add_argument("--data-root", type=Path, default=Path("data/atom3d_lba"))
    parser.add_argument("--complexes", type=int, default=2)
    parser.add_argument("--model-seed", type=int, default=41)
    args = parser.parse_args(argv)
    if args.complexes <= 0:
        parser.error("--complexes must be positive")
    return args


def _lba_batch(args: argparse.Namespace):
    samples = load_atom3d_lba_split_samples(
        args.data_root,
        split="train",
        revision=ATOM3D_LBA_REVISION,
        indices=tuple(range(args.complexes)),
    )
    prepared: list[GraphSample] = []
    for sample in samples:
        if sample.readout_mask is None:
            raise ValueError("ATOM3D-LBA sample requires a ligand readout mask")
        prepared.append(
            replace(
                sample,
                edge_index=segment_balanced_knn_edge_index(
                    sample.pos,
                    sample.readout_mask,
                    intra_k=INTRA_K,
                    cross_k=CROSS_K,
                    cutoff=LOCAL_CUTOFF_ANGSTROM,
                ),
            )
        )
    return collate_graphs(prepared)


def _synthetic_batch(args: argparse.Namespace):
    generator = torch.Generator().manual_seed(20260727)
    samples = []
    for index in range(args.complexes):
        nodes = 96
        samples.append(
            GraphSample(
                node_feats=torch.randn(
                    (nodes, ATOM3D_LBA_NODE_DIM), generator=generator
                ),
                pos=torch.randn((nodes, 3), generator=generator) * 6.0,
                target=torch.zeros(1),
                sample_id=f"synthetic-{index}",
                readout_mask=torch.zeros(nodes, dtype=torch.bool),
            )
        )
    return collate_graphs(samples)


def _row_statistics(weights: torch.Tensor) -> dict[str, float]:
    """Dispersion and entropy of one graph's equivalent attention rows."""
    rows = weights / weights.sum(dim=-1, keepdim=True)
    magnitude = rows.abs()
    probabilities = magnitude / magnitude.sum(dim=-1, keepdim=True)
    safe = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
    entropy = -(probabilities * safe.log()).sum(dim=-1)
    nodes = torch.tensor(float(weights.shape[-1]), dtype=rows.dtype)
    return {
        "row_cv_mean": float((rows.std(dim=-1) / rows.mean(dim=-1)).abs().mean()),
        "row_max_over_mean_mean": float(
            (rows.max(dim=-1).values / rows.mean(dim=-1)).mean()
        ),
        "entropy_over_log_n_mean": float((entropy / nodes.log()).mean()),
        "negative_weight_fraction": float((rows < 0.0).to(rows.dtype).mean()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    batch = _lba_batch(args) if args.dataset == "lba" else _synthetic_batch(args)

    torch.manual_seed(args.model_seed)
    model = (
        build_regression_model(
            node_dim=ATOM3D_LBA_NODE_DIM,
            hidden_dim=64,
            num_layers=3,
            num_heads=4,
            local_head_counts=(4, 0, 4),
            local_cutoff=LOCAL_CUTOFF_ANGSTROM,
            use_key_balancing=False,
            use_gated_local_transport=True,
            use_grouped_invariant_normalization=True,
            use_whitened_global_read=True,
        )
        .to(dtype=torch.float64)
        .eval()
    )

    captured: list[dict[str, torch.Tensor]] = []
    original = moment._global_moment_messages

    def capture(*call_args, **call_kwargs):
        captured.append(
            {
                "query_scalar": call_args[0].detach(),
                "key_scalar": call_args[1].detach(),
                "query_vector": call_args[2].detach(),
                "key_vector": call_args[3].detach(),
                "kernel_scale": call_args[4].detach(),
                "alignment_scale": call_kwargs["alignment_scale"].detach(),
                "alignment_dot_scale": call_kwargs["alignment_dot_scale"].detach(),
                "kernel_floor": torch.tensor(float(call_kwargs["kernel_floor"])),
            }
        )
        return original(*call_args, **call_kwargs)

    moment._global_moment_messages = capture
    try:
        with torch.no_grad():
            model(
                batch.node_feats.to(dtype=torch.float64),
                batch.pos.to(dtype=torch.float64),
                batch.batch,
                edge_index=batch.edge_index,
            )
    finally:
        moment._global_moment_messages = original

    if not captured:
        raise RuntimeError("no global stage was executed")
    stage = captured[0]
    graph_zero = batch.batch == 0
    head = 0

    query = moment._kernel_feature_map(
        stage["query_scalar"],
        stage["query_vector"],
        stage["kernel_scale"],
        stage["alignment_scale"],
        stage["alignment_dot_scale"],
        kernel_floor=float(stage["kernel_floor"]),
    )[graph_zero, head]
    key = moment._kernel_feature_map(
        stage["key_scalar"],
        stage["key_vector"],
        stage["kernel_scale"],
        stage["alignment_scale"],
        stage["alignment_dot_scale"],
        kernel_floor=float(stage["kernel_floor"]),
    )[graph_zero, head]

    nodes = int(graph_zero.sum())
    gram = key.T @ key / float(nodes)
    identity = torch.eye(gram.shape[0], dtype=gram.dtype)
    trace_scale = float(gram.diagonal().sum()) / gram.shape[0]
    incumbent = query @ key.T

    summary: dict[str, object] = {
        "dataset": args.dataset,
        "dataset_revision": (
            ATOM3D_LBA_REVISION if args.dataset == "lba" else "synthetic"
        ),
        "complexes": args.complexes,
        "model_seed": args.model_seed,
        "probe_nodes": nodes,
        "feature_dimension": int(gram.shape[0]),
        "gram_trace_over_features": trace_scale,
        "gram_condition_number": float(torch.linalg.cond(gram + 1e-12 * identity)),
        "incumbent": _row_statistics(incumbent),
        "whitened": {},
        "test_evaluated": False,
    }
    for ridge in RIDGE_GRID:
        weights = query @ torch.linalg.solve(
            gram + ridge * trace_scale * identity, key.T
        )
        summary["whitened"][str(ridge)] = _row_statistics(weights)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
