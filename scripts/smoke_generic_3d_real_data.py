#!/usr/bin/env python3
"""Real ATOM3D-LBA wiring smoke for the generic 3D high-order profile.

This is deliberately a capacity/wiring check, not an affinity-generalization
experiment.  It consumes only pinned training rows, constructs one typed sparse
candidate list, exercises role IDs, masks, exact global transport, the
homogeneous sparse residual, and the transient l=3 workspace, then records an
immutable execution receipt. Validation and test splits are not accepted.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import torch

from equivariant_attention.benchmarking import collate_graphs
from equivariant_attention.config import (
    ArchitectureConfig,
    GlobalTransportConfig,
    LocalResidualConfig,
    NeighborConfig,
    RepresentationConfig,
)
from equivariant_attention.execution import resolve_execution_metadata
from equivariant_attention.neighbor_providers import (
    PrecomputedNeighborProvider,
    neighbor_provider_capabilities,
)
from equivariant_attention.neighbors import pack_neighbor_graph
from equivariant_attention.pdbbind import (
    ATOM3D_LBA_NODE_DIM,
    load_atom3d_lba_split_samples,
    segment_balanced_knn_edge_index,
)
from equivariant_attention.reproducibility import configure_reproducibility
from equivariant_attention.training import (
    build_regression_model,
    predict_graph_scalar,
    train_regression_step,
)


def typed_relation_ids(
    edge_index: torch.Tensor,
    node_role_id: torch.Tensor,
) -> torch.Tensor:
    """Encode receiver/sender roles into one immutable four-relation list."""

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E)")
    if node_role_id.ndim != 1:
        raise ValueError("node_role_id must be one-dimensional")
    if node_role_id.numel() and (
        bool((node_role_id < 0).any().item())
        or bool((node_role_id > 1).any().item())
    ):
        raise ValueError("this LBA adapter expects exactly role IDs 0 and 1")
    receiver, sender = edge_index
    return 2 * node_role_id[receiver] + node_role_id[sender]


def build_real_data_architecture(
    *,
    local_cutoff: float,
    num_layers: int,
    num_heads: int,
    width: int,
) -> ArchitectureConfig:
    """Build the bounded, generic high-order smoke configuration."""

    if width % num_heads:
        raise ValueError("width must be divisible by num_heads")
    return ArchitectureConfig(
        node_dim=ATOM3D_LBA_NODE_DIM,
        num_layers=num_layers,
        num_heads=num_heads,
        profile="high_order",
        symmetry_group="O3",
        representation=RepresentationConfig(
            input_irreps=f"{ATOM3D_LBA_NODE_DIM}x0e",
            hidden_irreps=(
                f"{width}x0e + {num_heads + 2}x1o + "
                f"{max(2, num_heads // 2)}x2e"
            ),
            output_irreps="1x0e",
            num_node_roles=2,
            use_irrep_rms_normalization=True,
            angular_bandwidth=2,
            use_tensor_product_kernel=True,
            transient_max_degree=3,
            transient_workspace_channels=2,
            transient_workspace_layers=(0,),
        ),
        global_transport=GlobalTransportConfig(
            reduction_backend="auto",
        ),
        local=LocalResidualConfig(
            local_cutoff=local_cutoff,
            use_sparse_low_rank_local_residual=True,
            local_residual_rank=4,
            residual_layers=(0,),
            sparse_residual_normalization="positive",
            requested_backend="auto",
            distance_bands=(local_cutoff * 0.5, local_cutoff),
        ),
        neighbor=NeighborConfig(
            provider_kind="precomputed",
            geometry_cache_mode="auto",
            num_edge_relations=4,
            relation_cutoffs=(local_cutoff,) * 4,
        ),
        readout_mode="bipartite",
    )


def _annotate_samples(
    samples,
    *,
    intra_k: int,
    cross_k: int,
    cutoff: float,
):
    annotated = []
    for sample in samples:
        if sample.node_role_id is None or sample.node_masks is None:
            raise RuntimeError("LBA sample is missing generic role/mask metadata")
        edge_index = segment_balanced_knn_edge_index(
            sample.pos,
            sample.node_masks["ligand"],
            intra_k=intra_k,
            cross_k=cross_k,
            cutoff=cutoff,
        )
        relation_id = typed_relation_ids(edge_index, sample.node_role_id)
        annotated.append(
            replace(
                sample,
                edge_index=edge_index,
                edge_relation_id=relation_id,
            )
        )
    return annotated


def _fixed_orthogonal(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    value = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
        device=device,
    )
    if not torch.allclose(value.T @ value, torch.eye(3, device=device, dtype=dtype)):
        raise RuntimeError("internal orthogonal transform is invalid")
    if not torch.linalg.det(value) < 0:
        raise RuntimeError("O(3) smoke transform must include a reflection")
    return value


def _o3_invariance_tolerance(dtype: torch.dtype) -> float:
    """Return the absolute smoke tolerance for the prediction evaluation dtype."""
    if dtype == torch.float64:
        return 1e-8
    if dtype == torch.float32:
        return 1e-4
    if dtype == torch.float16:
        return 2e-2
    if dtype == torch.bfloat16:
        return 5e-2
    raise TypeError(f"unsupported O(3) smoke dtype: {dtype}")


def _verify_o3_invariance(max_abs: float, *, dtype: torch.dtype) -> float:
    tolerance = _o3_invariance_tolerance(dtype)
    if not math.isfinite(max_abs) or max_abs > tolerance:
        raise RuntimeError(
            "O(3) invariance smoke failed: "
            f"max_abs={max_abs!r}, atol={tolerance}, dtype={dtype}"
        )
    return tolerance


def run_smoke(args: argparse.Namespace) -> dict[str, object]:
    if args.split != "train":
        raise ValueError(
            "the wiring/capacity smoke is train-only; validation and test are closed"
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    reproducibility = configure_reproducibility(
        seed=args.seed,
        mode=args.determinism,
    )
    samples = load_atom3d_lba_split_samples(
        args.data_root,
        split=args.split,
        indices=tuple(args.indices),
    )
    samples = _annotate_samples(
        samples,
        intra_k=args.intra_k,
        cross_k=args.cross_k,
        cutoff=args.local_cutoff,
    )
    batch = collate_graphs(samples)
    if batch.edge_index is None or batch.edge_relation_id is None:
        raise RuntimeError("annotated real batch has no typed topology")
    packed = pack_neighbor_graph(
        batch.edge_index,
        num_nodes=batch.pos.shape[0],
        edge_relation_id=batch.edge_relation_id,
        build_reverse=True,
        build_ell=True,
    )
    batch = replace(
        batch,
        edge_index=None,
        edge_relation_id=None,
        packed_neighbors=packed,
    ).to(device=device, dtype=torch.float32)

    architecture = build_real_data_architecture(
        local_cutoff=args.local_cutoff,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        width=args.width,
    )
    model = build_regression_model(
        node_dim=ATOM3D_LBA_NODE_DIM,
        architecture_config=architecture,
    ).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
    )

    model.eval()
    with torch.no_grad():
        prediction = predict_graph_scalar(model, batch)
        transform = _fixed_orthogonal(batch.pos.dtype, batch.pos.device)
        moved = replace(
            batch,
            pos=batch.pos @ transform.T
            + torch.tensor(
                [3.5, -2.0, 1.25],
                dtype=batch.pos.dtype,
                device=batch.pos.device,
            ),
        )
        transformed_prediction = predict_graph_scalar(model, moved)
    invariance_max_abs = float(
        (prediction - transformed_prediction).abs().max().detach().cpu()
    )
    invariance_atol = _verify_o3_invariance(
        invariance_max_abs,
        dtype=prediction.dtype,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    losses: list[float] = []
    update_samples_ms: list[float] = []
    for _ in range(args.updates):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        update_started = time.perf_counter()
        losses.append(
            train_regression_step(
                model,
                batch,
                optimizer,
                grad_clip=None,
                amp_dtype=(
                    torch.bfloat16
                    if args.amp_dtype == "bfloat16"
                    else None
                ),
            )
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        update_samples_ms.append(
            1_000.0 * (time.perf_counter() - update_started)
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
    else:
        peak_allocated = None
        peak_reserved = None
    elapsed = time.perf_counter() - started
    if not losses or not all(torch.isfinite(torch.tensor(losses))):
        raise RuntimeError("real-data smoke produced a non-finite loss")

    moved_packed = batch.packed_neighbors
    if moved_packed is None:
        raise RuntimeError("packed topology disappeared during device transfer")
    provider = PrecomputedNeighborProvider(
        moved_packed.original_edge_index(),
        num_nodes=batch.pos.shape[0],
    )
    receipt = resolve_execution_metadata(
        requested_global_lane="auto",
        requested_local_backend="auto",
        requested_cache_mode="auto",
        graph_layout=batch.graph_layout,
        node_count=batch.pos.shape[0],
        edge_count=moved_packed.num_edges,
        neighbor_policy="fixed_precomputed",
        provider_capabilities=neighbor_provider_capabilities(provider),
        symmetry_group=architecture.symmetry_group,
        architecture_profile=architecture.profile,
        dtype=torch.float32,
        device=device,
        has_receiver_csr=True,
        has_ell=moved_packed.ell_sender is not None,
        max_degree=moved_packed.max_degree or 0,
    )
    architecture_json = architecture.to_json()
    result: dict[str, object] = {
        "schema_version": 1,
        "experiment": "generic_3d_high_order_real_lba_wiring_smoke",
        "dataset": "ATOM3D-LBA ID30",
        "split": args.split,
        "indices": list(args.indices),
        "validation_evaluated": False,
        "test_evaluated": False,
        "claim_boundary": (
            "train-only real-data wiring/capacity smoke; not validation, generalization, "
            "cold-target, affinity-SOTA, or architecture-superiority evidence"
        ),
        "sample_ids": list(batch.sample_ids),
        "node_count": int(batch.pos.shape[0]),
        "edge_count": moved_packed.num_edges,
        "relation_histogram": torch.bincount(
            moved_packed.relation_id.to(dtype=torch.long),
            minlength=4,
        )
        .detach()
        .cpu()
        .tolist()
        if moved_packed.relation_id is not None
        else None,
        "architecture": json.loads(architecture_json),
        "architecture_sha256": hashlib.sha256(
            architecture_json.encode("utf-8")
        ).hexdigest(),
        "execution": receipt.to_dict(),
        "reproducibility": reproducibility,
        "updates": args.updates,
        "learning_rate": args.learning_rate,
        "amp_dtype": args.amp_dtype,
        "losses": losses,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_decreased": losses[-1] < losses[0],
        "o3_invariance_max_abs": invariance_max_abs,
        "o3_invariance_atol": invariance_atol,
        "o3_invariance_passed": True,
        "o3_test_transform_determinant": float(
            torch.linalg.det(transform).detach().cpu()
        ),
        "elapsed_seconds": elapsed,
        "update_samples_ms": update_samples_ms,
        "median_update_ms": statistics.median(update_samples_ms),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
    }
    json.dumps(result, allow_nan=False)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/atom3d_lba"))
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--determinism",
        choices=("seeded", "strict"),
        default="seeded",
    )
    parser.add_argument("--amp-dtype", choices=("none", "bfloat16"), default="none")
    parser.add_argument("--local-cutoff", type=float, default=6.0)
    parser.add_argument("--intra-k", type=int, default=24)
    parser.add_argument("--cross-k", type=int, default=24)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.updates <= 0:
        raise ValueError("updates must be positive")
    result = run_smoke(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
