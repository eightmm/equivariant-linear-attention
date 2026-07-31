#!/usr/bin/env python3
"""Screen canonical ELA branch fusion on QM9 and train-only ATOM3D-LBA.

The comparison changes one optimization contract only. Both arms instantiate
the same canonical ELA schema from the same seed. ``identity_locked`` keeps all
branch-fusion parameters at their exact identity initialization, while
``trainable_fusion`` allows those same parameters to learn. No historical
routing model is part of this packet.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

import torch
import torch.nn.functional as F

import equivariant_attention as ela_package
from equivariant_attention.benchmarking import (
    GraphBatch,
    GraphSample,
    collate_graphs,
    load_qm9_samples,
    split_dataset,
)
from equivariant_attention.branch_fusion import RMSAwareBranchFusion
from equivariant_attention.canonical_regression import ELARegressionModel
from equivariant_attention.pdbbind import (
    ATOM3D_LBA_NODE_DIM,
    ATOM3D_LBA_REVISION,
    edge_topology_sha256,
    load_atom3d_lba_samples,
    sample_identity_sha256 as ordered_sample_identity_sha256,
    segment_balanced_knn_edge_index,
    topology_sha256,
)
from equivariant_attention.reproducibility import configure_reproducibility
from equivariant_attention.training import (
    TargetNormalizer,
    evaluate_regression,
    fit_target_normalizer,
    predict_graph_scalar,
)


PACKET_ID = "canonical-ela-branch-fusion-downstream-20260731"
ARMS = ("identity_locked", "trainable_fusion")
QM9_TARGET_INDEX = 4
QM9_MAX_STEPS = 500
LBA_MAX_STEPS = 3_000
LBA_MAX_SAMPLES = 16
LBA_INTRA_K = 16
LBA_CROSS_K = 16
LBA_OVERFIT_THRESHOLD_PK = 0.10


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/run_canonical_branch_fusion_downstream.py "
            "artifacts/canonical-fusion --task qm9 --device cuda\n"
            "  python scripts/run_canonical_branch_fusion_downstream.py "
            "artifacts/canonical-fusion --task lba --device cuda"
        ),
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--task", choices=("qm9", "lba", "both"), default="both")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=_positive_int, default=64)
    parser.add_argument("--depth", type=_positive_int, default=3)
    parser.add_argument("--num-rbf", type=_positive_int, default=16)
    parser.add_argument("--grad-clip", type=_positive_float, default=1.0)
    parser.add_argument(
        "--resource-receipt",
        type=Path,
        action="append",
        default=[],
        help=(
            "single passing two-shape AB/BA aggregate from "
            "summarize_canonical_resource_pairs.py"
        ),
    )
    parser.add_argument(
        "--diagnostic-allow-missing-resource-receipt",
        action="store_true",
    )
    parser.add_argument("--diagnostic-allow-dirty", action="store_true")

    parser.add_argument("--qm9-data-root", type=Path, default=Path("data/qm9"))
    parser.add_argument(
        "--qm9-num-samples",
        type=_positive_int,
        default=130_000,
    )
    parser.add_argument(
        "--qm9-train-size",
        type=_positive_int,
        default=110_000,
    )
    parser.add_argument(
        "--qm9-val-size",
        type=_positive_int,
        default=10_000,
    )
    parser.add_argument("--qm9-batch-size", type=_positive_int, default=64)
    parser.add_argument("--qm9-steps", type=_positive_int, default=500)
    parser.add_argument("--qm9-cutoff", type=_positive_float, default=5.0)
    parser.add_argument("--qm9-lr", type=_positive_float, default=3e-4)
    parser.add_argument(
        "--qm9-weight-decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--lba-data-root",
        type=Path,
        default=Path("data/atom3d_lba"),
    )
    parser.add_argument("--lba-samples", type=_positive_int, default=16)
    parser.add_argument("--lba-batch-size", type=_positive_int, default=2)
    parser.add_argument("--lba-steps", type=_positive_int, default=1_000)
    parser.add_argument("--lba-cutoff", type=_positive_float, default=6.0)
    parser.add_argument("--lba-lr", type=_positive_float, default=1e-3)
    parser.add_argument("--lba-weight-decay", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.width < 16:
        parser.error("--width must be at least 16")
    if args.task in {"qm9", "both"}:
        if args.qm9_steps > QM9_MAX_STEPS:
            parser.error(f"--qm9-steps may not exceed {QM9_MAX_STEPS}")
        if args.qm9_train_size + args.qm9_val_size >= args.qm9_num_samples:
            parser.error(
                "QM9 train and validation sizes must leave a closed test split"
            )
    if args.task in {"lba", "both"}:
        if args.lba_steps > LBA_MAX_STEPS:
            parser.error(f"--lba-steps may not exceed {LBA_MAX_STEPS}")
        if args.lba_samples > LBA_MAX_SAMPLES:
            parser.error(f"--lba-samples may not exceed {LBA_MAX_SAMPLES}")
    for name in ("qm9_weight_decay", "lba_weight_decay"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    return args


def build_plan(args: argparse.Namespace) -> dict[str, object]:
    selected_tasks = ("qm9", "lba") if args.task == "both" else (args.task,)
    protocol_checks = _registered_protocol_checks(args)
    return {
        "schema_version": 1,
        "packet_id": PACKET_ID,
        "status": "planned",
        "tasks": list(selected_tasks),
        "arms": list(ARMS),
        "causal_question": (
            "does learning the identity-initialized invariant global/local "
            "branch fusion improve a fixed canonical ELA?"
        ),
        "architecture": {
            "kind": "canonical_equivariant_linear_attention",
            "spatial_policy": (
                "exact_global_linear_attention_plus_sparse_short_range_local"
            ),
            "width": args.width,
            "depth": args.depth,
            "num_rbf": args.num_rbf,
            "arm_schema": "identical",
            "arm_initial_state": "byte_identical",
            "only_intervention": "branch_fusion_gradient_lock",
        },
        "determinism": "strict",
        "model_seed": args.seed,
        "device": args.device,
        "registered_protocol": {
            "checks": protocol_checks,
            "matched": all(protocol_checks.values()),
        },
        "registered_gates": {
            "resource_prerequisite": (
                "canonical resource receipt must pass before promotion"
            ),
            "router_activation": {
                "max_sector_weight_rms_deviation_min": 1e-3,
                "max_sector_message_relative_rms_min": 1e-5,
                "all_branch_gradients_finite": True,
            },
            "qm9": {
                "mae_improvement_eV_min": 0.010,
                "worst_regression_eV_max": 0.020,
                "nonfinite_is_failure": True,
            },
            "lba": {
                "train_mae_pK_max": LBA_OVERFIT_THRESHOLD_PK,
                "interpretation": "capacity_only_not_accuracy_promotion",
            },
        },
        "qm9": {
            "dataset": "QM9",
            "target_index": QM9_TARGET_INDEX,
            "num_samples": args.qm9_num_samples,
            "train_size": args.qm9_train_size,
            "validation_size": args.qm9_val_size,
            "batch_size": args.qm9_batch_size,
            "updates_per_arm": args.qm9_steps,
            "cutoff_angstrom": args.qm9_cutoff,
            "learning_rate": args.qm9_lr,
            "weight_decay": args.qm9_weight_decay,
            "split_seed": args.seed,
            "validation_evaluated": True,
            "test_evaluated": False,
            "claim_boundary": "one-seed architecture screen",
        },
        "lba": {
            "dataset": "vector-institute/atom3d-lba",
            "revision": ATOM3D_LBA_REVISION,
            "split": "train",
            "subset_indices": list(range(args.lba_samples)),
            "batch_size": args.lba_batch_size,
            "updates_per_arm": args.lba_steps,
            "cutoff_angstrom": args.lba_cutoff,
            "intra_k": LBA_INTRA_K,
            "cross_k": LBA_CROSS_K,
            "learning_rate": args.lba_lr,
            "weight_decay": args.lba_weight_decay,
            "order_seed": args.seed,
            "validation_evaluated": False,
            "test_evaluated": False,
            "claim_boundary": "train-only capacity/overfit",
        },
    }


def _registered_protocol_checks(
    args: argparse.Namespace,
) -> dict[str, bool]:
    checks = {
        "cuda_fp32": str(args.device).startswith("cuda"),
        "seed": args.seed == 42,
        "width": args.width == 64,
        "depth": args.depth == 3,
        "num_rbf": args.num_rbf == 16,
        "grad_clip": args.grad_clip == 1.0,
    }
    if args.task in {"qm9", "both"}:
        checks.update(
            {
                "qm9_num_samples": args.qm9_num_samples == 130_000,
                "qm9_train_size": args.qm9_train_size == 110_000,
                "qm9_validation_size": args.qm9_val_size == 10_000,
                "qm9_batch_size": args.qm9_batch_size == 64,
                "qm9_updates": args.qm9_steps == 500,
                "qm9_cutoff": args.qm9_cutoff == 5.0,
                "qm9_learning_rate": args.qm9_lr == 3e-4,
                "qm9_weight_decay": args.qm9_weight_decay == 0.01,
            }
        )
    if args.task in {"lba", "both"}:
        checks.update(
            {
                "lba_samples": args.lba_samples == 16,
                "lba_batch_size": args.lba_batch_size == 2,
                "lba_updates": args.lba_steps == 1_000,
                "lba_cutoff": args.lba_cutoff == 6.0,
                "lba_learning_rate": args.lba_lr == 1e-3,
                "lba_weight_decay": args.lba_weight_decay == 0.0,
            }
        )
    return checks


def _state_hashes(
    model: torch.nn.Module,
    *,
    include: Callable[[str], bool] | None = None,
) -> dict[str, str]:
    state_digest = hashlib.sha256()
    schema_digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if include is not None and not include(name):
            continue
        metadata = json.dumps(
            [name, str(tensor.dtype), list(tensor.shape)],
            separators=(",", ":"),
        ).encode("ascii")
        schema_digest.update(len(metadata).to_bytes(8, "big"))
        schema_digest.update(metadata)
        state_digest.update(len(metadata).to_bytes(8, "big"))
        state_digest.update(metadata)
        raw = (
            tensor.detach()
            .cpu()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
        state_digest.update(len(raw).to_bytes(8, "big"))
        state_digest.update(raw)
    return {
        "state_sha256": state_digest.hexdigest(),
        "schema_sha256": schema_digest.hexdigest(),
    }


def _is_branch_state(name: str) -> bool:
    return ".branch_fusion." in name


def _lock_branch_fusion(model: torch.nn.Module) -> int:
    count = 0
    for name, parameter in model.named_parameters():
        if _is_branch_state(name):
            parameter.register_hook(torch.zeros_like)
            count += parameter.numel()
    if count == 0:
        raise RuntimeError("canonical model has no branch-fusion parameters")
    return count


def build_paired_models(
    *,
    node_dim: int,
    width: int,
    depth: int,
    cutoff: float,
    num_rbf: int,
    seed: int,
) -> tuple[ELARegressionModel, ELARegressionModel, dict[str, object]]:
    kwargs = {
        "node_dim": node_dim,
        "width": width,
        "depth": depth,
        "cutoff": cutoff,
        "num_rbf": num_rbf,
    }
    torch.manual_seed(seed)
    control = ELARegressionModel(**kwargs)
    torch.manual_seed(seed)
    candidate = ELARegressionModel(**kwargs)

    control_state = control.state_dict()
    candidate_state = candidate.state_dict()
    names_match = tuple(control_state) == tuple(candidate_state)
    mismatches = [
        name
        for name, tensor in control_state.items()
        if name not in candidate_state or not torch.equal(tensor, candidate_state[name])
    ]
    control_full = _state_hashes(control)
    candidate_full = _state_hashes(candidate)
    control_branch = _state_hashes(control, include=_is_branch_state)
    candidate_branch = _state_hashes(candidate, include=_is_branch_state)
    control_common = _state_hashes(
        control,
        include=lambda name: not _is_branch_state(name),
    )
    candidate_common = _state_hashes(
        candidate,
        include=lambda name: not _is_branch_state(name),
    )
    byte_identical = (
        names_match
        and not mismatches
        and control_full == candidate_full
        and control_branch == candidate_branch
        and control_common == candidate_common
    )
    if not byte_identical:
        raise RuntimeError("paired canonical models do not share one initial state")

    branch_parameter_count = sum(
        parameter.numel()
        for name, parameter in control.named_parameters()
        if _is_branch_state(name)
    )
    locked_count = _lock_branch_fusion(control)
    receipt = {
        "schema_version": 1,
        "byte_identical": byte_identical,
        "tensor_name_order_identical": names_match,
        "tensor_mismatch_count": len(mismatches),
        "full_initial_state_sha256": control_full["state_sha256"],
        "candidate_full_initial_state_sha256": candidate_full["state_sha256"],
        "state_schema_sha256": control_full["schema_sha256"],
        "candidate_state_schema_sha256": candidate_full["schema_sha256"],
        "common_weight_initial_state_sha256": control_common["state_sha256"],
        "candidate_common_weight_initial_state_sha256": candidate_common[
            "state_sha256"
        ],
        "branch_initial_state_sha256": control_branch["state_sha256"],
        "candidate_branch_initial_state_sha256": candidate_branch["state_sha256"],
        "branch_parameter_count": branch_parameter_count,
        "identity_locked_parameter_count": locked_count,
        "total_parameter_count": sum(
            parameter.numel() for parameter in control.parameters()
        ),
    }
    return control, candidate, receipt


def _tensor_sha256(value: torch.Tensor) -> str:
    raw = (
        value.detach()
        .cpu()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )
    return hashlib.sha256(raw).hexdigest()


@torch.no_grad()
def fusion_probe(
    model: torch.nn.Module,
    batch: GraphBatch,
) -> dict[str, object]:
    parameter = next(model.parameters())
    moved = batch.to(device=parameter.device, dtype=parameter.dtype)
    records: list[dict[str, object]] = []
    handles = []

    for name, module in model.named_modules():
        if not isinstance(module, RMSAwareBranchFusion):
            continue

        def capture(
            current: RMSAwareBranchFusion,
            inputs: tuple[object, ...],
            output: tuple[object, ...],
            *,
            module_name: str = name,
        ) -> None:
            if len(inputs) != 3:
                raise RuntimeError("unexpected branch-fusion call signature")
            diagnostics = current.diagnostics(*inputs)
            global_message = inputs[1]
            local_message = inputs[2]
            routed_global = output[0]
            routed_local = output[1]
            if not all(
                isinstance(value, Sequence)
                for value in (
                    global_message,
                    local_message,
                    routed_global,
                    routed_local,
                )
            ):
                raise RuntimeError("unexpected branch-fusion message container")
            relative_message_rms = []
            for sector in range(len(RMSAwareBranchFusion.sector_names)):
                identity = global_message[sector] + local_message[sector]
                actual = routed_global[sector] + routed_local[sector]
                difference = actual.double() - identity.double()
                scale = identity.double().square().sum().clamp_min(
                    torch.finfo(torch.float64).tiny
                )
                relative_message_rms.append(
                    torch.sqrt(difference.square().sum() / scale)
                )
            records.append(
                {
                    "module": module_name,
                    "weights": diagnostics.weights.detach().float().cpu(),
                    "global_rms": diagnostics.global_rms.detach().float().cpu(),
                    "local_rms": diagnostics.local_rms.detach().float().cpu(),
                    "balance": diagnostics.balance_strength.detach().float().cpu(),
                    "relative_message_rms": torch.stack(
                        relative_message_rms
                    )
                    .detach()
                    .float()
                    .cpu(),
                }
            )

        handles.append(module.register_forward_hook(capture))

    training = model.training
    try:
        model.eval()
        prediction = predict_graph_scalar(model, moved).detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()
        model.train(training)
    if not records:
        raise RuntimeError("fusion probe observed no canonical branch-fusion call")

    weights = torch.cat(
        [
            record["weights"].reshape(-1, len(RMSAwareBranchFusion.sector_names), 2)
            for record in records
        ],
        dim=0,
    )
    balance = torch.stack([record["balance"] for record in records])
    deviation = (weights - 1.0).abs()
    weight_rms_by_layer_sector = torch.stack(
        [
            (record["weights"] - 1.0)
            .square()
            .mean(dim=(0, 2))
            .sqrt()
            for record in records
        ]
    )
    message_relative_rms_by_layer_sector = torch.stack(
        [record["relative_message_rms"] for record in records]
    )
    activation_cells = (
        (weight_rms_by_layer_sector >= 1e-3)
        & (message_relative_rms_by_layer_sector >= 1e-5)
    )
    layer_records = []
    for record in records:
        layer_weights = record["weights"]
        layer_records.append(
            {
                "module": record["module"],
                "mean_global_weight_by_sector": layer_weights[:, :, 0]
                .mean(dim=0)
                .tolist(),
                "mean_local_weight_by_sector": layer_weights[:, :, 1]
                .mean(dim=0)
                .tolist(),
                "mean_global_rms_by_sector": record["global_rms"].mean(dim=0).tolist(),
                "mean_local_rms_by_sector": record["local_rms"].mean(dim=0).tolist(),
                "balance_strength_by_sector": record["balance"].tolist(),
                "weight_rms_deviation_by_sector": (
                    (record["weights"] - 1.0)
                    .square()
                    .mean(dim=(0, 2))
                    .sqrt()
                    .tolist()
                ),
                "message_relative_rms_by_sector": record[
                    "relative_message_rms"
                ].tolist(),
            }
        )
    return {
        "schema_version": 1,
        "probe_sample_ids": list(batch.sample_ids),
        "layer_count": len(records),
        "sector_order": list(RMSAwareBranchFusion.sector_names),
        "mean_abs_weight_deviation_from_identity": float(deviation.mean()),
        "max_abs_weight_deviation_from_identity": float(deviation.max()),
        "fraction_weight_deviation_gt_1e-3": float((deviation > 1e-3).float().mean()),
        "max_abs_weight_sum_error": float((weights.sum(dim=-1) - 2.0).abs().max()),
        "max_abs_balance_strength": float(balance.abs().max()),
        "max_sector_weight_rms_deviation": float(
            weight_rms_by_layer_sector.max()
        ),
        "max_sector_message_relative_rms": float(
            message_relative_rms_by_layer_sector.max()
        ),
        "activation_thresholds": {
            "weight_rms_deviation": 1e-3,
            "message_relative_rms": 1e-5,
        },
        "router_active": bool(activation_cells.any()),
        "prediction_sha256": _tensor_sha256(prediction),
        "layers": layer_records,
    }


def _router_gradient(model: torch.nn.Module) -> dict[str, float | int]:
    trainable = 0
    with_gradient = 0
    nonzero = 0
    nonfinite = 0
    square_sum = 0.0
    balance_square_sum = 0.0
    network_square_sum = 0.0
    for name, parameter in model.named_parameters():
        if not _is_branch_state(name):
            continue
        if parameter.requires_grad:
            trainable += parameter.numel()
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        finite = torch.isfinite(gradient)
        with_gradient += gradient.numel()
        nonzero += int(torch.count_nonzero((gradient != 0) & finite).item())
        nonfinite += int(torch.count_nonzero(~finite).item())
        value = float(gradient.double().square().sum().cpu())
        square_sum += value
        if name.endswith(".balance_strength"):
            balance_square_sum += value
        else:
            network_square_sum += value
    return {
        "trainable_parameter_count": trainable,
        "with_gradient_count": with_gradient,
        "nonzero_gradient_count": nonzero,
        "nonfinite_gradient_count": nonfinite,
        "l2_norm": square_sum**0.5,
        "router_network_l2_norm": network_square_sum**0.5,
        "balance_strength_l2_norm": balance_square_sum**0.5,
    }


def train_step(
    model: torch.nn.Module,
    batch: GraphBatch,
    optimizer: torch.optim.Optimizer,
    *,
    target_normalizer: TargetNormalizer,
    grad_clip: float,
) -> dict[str, object]:
    parameter = next(model.parameters())
    moved = batch.to(device=parameter.device, dtype=parameter.dtype)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction = predict_graph_scalar(model, moved)
    target = target_normalizer.transform(moved.target.reshape_as(prediction))
    loss = F.mse_loss(prediction.float(), target.float())
    loss.backward()
    router_gradient = _router_gradient(model)
    pre_clip_norm = float(
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            grad_clip,
        )
        .detach()
        .cpu()
    )
    router_gradient["squared_norm_share"] = (
        float(router_gradient["l2_norm"]) ** 2
        / max(pre_clip_norm**2, torch.finfo(torch.float64).tiny)
    )
    optimizer.step()
    return {
        "loss_normalized_mse": float(loss.detach().cpu()),
        "pre_clip_gradient_l2_norm": pre_clip_norm,
        "gradient_clipped": pre_clip_norm > grad_clip,
        "router_gradient": router_gradient,
    }


def _update_training_diagnostics(
    monitor: dict[str, float | int],
    step: Mapping[str, object],
) -> None:
    monitor["step_count"] = int(monitor.get("step_count", 0)) + 1
    monitor["clipped_step_count"] = int(monitor.get("clipped_step_count", 0)) + int(
        bool(step["gradient_clipped"])
    )
    pre_clip = float(step["pre_clip_gradient_l2_norm"])
    monitor["pre_clip_gradient_l2_norm_sum"] = (
        float(monitor.get("pre_clip_gradient_l2_norm_sum", 0.0)) + pre_clip
    )
    monitor["pre_clip_gradient_l2_norm_max"] = max(
        float(monitor.get("pre_clip_gradient_l2_norm_max", 0.0)),
        pre_clip,
    )
    router = step["router_gradient"]
    assert isinstance(router, Mapping)
    norm = float(router["l2_norm"])
    monitor["router_gradient_l2_norm_sum"] = (
        float(monitor.get("router_gradient_l2_norm_sum", 0.0)) + norm
    )
    monitor["router_gradient_l2_norm_max"] = max(
        float(monitor.get("router_gradient_l2_norm_max", 0.0)),
        norm,
    )
    monitor["router_nonzero_gradient_step_count"] = int(
        monitor.get("router_nonzero_gradient_step_count", 0)
    ) + int(int(router["nonzero_gradient_count"]) > 0)
    monitor["router_nonfinite_gradient_count"] = int(
        monitor.get("router_nonfinite_gradient_count", 0)
    ) + int(router["nonfinite_gradient_count"])
    monitor["router_trainable_parameter_count"] = int(
        router["trainable_parameter_count"]
    )
    monitor["router_squared_gradient_share_sum"] = float(
        monitor.get("router_squared_gradient_share_sum", 0.0)
    ) + float(router["squared_norm_share"])


def _training_diagnostics_summary(
    monitor: Mapping[str, float | int],
) -> dict[str, float | int]:
    steps = int(monitor.get("step_count", 0))
    if steps <= 0:
        raise ValueError("training diagnostics require at least one step")
    return {
        **monitor,
        "clip_fraction": int(monitor["clipped_step_count"]) / steps,
        "pre_clip_gradient_l2_norm_mean": float(
            monitor["pre_clip_gradient_l2_norm_sum"]
        )
        / steps,
        "router_gradient_l2_norm_mean": float(monitor["router_gradient_l2_norm_sum"])
        / steps,
        "router_nonzero_gradient_step_fraction": int(
            monitor["router_nonzero_gradient_step_count"]
        )
        / steps,
        "router_squared_gradient_share_mean": float(
            monitor["router_squared_gradient_share_sum"]
        )
        / steps,
    }


def _branch_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if _is_branch_state(name)
    }


def _branch_delta(
    initial: Mapping[str, torch.Tensor],
    model: torch.nn.Module,
) -> dict[str, object]:
    current = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if _is_branch_state(name)
    }
    if tuple(initial) != tuple(current):
        raise RuntimeError("branch-fusion state schema changed during training")
    square_sum = 0.0
    max_abs = 0.0
    changed = 0
    for name, before in initial.items():
        difference = current[name].double() - before.double()
        square_sum += float(difference.square().sum())
        max_abs = max(max_abs, float(difference.abs().max()))
        changed += int(not torch.equal(current[name], before))
    hashes = _state_hashes(model, include=_is_branch_state)
    return {
        "l2_norm": square_sum**0.5,
        "max_abs": max_abs,
        "changed_tensor_count": changed,
        "final_state_sha256": hashes["state_sha256"],
        "state_schema_sha256": hashes["schema_sha256"],
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _runtime_fingerprint(device: torch.device) -> dict[str, object]:
    receipt: dict[str, object] = {
        "torch_version": torch.__version__,
        "device": str(device),
        "dtype": "float32",
    }
    if device.type != "cuda":
        receipt["cuda"] = None
        return receipt
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    receipt["cuda"] = {
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "compute_capability": [properties.major, properties.minor],
        "runtime_version": torch.version.cuda,
    }
    return receipt


def _paired_optimizer(
    model: torch.nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    common = []
    branch = []
    for name, parameter in model.named_parameters():
        (branch if _is_branch_state(name) else common).append(parameter)
    if not common or not branch:
        raise RuntimeError("canonical optimizer requires common and branch groups")
    return torch.optim.AdamW(
        [
            {"params": common, "weight_decay": weight_decay},
            {"params": branch, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )


def _iter_batches(
    samples: Sequence[GraphSample],
    indices: Sequence[int],
    batch_size: int,
) -> Iterable[GraphBatch]:
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        yield collate_graphs([samples[index] for index in selected])


def _cyclic_indices(
    indices: Sequence[int],
    *,
    step: int,
    batch_size: int,
) -> list[int]:
    if not indices:
        raise ValueError("cyclic batching requires nonempty indices")
    offset = (step * batch_size) % len(indices)
    return [indices[(offset + index) % len(indices)] for index in range(batch_size)]


def _lba_cyclic_indices(
    *,
    step: int,
    batch_size: int,
    sample_count: int,
    seed: int,
) -> list[int]:
    start = step * batch_size
    result: list[int] = []
    while len(result) < batch_size:
        epoch = start // sample_count
        offset = start % sample_count
        order = torch.randperm(
            sample_count,
            generator=torch.Generator().manual_seed(seed + epoch),
        ).tolist()
        take = min(batch_size - len(result), sample_count - offset)
        result.extend(order[offset : offset + take])
        start += take
    return result


def _run_arm(
    *,
    arm: str,
    task: str,
    model: torch.nn.Module,
    samples: Sequence[GraphSample],
    train_indices: Sequence[int],
    metric_indices: Sequence[int],
    probe_batch: GraphBatch,
    normalizer: TargetNormalizer,
    device: torch.device,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    seed: int,
) -> dict[str, object]:
    torch.manual_seed(seed)
    model = model.to(device=device, dtype=torch.float32)
    normalizer = normalizer.to(device=device, dtype=torch.float32)
    optimizer = _paired_optimizer(
        model,
        learning_rate=lr,
        weight_decay=weight_decay,
    )
    initial_branch = _branch_snapshot(model)
    initial_fusion = fusion_probe(model, probe_batch)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    started = time.perf_counter()
    latencies: list[float] = []
    monitor: dict[str, float | int] = {}
    history: list[dict[str, float | int]] = []
    history_interval = max(1, steps // 10)
    final_step: dict[str, object] | None = None
    for step_index in range(steps):
        if task == "lba":
            batch_indices = _lba_cyclic_indices(
                step=step_index,
                batch_size=batch_size,
                sample_count=len(train_indices),
                seed=seed,
            )
            batch_indices = [train_indices[index] for index in batch_indices]
        else:
            batch_indices = _cyclic_indices(
                train_indices,
                step=step_index,
                batch_size=batch_size,
            )
        batch = collate_graphs([samples[index] for index in batch_indices])
        _synchronize(device)
        step_started = time.perf_counter()
        final_step = train_step(
            model,
            batch,
            optimizer,
            target_normalizer=normalizer,
            grad_clip=grad_clip,
        )
        _synchronize(device)
        latencies.append(time.perf_counter() - step_started)
        _update_training_diagnostics(monitor, final_step)
        if (
            step_index == 0
            or (step_index + 1) % history_interval == 0
            or step_index + 1 == steps
        ):
            history.append(
                {
                    "step": step_index + 1,
                    "loss_normalized_mse": float(final_step["loss_normalized_mse"]),
                }
            )
    if final_step is None:
        raise RuntimeError("arm completed no training updates")

    metric_batches = _iter_batches(
        samples,
        metric_indices,
        batch_size,
    )
    metrics = evaluate_regression(
        model,
        metric_batches,
        target_normalizer=normalizer,
    )
    train_probe_count = min(256, len(train_indices))
    train_probe = evaluate_regression(
        model,
        _iter_batches(
            samples,
            train_indices[:train_probe_count],
            batch_size,
        ),
        target_normalizer=normalizer,
    )
    final_fusion = fusion_probe(model, probe_batch)
    _synchronize(device)
    elapsed = time.perf_counter() - started
    latency_window = latencies[min(10, len(latencies)) :]
    if not latency_window:
        latency_window = latencies
    sorted_latency = sorted(latency_window)
    p90_index = max(0, math.ceil(0.9 * len(sorted_latency)) - 1)
    result = {
        "schema_version": 1,
        "packet_id": PACKET_ID,
        "task": task,
        "arm": arm,
        "status": "completed",
        "updates_completed": steps,
        "model_seed": seed,
        "order_seed": seed,
        "initial_branch_fusion": initial_fusion,
        "final_branch_fusion": final_fusion,
        "branch_state_delta": _branch_delta(initial_branch, model),
        "training_diagnostics": _training_diagnostics_summary(monitor),
        "final_loss_normalized_mse": final_step["loss_normalized_mse"],
        "metric_split": "validation" if task == "qm9" else "train",
        "metric_sample_count": len(metric_indices),
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "train_probe": {
            "sample_count": train_probe_count,
            **train_probe,
        },
        "history": history,
        "elapsed_seconds": elapsed,
        "step_latency_median_seconds": statistics.median(latency_window),
        "step_latency_p90_seconds": sorted_latency[p90_index],
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "final_state_sha256": _state_hashes(model)["state_sha256"],
        "validation_evaluated": task == "qm9",
        "test_evaluated": False,
        "claim_boundary": (
            "one-seed architecture screen"
            if task == "qm9"
            else "train-only capacity/overfit"
        ),
    }
    if task == "lba":
        result["overfit_threshold_pK"] = LBA_OVERFIT_THRESHOLD_PK
        result["overfit_passed"] = metrics["mae"] <= LBA_OVERFIT_THRESHOLD_PK
    return result


def _sample_identity_sha256(samples: Sequence[GraphSample]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        payload = json.dumps(
            [
                sample.sample_id,
                list(sample.node_feats.shape),
                list(sample.pos.shape),
                (
                    None
                    if sample.readout_mask is None
                    else int(sample.readout_mask.sum())
                ),
            ],
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _sample_content_sha256(samples: Sequence[GraphSample]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(sample.sample_id.encode("utf-8"))
        for name, value in (
            ("node_feats", sample.node_feats),
            ("pos", sample.pos),
            ("target", sample.target),
            ("edge_index", sample.edge_index),
            ("readout_mask", sample.readout_mask),
        ):
            digest.update(name.encode("ascii"))
            if value is None:
                digest.update(b"<none>")
                continue
            cpu_value = value.detach().cpu()
            # A length-one strided view can report ``is_contiguous=True`` while
            # retaining a non-unit stride, which dtype-view rejects. Copy into
            # a fresh default-stride allocation before exposing raw bytes.
            packed = torch.empty(
                tuple(cpu_value.shape),
                dtype=cpu_value.dtype,
                device="cpu",
            )
            packed.copy_(cpu_value)
            digest.update(str(tuple(packed.shape)).encode("ascii"))
            digest.update(str(packed.dtype).encode("ascii"))
            raw = packed.view(torch.uint8).numpy().tobytes()
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
    return digest.hexdigest()


def _indices_sha256(indices: Sequence[int]) -> str:
    return hashlib.sha256(
        json.dumps(list(indices), separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _with_lba_edges(
    samples: Sequence[GraphSample],
    *,
    cutoff: float,
) -> list[GraphSample]:
    result = []
    for sample in samples:
        if sample.readout_mask is None:
            raise ValueError("ATOM3D-LBA samples require a ligand mask")
        result.append(
            replace(
                sample,
                edge_index=segment_balanced_knn_edge_index(
                    sample.pos,
                    sample.readout_mask,
                    intra_k=LBA_INTRA_K,
                    cross_k=LBA_CROSS_K,
                    cutoff=cutoff,
                ),
            )
        )
    return result


@torch.no_grad()
def _paired_initial_output_receipt(
    control: torch.nn.Module,
    candidate: torch.nn.Module,
    batch: GraphBatch,
) -> dict[str, object]:
    control.eval()
    candidate.eval()
    control_prediction = predict_graph_scalar(control, batch)
    candidate_prediction = predict_graph_scalar(candidate, batch)
    difference = (control_prediction.double() - candidate_prediction.double()).abs()
    return {
        "byte_identical": torch.equal(
            control_prediction,
            candidate_prediction,
        ),
        "max_abs_difference": float(difference.max()),
        "control_prediction_sha256": _tensor_sha256(control_prediction),
        "candidate_prediction_sha256": _tensor_sha256(candidate_prediction),
    }


def _require_paired_initial_output(receipt: Mapping[str, object]) -> None:
    if (
        receipt.get("byte_identical") is not True
        or float(receipt.get("max_abs_difference", math.inf)) != 0.0
    ):
        raise RuntimeError(
            "paired canonical models do not produce identical initial predictions"
        )


def _task_data(
    task: str,
    args: argparse.Namespace,
) -> tuple[
    list[GraphSample],
    list[int],
    list[int],
    GraphBatch,
    dict[str, object],
]:
    if task == "qm9":
        samples = load_qm9_samples(
            args.qm9_data_root,
            target_index=QM9_TARGET_INDEX,
            limit=args.qm9_num_samples,
            local_cutoff=args.qm9_cutoff,
        )
        train, validation, test = split_dataset(
            samples,
            train_size=args.qm9_train_size,
            val_size=args.qm9_val_size,
            seed=args.seed,
        )
        probe_indices = validation[: min(args.qm9_batch_size, len(validation))]
        receipt = {
            "resolved_data_root": str(args.qm9_data_root.resolve()),
            "sample_identity_sha256": _sample_identity_sha256(samples),
            "sample_content_sha256": _sample_content_sha256(samples),
            "split_hashes": {
                "train": _indices_sha256(train),
                "validation": _indices_sha256(validation),
                "test": _indices_sha256(test),
            },
            "train_size": len(train),
            "validation_size": len(validation),
            "closed_test_size": len(test),
            "validation_evaluated": True,
            "test_evaluated": False,
        }
        return (
            samples,
            train,
            validation,
            collate_graphs([samples[index] for index in probe_indices]),
            receipt,
        )

    samples = load_atom3d_lba_samples(
        args.lba_data_root,
        indices=tuple(range(args.lba_samples)),
        revision=ATOM3D_LBA_REVISION,
        split="train",
    )
    samples = _with_lba_edges(samples, cutoff=args.lba_cutoff)
    if any(sample.node_feats.shape[1] != ATOM3D_LBA_NODE_DIM for sample in samples):
        raise RuntimeError("ATOM3D-LBA node feature schema changed")
    train = list(range(len(samples)))
    probe_indices = train[: min(args.lba_batch_size, len(train))]
    receipt = {
        "resolved_data_root": str(args.lba_data_root.resolve()),
        "sample_identity_sha256": _sample_identity_sha256(samples),
        "sample_content_sha256": _sample_content_sha256(samples),
        "ordered_sample_id_sha256": ordered_sample_identity_sha256(samples),
        "edge_topology_sha256": edge_topology_sha256(samples),
        "legacy_joint_topology_sha256": topology_sha256(samples),
        "split": "train",
        "sample_count": len(samples),
        "validation_evaluated": False,
        "test_evaluated": False,
        "label_use": "train_labels_only",
    }
    return (
        samples,
        train,
        train,
        collate_graphs([samples[index] for index in probe_indices]),
        receipt,
    )


def _run_task(
    task: str,
    args: argparse.Namespace,
    *,
    reproducibility: Mapping[str, object],
    resource_prerequisite_passed: bool,
    registered_protocol_matched: bool,
    clean_source: bool,
) -> dict[str, object]:
    samples, train_indices, metric_indices, probe_batch, data_receipt = _task_data(
        task, args
    )
    if task == "qm9":
        cutoff = args.qm9_cutoff
        batch_size = args.qm9_batch_size
        steps = args.qm9_steps
        lr = args.qm9_lr
        weight_decay = args.qm9_weight_decay
    else:
        cutoff = args.lba_cutoff
        batch_size = args.lba_batch_size
        steps = args.lba_steps
        lr = args.lba_lr
        weight_decay = args.lba_weight_decay

    node_dim = samples[0].node_feats.shape[1]
    control, candidate, paired_state = build_paired_models(
        node_dim=node_dim,
        width=args.width,
        depth=args.depth,
        cutoff=cutoff,
        num_rbf=args.num_rbf,
        seed=args.seed,
    )
    paired_output = _paired_initial_output_receipt(
        control,
        candidate,
        probe_batch,
    )
    _require_paired_initial_output(paired_output)
    normalizer = fit_target_normalizer(samples[index] for index in train_indices)
    task_dir = args.output_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)
    arm_results = []
    for arm, model in zip(ARMS, (control, candidate), strict=True):
        result = _run_arm(
            arm=arm,
            task=task,
            model=model,
            samples=samples,
            train_indices=train_indices,
            metric_indices=metric_indices,
            probe_batch=probe_batch,
            normalizer=normalizer,
            device=torch.device(args.device),
            steps=steps,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip=args.grad_clip,
            seed=args.seed,
        )
        _write_json(task_dir / f"{arm}.json", result)
        arm_results.append(result)
        # Profile and train one arm at a time. Keeping the completed control on
        # CUDA would contaminate the candidate's peak-allocation receipt.
        model.to(device="cpu")
        if torch.device(args.device).type == "cuda":
            torch.cuda.empty_cache()

    control_result, candidate_result = arm_results
    improvement = float(control_result["mae"]) - float(candidate_result["mae"])
    mechanism = {
        "candidate_nonzero_gradient_step_fraction": candidate_result[
            "training_diagnostics"
        ]["router_nonzero_gradient_step_fraction"],
        "candidate_branch_gradients_finite": candidate_result[
            "training_diagnostics"
        ]["router_nonfinite_gradient_count"]
        == 0,
        "candidate_final_router_active": candidate_result[
            "final_branch_fusion"
        ]["router_active"],
        "control_branch_state_unchanged": control_result[
            "branch_state_delta"
        ]["changed_tensor_count"]
        == 0,
        "candidate_branch_state_changed": candidate_result[
            "branch_state_delta"
        ]["changed_tensor_count"]
        > 0,
    }
    mechanism_passed = all(
        (
            mechanism["candidate_nonzero_gradient_step_fraction"] > 0,
            mechanism["candidate_branch_gradients_finite"],
            mechanism["candidate_final_router_active"],
            mechanism["control_branch_state_unchanged"],
            mechanism["candidate_branch_state_changed"],
        )
    )
    if task == "qm9":
        gate_checks = {
            "resource_prerequisite": resource_prerequisite_passed,
            "registered_protocol": registered_protocol_matched,
            "clean_source": clean_source,
            "mechanism": mechanism_passed,
            "material_improvement": improvement >= 0.010,
            "worst_regression": improvement >= -0.020,
        }
    else:
        gate_checks = {
            "resource_prerequisite": resource_prerequisite_passed,
            "registered_protocol": registered_protocol_matched,
            "clean_source": clean_source,
            "mechanism": mechanism_passed,
            "train_overfit_capacity": (
                float(candidate_result["mae"]) <= LBA_OVERFIT_THRESHOLD_PK
            ),
        }
    summary = {
        "schema_version": 1,
        "packet_id": PACKET_ID,
        "task": task,
        "status": "completed",
        "data_receipt": data_receipt,
        "paired_common_weight_receipt": paired_state,
        "paired_initial_output_receipt": paired_output,
        "target_normalizer": normalizer.as_dict(),
        "reproducibility": dict(reproducibility),
        "arms": arm_results,
        "candidate_improvement": {
            "metric": "mae",
            "unit": "eV" if task == "qm9" else "pK",
            "identity_locked_minus_trainable_fusion": improvement,
            "candidate_better": improvement > 0.0,
        },
        "router_mechanism_observed": mechanism,
        "screen_gate": {
            "checks": gate_checks,
            "passed": all(gate_checks.values()),
            "interpretation": (
                "diagnostic_non_promotable"
                if not (
                    resource_prerequisite_passed
                    and registered_protocol_matched
                    and clean_source
                )
                else (
                    "eligible_for_multiseed_confirmation"
                    if task == "qm9"
                    else "train_only_capacity"
                )
            ),
        },
        "validation_evaluated": task == "qm9",
        "test_evaluated": False,
        "claim_boundary": (
            "one-seed architecture screen"
            if task == "qm9"
            else "train-only capacity/overfit"
        ),
    }
    _write_json(task_dir / "summary.json", summary)
    return summary


def _source_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for relative in (
        "scripts/run_canonical_branch_fusion_downstream.py",
        "src/equivariant_attention/branch_fusion.py",
        "src/equivariant_attention/canonical.py",
        "src/equivariant_attention/canonical_regression.py",
    ):
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_resource_receipts(paths: Sequence[Path]) -> list[dict[str, object]]:
    if len(paths) != 1:
        raise ValueError(
            "exactly one two-shape AB/BA aggregate resource receipt is required"
        )
    receipts = []
    for path in paths:
        raw = path.read_bytes()
        payload = json.loads(
            raw,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON constant: {value}")
            ),
        )
        if payload.get("schema_version") != 1:
            raise ValueError(f"unsupported aggregate resource schema: {path}")
        if payload.get("experiment") != "canonical_ela_resource_ab_ba":
            raise ValueError(f"resource receipt is not the registered aggregate: {path}")
        gate = payload.get("resource_gate")
        if not isinstance(gate, Mapping) or gate.get("passed") is not True:
            raise ValueError(f"resource gate did not pass: {path}")
        if payload.get("same_common_weights") is not True:
            raise ValueError(f"resource receipt lacks paired weights: {path}")
        environment_contract = payload.get("environment_contract")
        if not isinstance(environment_contract, Mapping):
            raise ValueError(f"resource receipt lacks environment contract: {path}")
        if (
            environment_contract.get("device") != "cuda"
            or environment_contract.get("dtype") != "float32"
            or int(environment_contract.get("minimum_warmup", 0)) < 10
            or int(environment_contract.get("minimum_repeats", 0)) < 30
            or not isinstance(
                environment_contract.get("device_fingerprint"), Mapping
            )
        ):
            raise ValueError(f"resource environment contract is invalid: {path}")
        shape_results = payload.get("shape_results")
        if not isinstance(shape_results, Sequence) or isinstance(
            shape_results, (str, bytes)
        ):
            raise ValueError(f"resource receipt lacks shape results: {path}")
        expected_shapes = {
            (128, 8, 64, 3),
            (512, 32, 64, 3),
        }
        observed_shapes: set[tuple[int, int, int, int]] = set()
        for result in shape_results:
            if not isinstance(result, Mapping):
                raise ValueError(f"invalid resource shape result: {path}")
            shape = (
                int(result["nodes"]),
                int(result["degree"]),
                int(result["width"]),
                int(result["depth"]),
            )
            if shape in observed_shapes:
                raise ValueError(f"duplicate resource shape result: {shape}")
            observed_shapes.add(shape)
            if int(result.get("pair_count", 0)) < 5:
                raise ValueError(
                    f"resource shape {shape} has fewer than five AB/BA pairs"
                )
            if result.get("passed") is not True:
                raise ValueError(f"resource shape {shape} did not pass")
        if observed_shapes != expected_shapes:
            raise ValueError(
                "resource receipt must contain exactly the registered shapes; "
                f"observed={sorted(observed_shapes)}"
            )
        receipts.append(
            {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "git_sha": payload.get("git_sha"),
                "source_file": payload.get("source_file"),
                "environment_contract": environment_contract,
                "shape_results": shape_results,
                "resource_gate": gate,
            }
        )
    return receipts


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text + "\n")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if (
        not args.resource_receipt
        and not args.diagnostic_allow_missing_resource_receipt
    ):
        raise ValueError(
            "at least one passing --resource-receipt is required before "
            "downstream interpretation"
        )
    resource_receipts = (
        _load_resource_receipts(args.resource_receipt)
        if args.resource_receipt
        else []
    )
    root = Path(__file__).resolve().parents[1]
    source_file = Path(ela_package.__file__).resolve()
    if source_file.parents[1] != (root / "src").resolve():
        raise RuntimeError("downstream runner imported the wrong source checkout")
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
    ).strip()
    git_dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=root,
            text=True,
        ).strip()
    )
    if git_dirty and not args.diagnostic_allow_dirty:
        raise ValueError("downstream screen requires a clean worktree")
    if any(
        receipt["git_sha"] != git_sha
        for receipt in resource_receipts
    ):
        raise ValueError("resource receipt git SHA differs from downstream source")
    if any(
        receipt["source_file"] != str(source_file)
        for receipt in resource_receipts
    ):
        raise ValueError("resource receipt source path differs from downstream source")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    reproducibility = configure_reproducibility(
        seed=args.seed,
        mode="strict",
    )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("output_dir must be new or empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        **plan,
        "status": "running",
        "source_sha256": _source_sha256(),
        "source_file": str(source_file),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "reproducibility": reproducibility,
        "runtime_fingerprint": _runtime_fingerprint(torch.device(args.device)),
        "resource_receipts": resource_receipts,
        "resource_prerequisite_passed": bool(resource_receipts),
    }
    _write_json(args.output_dir / "plan.json", plan)
    tasks = ("qm9", "lba") if args.task == "both" else (args.task,)
    registered_protocol_matched = bool(
        plan["registered_protocol"]["matched"]
    )
    started = time.perf_counter()
    results = [
        _run_task(
            task,
            args,
            reproducibility=reproducibility,
            resource_prerequisite_passed=bool(resource_receipts),
            registered_protocol_matched=registered_protocol_matched,
            clean_source=not git_dirty,
        )
        for task in tasks
    ]
    screen_packet_passed = (
        set(tasks) == {"qm9", "lba"}
        and bool(resource_receipts)
        and registered_protocol_matched
        and not git_dirty
        and all(result["screen_gate"]["passed"] for result in results)
    )
    manifest = {
        **plan,
        "status": "completed",
        "elapsed_seconds": time.perf_counter() - started,
        "results": results,
        "screen_packet_passed": screen_packet_passed,
        "confirmation_eligible": screen_packet_passed,
        "promotion_eligible": False,
        "promotion_requires": "registered multi-seed confirmation",
        "validation_evaluated": "qm9" in tasks,
        "test_evaluated": False,
    }
    _write_json(args.output_dir / "summary.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
