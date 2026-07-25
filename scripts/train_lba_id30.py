#!/usr/bin/env python3
"""Train matched models on the official ATOM3D-LBA ID30 train/val split."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence

import torch

from equivariant_attention._egnn_baseline import _StaticEGNNBaseline
from equivariant_attention.benchmarking import GraphSample, collate_graphs
from equivariant_attention.pdbbind import (
    ATOM3D_LBA_NODE_DIM,
    ATOM3D_LBA_REVISION,
    load_atom3d_lba_split_samples,
    segment_balanced_knn_edge_index,
)
from equivariant_attention.reproducibility import configure_reproducibility
from equivariant_attention.training import (
    TargetNormalizer,
    build_regression_model,
    fit_target_normalizer,
    predict_graph_scalar,
    train_regression_step,
)


RUN_ID = "lba-id30-validation-20260724"
MODEL_SEED = 20260724
ORDER_SEED = 20260724
HIDDEN_DIM = 64
NUM_LAYERS = 3
NUM_HEADS = 4
LOCAL_HEAD_COUNTS = (4, 0, 4)
LOCAL_CUTOFF_ANGSTROM = 6.0
INTRA_K = 16
CROSS_K = 16
DEFAULT_ARMS = ("candidate", "incumbent", "egnn")
V3_VARIANTS = {
    "irrep_norm": {
        "use_irrep_rms_normalization": True,
    },
    "quartic": {
        "use_quartic_kernel": True,
    },
    "rank2": {
        "angular_feature_rank": 2,
    },
    "combined": {
        "use_irrep_rms_normalization": True,
        "use_quartic_kernel": True,
        "angular_feature_rank": 2,
    },
}
PUBLISHED_ATOM3D_GNN_RMSE_PK = 1.601


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/atom3d_lba"))
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=(*DEFAULT_ARMS, "v3"),
        default=list(DEFAULT_ARMS),
    )
    parser.add_argument(
        "--v3-variant",
        choices=tuple(V3_VARIANTS),
        default="combined",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "bfloat16"),
        default="none",
    )
    parser.add_argument("--budget-seconds", type=float, default=7_200.0)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.max_epochs <= 0:
        parser.error("--max-epochs must be positive")
    if not 1 <= args.min_epochs <= args.max_epochs:
        parser.error("--min-epochs must lie in [1, max-epochs]")
    if args.patience <= 0:
        parser.error("--patience must be positive")
    if not 0 <= args.warmup_epochs <= args.max_epochs:
        parser.error("--warmup-epochs must lie in [0, max-epochs]")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        parser.error("learning rate must be positive and weight decay nonnegative")
    if args.grad_clip <= 0.0:
        parser.error("--grad-clip must be positive")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        parser.error("--min-lr-ratio must lie in [0, 1]")
    if args.budget_seconds <= 0.0:
        parser.error("--budget-seconds must be positive")
    for name in ("train_limit", "val_limit"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if len(set(args.arms)) != len(args.arms):
        parser.error("--arms must not contain duplicates")
    return args


def _plan(args: argparse.Namespace) -> dict[str, object]:
    limited = args.train_limit is not None or args.val_limit is not None
    primary_arm = "v3" if "v3" in args.arms else "candidate"
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "dataset": "vector-institute/atom3d-lba",
        "dataset_revision": ATOM3D_LBA_REVISION,
        "split": "official_ID30_train_validation",
        "prediction_unit": "one protein-pocket/ligand crystal complex",
        "target": "supplied affinity pK",
        "inference_features": (
            "bound pocket and ligand atom-token categories, segment identity, "
            "and Cartesian coordinates"
        ),
        "test_policy": "test split is structurally inadmissible to this runner",
        "arms": list(args.arms),
        "primary_arm": primary_arm,
        "primary_metric": "best validation RMSE in pK",
        "secondary_metrics": ["validation MAE", "Pearson", "Spearman"],
        "hypothesis": (
            f"{primary_arm} lowers official ID30 validation RMSE by at least "
            "0.02 pK relative to the matched incumbent LGL"
        ),
        "same_harness_baselines": [
            *([] if primary_arm == "candidate" else ["candidate"]),
            "incumbent",
            "egnn",
            "train-target mean",
        ],
        "published_reference": {
            "name": "ATOM3D GNN, ID30",
            "rmse_pK": PUBLISHED_ATOM3D_GNN_RMSE_PK,
            "decision_use": "descriptive_only",
        },
        "architecture": {
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "num_heads": NUM_HEADS,
            "local_head_counts": list(LOCAL_HEAD_COUNTS),
            "candidate_gated_local_transport": True,
            "candidate_grouped_invariant_normalization": True,
            "v3_variant": args.v3_variant,
            "v3_interventions": V3_VARIANTS[args.v3_variant],
            "coordinate_updates": False,
        },
        "topology": {
            "kind": "segment_balanced_knn_candidates",
            "intra_k": INTRA_K,
            "cross_k": CROSS_K,
            "cutoff_angstrom": LOCAL_CUTOFF_ANGSTROM,
            "self_edges": True,
            "identical_for_all_arms": True,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.grad_clip,
            "schedule": "linear warmup then cosine decay",
            "warmup_epochs": args.warmup_epochs,
            "min_lr_ratio": args.min_lr_ratio,
        },
        "selection": {
            "max_epochs": args.max_epochs,
            "min_epochs": args.min_epochs,
            "patience": args.patience,
            "validation_frequency_epochs": 1,
            "checkpoint_metric": "validation_rmse_pK",
        },
        "batch_size": args.batch_size,
        "model_seed": MODEL_SEED,
        "order_seed": ORDER_SEED,
        "determinism": "strict",
        "amp_dtype": args.amp_dtype,
        "budget_seconds": args.budget_seconds,
        "train_limit": args.train_limit,
        "val_limit": args.val_limit,
        "claim_boundary": (
            "smoke_only" if limited else "single-seed ID30 validation comparison"
        ),
        "validation_evaluated": True,
        "test_evaluated": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan = _plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    # This experiment is pinned to the already-materialized immutable cache.
    # Refusing Hub access makes the no-new-download boundary executable.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    reproducibility = configure_reproducibility(seed=MODEL_SEED, mode="strict")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.json"
    result: dict[str, object] = {
        **plan,
        "status": "loading_data",
        "reproducibility": reproducibility,
        "runtime_environment": _runtime_environment(device),
        "git": _git_provenance(),
        "source_sha256": _source_hash(),
        "arm_results": [],
        "validation_evaluated": True,
        "test_evaluated": False,
    }
    _write_json(result_path, result)
    run_started = time.perf_counter()

    train_indices = (
        None if args.train_limit is None else tuple(range(args.train_limit))
    )
    val_indices = None if args.val_limit is None else tuple(range(args.val_limit))
    train_samples = load_atom3d_lba_split_samples(
        args.data_root,
        split="train",
        revision=ATOM3D_LBA_REVISION,
        indices=train_indices,
    )
    val_samples = load_atom3d_lba_split_samples(
        args.data_root,
        split="val",
        revision=ATOM3D_LBA_REVISION,
        indices=val_indices,
    )
    _validate_splits(train_samples, val_samples)
    normalizer = fit_target_normalizer(train_samples)

    result["status"] = "building_topology"
    result["dataset_summary"] = {
        "train_size": len(train_samples),
        "validation_size": len(val_samples),
        "train_identity_sha256": _sample_identity_hash(train_samples),
        "validation_identity_sha256": _sample_identity_hash(val_samples),
        "target_normalizer": normalizer.as_dict(),
        "train_target_range_pK": _target_range(train_samples),
        "validation_target_range_pK": _target_range(val_samples),
    }
    result["constant_baseline"] = _constant_baseline(train_samples, val_samples)
    _write_json(result_path, result)

    topology_started = time.perf_counter()
    train_samples = _with_matched_sparse_edges(train_samples, split="train")
    val_samples = _with_matched_sparse_edges(val_samples, split="val")
    topology_seconds = time.perf_counter() - topology_started
    topology_sha256 = _topology_hash([*train_samples, *val_samples])
    edge_count = sum(
        int(sample.edge_index.shape[1])
        for sample in [*train_samples, *val_samples]
        if sample.edge_index is not None
    )

    candidate_parameters = _parameter_count(
        _build_model("candidate", None, args.v3_variant)
    )
    incumbent_parameters = _parameter_count(
        _build_model("incumbent", None, args.v3_variant)
    )
    v3_parameters = _parameter_count(_build_model("v3", None, args.v3_variant))
    match_target_parameters = (
        v3_parameters if "v3" in args.arms else candidate_parameters
    )
    egnn_width = _matched_egnn_width(match_target_parameters)
    egnn_parameters = _parameter_count(
        _build_model("egnn", egnn_width, args.v3_variant)
    )
    result["dataset_summary"].update(
        {
            "topology_sha256": topology_sha256,
            "candidate_edge_count": edge_count,
            "topology_build_seconds": topology_seconds,
        }
    )
    result["model_summary"] = {
        "candidate_parameters": candidate_parameters,
        "incumbent_parameters": incumbent_parameters,
        "v3_parameters": v3_parameters,
        "candidate_to_incumbent_parameter_ratio": (
            candidate_parameters / incumbent_parameters
        ),
        "v3_to_candidate_parameter_ratio": v3_parameters / candidate_parameters,
        "matched_egnn_width": egnn_width,
        "egnn_parameters": egnn_parameters,
        "egnn_to_match_target_parameter_ratio": (
            egnn_parameters / match_target_parameters
        ),
    }
    result["status"] = "training"
    _write_json(result_path, result)

    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else None
    for arm_index, arm in enumerate(args.arms):
        elapsed = time.perf_counter() - run_started
        remaining = args.budget_seconds - elapsed
        remaining_arms = len(args.arms) - arm_index
        if remaining <= 0.0:
            result["arm_results"].append(
                {"arm": arm, "status": "not_run_total_budget_exhausted"}
            )
            continue
        arm_budget = remaining / remaining_arms
        model = _build_model(arm, egnn_width, args.v3_variant)
        arm_result = _train_arm(
            arm=arm,
            model=model,
            train_samples=train_samples,
            val_samples=val_samples,
            normalizer=normalizer,
            device=device,
            amp_dtype=amp_dtype,
            args=args,
            output_dir=args.output_dir / arm,
            budget_seconds=arm_budget,
        )
        result["arm_results"].append(arm_result)
        result["elapsed_seconds"] = time.perf_counter() - run_started
        result["comparison"] = _comparison(result["arm_results"])
        _write_json(result_path, result)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result["elapsed_seconds"] = time.perf_counter() - run_started
    result["comparison"] = _comparison(result["arm_results"])
    completed = [
        record
        for record in result["arm_results"]
        if isinstance(record, Mapping) and record.get("status") == "completed"
    ]
    result["status"] = (
        "completed" if len(completed) == len(args.arms) else "partial"
    )
    _write_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


def _build_model(
    arm: str,
    egnn_width: int | None,
    v3_variant: str = "combined",
) -> torch.nn.Module:
    torch.manual_seed(MODEL_SEED)
    if arm == "egnn":
        if egnn_width is None:
            raise ValueError("egnn_width is required for the EGNN arm")
        return _StaticEGNNBaseline(
            node_dim=ATOM3D_LBA_NODE_DIM,
            hidden_dim=egnn_width,
            num_layers=NUM_LAYERS,
        )
    if arm not in {"candidate", "incumbent", "v3"}:
        raise ValueError(f"unknown arm: {arm}")
    v3_options = V3_VARIANTS[v3_variant] if arm == "v3" else {}
    return build_regression_model(
        node_dim=ATOM3D_LBA_NODE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        local_head_counts=LOCAL_HEAD_COUNTS,
        local_cutoff=LOCAL_CUTOFF_ANGSTROM,
        use_key_balancing=False,
        use_gated_local_transport=arm in {"candidate", "v3"},
        use_grouped_invariant_normalization=arm in {"candidate", "v3"},
        **v3_options,
    )


def _matched_egnn_width(target_parameter_count: int) -> int:
    candidates: list[tuple[int, int]] = []
    for width in range(8, 257):
        model = _StaticEGNNBaseline(
            node_dim=ATOM3D_LBA_NODE_DIM,
            hidden_dim=width,
            num_layers=NUM_LAYERS,
        )
        candidates.append(
            (abs(_parameter_count(model) - target_parameter_count), width)
        )
    return min(candidates)[1]


def _train_arm(
    *,
    arm: str,
    model: torch.nn.Module,
    train_samples: Sequence[GraphSample],
    val_samples: Sequence[GraphSample],
    normalizer: TargetNormalizer,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    args: argparse.Namespace,
    output_dir: Path,
    budget_seconds: float,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device=device, dtype=torch.float32)
    device_normalizer = normalizer.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = math.ceil(len(train_samples) / args.batch_size)
    total_steps = steps_per_epoch * args.max_epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    initial_state_sha256 = _state_hash(model)
    history: list[dict[str, object]] = []
    latencies: list[float] = []
    gradient_monitor: dict[str, float | int] = {}
    best_rmse = math.inf
    best_epoch = 0
    best_metrics: dict[str, float | None] | None = None
    epochs_since_best = 0
    global_step = 0
    stop_reason = "max_epochs"

    for epoch in range(1, args.max_epochs + 1):
        if time.perf_counter() - started >= budget_seconds:
            stop_reason = "arm_budget_exhausted"
            break
        order = torch.randperm(
            len(train_samples),
            generator=torch.Generator().manual_seed(ORDER_SEED + epoch - 1),
        ).tolist()
        epoch_losses: list[float] = []
        model.train()
        for start in range(0, len(order), args.batch_size):
            if time.perf_counter() - started >= budget_seconds:
                stop_reason = "arm_budget_exhausted"
                break
            learning_rate = _learning_rate(
                global_step,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                base_lr=args.learning_rate,
                min_lr_ratio=args.min_lr_ratio,
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            indices = order[start : start + args.batch_size]
            batch = collate_graphs([train_samples[index] for index in indices])
            _synchronize(device)
            step_started = time.perf_counter()
            loss = train_regression_step(
                model,
                batch,
                optimizer,
                grad_clip=args.grad_clip,
                target_normalizer=device_normalizer,
                amp_dtype=amp_dtype,
                gradient_monitor=gradient_monitor,
            )
            _synchronize(device)
            latencies.append(time.perf_counter() - step_started)
            if not math.isfinite(loss):
                stop_reason = "nonfinite_loss"
                break
            epoch_losses.append(loss)
            global_step += 1
        if stop_reason in {"nonfinite_loss", "arm_budget_exhausted"}:
            break

        validation_metrics = _evaluate(
            model,
            val_samples,
            batch_size=args.batch_size,
            normalizer=device_normalizer,
            amp_dtype=amp_dtype,
        )
        record: dict[str, object] = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss_normalized_mse": statistics.fmean(epoch_losses),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation": validation_metrics,
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(record)
        validation_rmse = float(validation_metrics["rmse_pK"])
        if validation_rmse < best_rmse:
            best_rmse = validation_rmse
            best_epoch = epoch
            best_metrics = validation_metrics
            epochs_since_best = 0
            _save_checkpoint(
                output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                arm=arm,
                epoch=epoch,
                global_step=global_step,
                best_rmse=best_rmse,
                config=_plan(args),
            )
        else:
            epochs_since_best += 1
        _write_json(
            output_dir / "history.json",
            {
                "arm": arm,
                "history": history,
                "best_epoch": best_epoch,
                "best_validation": best_metrics,
            },
        )
        if (
            epoch >= args.min_epochs
            and epochs_since_best >= args.patience
        ):
            stop_reason = "early_stopping"
            break

    epochs_completed = int(history[-1]["epoch"]) if history else 0
    _save_checkpoint(
        output_dir / "last.pt",
        model=model,
        optimizer=optimizer,
        arm=arm,
        epoch=epochs_completed,
        global_step=global_step,
        best_rmse=best_rmse,
        config=_plan(args),
    )
    if best_metrics is None:
        best_metrics = _evaluate(
            model,
            val_samples,
            batch_size=args.batch_size,
            normalizer=device_normalizer,
            amp_dtype=amp_dtype,
        )
        best_rmse = float(best_metrics["rmse_pK"])
        best_epoch = epochs_completed
    elif (output_dir / "best.pt").exists():
        checkpoint = torch.load(
            output_dir / "best.pt",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(checkpoint["model_state"])
        model.to(device=device, dtype=torch.float32)

    best_train_metrics = _evaluate(
        model,
        train_samples,
        batch_size=args.batch_size,
        normalizer=device_normalizer,
        amp_dtype=amp_dtype,
    )
    best_validation_metrics = _evaluate(
        model,
        val_samples,
        batch_size=args.batch_size,
        normalizer=device_normalizer,
        amp_dtype=amp_dtype,
    )
    _synchronize(device)
    latency_window = latencies[min(10, len(latencies)) :]
    if not latency_window:
        latency_window = latencies
    clip_count = int(gradient_monitor.get("clipped_step_count", 0))
    monitored_steps = int(gradient_monitor.get("step_count", 0))
    status = (
        "completed"
        if stop_reason in {"max_epochs", "early_stopping"}
        else "partial"
    )
    return {
        "arm": arm,
        "status": status,
        "stop_reason": stop_reason,
        "parameter_count": _parameter_count(model),
        "initial_state_sha256": initial_state_sha256,
        "best_state_sha256": _state_hash(model),
        "epochs_completed": epochs_completed,
        "global_steps": global_step,
        "best_epoch": best_epoch,
        "best_train": best_train_metrics,
        "best_validation": best_validation_metrics,
        "last_epoch_validation": (
            history[-1]["validation"] if history else None
        ),
        "validation_evaluation_count": len(history),
        "elapsed_seconds": time.perf_counter() - started,
        "step_latency_median_seconds": (
            statistics.median(latency_window) if latency_window else None
        ),
        "step_latency_p90_seconds": (
            _quantile(latency_window, 0.90) if latency_window else None
        ),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "gradient_monitor": {
            **gradient_monitor,
            "clip_fraction": (
                clip_count / monitored_steps if monitored_steps else None
            ),
            "pre_clip_grad_norm_mean": (
                float(gradient_monitor.get("pre_clip_grad_norm_sum", 0.0))
                / monitored_steps
                if monitored_steps
                else None
            ),
        },
        "history": history,
        "best_checkpoint": str(output_dir / "best.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
        "validation_evaluated": True,
        "test_evaluated": False,
    }


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    samples: Sequence[GraphSample],
    *,
    batch_size: int,
    normalizer: TargetNormalizer,
    amp_dtype: torch.dtype | None,
) -> dict[str, float | None]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    model.eval()
    for start in range(0, len(samples), batch_size):
        batch = collate_graphs(samples[start : start + batch_size])
        batch = batch.to(device=device, dtype=dtype)
        context = (
            torch.autocast(device_type=device.type, dtype=amp_dtype)
            if amp_dtype is not None
            else torch.autocast(device_type=device.type, enabled=False)
        )
        with context:
            prediction = predict_graph_scalar(model, batch)
        predictions.append(normalizer.inverse(prediction.float()).cpu().reshape(-1))
        targets.append(batch.target.float().cpu().reshape(-1))
    return _regression_metrics(
        torch.cat(predictions),
        torch.cat(targets),
    )


def _regression_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float | None]:
    if prediction.ndim != 1 or target.shape != prediction.shape:
        raise ValueError("prediction and target must be matching vectors")
    if prediction.numel() == 0:
        raise ValueError("metrics require at least one observation")
    prediction = prediction.double()
    target = target.double()
    difference = prediction - target
    return {
        "mae_pK": float(difference.abs().mean()),
        "rmse_pK": float(difference.square().mean().sqrt()),
        "pearson": _correlation(prediction, target),
        "spearman": _correlation(
            _average_ranks(prediction),
            _average_ranks(target),
        ),
        "count": int(prediction.numel()),
    }


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float | None:
    centered_left = left - left.mean()
    centered_right = right - right.mean()
    denominator = (
        centered_left.square().sum() * centered_right.square().sum()
    ).sqrt()
    if float(denominator) == 0.0:
        return None
    return float((centered_left * centered_right).sum() / denominator)


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    ranks = torch.empty_like(values, dtype=torch.float64)
    start = 0
    while start < values.numel():
        end = start + 1
        while (
            end < values.numel()
            and bool(sorted_values[end] == sorted_values[start])
        ):
            end += 1
        average = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average
        start = end
    return ranks


def _learning_rate(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    min_lr_ratio: float,
) -> float:
    if total_steps <= 0 or not 0 <= step < total_steps:
        raise ValueError("step must lie in [0, total_steps)")
    if not 0 <= warmup_steps <= total_steps:
        raise ValueError("warmup_steps must lie in [0, total_steps]")
    if warmup_steps and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - 1))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def _with_matched_sparse_edges(
    samples: Sequence[GraphSample],
    *,
    split: str,
) -> list[GraphSample]:
    result: list[GraphSample] = []
    for index, sample in enumerate(samples):
        if sample.readout_mask is None:
            raise ValueError("ATOM3D-LBA sample requires a ligand readout mask")
        edge_index = segment_balanced_knn_edge_index(
            sample.pos,
            sample.readout_mask,
            intra_k=INTRA_K,
            cross_k=CROSS_K,
            cutoff=LOCAL_CUTOFF_ANGSTROM,
        )
        result.append(replace(sample, edge_index=edge_index))
        if (index + 1) % 250 == 0 or index + 1 == len(samples):
            print(
                f"built {split} topology {index + 1}/{len(samples)}",
                file=sys.stderr,
                flush=True,
            )
    return result


def _validate_splits(
    train_samples: Sequence[GraphSample],
    val_samples: Sequence[GraphSample],
) -> None:
    if not train_samples or not val_samples:
        raise ValueError("train and validation splits must both be nonempty")
    train_ids = {sample.sample_id for sample in train_samples}
    val_ids = {sample.sample_id for sample in val_samples}
    if len(train_ids) != len(train_samples) or len(val_ids) != len(val_samples):
        raise ValueError("sample IDs must be unique within each split")
    if train_ids & val_ids:
        raise ValueError("train and validation sample IDs overlap")
    if any(not sample.sample_id.startswith("atom3d-lba:train:") for sample in train_samples):
        raise ValueError("training sample carries the wrong split identity")
    if any(not sample.sample_id.startswith("atom3d-lba:val:") for sample in val_samples):
        raise ValueError("validation sample carries the wrong split identity")


def _constant_baseline(
    train_samples: Sequence[GraphSample],
    val_samples: Sequence[GraphSample],
) -> dict[str, float | None]:
    train_target = torch.cat([sample.target.reshape(-1) for sample in train_samples])
    val_target = torch.cat([sample.target.reshape(-1) for sample in val_samples])
    prediction = torch.full_like(val_target, float(train_target.mean()))
    return _regression_metrics(prediction, val_target)


def _comparison(records: object) -> dict[str, object]:
    if not isinstance(records, list):
        return {}
    by_arm = {
        str(record["arm"]): record
        for record in records
        if isinstance(record, Mapping)
        and isinstance(record.get("best_validation"), Mapping)
    }
    primary_arm = "v3" if "v3" in by_arm else "candidate"
    candidate = by_arm.get(primary_arm)
    comparison: dict[str, object] = {
        "published_atom3d_gnn_rmse_pK": PUBLISHED_ATOM3D_GNN_RMSE_PK,
        "published_reference_is_same_harness": False,
    }
    if candidate is None:
        return comparison
    candidate_rmse = float(candidate["best_validation"]["rmse_pK"])
    comparison["primary_arm"] = primary_arm
    comparison[f"{primary_arm}_rmse_pK"] = candidate_rmse
    comparison[f"{primary_arm}_below_published_gnn_reference"] = (
        candidate_rmse < PUBLISHED_ATOM3D_GNN_RMSE_PK
    )
    for baseline in ("candidate", "incumbent", "egnn"):
        if baseline == primary_arm:
            continue
        record = by_arm.get(baseline)
        if record is None:
            continue
        baseline_rmse = float(record["best_validation"]["rmse_pK"])
        delta = candidate_rmse - baseline_rmse
        comparison[f"{baseline}_rmse_pK"] = baseline_rmse
        comparison[f"{primary_arm}_minus_{baseline}_rmse_pK"] = delta
        comparison[f"{primary_arm}_beats_{baseline}"] = delta < 0.0
    incumbent_delta = comparison.get(f"{primary_arm}_minus_incumbent_rmse_pK")
    if isinstance(incumbent_delta, float):
        comparison["registered_primary_improvement_passed"] = (
            incumbent_delta <= -0.02
        )
        if primary_arm == "candidate":
            comparison["registered_candidate_improvement_passed"] = (
                incumbent_delta <= -0.02
            )
    return comparison


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    arm: str,
    epoch: int,
    global_step: int,
    best_rmse: float,
    config: Mapping[str, object],
) -> None:
    payload = {
        "model_state": _cpu_tree(model.state_dict()),
        "optimizer_state": _cpu_tree(optimizer.state_dict()),
        "arm": arm,
        "epoch": epoch,
        "global_step": global_step,
        "best_validation_rmse_pK": best_rmse,
        "config": dict(config),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            [state.cpu() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _cpu_tree(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _sample_identity_hash(samples: Sequence[GraphSample]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(sample.sample_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _topology_hash(samples: Sequence[GraphSample]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        if sample.edge_index is None:
            raise ValueError("topology hash requires precomputed edges")
        digest.update(sample.sample_id.encode("utf-8"))
        digest.update(sample.edge_index.contiguous().numpy().tobytes())
    return digest.hexdigest()


def _target_range(samples: Sequence[GraphSample]) -> list[float]:
    values = torch.cat([sample.target.reshape(-1) for sample in samples])
    return [float(values.min()), float(values.max())]


def _state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires at least one value")
    return ordered[round(probability * (len(ordered) - 1))]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _git_provenance() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    diff = subprocess.run(
        ["git", "diff", "--no-ext-diff", "--binary"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    return {
        "commit": commit,
        "dirty": bool(status),
        "dirty_path_count": len(status.splitlines()),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _runtime_environment(device: torch.device) -> dict[str, object]:
    environment: dict[str, object] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        environment.update(
            {
                "device_name": properties.name,
                "device_total_memory_bytes": properties.total_memory,
                "cudnn_version": torch.backends.cudnn.version(),
            }
        )
    return environment


def _source_hash() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__).resolve(),
        root / "src" / "equivariant_attention" / "_egnn_baseline.py",
        root / "src" / "equivariant_attention" / "moment.py",
        root / "src" / "equivariant_attention" / "pdbbind.py",
        root / "src" / "equivariant_attention" / "training.py",
        root / "PROJECT.md",
        root / "uv.lock",
    ]
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text + "\n")
    temporary.replace(path)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ATOM3D-LBA ID30 training failed: {error}", file=sys.stderr)
        raise
