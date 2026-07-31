#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from equivariant_attention import prepare_3d_graph
from equivariant_attention.equivariant_linear_attention import (
    EquivariantLinearAttentionConfig,
)
from equivariant_attention.implicit_spatial import ImplicitSpatialKernelConfig
from equivariant_attention.spatial_ablation import (
    SpatialOperatorAblationConfig,
    SpatialOperatorAblationModel,
    empty_prepared_graph_like,
    state_dict_sha256,
)
from equivariant_attention.spatial_benchmarks import (
    SyntheticSpatialBatch,
    make_synthetic_spatial_batch,
    synthetic_batch_sha256,
)


ARMS = ("explicit", "implicit", "hybrid")
TASKS = ("local_directional", "smooth_gaussian", "mixed")


def _csv_ints(value: str) -> list[int]:
    result = [int(item) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return result


def _csv_strings(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected comma-separated strings")
    return result


def _csv_floats(value: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value.split(",") if item.strip())
    if not result or any(item <= 0.0 for item in result):
        raise argparse.ArgumentTypeError(
            "expected comma-separated positive floats"
        )
    return result


def _dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float64": torch.float64,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _autocast(
    device: torch.device,
    compute_dtype: torch.dtype,
) -> contextlib.AbstractContextManager[Any]:
    if compute_dtype == torch.bfloat16:
        return torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type in {"cpu", "cuda"},
        )
    return contextlib.nullcontext()


def _model_dtype(compute_dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if compute_dtype == torch.bfloat16 else compute_dtype


def _clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _pearson(actual: torch.Tensor, target: torch.Tensor) -> float:
    actual = actual.detach().double().flatten().cpu()
    target = target.detach().double().flatten().cpu()
    actual = actual - actual.mean()
    target = target - target.mean()
    denominator = torch.linalg.vector_norm(actual) * torch.linalg.vector_norm(
        target
    )
    if denominator <= 0.0:
        return float("nan")
    return float((actual * target).sum().item() / denominator.item())


def _metrics(
    prediction_normalized: torch.Tensor,
    target_normalized: torch.Tensor,
    *,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> dict[str, float]:
    prediction = (
        prediction_normalized.float() * target_std.float() + target_mean.float()
    )
    target = target_normalized.float() * target_std.float() + target_mean.float()
    error = prediction - target
    return {
        "normalized_mse": float(
            F.mse_loss(
                prediction_normalized.float(),
                target_normalized.float(),
            ).item()
        ),
        "mae": float(error.abs().mean().item()),
        "rmse": float(error.square().mean().sqrt().item()),
        "pearson": _pearson(prediction, target),
    }


def _predict(
    model: SpatialOperatorAblationModel,
    batch: SyntheticSpatialBatch,
    graph: Any,
    no_edge_graph: Any,
    *,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    with _autocast(device, compute_dtype):
        return model(
            batch.node_irreps,
            batch.positions,
            graph,
            no_edge_graph=no_edge_graph,
        )["graph_irreps"]


def _evaluate(
    model: SpatialOperatorAblationModel,
    batch: SyntheticSpatialBatch,
    graph: Any,
    no_edge_graph: Any,
    target_normalized: torch.Tensor,
    *,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> dict[str, float]:
    model.eval()
    with torch.inference_mode():
        prediction = _predict(
            model,
            batch,
            graph,
            no_edge_graph,
            device=device,
            compute_dtype=compute_dtype,
        )
    return _metrics(
        prediction,
        target_normalized,
        target_mean=target_mean,
        target_std=target_std,
    )


def _inference_profile(
    model: SpatialOperatorAblationModel,
    batch: SyntheticSpatialBatch,
    graph: Any,
    no_edge_graph: Any,
    *,
    device: torch.device,
    compute_dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> tuple[float, int]:
    model.eval()
    for _ in range(warmup):
        with torch.inference_mode():
            _predict(
                model,
                batch,
                graph,
                no_edge_graph,
                device=device,
                compute_dtype=compute_dtype,
            )
    _synchronize(device)

    samples: list[float] = []
    peak = 0
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        with torch.inference_mode():
            _predict(
                model,
                batch,
                graph,
                no_edge_graph,
                device=device,
                compute_dtype=compute_dtype,
            )
        _synchronize(device)
        samples.append((time.perf_counter() - start) * 1000.0)
        if device.type == "cuda":
            peak = max(peak, torch.cuda.max_memory_allocated(device))
    return statistics.median(samples), peak


def _initial_equivalence(
    config: SpatialOperatorAblationConfig,
    state_dict: dict[str, torch.Tensor],
    batch: SyntheticSpatialBatch,
    graph: Any,
    no_edge_graph: Any,
    *,
    device: torch.device,
    model_dtype: torch.dtype,
    compute_dtype: torch.dtype,
) -> dict[str, float]:
    outputs: dict[str, torch.Tensor] = {}
    for arm in ARMS:
        model = SpatialOperatorAblationModel(config, arm=arm)
        model.load_state_dict(state_dict, strict=True)
        model.to(device=device, dtype=model_dtype).eval()
        with torch.inference_mode():
            outputs[arm] = _predict(
                model,
                batch,
                graph,
                no_edge_graph,
                device=device,
                compute_dtype=compute_dtype,
            ).float()

    implicit = SpatialOperatorAblationModel(config, arm="implicit")
    implicit.load_state_dict(state_dict, strict=True)
    implicit.to(device=device, dtype=model_dtype).eval()
    with torch.inference_mode():
        implicit_with_explicit_metadata = _predict(
            implicit,
            batch,
            graph,
            no_edge_graph,
            device=device,
            compute_dtype=compute_dtype,
        ).float()
        implicit_with_no_edge_metadata = _predict(
            implicit,
            batch,
            no_edge_graph,
            no_edge_graph,
            device=device,
            compute_dtype=compute_dtype,
        ).float()

    return {
        "explicit_vs_hybrid_max_abs": float(
            (outputs["explicit"] - outputs["hybrid"]).abs().max().item()
        ),
        "explicit_vs_implicit_max_abs": float(
            (outputs["explicit"] - outputs["implicit"]).abs().max().item()
        ),
        "implicit_edge_independence_max_abs": float(
            (
                implicit_with_explicit_metadata
                - implicit_with_no_edge_metadata
            )
            .abs()
            .max()
            .item()
        ),
    }


def _train_arm(
    *,
    arm: str,
    config: SpatialOperatorAblationConfig,
    initial_state: dict[str, torch.Tensor],
    initial_hash: str,
    train: SyntheticSpatialBatch,
    validation: SyntheticSpatialBatch,
    train_graph: Any,
    validation_graph: Any,
    train_no_edge_graph: Any,
    validation_no_edge_graph: Any,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    steps: int,
    evaluation_interval: int,
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    device: torch.device,
    model_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    profile_warmup: int,
    profile_repeats: int,
) -> dict[str, Any]:
    model = SpatialOperatorAblationModel(config, arm=arm)
    model.load_state_dict(initial_state, strict=True)
    if state_dict_sha256(model) != initial_hash:
        raise RuntimeError("arm initialization hash mismatch")
    model.to(device=device, dtype=model_dtype)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    train_target = (train.targets - target_mean) / target_std
    validation_target = (validation.targets - target_mean) / target_std

    best_step = 0
    best_metrics: dict[str, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    curve: list[dict[str, Any]] = []
    step_times: list[float] = []
    clipped_steps = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        start = time.perf_counter()
        prediction = _predict(
            model,
            train,
            train_graph,
            train_no_edge_graph,
            device=device,
            compute_dtype=compute_dtype,
        )
        loss = F.mse_loss(prediction.float(), train_target.float())
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_grad_norm,
        )
        if float(gradient_norm.detach().cpu()) > max_grad_norm:
            clipped_steps += 1
        optimizer.step()
        _synchronize(device)
        step_times.append((time.perf_counter() - start) * 1000.0)

        if step == 1 or step % evaluation_interval == 0 or step == steps:
            train_metrics = _metrics(
                prediction.detach(),
                train_target,
                target_mean=target_mean,
                target_std=target_std,
            )
            validation_metrics = _evaluate(
                model,
                validation,
                validation_graph,
                validation_no_edge_graph,
                validation_target,
                target_mean=target_mean,
                target_std=target_std,
                device=device,
                compute_dtype=compute_dtype,
            )
            curve.append(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "gradient_norm": float(gradient_norm.detach().cpu()),
                    "train": train_metrics,
                    "validation": validation_metrics,
                }
            )
            if best_metrics is None or (
                validation_metrics["mae"] < best_metrics["mae"]
            ):
                best_step = step
                best_metrics = validation_metrics
                best_state = _clone_state_dict(model)

    if best_metrics is None or best_state is None:
        raise RuntimeError("training produced no validation checkpoint")
    final_metrics = _evaluate(
        model,
        validation,
        validation_graph,
        validation_no_edge_graph,
        validation_target,
        target_mean=target_mean,
        target_std=target_std,
        device=device,
        compute_dtype=compute_dtype,
    )
    training_peak = (
        torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0
    )

    model.load_state_dict(best_state, strict=True)
    model.to(device=device, dtype=model_dtype)
    inference_ms, inference_peak = _inference_profile(
        model,
        validation,
        validation_graph,
        validation_no_edge_graph,
        device=device,
        compute_dtype=compute_dtype,
        warmup=profile_warmup,
        repeats=profile_repeats,
    )
    return {
        "arm": arm,
        "audit": model.audit(),
        "initial_state_sha256": initial_hash,
        "best_step": best_step,
        "best_validation": best_metrics,
        "final_validation": final_metrics,
        "median_train_step_ms": statistics.median(step_times),
        "p90_train_step_ms": statistics.quantiles(
            step_times,
            n=10,
            method="inclusive",
        )[8]
        if len(step_times) >= 2
        else step_times[0],
        "inference_ms": inference_ms,
        "training_peak_allocated_bytes": training_peak,
        "inference_peak_allocated_bytes": inference_peak,
        "clipped_steps": clipped_steps,
        "clip_fraction": clipped_steps / steps,
        "curve": curve,
    }


def _summaries(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault((run["task"], run["arm"]), []).append(run)
    output: list[dict[str, Any]] = []
    for (task, arm), group in sorted(grouped.items()):
        row: dict[str, Any] = {
            "task": task,
            "arm": arm,
            "seeds": len(group),
        }
        for key in (
            "mae",
            "rmse",
            "normalized_mse",
            "pearson",
        ):
            values = [run["best_validation"][key] for run in group]
            finite = [value for value in values if math.isfinite(value)]
            row[f"mean_{key}"] = statistics.mean(finite) if finite else None
            row[f"std_{key}"] = (
                statistics.stdev(finite) if len(finite) >= 2 else 0.0
            )
        for key in (
            "median_train_step_ms",
            "inference_ms",
            "training_peak_allocated_bytes",
            "inference_peak_allocated_bytes",
            "clip_fraction",
        ):
            values = [float(run[key]) for run in group]
            row[f"mean_{key}"] = statistics.mean(values)
            row[f"std_{key}"] = (
                statistics.stdev(values) if len(values) >= 2 else 0.0
            )
        output.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resource-matched explicit/implicit/hybrid comparison"
    )
    parser.add_argument(
        "--tasks",
        type=_csv_strings,
        default=list(TASKS),
    )
    parser.add_argument("--seeds", type=_csv_ints, default=[0, 1, 2])
    parser.add_argument("--train-graphs", type=int, default=64)
    parser.add_argument("--validation-graphs", type=int, default=32)
    parser.add_argument("--nodes-per-graph", type=int, default=24)
    parser.add_argument("--scalar-dim", type=int, default=4)
    parser.add_argument("--cutoff", type=float, default=1.75)
    parser.add_argument("--candidate-skin", type=float, default=0.25)
    parser.add_argument("--gaussian-scale", type=float, default=2.5)
    parser.add_argument("--coordinate-scale", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--local-rank", type=int, default=4)
    parser.add_argument("--num-rbf", type=int, default=16)
    parser.add_argument(
        "--implicit-scales",
        type=_csv_floats,
        default=(2.0, 4.0, 8.0),
    )
    parser.add_argument("--implicit-every", type=int, default=1)
    parser.add_argument("--implicit-scale-init", type=float, default=0.0)
    parser.add_argument("--implicit-chunk-size", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--evaluation-interval", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--profile-warmup", type=int, default=5)
    parser.add_argument("--profile-repeats", type=int, default=20)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float64"],
        default="float32",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    invalid_tasks = set(args.tasks) - set(TASKS)
    if invalid_tasks:
        raise ValueError(f"unsupported tasks: {sorted(invalid_tasks)}")
    device = torch.device(args.device)
    compute_dtype = _dtype(args.dtype)
    model_dtype = _model_dtype(compute_dtype)
    runs: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []

    for task_index, task in enumerate(args.tasks):
        for seed in args.seeds:
            train = make_synthetic_spatial_batch(
                task=task,
                num_graphs=args.train_graphs,
                nodes_per_graph=args.nodes_per_graph,
                seed=10_000 * task_index + seed,
                scalar_dim=args.scalar_dim,
                cutoff=args.cutoff,
                candidate_skin=args.candidate_skin,
                gaussian_scale=args.gaussian_scale,
                coordinate_scale=args.coordinate_scale,
                dtype=model_dtype,
            ).to(device, dtype=model_dtype)
            validation = make_synthetic_spatial_batch(
                task=task,
                num_graphs=args.validation_graphs,
                nodes_per_graph=args.nodes_per_graph,
                seed=100_000 + 10_000 * task_index + seed,
                scalar_dim=args.scalar_dim,
                cutoff=args.cutoff,
                candidate_skin=args.candidate_skin,
                gaussian_scale=args.gaussian_scale,
                coordinate_scale=args.coordinate_scale,
                dtype=model_dtype,
            ).to(device, dtype=model_dtype)
            train_graph = prepare_3d_graph(train.batch, train.edge_index)
            validation_graph = prepare_3d_graph(
                validation.batch,
                validation.edge_index,
            )
            train_no_edge_graph = empty_prepared_graph_like(train_graph)
            validation_no_edge_graph = empty_prepared_graph_like(validation_graph)
            target_mean = train.targets.mean(dim=0, keepdim=True)
            target_std = train.targets.std(dim=0, keepdim=True).clamp_min(1e-6)

            model_config = EquivariantLinearAttentionConfig(
                input_irreps=train.input_irreps,
                output_irreps="1x0e",
                hidden_dim=args.hidden_dim,
                num_layers=args.layers,
                num_heads=args.heads,
                local_rank=args.local_rank,
                local_cutoff=args.cutoff + args.candidate_skin,
                num_rbf=args.num_rbf,
                coordinate_updates=False,
            )
            ablation_config = SpatialOperatorAblationConfig(
                model=model_config,
                implicit=ImplicitSpatialKernelConfig(
                    scales=args.implicit_scales,
                    order=2,
                    exclude_self=True,
                    normalization="one_plus_mass",
                    learnable_scale_weights=True,
                    chunk_size=args.implicit_chunk_size,
                ),
                implicit_residual_scale_init=args.implicit_scale_init,
                implicit_every=args.implicit_every,
            )
            torch.manual_seed(1_000_000 + seed)
            template = SpatialOperatorAblationModel(
                ablation_config,
                arm="explicit",
            )
            initial_hash = state_dict_sha256(template)
            initial_state = _clone_state_dict(template)
            equivalence = _initial_equivalence(
                ablation_config,
                initial_state,
                train,
                train_graph,
                train_no_edge_graph,
                device=device,
                model_dtype=model_dtype,
                compute_dtype=compute_dtype,
            )
            audits.append(
                {
                    "task": task,
                    "seed": seed,
                    "train_data_sha256": synthetic_batch_sha256(train),
                    "validation_data_sha256": synthetic_batch_sha256(validation),
                    "initial_state_sha256": initial_hash,
                    "initial_equivalence": equivalence,
                    "parameter_count": sum(
                        parameter.numel() for parameter in template.parameters()
                    ),
                    "train_nodes": train.num_nodes,
                    "train_edges": train.num_edges,
                    "validation_nodes": validation.num_nodes,
                    "validation_edges": validation.num_edges,
                }
            )

            for arm in ARMS:
                result = _train_arm(
                    arm=arm,
                    config=ablation_config,
                    initial_state=initial_state,
                    initial_hash=initial_hash,
                    train=train,
                    validation=validation,
                    train_graph=train_graph,
                    validation_graph=validation_graph,
                    train_no_edge_graph=train_no_edge_graph,
                    validation_no_edge_graph=validation_no_edge_graph,
                    target_mean=target_mean,
                    target_std=target_std,
                    steps=args.steps,
                    evaluation_interval=args.evaluation_interval,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    max_grad_norm=args.max_grad_norm,
                    device=device,
                    model_dtype=model_dtype,
                    compute_dtype=compute_dtype,
                    profile_warmup=args.profile_warmup,
                    profile_repeats=args.profile_repeats,
                )
                result.update(
                    {
                        "task": task,
                        "seed": seed,
                        "target_mean": float(target_mean.item()),
                        "target_std": float(target_std.item()),
                    }
                )
                runs.append(result)

    payload = {
        "schema_version": 2,
        "experiment": "spatial_operator_comparison",
        "arms": list(ARMS),
        "tasks": args.tasks,
        "seeds": args.seeds,
        "device": str(device),
        "compute_dtype": args.dtype,
        "neighbor_discovery_included": False,
        "protocol": {
            "same_parameter_schema": True,
            "same_initial_state_per_task_seed": True,
            "same_train_validation_data_per_task_seed": True,
            "validation_or_test_labels_used_for_training": False,
            "full_batch_training": True,
            "no_edge_graph_prepared_outside_timed_forward": True,
            "arguments": vars(args) | {"output": str(args.output)},
        },
        "audits": audits,
        "runs": runs,
        "summaries": _summaries(runs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
