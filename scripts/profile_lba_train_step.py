#!/usr/bin/env python3
"""Profile full train-step operators on one cached ATOM3D-LBA batch."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import runpy
import statistics
import time
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from equivariant_attention.benchmarking import GraphBatch, collate_graphs
from equivariant_attention.pdbbind import (
    ATOM3D_LBA_REVISION,
    load_atom3d_lba_split_samples,
)
from equivariant_attention.reproducibility import configure_reproducibility
from equivariant_attention.training import (
    TargetNormalizer,
    fit_target_normalizer,
    predict_graph_scalar,
)


TRAIN_LBA = runpy.run_path(str(Path(__file__).with_name("train_lba_id30.py")))
ARMS = ("candidate", "candidate_checkpointed", "incumbent")
SUPPORTED_ARMS = (
    *ARMS,
    "persistent_2e",
    "ctp",
    "geometry_o3",
    "geometry_se3",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/atom3d_lba"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timing-repeats", type=int, default=20)
    parser.add_argument("--model-seed", type=int, default=41)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=SUPPORTED_ARMS,
        default=list(ARMS),
    )
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.warmup < 0 or args.repeats <= 0 or args.timing_repeats <= 0:
        parser.error(
            "--warmup must be nonnegative and repeat counts must be positive"
        )
    if args.model_seed < 0:
        parser.error("--model-seed must be nonnegative")
    if len(set(args.arms)) != len(args.arms):
        parser.error("--arms must not contain duplicates")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_reproducibility(seed=args.model_seed, mode="strict")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    samples = load_atom3d_lba_split_samples(
        args.data_root,
        split="train",
        revision=ATOM3D_LBA_REVISION,
        indices=tuple(range(args.batch_size)),
    )
    normalizer = fit_target_normalizer(samples)
    samples = TRAIN_LBA["_with_matched_sparse_edges"](samples, split="profile")
    batch = collate_graphs(samples).to(device=device, dtype=torch.float32)
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "profiling",
        "dataset": "vector-institute/atom3d-lba",
        "dataset_revision": ATOM3D_LBA_REVISION,
        "split": "official_ID30_train_prefix",
        "sample_ids": list(batch.sample_ids),
        "sample_identity_sha256": _sample_identity_hash(batch.sample_ids),
        "batch_size": args.batch_size,
        "node_count": int(batch.node_feats.shape[0]),
        "edge_count": (
            0 if batch.edge_index is None else int(batch.edge_index.shape[1])
        ),
        "edge_index_sha256": _tensor_hash(batch.edge_index),
        "device": str(device),
        "dtype": "float32",
        "model_seed": args.model_seed,
        "warmup": args.warmup,
        "profiled_steps": args.repeats,
        "timed_steps": args.timing_repeats,
        "arms": {},
        "validation_evaluated": False,
        "test_evaluated": False,
        "limitations": {
            "train_prefix_only": True,
            "profiler_overhead_included": True,
            "operator_times_are_diagnostic_not_benchmark_latency": True,
            "optimizer_updates_change_weights_between_profiled_steps": True,
        },
    }
    for arm in args.arms:
        result["arms"][arm] = _profile_arm(
            arm=arm,
            batch=batch,
            normalizer=normalizer,
            device=device,
            model_seed=args.model_seed,
            warmup=args.warmup,
            repeats=args.repeats,
            timing_repeats=args.timing_repeats,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result["comparison"] = _comparison(result["arms"])
    result["status"] = (
        "completed"
        if all(result["arms"][arm]["status"] == "completed" for arm in args.arms)
        else "failed"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(result["comparison"], indent=2, sort_keys=True))
    return 0 if result["status"] == "completed" else 1


def _profile_arm(
    *,
    arm: str,
    batch: GraphBatch,
    normalizer: TargetNormalizer,
    device: torch.device,
    model_seed: int,
    warmup: int,
    repeats: int,
    timing_repeats: int,
) -> dict[str, object]:
    build_arm = "candidate" if arm == "candidate_checkpointed" else arm
    model = TRAIN_LBA["_build_model"](
        build_arm,
        None,
        "combined",
        model_seed=model_seed,
    ).to(device=device, dtype=torch.float32)
    if arm == "candidate_checkpointed":
        for layer in model.layers:
            if layer.gated_local is not None:
                layer.gated_local.checkpoint_mlp = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    device_normalizer = normalizer.to(device=device, dtype=torch.float32)
    for _ in range(warmup):
        _step(
            model=model,
            optimizer=optimizer,
            batch=batch,
            normalizer=device_normalizer,
            device=device,
            profiled=False,
        )
    _synchronize(device)
    latencies: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(timing_repeats):
        _synchronize(device)
        started = time.perf_counter()
        _step(
            model=model,
            optimizer=optimizer,
            batch=batch,
            normalizer=device_normalizer,
            device=device,
            profiled=False,
        )
        _synchronize(device)
        latencies.append(time.perf_counter() - started)
    timed_peak_cuda_memory_bytes = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    losses: list[float] = []
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        for _ in range(repeats):
            losses.append(
                _step(
                    model=model,
                    optimizer=optimizer,
                    batch=batch,
                    normalizer=device_normalizer,
                    device=device,
                    profiled=True,
                )
            )
    _synchronize(device)
    events = [_event_record(event) for event in profiler.key_averages()]
    events.sort(
        key=lambda event: (
            event["device_time_total_us"]
            if device.type == "cuda"
            else event["cpu_time_total_us"]
        ),
        reverse=True,
    )
    stages = {
        event["name"]: event
        for event in events
        if str(event["name"]).startswith("stage.")
    }
    operator_events = [
        event for event in events if not str(event["name"]).startswith("stage.")
    ]
    operators = operator_events[:40]
    required_stages = {
        "stage.zero_grad",
        "stage.forward",
        "stage.loss",
        "stage.backward",
        "stage.clip",
        "stage.optimizer_step",
    }
    finite = all(math.isfinite(loss) for loss in losses)
    profiled_peak_cuda_memory_bytes = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )
    return {
        "status": (
            "completed"
            if finite and required_stages.issubset(stages)
            else "failed"
        ),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "losses": losses,
        "median_synchronized_step_seconds": statistics.median(latencies),
        "profiled_step_synchronized_wall_time_us": sum(
            float(stages[name]["cpu_time_total_us"]) for name in required_stages
        ),
        "operator_self_device_time_us": sum(
            float(event["self_device_time_total_us"]) for event in operator_events
        ),
        "timed_peak_cuda_memory_bytes": timed_peak_cuda_memory_bytes,
        "profiled_peak_cuda_memory_bytes": profiled_peak_cuda_memory_bytes,
        "peak_cuda_memory_bytes": (
            max(
                timed_peak_cuda_memory_bytes,
                profiled_peak_cuda_memory_bytes,
            )
            if timed_peak_cuda_memory_bytes is not None
            and profiled_peak_cuda_memory_bytes is not None
            else None
        ),
        "stages": stages,
        "operators": operators,
        "gradient_probe": _geometry_gradient_probe(model),
    }


def _geometry_gradient_probe(model: torch.nn.Module) -> dict[str, object]:
    geometry_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if ".geometry_attention." in name
    ]
    axial_parameters = [
        (name, parameter)
        for name, parameter in geometry_parameters
        if ".axial_gate." in name
    ]

    def summarize(
        parameters: list[tuple[str, torch.nn.Parameter]],
    ) -> dict[str, object]:
        gradients = [
            parameter.grad.detach()
            for _, parameter in parameters
            if parameter.grad is not None
        ]
        square_sum = sum(
            float(gradient.float().square().sum().cpu())
            for gradient in gradients
        )
        return {
            "parameter_tensor_count": len(parameters),
            "gradient_tensor_count": len(gradients),
            "finite": all(bool(torch.isfinite(gradient).all()) for gradient in gradients),
            "nonzero": any(bool(torch.count_nonzero(gradient)) for gradient in gradients),
            "l2_norm": math.sqrt(square_sum),
        }

    return {
        "geometry": summarize(geometry_parameters),
        "axial": summarize(axial_parameters),
    }


def _comparison(arms: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    candidate = arms.get("candidate")
    incumbent = arms.get("incumbent")
    checkpointed = arms.get("candidate_checkpointed")
    if isinstance(candidate, Mapping) and isinstance(incumbent, Mapping):
        candidate_stages = candidate["stages"]
        incumbent_stages = incumbent["stages"]
        result.update(
            {
                "forward_synchronized_wall_time_ratio": _ratio(
                    candidate_stages["stage.forward"]["cpu_time_total_us"],
                    incumbent_stages["stage.forward"]["cpu_time_total_us"],
                ),
                "backward_synchronized_wall_time_ratio": _ratio(
                    candidate_stages["stage.backward"]["cpu_time_total_us"],
                    incumbent_stages["stage.backward"]["cpu_time_total_us"],
                ),
                "operator_self_device_time_ratio": _ratio(
                    candidate["operator_self_device_time_us"],
                    incumbent["operator_self_device_time_us"],
                ),
                "profiled_step_synchronized_wall_time_ratio": _ratio(
                    candidate["profiled_step_synchronized_wall_time_us"],
                    incumbent["profiled_step_synchronized_wall_time_us"],
                ),
                "median_synchronized_step_ratio": _ratio(
                    candidate["median_synchronized_step_seconds"],
                    incumbent["median_synchronized_step_seconds"],
                ),
                "peak_cuda_allocation_ratio": _ratio(
                    candidate["peak_cuda_memory_bytes"],
                    incumbent["peak_cuda_memory_bytes"],
                ),
            }
        )
    if isinstance(candidate, Mapping) and isinstance(checkpointed, Mapping):
        result.update(
            {
                "checkpointed_vs_candidate_wall_time_ratio": _ratio(
                    checkpointed["profiled_step_synchronized_wall_time_us"],
                    candidate["profiled_step_synchronized_wall_time_us"],
                ),
                "checkpointed_vs_candidate_median_step_ratio": _ratio(
                    checkpointed["median_synchronized_step_seconds"],
                    candidate["median_synchronized_step_seconds"],
                ),
                "checkpointed_vs_candidate_peak_allocation_ratio": _ratio(
                    checkpointed["peak_cuda_memory_bytes"],
                    candidate["peak_cuda_memory_bytes"],
                ),
            }
        )
    if isinstance(candidate, Mapping):
        for name in ("geometry_o3", "geometry_se3"):
            geometry = arms.get(name)
            if not isinstance(geometry, Mapping):
                continue
            result.update(
                {
                    f"{name}_to_candidate_parameter_ratio": _ratio(
                        geometry["parameter_count"],
                        candidate["parameter_count"],
                    ),
                    f"{name}_to_candidate_median_step_ratio": _ratio(
                        geometry["median_synchronized_step_seconds"],
                        candidate["median_synchronized_step_seconds"],
                    ),
                    f"{name}_to_candidate_peak_allocation_ratio": _ratio(
                        geometry["timed_peak_cuda_memory_bytes"],
                        candidate["timed_peak_cuda_memory_bytes"],
                    ),
                }
            )
    if all(isinstance(arms.get(name), Mapping) for name in ("candidate", "persistent_2e", "ctp")):
        result.update(_ctp_resource_comparison(arms))
    return result


def _ctp_resource_comparison(arms: Mapping[str, object]) -> dict[str, object]:
    candidate = arms["candidate"]
    persistent = arms["persistent_2e"]
    ctp = arms["ctp"]
    if not all(
        isinstance(record, Mapping)
        for record in (candidate, persistent, ctp)
    ):
        raise TypeError("CTP resource arms must be mappings")
    comparison = {
        "persistent_2e_to_candidate_parameter_ratio": _ratio(
            persistent["parameter_count"],
            candidate["parameter_count"],
        ),
        "persistent_2e_to_candidate_median_step_ratio": _ratio(
            persistent["median_synchronized_step_seconds"],
            candidate["median_synchronized_step_seconds"],
        ),
        "persistent_2e_to_candidate_peak_allocation_ratio": _ratio(
            persistent["timed_peak_cuda_memory_bytes"],
            candidate["timed_peak_cuda_memory_bytes"],
        ),
        "ctp_to_candidate_parameter_ratio": _ratio(
            ctp["parameter_count"],
            candidate["parameter_count"],
        ),
        "ctp_to_candidate_median_step_ratio": _ratio(
            ctp["median_synchronized_step_seconds"],
            candidate["median_synchronized_step_seconds"],
        ),
        "ctp_to_candidate_peak_allocation_ratio": _ratio(
            ctp["timed_peak_cuda_memory_bytes"],
            candidate["timed_peak_cuda_memory_bytes"],
        ),
    }
    parameter_ratio = comparison["ctp_to_candidate_parameter_ratio"]
    latency_ratio = comparison["ctp_to_candidate_median_step_ratio"]
    memory_ratio = comparison["ctp_to_candidate_peak_allocation_ratio"]
    checks = {
        "parameter_ratio_at_most_1.10": (
            parameter_ratio is not None and parameter_ratio <= 1.10
        ),
        "median_step_ratio_at_most_1.25": (
            latency_ratio is not None and latency_ratio <= 1.25
        ),
        "peak_allocation_ratio_at_most_1.25": (
            memory_ratio is not None and memory_ratio <= 1.25
        ),
    }
    comparison["resource_gate"] = {
        **checks,
        "passed": all(checks.values()),
    }
    return comparison


def _step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: GraphBatch,
    normalizer: TargetNormalizer,
    device: torch.device,
    profiled: bool,
) -> float:
    context = record_function if profiled else _null_record
    with context("stage.zero_grad"):
        optimizer.zero_grad(set_to_none=True)
    with context("stage.forward"):
        prediction = predict_graph_scalar(model, batch)
        if profiled:
            _synchronize(device)
    with context("stage.loss"):
        target = normalizer.transform(batch.target.reshape_as(prediction))
        loss = torch.nn.functional.mse_loss(prediction.float(), target.float())
    with context("stage.backward"):
        loss.backward()
        if profiled:
            _synchronize(device)
    with context("stage.clip"):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if profiled:
            _synchronize(device)
    with context("stage.optimizer_step"):
        optimizer.step()
        if profiled:
            _synchronize(device)
    return float(loss.detach().cpu())


class _null_record:
    def __init__(self, _: str) -> None:
        pass

    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None


def _event_record(event: Any) -> dict[str, object]:
    device_total = getattr(
        event,
        "device_time_total",
        getattr(event, "cuda_time_total", 0.0),
    )
    self_device = getattr(
        event,
        "self_device_time_total",
        getattr(event, "self_cuda_time_total", 0.0),
    )
    return {
        "name": str(event.key),
        "count": int(event.count),
        "cpu_time_total_us": float(event.cpu_time_total),
        "self_cpu_time_total_us": float(event.self_cpu_time_total),
        "device_time_total_us": float(device_total),
        "self_device_time_total_us": float(self_device),
        "cpu_memory_usage_bytes": int(event.cpu_memory_usage),
        "device_memory_usage_bytes": int(getattr(event, "device_memory_usage", 0)),
    }


def _sample_identity_hash(sample_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in sample_ids:
        digest.update(sample_id.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _tensor_hash(tensor: torch.Tensor | None) -> str | None:
    if tensor is None:
        return None
    value = tensor.detach().to(device="cpu").contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _ratio(numerator: object, denominator: object) -> float | None:
    if numerator is None or denominator is None:
        return None
    denominator_value = float(denominator)
    if denominator_value <= 0.0:
        raise ValueError("ratio denominator must be positive")
    return float(numerator) / denominator_value


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


if __name__ == "__main__":
    raise SystemExit(main())
