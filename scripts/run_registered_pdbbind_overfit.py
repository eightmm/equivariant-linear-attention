#!/usr/bin/env python3
"""Run the frozen train-only ATOM3D-LBA overfit comparison."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from collections.abc import Sequence

import torch

from equivariant_attention._egnn_baseline import _StaticEGNNBaseline
from equivariant_attention.benchmarking import (
    GraphSample,
    collate_graphs,
)
from equivariant_attention.pdbbind import (
    ATOM3D_LBA_NODE_DIM,
    ATOM3D_LBA_REVISION,
    load_atom3d_lba_samples,
)
from equivariant_attention.training import (
    TargetNormalizer,
    build_regression_model,
    evaluate_regression,
    fit_target_normalizer,
    train_regression_step,
)


PACKET_ID = "pdbbind-overfit-persistent2e-20260723"
DATASET_REVISION = ATOM3D_LBA_REVISION
SUBSET_INDICES = tuple(range(16))
FROZEN_SAMPLE_IDS = (
    "atom3d-lba:train:0000000:f4b50b7f750536eb",
    "atom3d-lba:train:0000001:692313822d508b6e",
    "atom3d-lba:train:0000002:15125e5a557a555d",
    "atom3d-lba:train:0000003:406aa0a7d117016d",
    "atom3d-lba:train:0000004:2f9c0cfddea66fb4",
    "atom3d-lba:train:0000005:f2f3723537437ed5",
    "atom3d-lba:train:0000006:7ed9cbf7b02f4f79",
    "atom3d-lba:train:0000007:a126aaaf321f6a5a",
    "atom3d-lba:train:0000008:eadaa56e544f2a15",
    "atom3d-lba:train:0000009:e8ec563c766fc777",
    "atom3d-lba:train:0000010:fb3dcdcdb15d2ad4",
    "atom3d-lba:train:0000011:91374cf35c81aa57",
    "atom3d-lba:train:0000012:d969fce338703204",
    "atom3d-lba:train:0000013:e66008b245277d8d",
    "atom3d-lba:train:0000014:1fc38247d746f251",
    "atom3d-lba:train:0000015:c852ae5217f21b9e",
)
FROZEN_NODE_COUNTS = (
    331,
    243,
    412,
    721,
    386,
    587,
    708,
    394,
    601,
    498,
    551,
    227,
    625,
    223,
    343,
    528,
)
FROZEN_LIGAND_COUNTS = (
    27,
    32,
    14,
    54,
    32,
    25,
    47,
    12,
    40,
    32,
    39,
    14,
    29,
    15,
    40,
    37,
)
MAX_STEPS = 3_000
TRAIN_MAE_THRESHOLD_PK = 0.10
MAX_GPU_SECONDS = 1_800
MAX_ARM_GPU_SECONDS = 900
EGNN_CUTOFF_ANGSTROM = 6.0
MODEL_SEED = 20260723
ORDER_SEED = 20260723
HIDDEN_DIM = 64
HIDDEN_TENSOR_DIM = 4
NUM_LAYERS = 3
NUM_HEADS = 4
BATCH_SIZE = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0
EVAL_INTERVAL = 50


def registered_arms() -> tuple[str, str]:
    return ("attention", "egnn")


def radius_edge_index(pos: torch.Tensor, *, cutoff: float) -> torch.Tensor:
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("pos must have shape (N, 3)")
    if not torch.is_floating_point(pos):
        raise TypeError("pos must be floating point")
    if (
        isinstance(cutoff, bool)
        or not isinstance(cutoff, (int, float))
        or not math.isfinite(float(cutoff))
        or float(cutoff) <= 0.0
    ):
        raise ValueError("cutoff must be finite and positive")
    displacement = pos[:, None, :] - pos[None, :, :]
    squared_distance = displacement.square().sum(dim=-1)
    receiver, sender = torch.nonzero(
        squared_distance < float(cutoff) ** 2,
        as_tuple=True,
    )
    return torch.stack([receiver, sender]).to(dtype=torch.long)


def matched_egnn_width(
    *,
    target_parameter_count: int,
    node_dim: int,
    num_layers: int,
) -> int:
    if target_parameter_count <= 0:
        raise ValueError("target_parameter_count must be positive")
    candidates = []
    for width in range(8, 257):
        model = _StaticEGNNBaseline(
            node_dim=node_dim,
            hidden_dim=width,
            num_layers=num_layers,
        )
        parameter_count = sum(
            parameter.numel() for parameter in model.parameters()
        )
        candidates.append((abs(parameter_count - target_parameter_count), width))
    return min(candidates)[1]


def cyclic_batch_indices(
    *,
    step: int,
    batch_size: int,
    sample_count: int,
    seed: int,
) -> list[int]:
    if step < 0 or batch_size <= 0 or sample_count <= 0:
        raise ValueError("step must be nonnegative and counts must be positive")
    start = step * batch_size
    indices: list[int] = []
    while len(indices) < batch_size:
        epoch = start // sample_count
        offset = start % sample_count
        order = torch.randperm(
            sample_count,
            generator=torch.Generator().manual_seed(seed + epoch),
        ).tolist()
        take = min(batch_size - len(indices), sample_count - offset)
        indices.extend(order[offset : offset + take])
        start += take
    return indices


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen 16-complex train-only PDBBind overfit packet."
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/atom3d_lba"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument(
        "--threshold",
        type=float,
        default=TRAIN_MAE_THRESHOLD_PK,
    )
    parser.add_argument(
        "--budget-seconds",
        type=float,
        default=MAX_GPU_SECONDS,
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not 0 < args.max_steps <= MAX_STEPS:
        parser.error(f"--max-steps must lie in [1, {MAX_STEPS}]")
    if not 0.0 < args.threshold <= TRAIN_MAE_THRESHOLD_PK:
        parser.error(
            f"--threshold must lie in (0, {TRAIN_MAE_THRESHOLD_PK}]"
        )
    if not 0.0 < args.budget_seconds <= MAX_GPU_SECONDS:
        parser.error(
            f"--budget-seconds must lie in (0, {MAX_GPU_SECONDS}]"
        )
    if not 0 < args.batch_size <= BATCH_SIZE:
        parser.error(f"--batch-size must lie in [1, {BATCH_SIZE}]")
    if args.eval_interval <= 0:
        parser.error("--eval-interval must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan = _run_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("registered CUDA run requested but CUDA is unavailable")

    samples = load_atom3d_lba_samples(
        args.data_root,
        indices=SUBSET_INDICES,
        revision=DATASET_REVISION,
    )
    _validate_frozen_samples(samples)
    normalizer = fit_target_normalizer(samples)
    attention_probe = _build_attention()
    attention_parameter_count = _parameter_count(attention_probe)
    egnn_width = matched_egnn_width(
        target_parameter_count=attention_parameter_count,
        node_dim=ATOM3D_LBA_NODE_DIM,
        num_layers=NUM_LAYERS,
    )
    del attention_probe

    result: dict[str, object] = {
        **plan,
        "status": "running",
        "sample_ids": [sample.sample_id for sample in samples],
        "node_counts": [int(sample.pos.shape[0]) for sample in samples],
        "ligand_counts": [
            int(sample.readout_mask.sum().item())
            for sample in samples
            if sample.readout_mask is not None
        ],
        "target_normalizer": normalizer.as_dict(),
        "matched_egnn_width": egnn_width,
        "arms": [],
        "validation_evaluated": False,
        "test_evaluated": False,
    }
    _write_result(args.output, result)
    packet_started = time.perf_counter()
    try:
        for arm in registered_arms():
            packet_elapsed = time.perf_counter() - packet_started
            remaining = args.budget_seconds - packet_elapsed
            if remaining <= 0.0:
                result["arms"].append(
                    {
                        "arm": arm,
                        "status": "not_run_packet_budget_exhausted",
                    }
                )
                continue
            arm_samples = (
                samples
                if arm == "attention"
                else _with_egnn_radius_edges(samples)
            )
            model = (
                _build_attention()
                if arm == "attention"
                else _build_egnn(egnn_width)
            )
            arm_result = _run_arm(
                arm=arm,
                model=model,
                samples=arm_samples,
                normalizer=normalizer,
                device=device,
                max_steps=args.max_steps,
                threshold=args.threshold,
                batch_size=args.batch_size,
                eval_interval=args.eval_interval,
                budget_seconds=min(MAX_ARM_GPU_SECONDS, remaining),
            )
            result["arms"].append(arm_result)
            result["packet_elapsed_seconds"] = time.perf_counter() - packet_started
            _write_result(args.output, result)
    except KeyboardInterrupt:
        result["status"] = "interrupted"
        result["packet_elapsed_seconds"] = time.perf_counter() - packet_started
        _write_result(args.output, result)
        raise

    result["packet_elapsed_seconds"] = time.perf_counter() - packet_started
    result["status"] = "completed"
    result["attention_overfit_passed"] = _arm_passed(result, "attention")
    result["egnn_overfit_passed"] = _arm_passed(result, "egnn")
    _write_result(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


def _run_plan(args: argparse.Namespace) -> dict[str, object]:
    return {
        "schema_version": 1,
        "packet_id": PACKET_ID,
        "dataset": "vector-institute/atom3d-lba",
        "dataset_revision": DATASET_REVISION,
        "split": "train",
        "subset_indices": list(SUBSET_INDICES),
        "device": args.device,
        "arms_registered": list(registered_arms()),
        "max_steps": args.max_steps,
        "threshold_train_mae_pK": args.threshold,
        "packet_budget_seconds": args.budget_seconds,
        "arm_budget_seconds": min(
            MAX_ARM_GPU_SECONDS,
            args.budget_seconds,
        ),
        "batch_size": args.batch_size,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRAD_CLIP,
        },
        "attention": {
            "hidden_dim": HIDDEN_DIM,
            "hidden_tensor_dim": HIDDEN_TENSOR_DIM,
            "num_layers": NUM_LAYERS,
            "num_heads": NUM_HEADS,
            "route": "ggg",
            "key_balancing": False,
            "multiscale_spatial_kernel": True,
            "coordinate_updates": False,
            "edge_index": None,
        },
        "egnn": {
            "kind": "internal_static_egnn_baseline",
            "num_layers": NUM_LAYERS,
            "cutoff_angstrom": EGNN_CUTOFF_ANGSTROM,
            "coordinate_updates": False,
        },
        "model_seed": MODEL_SEED,
        "order_seed": ORDER_SEED,
        "validation_evaluated": False,
        "test_evaluated": False,
        "claim_boundary": "train-only wiring/capacity sanity check",
    }


def _build_attention() -> torch.nn.Module:
    torch.manual_seed(MODEL_SEED)
    return build_regression_model(
        node_dim=ATOM3D_LBA_NODE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        use_key_balancing=False,
        use_multiscale_spatial_kernel=True,
        hidden_tensor_dim=HIDDEN_TENSOR_DIM,
    )


def _build_egnn(width: int) -> torch.nn.Module:
    torch.manual_seed(MODEL_SEED)
    return _StaticEGNNBaseline(
        node_dim=ATOM3D_LBA_NODE_DIM,
        hidden_dim=width,
        num_layers=NUM_LAYERS,
    )


def _with_egnn_radius_edges(
    samples: Sequence[GraphSample],
) -> list[GraphSample]:
    return [
        replace(
            sample,
            edge_index=radius_edge_index(
                sample.pos,
                cutoff=EGNN_CUTOFF_ANGSTROM,
            ),
        )
        for sample in samples
    ]


def _run_arm(
    *,
    arm: str,
    model: torch.nn.Module,
    samples: Sequence[GraphSample],
    normalizer: TargetNormalizer,
    device: torch.device,
    max_steps: int,
    threshold: float,
    batch_size: int,
    eval_interval: int,
    budget_seconds: float,
) -> dict[str, object]:
    model = model.to(device=device, dtype=torch.float32)
    normalizer = normalizer.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    initial_state_sha256 = _state_hash(model)
    latencies: list[float] = []
    history: list[dict[str, float | int]] = []
    final_loss: float | None = None
    final_metrics: dict[str, float] | None = None
    threshold_step: int | None = None
    stop_reason = "max_steps"
    step = 0
    steps_completed = 0
    for step in range(1, max_steps + 1):
        elapsed = time.perf_counter() - started
        if elapsed >= budget_seconds:
            stop_reason = "arm_budget_exhausted"
            break
        batch_indices = cyclic_batch_indices(
            step=step - 1,
            batch_size=batch_size,
            sample_count=len(samples),
            seed=ORDER_SEED,
        )
        batch = collate_graphs([samples[index] for index in batch_indices])
        _synchronize(device)
        step_started = time.perf_counter()
        final_loss = train_regression_step(
            model,
            batch,
            optimizer,
            grad_clip=GRAD_CLIP,
            target_normalizer=normalizer,
        )
        steps_completed = step
        _synchronize(device)
        latencies.append(time.perf_counter() - step_started)
        if not math.isfinite(final_loss):
            stop_reason = "nonfinite_loss"
            break
        should_evaluate = (
            step == 1
            or step % eval_interval == 0
            or step == max_steps
        )
        if should_evaluate:
            final_metrics = evaluate_regression(
                model,
                _iter_batches(samples, batch_size),
                target_normalizer=normalizer,
            )
            history.append(
                {
                    "step": step,
                    "train_mae_pK": final_metrics["mae"],
                    "train_rmse_pK": final_metrics["rmse"],
                    "loss_normalized_mse": final_loss,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            if final_metrics["mae"] <= threshold:
                threshold_step = step
                stop_reason = "threshold_reached"
                break
    if final_metrics is None or history[-1]["step"] != step:
        final_metrics = evaluate_regression(
            model,
            _iter_batches(samples, batch_size),
            target_normalizer=normalizer,
        )
        history.append(
            {
                "step": step,
                "train_mae_pK": final_metrics["mae"],
                "train_rmse_pK": final_metrics["rmse"],
                "loss_normalized_mse": (
                    final_loss if final_loss is not None else 0.0
                ),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    _synchronize(device)
    elapsed_seconds = time.perf_counter() - started
    latency_window = latencies[min(10, len(latencies)) :]
    if not latency_window:
        latency_window = latencies
    gradient = _gradient_summary(model)
    edge_count = (
        sum(
            int(sample.edge_index.shape[1])
            for sample in samples
            if sample.edge_index is not None
        )
        if arm == "egnn"
        else 0
    )
    return {
        "arm": arm,
        "status": "completed",
        "stop_reason": stop_reason,
        "overfit_passed": final_metrics["mae"] <= threshold,
        "steps_completed": steps_completed,
        "threshold_step": threshold_step,
        "time_to_threshold_seconds": (
            history[-1]["elapsed_seconds"] if threshold_step is not None else None
        ),
        "train_mae_pK": final_metrics["mae"],
        "train_rmse_pK": final_metrics["rmse"],
        "final_loss_normalized_mse": final_loss,
        "elapsed_seconds": elapsed_seconds,
        "step_latency_median_seconds": (
            statistics.median(latency_window) if latency_window else None
        ),
        "step_latency_p90_seconds": (
            _quantile(latency_window, 0.90) if latency_window else None
        ),
        "peak_cuda_memory_bytes": _peak_cuda_memory_bytes(device),
        "parameter_count": _parameter_count(model),
        "gradient_parameters": gradient,
        "initial_state_sha256": initial_state_sha256,
        "final_state_sha256": _state_hash(model),
        "edge_count_with_self": edge_count,
        "history": history,
        "validation_evaluated": False,
        "test_evaluated": False,
    }


def _iter_batches(
    samples: Sequence[GraphSample],
    batch_size: int,
) -> Sequence:
    return [
        collate_graphs(samples[start : start + batch_size])
        for start in range(0, len(samples), batch_size)
    ]


def _validate_frozen_samples(samples: Sequence[GraphSample]) -> None:
    if len(samples) != len(SUBSET_INDICES):
        raise ValueError("registered subset must contain exactly 16 complexes")
    if tuple(sample.sample_id for sample in samples) != FROZEN_SAMPLE_IDS:
        raise ValueError("registered sample identity or order changed")
    if tuple(int(sample.pos.shape[0]) for sample in samples) != FROZEN_NODE_COUNTS:
        raise ValueError("registered node count changed")
    if any(sample.node_feats.shape[1] != ATOM3D_LBA_NODE_DIM for sample in samples):
        raise ValueError("registered atom-token feature dimension changed")
    if any(sample.readout_mask is None for sample in samples):
        raise ValueError("every complex requires a ligand readout mask")
    ligand_counts = tuple(
        int(sample.readout_mask.sum().item())
        for sample in samples
        if sample.readout_mask is not None
    )
    if ligand_counts != FROZEN_LIGAND_COUNTS:
        raise ValueError("registered ligand count changed")
    if any(not torch.isfinite(sample.target).all() for sample in samples):
        raise ValueError("registered targets must be finite")


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _gradient_summary(model: torch.nn.Module) -> dict[str, int]:
    with_gradient = 0
    nonzero = 0
    nonfinite = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        finite = torch.isfinite(gradient)
        with_gradient += gradient.numel()
        nonzero += int(torch.count_nonzero((gradient != 0) & finite).item())
        nonfinite += int(torch.count_nonzero(~finite).item())
    return {
        "parameter_count": _parameter_count(model),
        "with_gradient_count": with_gradient,
        "nonzero_gradient_count": nonzero,
        "nonfinite_gradient_count": nonfinite,
    }


def _state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires at least one value")
    index = round(probability * (len(ordered) - 1))
    return ordered[index]


def _peak_cuda_memory_bytes(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _arm_passed(result: dict[str, object], arm: str) -> bool:
    arms = result["arms"]
    if not isinstance(arms, list):
        return False
    return any(
        record.get("arm") == arm and record.get("overfit_passed") is True
        for record in arms
        if isinstance(record, dict)
    )


def _write_result(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    path.write_text(text + "\n")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"registered PDBBind overfit failed: {error}", file=sys.stderr)
        raise
