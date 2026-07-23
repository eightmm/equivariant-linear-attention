#!/usr/bin/env python3
"""Compare architecture-v2 with matched train-only ATOM3D-LBA controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time
from collections.abc import Sequence

import torch

from equivariant_attention.pdbbind import (
    ATOM3D_LBA_NODE_DIM,
    load_atom3d_lba_samples,
)
from equivariant_attention.reproducibility import configure_reproducibility
from equivariant_attention.training import (
    build_regression_model,
    fit_target_normalizer,
)


LEGACY = runpy.run_path(
    str(Path(__file__).with_name("run_registered_pdbbind_overfit.py"))
)

DATASET_REVISION = LEGACY["DATASET_REVISION"]
SUBSET_INDICES = LEGACY["SUBSET_INDICES"]
MODEL_SEED = LEGACY["MODEL_SEED"]
ORDER_SEED = LEGACY["ORDER_SEED"]
HIDDEN_DIM = LEGACY["HIDDEN_DIM"]
HIDDEN_TENSOR_DIM = LEGACY["HIDDEN_TENSOR_DIM"]
NUM_LAYERS = LEGACY["NUM_LAYERS"]
NUM_HEADS = LEGACY["NUM_HEADS"]
BATCH_SIZE = LEGACY["BATCH_SIZE"]
MAX_STEPS = LEGACY["MAX_STEPS"]
TRAIN_MAE_THRESHOLD_PK = LEGACY["TRAIN_MAE_THRESHOLD_PK"]
MAX_PACKET_SECONDS = LEGACY["MAX_GPU_SECONDS"]
MAX_ARM_SECONDS = MAX_PACKET_SECONDS / 3
EVAL_INTERVAL = LEGACY["EVAL_INTERVAL"]

matched_egnn_width = LEGACY["matched_egnn_width"]
_build_egnn = LEGACY["_build_egnn"]
_with_egnn_radius_edges = LEGACY["_with_egnn_radius_edges"]
_run_arm = LEGACY["_run_arm"]
_validate_frozen_samples = LEGACY["_validate_frozen_samples"]
_parameter_count = LEGACY["_parameter_count"]
_arm_passed = LEGACY["_arm_passed"]
_write_result = LEGACY["_write_result"]

PACKET_ID = "architecture-v2-pdbbind-train-overfit"


def registered_arms() -> tuple[str, str, str]:
    return ("incumbent", "candidate", "egnn")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare incumbent, architecture-v2, and private EGNN on the frozen "
            "16-complex ATOM3D-LBA train subset."
        )
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/atom3d_lba"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--threshold", type=float, default=TRAIN_MAE_THRESHOLD_PK)
    parser.add_argument("--budget-seconds", type=float, default=MAX_PACKET_SECONDS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not 0 < args.max_steps <= MAX_STEPS:
        parser.error(f"--max-steps must lie in [1, {MAX_STEPS}]")
    if not 0.0 < args.threshold <= TRAIN_MAE_THRESHOLD_PK:
        parser.error(f"--threshold must lie in (0, {TRAIN_MAE_THRESHOLD_PK}]")
    if not 0.0 < args.budget_seconds <= MAX_PACKET_SECONDS:
        parser.error(f"--budget-seconds must lie in (0, {MAX_PACKET_SECONDS}]")
    if not 0 < args.batch_size <= BATCH_SIZE:
        parser.error(f"--batch-size must lie in [1, {BATCH_SIZE}]")
    if args.eval_interval <= 0:
        parser.error("--eval-interval must be positive")
    return args


def _attention_kwargs() -> dict[str, object]:
    return {
        "node_dim": ATOM3D_LBA_NODE_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "use_key_balancing": False,
        "use_multiscale_spatial_kernel": True,
        "hidden_tensor_dim": HIDDEN_TENSOR_DIM,
    }


def _build_incumbent_attention() -> torch.nn.Module:
    torch.manual_seed(MODEL_SEED)
    return build_regression_model(
        **_attention_kwargs(),
        scalar_content_mode="unit",
        use_tensor_product_kernel=False,
    )


def _build_candidate_attention() -> torch.nn.Module:
    torch.manual_seed(MODEL_SEED)
    return build_regression_model(
        **_attention_kwargs(),
        scalar_content_mode="bounded",
        use_tensor_product_kernel=True,
    )


def _run_plan(args: argparse.Namespace) -> dict[str, object]:
    return {
        "schema_version": 1,
        "packet_id": PACKET_ID,
        "dataset": "vector-institute/atom3d-lba",
        "dataset_revision": DATASET_REVISION,
        "split": "train",
        "subset_indices": list(SUBSET_INDICES),
        "device": args.device,
        "determinism": "strict",
        "arms_registered": list(registered_arms()),
        "max_steps": args.max_steps,
        "threshold_train_mae_pK": args.threshold,
        "packet_budget_seconds": args.budget_seconds,
        "arm_budget_seconds": min(MAX_ARM_SECONDS, args.budget_seconds),
        "batch_size": args.batch_size,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": LEGACY["LEARNING_RATE"],
            "weight_decay": LEGACY["WEIGHT_DECAY"],
            "gradient_clip": LEGACY["GRAD_CLIP"],
        },
        "shared_attention": {
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
        "incumbent": {
            "scalar_content_mode": "unit",
            "tensor_product_kernel": False,
        },
        "candidate": {
            "scalar_content_mode": "bounded",
            "tensor_product_kernel": True,
        },
        "egnn": {
            "kind": "internal_static_egnn_baseline",
            "num_layers": NUM_LAYERS,
            "cutoff_angstrom": LEGACY["EGNN_CUTOFF_ANGSTROM"],
            "coordinate_updates": False,
            "parameter_match_target": "candidate",
        },
        "model_seed": MODEL_SEED,
        "order_seed": ORDER_SEED,
        "validation_evaluated": False,
        "test_evaluated": False,
        "claim_boundary": "train-only wiring/capacity comparison",
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan = _run_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    reproducibility = configure_reproducibility(seed=MODEL_SEED, mode="strict")
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
    candidate_probe = _build_candidate_attention()
    candidate_parameter_count = _parameter_count(candidate_probe)
    egnn_width = matched_egnn_width(
        target_parameter_count=candidate_parameter_count,
        node_dim=ATOM3D_LBA_NODE_DIM,
        num_layers=NUM_LAYERS,
    )
    del candidate_probe

    result: dict[str, object] = {
        **plan,
        "status": "running",
        "reproducibility": reproducibility,
        "source_sha256": _source_hash(),
        "sample_identity_sha256": _sample_identity_hash(samples),
        "sample_ids": [sample.sample_id for sample in samples],
        "node_counts": [int(sample.pos.shape[0]) for sample in samples],
        "ligand_counts": [
            int(sample.readout_mask.sum().item())
            for sample in samples
            if sample.readout_mask is not None
        ],
        "target_normalizer": normalizer.as_dict(),
        "candidate_parameter_count": candidate_parameter_count,
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
                        "validation_evaluated": False,
                        "test_evaluated": False,
                    }
                )
                continue
            if arm == "incumbent":
                model = _build_incumbent_attention()
                arm_samples = samples
            elif arm == "candidate":
                model = _build_candidate_attention()
                arm_samples = samples
            else:
                model = _build_egnn(egnn_width)
                arm_samples = _with_egnn_radius_edges(samples)
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
                budget_seconds=min(MAX_ARM_SECONDS, remaining),
            )
            arm_result["model"] = (
                "internal_static_egnn_baseline"
                if arm == "egnn"
                else "factorized_moment"
            )
            arm_result["architecture"] = _arm_architecture(arm)
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
    for arm in registered_arms():
        result[f"{arm}_overfit_passed"] = _arm_passed(result, arm)
    _write_result(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


def _arm_architecture(arm: str) -> dict[str, object]:
    if arm == "incumbent":
        return {
            "scalar_content_mode": "unit",
            "tensor_product_kernel": False,
        }
    if arm == "candidate":
        return {
            "scalar_content_mode": "bounded",
            "tensor_product_kernel": True,
        }
    if arm == "egnn":
        return {
            "kind": "internal_static_egnn_baseline",
            "cutoff_angstrom": LEGACY["EGNN_CUTOFF_ANGSTROM"],
        }
    raise ValueError(f"unknown arm: {arm}")


def _sample_identity_hash(samples: Sequence[object]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        payload = json.dumps(
            {
                "sample_id": sample.sample_id,
                "node_count": int(sample.pos.shape[0]),
                "ligand_count": int(sample.readout_mask.sum().item()),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _source_hash() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("run_registered_pdbbind_overfit.py").resolve(),
        root / "src" / "equivariant_attention" / "_egnn_baseline.py",
        root / "src" / "equivariant_attention" / "benchmarking.py",
        root / "src" / "equivariant_attention" / "moment.py",
        root / "src" / "equivariant_attention" / "pdbbind.py",
        root / "src" / "equivariant_attention" / "reproducibility.py",
        root / "src" / "equivariant_attention" / "training.py",
        root / "artifacts" / "architecture-v2-positive-tensor-20260723" / "scope.md",
        root
        / "artifacts"
        / "architecture-v2-positive-tensor-20260723"
        / "initialization-preserving-followup.md",
        root / "PROJECT.md",
        root / "pyproject.toml",
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


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"architecture-v2 PDBBind run failed: {error}", file=sys.stderr)
        raise
