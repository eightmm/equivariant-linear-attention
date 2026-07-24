#!/usr/bin/env python3
"""Re-evaluate LBA checkpoints and run paired validation-set bootstrap."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import runpy
import sys
from collections.abc import Sequence

import torch

from equivariant_attention.benchmarking import GraphSample, collate_graphs
from equivariant_attention.pdbbind import (
    ATOM3D_LBA_REVISION,
    load_atom3d_lba_split_samples,
)
from equivariant_attention.reproducibility import configure_reproducibility
from equivariant_attention.training import TargetNormalizer, predict_graph_scalar


ROOT = Path(__file__).resolve().parents[1]
TRAIN = runpy.run_path(str(Path(__file__).with_name("train_lba_id30.py")))
ANALYSIS_SEED = 20260724


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/atom3d_lba"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    args = parser.parse_args(argv)
    if args.bootstrap_replicates <= 0:
        parser.error("--bootstrap-replicates must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    configure_reproducibility(seed=ANALYSIS_SEED, mode="strict")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    result = json.loads(args.result.read_text())
    if result.get("status") != "completed":
        raise ValueError("training result must be complete")
    if result.get("test_evaluated") is not False:
        raise ValueError("analysis requires a test-unopened training result")
    if result.get("dataset_revision") != ATOM3D_LBA_REVISION:
        raise ValueError("dataset revision differs from the pinned LBA revision")

    validation_samples = load_atom3d_lba_split_samples(
        args.data_root,
        split="val",
        revision=ATOM3D_LBA_REVISION,
    )
    expected_identity = result["dataset_summary"]["validation_identity_sha256"]
    observed_identity = TRAIN["_sample_identity_hash"](validation_samples)
    if observed_identity != expected_identity:
        raise ValueError("validation sample identity changed")
    validation_samples = TRAIN["_with_matched_sparse_edges"](
        validation_samples,
        split="val-analysis",
    )
    observed_topology = TRAIN["_topology_hash"](validation_samples)

    normalizer_record = result["dataset_summary"]["target_normalizer"]
    normalizer = TargetNormalizer(
        mean=torch.tensor(normalizer_record["mean"], dtype=torch.float32),
        std=torch.tensor(normalizer_record["std"], dtype=torch.float32),
    ).to(device=device, dtype=torch.float32)
    egnn_width = int(result["model_summary"]["matched_egnn_width"])
    arm_records = {
        record["arm"]: record for record in result["arm_results"]
    }
    predictions: dict[str, torch.Tensor] = {}
    target: torch.Tensor | None = None
    metrics: dict[str, object] = {}
    prediction_files: dict[str, str] = {}

    for arm in ("candidate", "incumbent", "egnn"):
        record = arm_records[arm]
        checkpoint_path = _resolve_path(record["best_checkpoint"])
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        model = TRAIN["_build_model"](arm, egnn_width)
        model.load_state_dict(checkpoint["model_state"])
        observed_state = TRAIN["_state_hash"](model)
        if observed_state != record["best_state_sha256"]:
            raise ValueError(f"{arm} best checkpoint state hash changed")
        model = model.to(device=device, dtype=torch.float32)
        arm_prediction, arm_target = _predict(
            model,
            validation_samples,
            normalizer=normalizer,
            batch_size=int(result["batch_size"]),
        )
        if target is None:
            target = arm_target
        elif not torch.equal(target, arm_target):
            raise ValueError("validation targets differ between model arms")
        predictions[arm] = arm_prediction
        observed_metrics = TRAIN["_regression_metrics"](
            arm_prediction,
            arm_target,
        )
        expected_metrics = record["best_validation"]
        _assert_metrics_match(arm, observed_metrics, expected_metrics)
        metrics[arm] = observed_metrics
        prediction_path = args.output.with_name(
            f"{args.output.stem}-{arm}-predictions.jsonl"
        )
        _write_predictions(
            prediction_path,
            validation_samples,
            arm_prediction,
            arm_target,
        )
        prediction_files[arm] = str(prediction_path)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if target is None:
        raise RuntimeError("no validation predictions were produced")
    paired = {
        baseline: _paired_rmse_bootstrap(
            predictions["candidate"],
            predictions[baseline],
            target,
            replicates=args.bootstrap_replicates,
            seed=ANALYSIS_SEED,
        )
        for baseline in ("incumbent", "egnn")
    }
    analysis = {
        "schema_version": 1,
        "training_result": str(args.result),
        "training_result_sha256": _file_sha256(args.result),
        "dataset_revision": ATOM3D_LBA_REVISION,
        "validation_size": len(validation_samples),
        "validation_identity_sha256": observed_identity,
        "validation_topology_sha256": observed_topology,
        "note": (
            "Paired nonparametric bootstrap over validation complexes only; "
            "this is not uncertainty over model seeds or protein clusters."
        ),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": ANALYSIS_SEED,
        "metrics": metrics,
        "candidate_paired_rmse": paired,
        "prediction_files": prediction_files,
        "validation_evaluated": True,
        "test_evaluated": False,
    }
    _write_json(args.output, analysis)
    print(json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False))
    return 0


@torch.no_grad()
def _predict(
    model: torch.nn.Module,
    samples: Sequence[GraphSample],
    *,
    normalizer: TargetNormalizer,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    model.eval()
    for start in range(0, len(samples), batch_size):
        batch = collate_graphs(samples[start : start + batch_size])
        batch = batch.to(device=device, dtype=dtype)
        with nullcontext():
            prediction = predict_graph_scalar(model, batch)
        predictions.append(normalizer.inverse(prediction.float()).cpu().reshape(-1))
        targets.append(batch.target.float().cpu().reshape(-1))
    return torch.cat(predictions), torch.cat(targets)


def _paired_rmse_bootstrap(
    candidate_prediction: torch.Tensor,
    baseline_prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    if (
        candidate_prediction.shape != target.shape
        or baseline_prediction.shape != target.shape
        or target.ndim != 1
        or target.numel() == 0
    ):
        raise ValueError("paired bootstrap requires matching nonempty vectors")
    candidate_square = (candidate_prediction.double() - target.double()).square()
    baseline_square = (baseline_prediction.double() - target.double()).square()
    point_delta = float(
        candidate_square.mean().sqrt() - baseline_square.mean().sqrt()
    )
    generator = torch.Generator().manual_seed(seed)
    chunks: list[torch.Tensor] = []
    remaining = replicates
    while remaining:
        count = min(1_000, remaining)
        index = torch.randint(
            target.numel(),
            (count, target.numel()),
            generator=generator,
        )
        delta = (
            candidate_square[index].mean(dim=1).sqrt()
            - baseline_square[index].mean(dim=1).sqrt()
        )
        chunks.append(delta)
        remaining -= count
    distribution = torch.cat(chunks)
    return {
        "candidate_minus_baseline_rmse_pK": point_delta,
        "bootstrap_mean_delta_pK": float(distribution.mean()),
        "ci95_low_pK": float(torch.quantile(distribution, 0.025)),
        "ci95_high_pK": float(torch.quantile(distribution, 0.975)),
        "probability_candidate_lower_rmse": float(
            (distribution < 0.0).double().mean()
        ),
        "replicates": replicates,
    }


def _assert_metrics_match(
    arm: str,
    observed: dict[str, float | int | None],
    expected: dict[str, float | int | None],
) -> None:
    for name in ("mae_pK", "rmse_pK", "pearson", "spearman"):
        left = observed[name]
        right = expected[name]
        if left is None or right is None:
            if left != right:
                raise ValueError(f"{arm} {name} nullability changed")
            continue
        if not math_isclose(float(left), float(right), tolerance=1e-12):
            raise ValueError(
                f"{arm} {name} changed: observed {left}, expected {right}"
            )


def math_isclose(left: float, right: float, *, tolerance: float) -> bool:
    return abs(left - right) <= tolerance * max(1.0, abs(left), abs(right))


def _write_predictions(
    path: Path,
    samples: Sequence[GraphSample],
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> None:
    rows = []
    for sample, predicted, observed in zip(
        samples,
        prediction.tolist(),
        target.tolist(),
        strict=True,
    ):
        rows.append(
            json.dumps(
                {
                    "sample_id": sample.sample_id,
                    "target_pK": observed,
                    "prediction_pK": predicted,
                    "error_pK": predicted - observed,
                },
                sort_keys=True,
                allow_nan=False,
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(rows) + "\n")
    temporary.replace(path)


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _file_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ATOM3D-LBA paired analysis failed: {error}", file=sys.stderr)
        raise
