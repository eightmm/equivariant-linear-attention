#!/usr/bin/env python3
"""Screen the whitened global read over its ridge grid on ATOM3D-LBA ID30.

The whitened lane replaces the metric of the global read rather than its kernel
weights: it evaluates `phi_i^T (G + lambda I)^-1 S` with
`lambda = ridge * tr(G)/F`, so `ridge` is the single hyperparameter and the
large-shrinkage limit is a scaled unnormalized kernel numerator, not the
query-normalized incumbent read. A bounded probe on real cached data showed the
mechanism is active and monotone in `ridge`
(`docs/WHITENED_GLOBAL_READ_20260727.md`), but no accuracy evidence existed.

This runner executes the registered screen in one process: it loads the official
split once, materializes and hashes one candidate topology, and trains the
current candidate plus one whitened arm per ridge value on that same in-memory
topology with identical features, batches, optimizer, and update budget.

Thresholds are fixed before any outcome is inspected. Validation only; the test
split is structurally inadmissible to the underlying runner.
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import time

import torch

from equivariant_attention.benchmarking import GraphSample
from equivariant_attention.pdbbind import (
    ATOM3D_LBA_REVISION,
    load_atom3d_lba_split_samples,
    topology_sha256,
)
from equivariant_attention.reproducibility import configure_reproducibility
from equivariant_attention.training import fit_target_normalizer


RUN_ID = "whitened-ridge-screen-20260728"
MODEL_SEED = 44
ORDER_SEED = 44
RIDGE_GRID: tuple[float, ...] = (0.5, 0.1, 0.01)
MATCHED_EPOCHS = 20
BASELINE_ARM = "candidate"
MAXIMUM_REGRESSION_PK = 0.050
MAXIMUM_LATENCY_RATIO = 1.25
MAXIMUM_MEMORY_RATIO = 1.25
FROZEN_TOPOLOGY_SHA256 = (
    "57f40fb157e6416558db5507d95c3a5e4f828881e0bc92e142e1b85de802dc6c"
)
TRAIN_LBA = runpy.run_path(str(Path(__file__).with_name("train_lba_id30.py")))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/atom3d_lba"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=MATCHED_EPOCHS)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--model-seed", type=int, default=MODEL_SEED)
    parser.add_argument("--order-seed", type=int, default=ORDER_SEED)
    parser.add_argument(
        "--ridge-grid",
        nargs="+",
        type=float,
        default=list(RIDGE_GRID),
        help="dimensionless whitened shrinkage values, one arm each",
    )
    parser.add_argument("--budget-seconds", type=float, default=2_400.0)
    parser.add_argument("--finalization-grace-seconds", type=float, default=90.0)
    parser.add_argument(
        "--expect-topology",
        default=FROZEN_TOPOLOGY_SHA256,
        help="required topology SHA-256; 'none' skips the check",
    )
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.epochs <= 0:
        parser.error("batch size and epochs must be positive")
    if not 0 <= args.warmup_epochs <= args.epochs:
        parser.error("--warmup-epochs must lie in [0, epochs]")
    if args.model_seed < 0 or args.order_seed < 0:
        parser.error("seeds must be nonnegative")
    if args.budget_seconds <= 0.0 or args.finalization_grace_seconds < 0.0:
        parser.error("budget must be positive and grace must be nonnegative")
    if not args.ridge_grid:
        parser.error("--ridge-grid must contain at least one value")
    if len(set(args.ridge_grid)) != len(args.ridge_grid):
        parser.error("--ridge-grid must not repeat a value")
    for ridge in args.ridge_grid:
        if not math.isfinite(ridge) or ridge <= 0.0:
            parser.error("every ridge must be finite and positive")
    for name in ("train_limit", "val_limit"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def _arm_name(ridge: float) -> str:
    return f"whitened_ridge_{ridge:g}".replace(".", "p")


def _plan(args: argparse.Namespace) -> dict[str, object]:
    limited = args.train_limit is not None or args.val_limit is not None
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "question": (
            "Does whitening the global read improve fixed-budget ATOM3D-LBA ID30 "
            "validation RMSE, and at which shrinkage?"
        ),
        "hypothesis": (
            "Dividing out the dominant near-constant kernel direction gives the "
            "global path usable mid-range selectivity on large complexes, so at "
            "least one ridge improves last-epoch validation RMSE over the "
            "current candidate."
        ),
        "prediction": (
            "Conservatively, no ridge improves by more than 0.020 pK at this "
            "budget, and the smallest ridge is the most likely to hurt because "
            "about 40% of its equivalent row weights are negative."
        ),
        "dataset": "vector-institute/atom3d-lba",
        "dataset_revision": ATOM3D_LBA_REVISION,
        "split": "official_ID30_train_validation",
        "model": "current squared-RBF gated-plus-grouped LGL candidate",
        "intervention": "whitened global read on the middle global stage",
        "arms": [BASELINE_ARM, *(_arm_name(ridge) for ridge in args.ridge_grid)],
        "ridge_grid": list(args.ridge_grid),
        "shared_topology": "one in-memory hashed candidate list for every arm",
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "batch_size": args.batch_size,
        "model_seed": args.model_seed,
        "order_seed": args.order_seed,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 3e-4,
            "weight_decay": 0.01,
            "grad_clip": 1.0,
            "schedule": "linear warmup then cosine decay",
        },
        "determinism": "strict",
        "dtype": "float32",
        "primary_metric": "last-epoch validation RMSE in pK",
        "acceptance": {
            "must_improve_last_rmse_over_candidate": True,
            "maximum_regression_pK": MAXIMUM_REGRESSION_PK,
            "maximum_step_latency_ratio": MAXIMUM_LATENCY_RATIO,
            "maximum_peak_memory_ratio": MAXIMUM_MEMORY_RATIO,
        },
        "advancement": (
            "an improving ridge inside the resource ceilings advances to a "
            "separate seeds 41--43 confirmation packet; this screen alone "
            "authorizes no default change"
        ),
        "budget_seconds": args.budget_seconds,
        "expected_topology_sha256": (
            None if args.expect_topology == "none" else args.expect_topology
        ),
        "train_limit": args.train_limit,
        "val_limit": args.val_limit,
        "claim_boundary": "smoke_only" if limited else "exploratory_one_seed",
        "validation_evaluated": True,
        "test_evaluated": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan = _plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    reproducibility = configure_reproducibility(seed=args.model_seed, mode="strict")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "summary.json"
    started = time.perf_counter()
    result: dict[str, object] = {
        **plan,
        "status": "loading_data",
        "reproducibility": reproducibility,
        "runtime_environment": TRAIN_LBA["_runtime_environment"](device),
        "git": TRAIN_LBA["_git_provenance"](),
        "source_sha256": _source_hash(),
        "arm_results": [],
    }
    _write_json(result_path, result)

    train_samples, val_samples = _load_splits(args)
    TRAIN_LBA["_validate_splits"](train_samples, val_samples)
    normalizer = fit_target_normalizer(train_samples)

    result["status"] = "building_topology"
    _write_json(result_path, result)
    topology_started = time.perf_counter()
    train_samples = TRAIN_LBA["_with_matched_sparse_edges"](train_samples, split="train")
    val_samples = TRAIN_LBA["_with_matched_sparse_edges"](val_samples, split="val")
    observed_topology = topology_sha256([*train_samples, *val_samples])
    result["dataset_summary"] = {
        "train_size": len(train_samples),
        "validation_size": len(val_samples),
        "train_identity_sha256": TRAIN_LBA["_sample_identity_hash"](train_samples),
        "validation_identity_sha256": TRAIN_LBA["_sample_identity_hash"](val_samples),
        "topology_sha256": observed_topology,
        "candidate_edge_count": sum(
            int(sample.edge_index.shape[1])
            for sample in [*train_samples, *val_samples]
            if sample.edge_index is not None
        ),
        "topology_build_seconds": time.perf_counter() - topology_started,
        "target_normalizer": normalizer.as_dict(),
    }
    _write_json(result_path, result)
    expected = plan["expected_topology_sha256"]
    if expected is not None and observed_topology != expected:
        result["status"] = "failed"
        result["failure"] = "topology identity differs from the frozen hash"
        _write_json(result_path, result)
        raise RuntimeError(f"topology {observed_topology} does not match {expected}")

    remaining = (
        args.budget_seconds
        - (time.perf_counter() - started)
        - args.finalization_grace_seconds
    )
    arms: list[tuple[str, float | None]] = [
        (BASELINE_ARM, None),
        *((_arm_name(ridge), ridge) for ridge in args.ridge_grid),
    ]
    if remaining <= 0.0:
        raise RuntimeError("budget exhausted before arm training")
    per_arm_budget = remaining / len(arms)

    result["status"] = "training"
    _write_json(result_path, result)
    base_args = _train_args(args, budget_seconds=per_arm_budget)
    for name, ridge in arms:
        arm_args = copy.copy(base_args)
        model = TRAIN_LBA["_build_model"](
            BASELINE_ARM if ridge is None else "whitened",
            None,
            model_seed=args.model_seed,
            whitened_ridge=0.1 if ridge is None else ridge,
        )
        record = TRAIN_LBA["_train_arm"](
            arm=BASELINE_ARM if ridge is None else "whitened",
            model=model,
            train_samples=train_samples,
            val_samples=val_samples,
            normalizer=normalizer,
            device=device,
            amp_dtype=None,
            args=arm_args,
            output_dir=args.output_dir / name,
            budget_seconds=per_arm_budget,
        )
        record["screen_arm"] = name
        record["whitened_global_ridge"] = ridge
        record["whitened_mix_magnitude"] = _mix_magnitude(model)
        _append(result, "arm_results", record)
        result["elapsed_seconds"] = time.perf_counter() - started
        _write_json(result_path, result)
        print(
            f"{name}: status={record['status']} "
            f"last_rmse={_metric(record, 'last_epoch_validation', 'rmse_pK'):.6f} "
            f"steps={record.get('global_steps')}",
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result["decision"] = decision(result["arm_results"])
    result["elapsed_seconds"] = time.perf_counter() - started
    result["status"] = (
        "completed"
        if all(
            isinstance(record, Mapping) and record.get("status") == "completed"
            for record in _records(result)
        )
        else "partial"
    )
    _write_json(result_path, result)
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    return 0 if result["status"] == "completed" else 1


def decision(records: object) -> dict[str, object]:
    """Apply the frozen per-ridge advancement rule to the collected arms."""
    if not isinstance(records, list):
        return {"passed": False, "reason": "arm results are missing"}
    by_arm = {
        str(record["screen_arm"]): record
        for record in records
        if isinstance(record, Mapping)
    }
    baseline = by_arm.get(BASELINE_ARM)
    if baseline is None:
        return {"passed": False, "reason": "the candidate baseline is missing"}
    baseline_last = _metric(baseline, "last_epoch_validation", "rmse_pK")
    baseline_best = _metric(baseline, "best_validation", "rmse_pK")
    baseline_latency = float(baseline["step_latency_median_seconds"])
    baseline_memory = baseline.get("peak_cuda_memory_bytes")

    ridges: dict[str, object] = {}
    admitted: list[tuple[float, str]] = []
    for name, record in by_arm.items():
        if name == BASELINE_ARM:
            continue
        improvement = baseline_last - _metric(
            record, "last_epoch_validation", "rmse_pK"
        )
        best_improvement = baseline_best - _metric(
            record, "best_validation", "rmse_pK"
        )
        latency_ratio = (
            float(record["step_latency_median_seconds"]) / baseline_latency
        )
        memory_ratio = _ratio(
            record.get("peak_cuda_memory_bytes"), baseline_memory
        )
        mix = record.get("whitened_mix_magnitude")
        criteria = {
            "improves_last_rmse": improvement > 0.0,
            f"regression_at_most_{MAXIMUM_REGRESSION_PK}_pK": (
                -improvement <= MAXIMUM_REGRESSION_PK
            ),
            f"latency_ratio_at_most_{MAXIMUM_LATENCY_RATIO}": (
                latency_ratio <= MAXIMUM_LATENCY_RATIO
            ),
            f"peak_memory_ratio_at_most_{MAXIMUM_MEMORY_RATIO}": (
                memory_ratio is None or memory_ratio <= MAXIMUM_MEMORY_RATIO
            ),
            "completed_and_finite": (
                record.get("status") == "completed"
                and math.isfinite(improvement)
                and record.get("test_evaluated") is not True
            ),
            "matched_update_count": (
                record.get("global_steps") == baseline.get("global_steps")
            ),
        }
        passed = all(criteria.values())
        ridges[name] = {
            "ridge": record.get("whitened_global_ridge"),
            "last_rmse_improvement_pK": improvement,
            "best_rmse_improvement_pK": best_improvement,
            "step_latency_ratio": latency_ratio,
            "peak_memory_ratio": memory_ratio,
            "whitened_mix_magnitude": mix,
            "lane_active": (
                None
                if not isinstance(mix, Mapping)
                else bool(float(mix.get("scalar_absmax", 0.0)) > 0.0)
            ),
            "criteria": criteria,
            "passed": passed,
        }
        if passed:
            admitted.append((improvement, name))
    selected = max(admitted)[1] if admitted else None
    return {
        "baseline_last_validation_rmse_pK": baseline_last,
        "baseline_best_validation_rmse_pK": baseline_best,
        "ridges": ridges,
        "selected_arm": selected,
        "selected_ridge": (
            None if selected is None else ridges[selected]["ridge"]  # type: ignore[index]
        ),
        "passed": selected is not None,
        "default_change_authorized": False,
        "advances_to_multiseed_confirmation": selected is not None,
        "evidence_grade": "exploratory_one_seed",
    }


def _mix_magnitude(model: torch.nn.Module) -> dict[str, object]:
    """Best-checkpoint magnitude of the zero-initialized whitened lane gates."""
    scalar: list[float] = []
    vector: list[float] = []
    for layer in getattr(model, "layers", []):
        scalar_mix = getattr(layer, "whitened_scalar_mix", None)
        vector_mix = getattr(layer, "whitened_vector_mix", None)
        if scalar_mix is None or vector_mix is None:
            continue
        scalar.append(float(scalar_mix.detach().abs().max()))
        vector.append(float(vector_mix.detach().abs().max()))
    return {
        "per_layer_scalar_absmax": scalar,
        "per_layer_vector_absmax": vector,
        "scalar_absmax": max(scalar) if scalar else 0.0,
        "vector_absmax": max(vector) if vector else 0.0,
    }


def _load_splits(
    args: argparse.Namespace,
) -> tuple[list[GraphSample], list[GraphSample]]:
    train_indices = None if args.train_limit is None else tuple(range(args.train_limit))
    val_indices = None if args.val_limit is None else tuple(range(args.val_limit))
    return (
        load_atom3d_lba_split_samples(
            args.data_root,
            split="train",
            revision=ATOM3D_LBA_REVISION,
            indices=train_indices,
        ),
        load_atom3d_lba_split_samples(
            args.data_root,
            split="val",
            revision=ATOM3D_LBA_REVISION,
            indices=val_indices,
        ),
    )


def _train_args(
    args: argparse.Namespace, *, budget_seconds: float
) -> argparse.Namespace:
    train_args = TRAIN_LBA["parse_args"](
        [
            str(args.output_dir),
            "--arms",
            "candidate",
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
            "--max-epochs",
            str(args.epochs),
            "--min-epochs",
            str(args.epochs),
            "--patience",
            str(args.epochs),
            "--warmup-epochs",
            str(args.warmup_epochs),
            "--learning-rate",
            "0.0003",
            "--weight-decay",
            "0.01",
            "--model-seed",
            str(args.model_seed),
            "--order-seed",
            str(args.order_seed),
            "--amp-dtype",
            "none",
            "--budget-seconds",
            str(budget_seconds),
        ]
    )
    train_args.train_limit = args.train_limit
    train_args.val_limit = args.val_limit
    return train_args


def _records(result: Mapping[str, object]) -> list[Mapping[str, object]]:
    records = result.get("arm_results")
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, Mapping)]


def _append(result: dict[str, object], key: str, value: object) -> None:
    records = result.get(key)
    if not isinstance(records, list):
        records = []
        result[key] = records
    records.append(value)


def _metric(record: Mapping[str, object], group: str, metric: str) -> float:
    values = record.get(group)
    if not isinstance(values, Mapping):
        raise ValueError(f"missing metric group: {group}")
    return float(values[metric])


def _ratio(numerator: object, denominator: object) -> float | None:
    if numerator is None or denominator is None:
        return None
    denominator_value = float(denominator)
    if denominator_value <= 0.0:
        return None
    return float(numerator) / denominator_value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _source_hash() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for path in (
        Path(__file__).resolve(),
        Path(__file__).with_name("train_lba_id30.py").resolve(),
        root / "src" / "equivariant_attention" / "moment.py",
        root / "src" / "equivariant_attention" / "pdbbind.py",
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
