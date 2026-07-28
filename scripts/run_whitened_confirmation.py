#!/usr/bin/env python3
"""Run the frozen three-seed LBA confirmation for the whitened global read.

The preregistered primary is ridge 0.1.  Ridge 0.01 is carried as a secondary
sensitivity arm because seed 44 could not separate the two; it cannot rescue a
failed primary ridge.  Every arm uses one in-memory, hashed ID30 topology and
the test split remains structurally inaccessible.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import statistics
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


RUN_ID = "whitened-lba-confirmation-20260728"
SEEDS = (41, 42, 43)
PRIMARY_RIDGE = 0.1
SECONDARY_RIDGE = 0.01
RIDGES = (PRIMARY_RIDGE, SECONDARY_RIDGE)
MATCHED_EPOCHS = 20
MINIMUM_MEAN_IMPROVEMENT_PK = 0.020
MINIMUM_IMPROVING_SEEDS = 2
MAXIMUM_PAIRED_REGRESSION_PK = 0.050
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
    parser.add_argument("--budget-seconds", type=float, default=4_800.0)
    parser.add_argument("--finalization-grace-seconds", type=float, default=120.0)
    parser.add_argument(
        "--expect-topology",
        default=FROZEN_TOPOLOGY_SHA256,
        help="required topology SHA-256; 'none' skips the check for debug smokes",
    )
    parser.add_argument(
        "--qm9-safety-summary",
        type=Path,
        help="completed summary from run_whitened_qm9_smoke.py",
    )
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.epochs <= 0:
        parser.error("batch size and epochs must be positive")
    limited = args.train_limit is not None or args.val_limit is not None
    if args.epochs != MATCHED_EPOCHS and not limited:
        parser.error(f"full confirmation requires exactly {MATCHED_EPOCHS} epochs")
    if not 0 <= args.warmup_epochs <= args.epochs:
        parser.error("--warmup-epochs must lie in [0, epochs]")
    if args.budget_seconds <= 0.0 or args.finalization_grace_seconds < 0.0:
        parser.error("budget must be positive and grace must be nonnegative")
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
            "Does the ridge-0.1 whitened global read improve fixed-budget "
            "ATOM3D-LBA ID30 validation RMSE across seeds 41--43?"
        ),
        "hypothesis": (
            "Ridge 0.1 improves mean paired last-epoch validation RMSE by at "
            "least 0.020 pK, wins at least two seeds, and has no paired "
            "regression larger than 0.050 pK."
        ),
        "prediction": (
            "The seed-44 gain shrinks but remains at or above 0.020 pK on the "
            "paired mean; ridge 0.01 remains directionally similar."
        ),
        "dataset": "vector-institute/atom3d-lba",
        "dataset_revision": ATOM3D_LBA_REVISION,
        "split": "official_ID30_train_validation",
        "prediction_unit": "one protein-pocket/ligand crystal complex",
        "inference_features": (
            "bound pocket and ligand atom categories, segment identity, and "
            "Cartesian coordinates"
        ),
        "model": "current squared-RBF gated-plus-grouped LGL candidate",
        "intervention": "whitened scalar and 1o global value reads",
        "model_seeds": list(SEEDS),
        "order_seeds": list(SEEDS),
        "primary_ridge": PRIMARY_RIDGE,
        "secondary_ridge": SECONDARY_RIDGE,
        "secondary_policy": (
            "sensitivity only; ridge 0.01 cannot rescue a failed ridge-0.1 primary"
        ),
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "batch_size": args.batch_size,
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
        "secondary_metrics": [
            "best-checkpoint validation RMSE and MAE",
            "last and best Pearson/Spearman",
            "pre-clip norm and effective gradient scale",
        ],
        "acceptance": {
            "minimum_mean_improvement_pK": MINIMUM_MEAN_IMPROVEMENT_PK,
            "minimum_improving_seeds": MINIMUM_IMPROVING_SEEDS,
            "maximum_paired_regression_pK": MAXIMUM_PAIRED_REGRESSION_PK,
            "maximum_step_latency_ratio": MAXIMUM_LATENCY_RATIO,
            "maximum_peak_memory_ratio": MAXIMUM_MEMORY_RATIO,
        },
        "topology_contract": "deterministic float64 squared distance, ties retained",
        "expected_topology_sha256": (
            None if args.expect_topology == "none" else args.expect_topology
        ),
        "qm9_safety_summary": (
            None
            if args.qm9_safety_summary is None
            else str(args.qm9_safety_summary)
        ),
        "budget_seconds": args.budget_seconds,
        "finalization_grace_seconds": args.finalization_grace_seconds,
        "train_limit": args.train_limit,
        "val_limit": args.val_limit,
        "claim_boundary": "debug_smoke_only" if limited else "paired_three_seed",
        "validation_evaluated": True,
        "test_evaluated": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan = _plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    limited = args.train_limit is not None or args.val_limit is not None
    qm9_safety = _load_qm9_safety(args.qm9_safety_summary, required=not limited)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    reproducibility = configure_reproducibility(seed=SEEDS[0], mode="strict")
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
        "qm9_safety": qm9_safety,
        "arm_results": [],
    }
    _write_json(result_path, result)

    train_samples, val_samples = _load_splits(args)
    TRAIN_LBA["_validate_splits"](train_samples, val_samples)
    normalizer = fit_target_normalizer(train_samples)

    result["status"] = "building_topology"
    _write_json(result_path, result)
    topology_started = time.perf_counter()
    train_samples = TRAIN_LBA["_with_matched_sparse_edges"](
        train_samples, split="train"
    )
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
    arm_count = len(SEEDS) * (1 + len(RIDGES))
    if remaining <= 0.0:
        raise RuntimeError("budget exhausted before arm training")
    per_arm_budget = remaining / arm_count

    result["status"] = "training"
    _write_json(result_path, result)
    for seed in SEEDS:
        base_args = _train_args(args, seed=seed, budget_seconds=per_arm_budget)
        for name, ridge in [
            ("candidate", None),
            *((_arm_name(value), value) for value in RIDGES),
        ]:
            arm_args = copy.copy(base_args)
            model = TRAIN_LBA["_build_model"](
                "candidate" if ridge is None else "whitened",
                None,
                model_seed=seed,
                whitened_ridge=PRIMARY_RIDGE if ridge is None else ridge,
            )
            paired_hash = _paired_base_initial_state_sha256(model)
            record = TRAIN_LBA["_train_arm"](
                arm="candidate" if ridge is None else "whitened",
                model=model,
                train_samples=train_samples,
                val_samples=val_samples,
                normalizer=normalizer,
                device=device,
                amp_dtype=None,
                args=arm_args,
                output_dir=args.output_dir / f"seed{seed}" / name,
                budget_seconds=per_arm_budget,
            )
            record["confirmation_arm"] = name
            record["whitened_global_ridge"] = ridge
            record["model_seed"] = seed
            record["order_seed"] = seed
            record["paired_base_initial_state_sha256"] = paired_hash
            record["whitened_mix_magnitude"] = _mix_magnitude(model)
            _append(result, "arm_results", record)
            result["elapsed_seconds"] = time.perf_counter() - started
            _write_json(result_path, result)
            print(
                f"seed={seed} arm={name} status={record['status']} "
                f"last_rmse={_metric(record, 'last_epoch_validation', 'rmse_pK'):.6f} "
                f"steps={record.get('global_steps')}",
                flush=True,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    safety_passed = bool(
        isinstance(qm9_safety, Mapping) and qm9_safety.get("passed")
    )
    result["decision"] = decision(
        result["arm_results"],
        qm9_safety_passed=safety_passed,
    )
    result["elapsed_seconds"] = time.perf_counter() - started
    result["status"] = (
        "completed"
        if all(
            record.get("status") == "completed"
            for record in _records(result)
        )
        else "partial"
    )
    _write_json(result_path, result)
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    return 0 if result["status"] == "completed" else 1


def decision(
    records: object,
    *,
    qm9_safety_passed: bool = False,
) -> dict[str, object]:
    if not isinstance(records, list):
        return {"passed": False, "reason": "arm results are missing"}
    by_seed: dict[int, dict[str, Mapping[str, object]]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        seed = int(record["model_seed"])
        by_seed.setdefault(seed, {})[str(record["confirmation_arm"])] = record
    required = {"candidate", *(_arm_name(ridge) for ridge in RIDGES)}
    missing = [
        seed for seed in SEEDS if set(by_seed.get(seed, {})) != required
    ]
    if missing:
        return {"passed": False, "reason": f"incomplete seeds: {missing}"}

    primary = _ridge_decision(by_seed, ridge=PRIMARY_RIDGE)
    secondary = _ridge_decision(by_seed, ridge=SECONDARY_RIDGE)
    primary_passed = bool(primary["passed"])
    passed = primary_passed and qm9_safety_passed
    return {
        "primary": primary,
        "secondary": secondary,
        "qm9_safety_passed": qm9_safety_passed,
        "passed": passed,
        "lba_confirmation_passed": primary_passed,
        "selected_ridge": PRIMARY_RIDGE if primary_passed else None,
        "exact_ridge_resolved": primary_passed,
        "secondary_can_rescue_primary": False,
        "default_change_authorized": passed,
        "next_stage_authorized": passed,
        "evidence_grade": "paired_three_seed_same_validation_harness",
    }


def _ridge_decision(
    by_seed: Mapping[int, Mapping[str, Mapping[str, object]]],
    *,
    ridge: float,
) -> dict[str, object]:
    name = _arm_name(ridge)
    improvements: list[float] = []
    best_improvements: list[float] = []
    latency_ratios: list[float] = []
    memory_ratios: list[float] = []
    per_seed: dict[str, object] = {}
    complete = True
    matched_updates = True
    paired_initialization = True
    finite = True
    for seed in SEEDS:
        baseline = by_seed[seed]["candidate"]
        candidate = by_seed[seed][name]
        baseline_last = _metric(
            baseline, "last_epoch_validation", "rmse_pK"
        )
        candidate_last = _metric(
            candidate, "last_epoch_validation", "rmse_pK"
        )
        baseline_best = _metric(baseline, "best_validation", "rmse_pK")
        candidate_best = _metric(candidate, "best_validation", "rmse_pK")
        improvement = baseline_last - candidate_last
        best_improvement = baseline_best - candidate_best
        latency = _ratio(
            candidate.get("step_latency_median_seconds"),
            baseline.get("step_latency_median_seconds"),
        )
        memory = _ratio(
            candidate.get("peak_cuda_memory_bytes"),
            baseline.get("peak_cuda_memory_bytes"),
        )
        improvements.append(improvement)
        best_improvements.append(best_improvement)
        latency_ratios.append(latency)
        memory_ratios.append(memory)
        complete = complete and all(
            record.get("status") == "completed"
            and record.get("test_evaluated") is False
            for record in (baseline, candidate)
        )
        matched_updates = matched_updates and (
            baseline.get("global_steps") == candidate.get("global_steps")
        )
        initial_matches = (
            baseline.get("paired_base_initial_state_sha256")
            == candidate.get("paired_base_initial_state_sha256")
        )
        paired_initialization = paired_initialization and initial_matches
        values = (
            baseline_last,
            candidate_last,
            baseline_best,
            candidate_best,
            improvement,
            best_improvement,
            latency,
            memory,
        )
        finite = finite and all(math.isfinite(value) for value in values)
        per_seed[str(seed)] = {
            "baseline_last_validation_rmse_pK": baseline_last,
            "whitened_last_validation_rmse_pK": candidate_last,
            "last_rmse_improvement_pK": improvement,
            "baseline_best_validation_rmse_pK": baseline_best,
            "whitened_best_validation_rmse_pK": candidate_best,
            "best_rmse_improvement_pK": best_improvement,
            "step_latency_ratio": latency,
            "peak_memory_ratio": memory,
            "paired_base_initialization_matches": initial_matches,
            "baseline_gradient_monitor": baseline.get("gradient_monitor"),
            "whitened_gradient_monitor": candidate.get("gradient_monitor"),
            "whitened_mix_magnitude": candidate.get("whitened_mix_magnitude"),
        }

    mean_improvement = statistics.fmean(improvements)
    worst_regression = max(0.0, -min(improvements))
    criteria = {
        f"mean_improvement_at_least_{MINIMUM_MEAN_IMPROVEMENT_PK}_pK": (
            mean_improvement >= MINIMUM_MEAN_IMPROVEMENT_PK
        ),
        f"improving_seeds_at_least_{MINIMUM_IMPROVING_SEEDS}": (
            sum(value > 0.0 for value in improvements)
            >= MINIMUM_IMPROVING_SEEDS
        ),
        f"worst_regression_at_most_{MAXIMUM_PAIRED_REGRESSION_PK}_pK": (
            worst_regression <= MAXIMUM_PAIRED_REGRESSION_PK
        ),
        f"latency_ratio_at_most_{MAXIMUM_LATENCY_RATIO}": (
            max(latency_ratios) <= MAXIMUM_LATENCY_RATIO
        ),
        f"peak_memory_ratio_at_most_{MAXIMUM_MEMORY_RATIO}": (
            max(memory_ratios) <= MAXIMUM_MEMORY_RATIO
        ),
        "all_arms_completed_without_test_access": complete,
        "matched_update_counts": matched_updates,
        "paired_base_initialization": paired_initialization,
        "finite_metrics": finite,
    }
    return {
        "ridge": ridge,
        "per_seed": per_seed,
        "mean_improvement_pK": mean_improvement,
        "sample_std_improvement_pK": statistics.stdev(improvements),
        "mean_best_checkpoint_improvement_pK": statistics.fmean(
            best_improvements
        ),
        "improving_seed_count": sum(value > 0.0 for value in improvements),
        "worst_paired_regression_pK": worst_regression,
        "maximum_step_latency_ratio": max(latency_ratios),
        "maximum_peak_memory_ratio": max(memory_ratios),
        "criteria": criteria,
        "passed": all(criteria.values()),
    }


def _load_qm9_safety(
    path: Path | None,
    *,
    required: bool,
) -> dict[str, object] | None:
    if path is None:
        if required:
            raise ValueError("--qm9-safety-summary is required for full confirmation")
        return None
    summary = json.loads(path.read_text())
    if not isinstance(summary, dict):
        raise ValueError("QM9 safety summary must be a JSON object")
    decision_record = summary.get("decision")
    if not isinstance(decision_record, Mapping) or not decision_record.get("passed"):
        raise ValueError("QM9 safety smoke did not pass")
    if summary.get("test_evaluated") is not False:
        raise ValueError("QM9 safety summary violates the test boundary")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "passed": True,
        "decision": dict(decision_record),
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
    args: argparse.Namespace,
    *,
    seed: int,
    budget_seconds: float,
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
            "--grad-clip",
            "1.0",
            "--model-seed",
            str(seed),
            "--order-seed",
            str(seed),
            "--amp-dtype",
            "none",
            "--budget-seconds",
            str(budget_seconds),
        ]
    )
    train_args.train_limit = args.train_limit
    train_args.val_limit = args.val_limit
    return train_args


def _paired_base_initial_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if "whitened_scalar_mix" in name or "whitened_vector_mix" in name:
            continue
        metadata = json.dumps(
            [name, str(value.dtype), list(value.shape)],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        raw = (
            value.detach()
            .cpu()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _mix_magnitude(model: torch.nn.Module) -> dict[str, object]:
    scalar: list[float] = []
    vector: list[float] = []
    for layer in getattr(model, "layers", []):
        scalar_mix = getattr(layer, "whitened_scalar_mix", None)
        vector_mix = getattr(layer, "whitened_vector_mix", None)
        if scalar_mix is None or vector_mix is None:
            continue
        scalar.append(float(scalar_mix.detach().abs().max().cpu()))
        vector.append(float(vector_mix.detach().abs().max().cpu()))
    return {
        "per_layer_scalar_absmax": scalar,
        "per_layer_vector_absmax": vector,
        "scalar_absmax": max(scalar) if scalar else 0.0,
        "vector_absmax": max(vector) if vector else 0.0,
    }


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


def _ratio(numerator: object, denominator: object) -> float:
    numerator_value = float(numerator)
    denominator_value = float(denominator)
    if denominator_value <= 0.0:
        raise ValueError("ratio denominator must be positive")
    return numerator_value / denominator_value


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
