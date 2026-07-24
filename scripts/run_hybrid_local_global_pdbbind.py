#!/usr/bin/env python3
"""Run same-feature train-only ATOM3D-LBA capacity checks for the hybrid."""

from __future__ import annotations

import argparse
from dataclasses import replace
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
    segment_balanced_knn_edge_index,
)
from equivariant_attention.reproducibility import configure_reproducibility
from equivariant_attention.training import (
    build_regression_model,
    fit_target_normalizer,
)


LEGACY = runpy.run_path(
    str(Path(__file__).with_name("run_registered_pdbbind_overfit.py"))
)
PACKET_ID = "hybrid-local-global-20260724"
DATASET_REVISION = LEGACY["DATASET_REVISION"]
SUBSET_INDICES = LEGACY["SUBSET_INDICES"]
MODEL_SEED = LEGACY["MODEL_SEED"]
MAX_STEPS = LEGACY["MAX_STEPS"]
THRESHOLD_PK = LEGACY["TRAIN_MAE_THRESHOLD_PK"]
MAX_GPU_SECONDS = 600.0
INTRA_K = 16
CROSS_K = 16
CUTOFF_ANGSTROM = 6.0
ARMS = ("incumbent", "candidate", "egnn")

_run_arm = LEGACY["_run_arm"]
_validate_frozen_samples = LEGACY["_validate_frozen_samples"]
_parameter_count = LEGACY["_parameter_count"]
_arm_passed = LEGACY["_arm_passed"]
_write_result = LEGACY["_write_result"]
matched_egnn_width = LEGACY["matched_egnn_width"]
_build_egnn = LEGACY["_build_egnn"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/atom3d_lba"))
    parser.add_argument(
        "--candidate",
        choices=["gated", "gated_grouped"],
        default="gated_grouped",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--budget-seconds", type=float, default=MAX_GPU_SECONDS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.max_steps <= MAX_STEPS:
        parser.error(f"--max-steps must lie in [1, {MAX_STEPS}]")
    if not 0.0 < args.budget_seconds <= MAX_GPU_SECONDS:
        parser.error(f"--budget-seconds must lie in (0, {MAX_GPU_SECONDS}]")
    return args


def _attention_kwargs() -> dict[str, object]:
    return {
        "node_dim": ATOM3D_LBA_NODE_DIM,
        "hidden_dim": LEGACY["HIDDEN_DIM"],
        "num_layers": LEGACY["NUM_LAYERS"],
        "num_heads": LEGACY["NUM_HEADS"],
        "local_head_counts": (4, 0, 4),
        "local_cutoff": CUTOFF_ANGSTROM,
        "use_key_balancing": False,
    }


def _build_attention(*, candidate: str | None) -> torch.nn.Module:
    torch.manual_seed(MODEL_SEED)
    return build_regression_model(
        **_attention_kwargs(),
        use_gated_local_transport=candidate is not None,
        use_grouped_invariant_normalization=candidate == "gated_grouped",
    )


def _run_plan(args: argparse.Namespace) -> dict[str, object]:
    return {
        "schema_version": 1,
        "packet_id": PACKET_ID,
        "dataset": "vector-institute/atom3d-lba",
        "dataset_revision": DATASET_REVISION,
        "split": "train",
        "subset_indices": list(SUBSET_INDICES),
        "candidate": args.candidate,
        "device": args.device,
        "determinism": "strict",
        "arms": list(ARMS),
        "max_steps": args.max_steps,
        "threshold_train_mae_pK": THRESHOLD_PK,
        "budget_seconds": args.budget_seconds,
        "topology": {
            "kind": "segment_balanced_knn_candidates",
            "self_edges": True,
            "intra_k": INTRA_K,
            "cross_k": CROSS_K,
            "cutoff_angstrom": CUTOFF_ANGSTROM,
            "identical_for_all_arms": True,
        },
        "raw_feature_contract": (
            "identical node_feats, pos, readout_mask, target, and edge_index"
        ),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": LEGACY["LEARNING_RATE"],
            "weight_decay": LEGACY["WEIGHT_DECAY"],
            "gradient_clip": LEGACY["GRAD_CLIP"],
        },
        "batch_size": LEGACY["BATCH_SIZE"],
        "model_seed": MODEL_SEED,
        "order_seed": LEGACY["ORDER_SEED"],
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
    raw_samples = load_atom3d_lba_samples(
        args.data_root,
        indices=SUBSET_INDICES,
        revision=DATASET_REVISION,
    )
    _validate_frozen_samples(raw_samples)
    samples = _with_matched_sparse_edges(raw_samples)
    normalizer = fit_target_normalizer(samples)

    candidate_probe = _build_attention(candidate=args.candidate)
    candidate_parameters = _parameter_count(candidate_probe)
    incumbent_parameters = _parameter_count(_build_attention(candidate=None))
    parameter_ratio = candidate_parameters / incumbent_parameters
    if parameter_ratio > 1.05:
        raise RuntimeError("hybrid candidate exceeds the frozen 1.05 parameter ratio")
    egnn_width = matched_egnn_width(
        target_parameter_count=candidate_parameters,
        node_dim=ATOM3D_LBA_NODE_DIM,
        num_layers=LEGACY["NUM_LAYERS"],
    )
    del candidate_probe

    result: dict[str, object] = {
        **plan,
        "status": "running",
        "reproducibility": reproducibility,
        "source_sha256": _source_hash(),
        "sample_ids": [sample.sample_id for sample in samples],
        "sample_identity_sha256": _sample_identity_hash(samples),
        "node_counts": [int(sample.pos.shape[0]) for sample in samples],
        "ligand_counts": [
            int(sample.readout_mask.sum())
            for sample in samples
            if sample.readout_mask is not None
        ],
        "edge_counts": [
            int(sample.edge_index.shape[1])
            for sample in samples
            if sample.edge_index is not None
        ],
        "incumbent_parameter_count": incumbent_parameters,
        "candidate_parameter_count": candidate_parameters,
        "candidate_parameter_ratio": parameter_ratio,
        "matched_egnn_width": egnn_width,
        "target_normalizer": normalizer.as_dict(),
        "arm_results": [],
        "validation_evaluated": False,
        "test_evaluated": False,
    }
    _write_result(args.output, result)
    started = time.perf_counter()
    per_arm_budget = args.budget_seconds / len(ARMS)
    for arm in ARMS:
        if arm == "incumbent":
            model = _build_attention(candidate=None)
        elif arm == "candidate":
            model = _build_attention(candidate=args.candidate)
        else:
            model = _build_egnn(egnn_width)
        arm_result = _run_arm(
            arm=arm,
            model=model,
            samples=samples,
            normalizer=normalizer,
            device=device,
            max_steps=args.max_steps,
            threshold=THRESHOLD_PK,
            batch_size=LEGACY["BATCH_SIZE"],
            eval_interval=LEGACY["EVAL_INTERVAL"],
            budget_seconds=per_arm_budget,
        )
        # The legacy runner only reported this field for its EGNN arm even
        # though collate_graphs forwards the supplied topology to every model.
        # This experiment deliberately gives every arm the same sparse edges,
        # so record the consumed input edge count uniformly.
        arm_result["edge_count_with_self"] = sum(
            int(sample.edge_index.shape[1])
            for sample in samples
            if sample.edge_index is not None
        )
        arm_result["architecture"] = _architecture(arm, args.candidate)
        result["arm_results"].append(arm_result)
        # Keep the legacy key for its threshold helper.
        result["arms"] = result["arm_results"]
        result["elapsed_seconds"] = time.perf_counter() - started
        _write_result(args.output, result)

    result["status"] = "completed"
    result["elapsed_seconds"] = time.perf_counter() - started
    result["overfit_passed"] = {
        arm: _arm_passed(result, arm)
        for arm in ARMS
    }
    _write_result(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


def _with_matched_sparse_edges(samples: Sequence[object]) -> list[object]:
    result = []
    for sample in samples:
        if sample.readout_mask is None:
            raise ValueError("ATOM3D-LBA sample requires a ligand readout mask")
        edge_index = segment_balanced_knn_edge_index(
            sample.pos,
            sample.readout_mask,
            intra_k=INTRA_K,
            cross_k=CROSS_K,
            cutoff=CUTOFF_ANGSTROM,
        )
        result.append(replace(sample, edge_index=edge_index))
    return result


def _architecture(arm: str, candidate: str) -> dict[str, object]:
    if arm == "incumbent":
        return {
            "kind": "factorized_moment",
            "route": "lgl",
            "gated_local_transport": False,
            "grouped_invariant_normalization": False,
        }
    if arm == "candidate":
        return {
            "kind": "factorized_moment",
            "route": "lgl",
            "gated_local_transport": True,
            "grouped_invariant_normalization": candidate == "gated_grouped",
        }
    return {
        "kind": "internal_static_egnn_baseline",
        "official_reproduction": False,
    }


def _sample_identity_hash(samples: Sequence[object]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        payload = json.dumps(
            {
                "id": sample.sample_id,
                "nodes": int(sample.pos.shape[0]),
                "edges": int(sample.edge_index.shape[1]),
            },
            sort_keys=True,
        ).encode()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _source_hash() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__).resolve(),
        root / "scripts" / "run_registered_pdbbind_overfit.py",
        root / "src" / "equivariant_attention" / "_egnn_baseline.py",
        root / "src" / "equivariant_attention" / "moment.py",
        root / "src" / "equivariant_attention" / "pdbbind.py",
        root / "src" / "equivariant_attention" / "training.py",
        root / "artifacts" / PACKET_ID / "scope.md",
        root / "PROJECT.md",
        root / "uv.lock",
    ]
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode()
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
        print(f"hybrid PDBBind run failed: {error}", file=sys.stderr)
        raise
