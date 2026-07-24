#!/usr/bin/env python3
"""Compare mean and interaction readouts on frozen train-only ATOM3D-LBA rows."""

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

from equivariant_attention.benchmarking import collate_graphs
from equivariant_attention.pdbbind import (
    ATOM3D_LBA_NODE_DIM,
    ATOM3D_LBA_REVISION,
    load_atom3d_lba_samples,
    segment_balanced_knn_edge_index,
)
from equivariant_attention.reproducibility import configure_reproducibility
from equivariant_attention.training import (
    build_regression_model,
    evaluate_regression,
    fit_target_normalizer,
    predict_graph_scalar,
)


LEGACY = runpy.run_path(
    str(Path(__file__).with_name("run_registered_pdbbind_overfit.py"))
)
PACKET_ID = "model-feedback-followup-20260724"
SUBSET_INDICES = tuple(range(16))
MODEL_SEED = 20260723
INTRA_K = 16
CROSS_K = 16
CUTOFF_ANGSTROM = 6.0
MAX_STEPS = 1_000
MAX_GPU_SECONDS = 120.0

_run_arm = LEGACY["_run_arm"]
_validate_frozen_samples = LEGACY["_validate_frozen_samples"]
_write_result = LEGACY["_write_result"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/atom3d_lba"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--budget-seconds", type=float, default=MAX_GPU_SECONDS)
    args = parser.parse_args(argv)
    if not 1 <= args.max_steps <= MAX_STEPS:
        parser.error(f"--max-steps must lie in [1, {MAX_STEPS}]")
    if not 0.0 < args.budget_seconds <= MAX_GPU_SECONDS:
        parser.error(f"--budget-seconds must lie in (0, {MAX_GPU_SECONDS}]")
    return args


def _build_model(readout_mode: str) -> torch.nn.Module:
    torch.manual_seed(MODEL_SEED)
    return build_regression_model(
        node_dim=ATOM3D_LBA_NODE_DIM,
        hidden_dim=64,
        num_layers=3,
        num_heads=4,
        local_head_counts=(4, 0, 4),
        local_cutoff=CUTOFF_ANGSTROM,
        use_key_balancing=False,
        use_gated_local_transport=True,
        use_grouped_invariant_normalization=True,
        readout_mode=readout_mode,
    )


def _with_sparse_edges(samples: Sequence[object]) -> list[object]:
    from dataclasses import replace

    result = []
    for sample in samples:
        if sample.readout_mask is None:
            raise ValueError("ATOM3D-LBA sample requires a ligand readout mask")
        result.append(
            replace(
                sample,
                edge_index=segment_balanced_knn_edge_index(
                    sample.pos,
                    sample.readout_mask,
                    intra_k=INTRA_K,
                    cross_k=CROSS_K,
                    cutoff=CUTOFF_ANGSTROM,
                ),
            )
        )
    return result


def _iter_batches(samples: Sequence[object], batch_size: int) -> list[object]:
    return [
        collate_graphs(samples[start : start + batch_size])
        for start in range(0, len(samples), batch_size)
    ]


@torch.no_grad()
def _prediction_hash(
    model: torch.nn.Module,
    samples: Sequence[object],
    device: torch.device,
) -> str:
    model = model.to(device=device, dtype=torch.float32).eval()
    digest = hashlib.sha256()
    for batch in _iter_batches(samples, LEGACY["BATCH_SIZE"]):
        batch = batch.to(device=device, dtype=torch.float32)
        prediction = predict_graph_scalar(model, batch).detach().cpu().contiguous()
        digest.update(prediction.numpy().tobytes())
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA run requested but CUDA is unavailable")
    reproducibility = configure_reproducibility(seed=MODEL_SEED, mode="strict")
    raw_samples = load_atom3d_lba_samples(
        args.data_root,
        indices=SUBSET_INDICES,
        revision=ATOM3D_LBA_REVISION,
    )
    _validate_frozen_samples(raw_samples)
    samples = _with_sparse_edges(raw_samples)
    normalizer = fit_target_normalizer(samples)

    initial_models = {
        mode: _build_model(mode).to(device=device, dtype=torch.float32)
        for mode in ("mean", "interaction")
    }
    initial_metrics = {
        mode: evaluate_regression(
            model,
            _iter_batches(samples, LEGACY["BATCH_SIZE"]),
            target_normalizer=normalizer,
        )
        for mode, model in initial_models.items()
    }
    initial_prediction_hashes = {
        mode: _prediction_hash(model, samples, device)
        for mode, model in initial_models.items()
    }
    if len(set(initial_prediction_hashes.values())) != 1:
        raise RuntimeError(
            "zero-initialized interaction readout changed initial outputs"
        )

    result: dict[str, object] = {
        "schema_version": 1,
        "packet_id": PACKET_ID,
        "status": "running",
        "dataset": "vector-institute/atom3d-lba",
        "dataset_revision": ATOM3D_LBA_REVISION,
        "split": "train",
        "subset_indices": list(SUBSET_INDICES),
        "sample_ids": [sample.sample_id for sample in samples],
        "node_counts": [int(sample.pos.shape[0]) for sample in samples],
        "edge_counts": [int(sample.edge_index.shape[1]) for sample in samples],
        "initial_metrics": initial_metrics,
        "initial_prediction_sha256": next(iter(initial_prediction_hashes.values())),
        "determinism": reproducibility,
        "max_steps": args.max_steps,
        "budget_seconds": args.budget_seconds,
        "validation_evaluated": False,
        "test_evaluated": False,
        "claim_boundary": "train-only readout optimization and cost diagnostic",
        "arm_results": [],
    }
    _write_result(args.output, result)

    started = time.perf_counter()
    per_arm_budget = args.budget_seconds / 2.0
    for mode in ("mean", "interaction"):
        arm_result = _run_arm(
            arm=mode,
            model=_build_model(mode),
            samples=samples,
            normalizer=normalizer,
            device=device,
            max_steps=args.max_steps,
            threshold=0.0,
            batch_size=LEGACY["BATCH_SIZE"],
            eval_interval=max(1, min(100, args.max_steps)),
            budget_seconds=per_arm_budget,
        )
        arm_result["readout_mode"] = mode
        arm_result["edge_count_with_self"] = sum(
            int(sample.edge_index.shape[1]) for sample in samples
        )
        result["arm_results"].append(arm_result)
        result["elapsed_seconds"] = time.perf_counter() - started
        _write_result(args.output, result)

    result["status"] = "completed"
    result["elapsed_seconds"] = time.perf_counter() - started
    _write_result(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"interaction-readout PDBBind run failed: {error}", file=sys.stderr)
        raise
