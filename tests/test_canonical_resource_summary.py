from __future__ import annotations

import json
from pathlib import Path
import runpy


def _receipt(
    *,
    nodes: int,
    degree: int,
    seed: int,
    order: str,
    inference: float,
    train: float,
) -> dict[str, object]:
    return {
        "schema_version": 3,
        "experiment": "canonical_ela_overhead",
        "git_sha": "a" * 40,
        "git_dirty": False,
        "source_file": "/repo/src/equivariant_attention/__init__.py",
        "source_verified": True,
        "reproducibility": {
            "seed": seed,
            "mode": "strict",
            "deterministic_algorithms": True,
            "deterministic_warn_only": False,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cublas_workspace_config": ":4096:8",
        },
        "device": "cuda",
        "device_fingerprint": {
            "type": "cuda",
            "name": "test-gpu",
            "total_memory_bytes": 1_000_000,
            "compute_capability": [9, 0],
            "torch_version": "test",
            "torch_cuda_runtime": "test",
        },
        "dtype": "float32",
        "warmup": 10,
        "repeats": 30,
        "neighbor_discovery_included": False,
        "graph_packing_included": False,
        "host_device_preparation_included": False,
        "models_profiled_one_at_a_time": True,
        "same_common_weights": True,
        "common_state_sha256": f"{nodes:064x}",
        "input_sha256": f"{nodes + seed:064x}",
        "functional_equivalence": {
            "node_output_max_abs": 0.0,
            "graph_output_max_abs": 0.0,
            "feature_gradient_max_abs": 0.0,
            "position_gradient_max_abs": 0.0,
            "common_parameter_gradient_max_abs": 0.0,
            "candidate_branch_gradients_finite": True,
            "candidate_branch_gradients_nonzero": True,
        },
        "nodes": nodes,
        "supplied_candidate_degree": degree,
        "width": 64,
        "depth": 3,
        "seed": seed,
        "profile_order": order,
        "ratios": {
            "parameters": 1.04,
            "inference_median": inference,
            "optimizer_train_step_median": train,
            "inference_peak_allocated": 1.02,
            "optimizer_train_step_peak_allocated": 1.03,
        },
    }


def test_ab_ba_resource_summary_applies_registered_limits(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    symbols = runpy.run_path(
        root / "scripts/summarize_canonical_resource_pairs.py"
    )
    receipts = []
    for nodes, degree in ((128, 8), (512, 32)):
        for seed in range(5):
            for order, inference, train in (
                ("control-first", 1.06, 1.10),
                ("candidate-first", 1.04, 1.08),
            ):
                path = tmp_path / f"{nodes}-{seed}-{order}.json"
                path.write_text(
                    json.dumps(
                        _receipt(
                            nodes=nodes,
                            degree=degree,
                            seed=seed,
                            order=order,
                            inference=inference,
                            train=train,
                        )
                    ),
                    encoding="utf-8",
                )
                receipts.append(path)

    summary = symbols["summarize"](receipts)

    assert summary["resource_gate"]["passed"] is True
    assert len(summary["receipts"]) == 20
    for shape in summary["shape_results"]:
        assert shape["pair_count"] == 5
        assert shape["median_inference_ratio"] < 1.10
        assert shape["median_optimizer_train_step_ratio"] < 1.15
