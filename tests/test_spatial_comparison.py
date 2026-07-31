from __future__ import annotations

from equivariant_attention.spatial_comparison import (
    SpatialPromotionThresholds,
    paired_spatial_deltas,
    render_spatial_comparison_report,
    spatial_promotion_decision,
    validate_spatial_comparison,
)


def _payload(*, hybrid_pass: bool = True) -> dict[str, object]:
    tasks = ["local_directional", "smooth_gaussian", "mixed"]
    seeds = [0, 1, 2]
    runs = []
    audits = []
    explicit_mae = {
        "local_directional": 1.0,
        "smooth_gaussian": 1.0,
        "mixed": 1.0,
    }
    hybrid_mae = {
        "local_directional": 1.01 if hybrid_pass else 1.10,
        "smooth_gaussian": 0.90 if hybrid_pass else 1.00,
        "mixed": 0.95 if hybrid_pass else 1.02,
    }
    implicit_mae = {
        "local_directional": 1.20,
        "smooth_gaussian": 0.85,
        "mixed": 1.05,
    }
    for task in tasks:
        for seed in seeds:
            audits.append(
                {
                    "task": task,
                    "seed": seed,
                    "initial_equivalence": {
                        "explicit_vs_hybrid_max_abs": 0.0,
                        "explicit_vs_implicit_max_abs": 0.0,
                        "implicit_edge_independence_max_abs": 0.0,
                    },
                }
            )
            for arm, mae in (
                ("explicit", explicit_mae[task]),
                ("implicit", implicit_mae[task]),
                ("hybrid", hybrid_mae[task]),
            ):
                runs.append(
                    {
                        "task": task,
                        "seed": seed,
                        "arm": arm,
                        "initial_state_sha256": f"hash-{task}-{seed}",
                        "audit": {"parameter_count": 1000},
                        "best_validation": {
                            "mae": mae,
                            "rmse": mae,
                            "normalized_mse": mae**2,
                            "pearson": 0.9,
                        },
                        "final_validation": {
                            "mae": mae,
                            "rmse": mae,
                            "normalized_mse": mae**2,
                            "pearson": 0.9,
                        },
                        "median_train_step_ms": {
                            "explicit": 10.0,
                            "implicit": 9.0,
                            "hybrid": 11.0,
                        }[arm],
                        "inference_ms": {
                            "explicit": 5.0,
                            "implicit": 4.5,
                            "hybrid": 5.5,
                        }[arm],
                        "training_peak_allocated_bytes": {
                            "explicit": 100,
                            "implicit": 90,
                            "hybrid": 110,
                        }[arm],
                        "inference_peak_allocated_bytes": {
                            "explicit": 80,
                            "implicit": 70,
                            "hybrid": 85,
                        }[arm],
                        "clip_fraction": 0.0,
                    }
                )
    return {
        "schema_version": 2,
        "experiment": "spatial_operator_comparison",
        "arms": ["explicit", "implicit", "hybrid"],
        "tasks": tasks,
        "seeds": seeds,
        "device": "cuda",
        "compute_dtype": "bfloat16",
        "neighbor_discovery_included": False,
        "protocol": {
            "same_parameter_schema": True,
            "same_initial_state_per_task_seed": True,
            "same_train_validation_data_per_task_seed": True,
            "validation_or_test_labels_used_for_training": False,
            "no_edge_graph_prepared_outside_timed_forward": True,
        },
        "audits": audits,
        "runs": runs,
        "summaries": [],
    }


def test_complete_payload_passes_protocol_validation() -> None:
    assert validate_spatial_comparison(_payload()) == []
    assert len(paired_spatial_deltas(_payload())) == 18


def test_hybrid_can_pass_only_the_synthetic_candidate_gate() -> None:
    decision = spatial_promotion_decision(_payload())
    assert decision["verdict"] == "hybrid_passes_synthetic_candidate_gate"
    assert decision["synthetic_only"] is True
    assert decision["real_task_validation_required"] is True


def test_failed_local_fidelity_retains_explicit() -> None:
    decision = spatial_promotion_decision(_payload(hybrid_pass=False))
    assert decision["verdict"] == "retain_explicit_as_canonical"
    assert decision["hybrid_checks"]["local_regression"] is False


def test_incomplete_seed_count_is_insufficient_evidence() -> None:
    payload = _payload()
    payload["seeds"] = [0, 1]
    thresholds = SpatialPromotionThresholds(min_seeds=3)
    decision = spatial_promotion_decision(payload, thresholds)
    assert decision["verdict"] == "insufficient_synthetic_evidence"


def test_nonfinite_seed_cannot_be_dropped_from_promotion_gate() -> None:
    payload = _payload()
    for run in payload["runs"]:
        if run["arm"] == "hybrid" and run["seed"] in {1, 2}:
            run["best_validation"]["mae"] = float("nan")
    errors = validate_spatial_comparison(payload)
    assert any("non-finite best_validation.mae" in error for error in errors)
    decision = spatial_promotion_decision(payload)
    assert decision["verdict"] == "insufficient_synthetic_evidence"


def test_cpu_zero_memory_smoke_cannot_promote_architecture() -> None:
    payload = _payload()
    payload["device"] = "cpu"
    for run in payload["runs"]:
        run["training_peak_allocated_bytes"] = 0
        run["inference_peak_allocated_bytes"] = 0
    decision = spatial_promotion_decision(payload)
    assert decision["resource_evidence_complete"] is False
    assert decision["verdict"] == "insufficient_synthetic_evidence"


def test_report_contains_audit_and_gate_sections() -> None:
    report = render_spatial_comparison_report(_payload())
    assert "Synthetic gate verdict" in report
    assert "Protocol audit" in report
    assert "Paired differences versus explicit" in report
    assert "Hybrid" in report
    assert "Implicit replacement" in report
