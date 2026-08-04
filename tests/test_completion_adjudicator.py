from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "adjudicate_completion.py"
_SPEC = importlib.util.spec_from_file_location("_ela_completion_adjudicator", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
adjudicator = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = adjudicator
_SPEC.loader.exec_module(adjudicator)
SHA = "a" * 64


def _arm(*, updates: int, split: str, count: int = 16) -> dict[str, object]:
    return {
        "status": "completed",
        "updates_completed": updates,
        "evaluation_split": split,
        "evaluation": {
            "count": count,
            "mae": 1.0,
            "rmse": 1.1,
            "normalized_mse": 0.5,
            "coordinate_delta_mean": None,
            "coordinate_delta_max": None,
        },
        "initial_evaluation": {
            "count": count,
            "mae": 2.0,
            "rmse": 2.1,
            "normalized_mse": 1.5,
        },
        "initial_state_sha256": SHA,
        "peak_cuda_memory_bytes": 1024,
    }


def _lba_access(*, validation: bool) -> dict[str, object]:
    return {
        "test_shard_opened": False,
        "test_indices_indexed": False,
        "test_labels_used": False,
        "test_evaluated": False,
        "access_scope": "this_run",
        "test_label_storage_materialized_by_this_run": False,
        "historical_local_test_row_and_label_materialized": True,
        "test_labels_accessed": False,
        "train_labels_accessed": True,
        "validation_labels_accessed": validation,
    }


def _pairing(
    static_arms: tuple[str, ...],
    *,
    edge_types: int = 3,
    stagewise: bool = False,
) -> dict[str, object]:
    controls: dict[str, object] = {
        name: {
            "paired_schema": True,
            "base_schema_sha256": SHA,
            "base_state_sha256": SHA,
            "initial_state_sha256": SHA,
            "disabled_lane_parameters": adjudicator._expected_disabled_parameters(
                name,
                depth=3,
                edge_types=edge_types,
            ),
        }
        for name in static_arms
    }
    if stagewise:
        controls["stagewise"] = {
            "paired_schema": False,
            "role": "separate_stagewise_coordinate_functionality_arm",
        }
    return {
        "static_initial_predictions_identical": True,
        "base_schema_sha256": SHA,
        "base_state_sha256": SHA,
        "controls": controls,
    }


def _configuration(
    expected: tuple[str, ...],
    *,
    batch_size: int,
    steps: int,
    cutoff: float,
    learning_rate: float,
    weight_decay: float,
    model_seed: int,
    order_seed: int,
    split_seed: int | None,
    stagewise: bool = False,
    edge_types: int = 3,
) -> dict[str, object]:
    return {
        "width": 64,
        "depth": 3,
        "edge_types": edge_types,
        "cutoff": cutoff,
        "batch_size": batch_size,
        "updates_per_arm": steps,
        "paired_arms": [name for name in expected if name != "stagewise"],
        "arm_execution_order": list(expected),
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "grad_clip": 1.0,
        "model_seed": model_seed,
        "order_seed": order_seed,
        "threads": 1,
        "split_seed": split_seed,
        "stagewise_functionality_arm": stagewise,
    }


def _timing_leaf(
    *,
    count: int = 7,
    milliseconds: float = 1.0,
) -> dict[str, object]:
    summary = {
        "median_ms": milliseconds,
        "p95_ms": milliseconds,
        "min_ms": milliseconds,
        "max_ms": milliseconds,
        "samples_ms": [milliseconds] * count,
    }
    return {
        "wall": dict(summary),
        "cuda_event": dict(summary),
        "peak_allocated_bytes": 1024,
        "peak_incremental_allocated_bytes": 512,
    }


def _profiler_plan_and_report() -> tuple[dict[str, object], dict[str, object]]:
    acceptance = {
        "trusted_prepared_cache": {
            "prepared_object_identity_reused": True,
            "immutable_trusted_prepared_admitted": True,
            "packed_template_identity_reused": True,
            "unsealed_dlpack_alias_invalidates_cache": True,
            "peak_allocated_bytes_lt": adjudicator.PEAK_ALLOCATED_LIMIT_BYTES,
        },
        "ragged_global": {
            "transport_relative_l2_max": 0.05,
            "balanced_relative_l2_max": 0.05,
            "native_over_segmented_inference_median_max": 1.0,
            "integrated_native_calls_exact": 1,
            "dispatch_vs_direct_max_abs": 0.0,
            "peak_allocated_bytes_lt": adjudicator.PEAK_ALLOCATED_LIMIT_BYTES,
        },
        "radius_ingestion": {
            "topology_equal": True,
            "output_max_abs_error": 1e-5,
            "output_relative_l2_error": 1e-5,
            "latency_promotion_gate": None,
            "peak_allocated_bytes_lt": adjudicator.PEAK_ALLOCATED_LIMIT_BYTES,
        },
        "triton_training_local": {
            "maximum_relative_l2": 1e-3,
            "maximum_absolute_error": 5e-3,
            "complete_fused_dispatch_proven": True,
            "parameter_gradient_name_sets_equal": True,
            "peak_allocated_bytes_lt": adjudicator.PEAK_ALLOCATED_LIMIT_BYTES,
        },
        "compiled_numerical_core": {
            "fallback_allowed": False,
            "fallback_warning_allowed": False,
            "cold_unique_graph_delta_min": 1,
            "warm_unique_graph_delta": 0,
            "same_shape_new_topology_unique_graph_delta": 0,
            "maximum_absolute_error": 1e-3,
            "compiled_over_eager_inference_median_max": 1.0,
            "peak_allocated_bytes_lt": adjudicator.PEAK_ALLOCATED_LIMIT_BYTES,
        },
    }
    observed = {
        "trusted_prepared_cache": {
            "prepared_object_identity_reused": True,
            "immutable_trusted_prepared_admitted": True,
            "packed_template_identity_reused": True,
            "unsealed_dlpack_alias_invalidates_cache": True,
        },
        "ragged_global": {
            "transport_relative_l2": 0.01,
            "balanced_relative_l2": 0.01,
            "native_over_segmented_inference_median": 1.0,
            "integrated_native_calls": 1,
            "dispatch_vs_direct_max_abs": 0.0,
        },
        "radius_ingestion": {
            "topology_equal": True,
            "output_max_abs_error": 1e-6,
            "output_relative_l2_error": 1e-6,
        },
        "triton_training_local": {
            "maximum_relative_l2": 1e-4,
            "maximum_absolute_error": 1e-4,
            "complete_fused_dispatch_proven": True,
            "parameter_gradient_name_sets_equal": True,
        },
        "compiled_numerical_core": {
            "fallback_occurred": False,
            "fallback_warning_observed": False,
            "cold_unique_graph_delta": 1,
            "warm_unique_graph_delta": 0,
            "same_shape_new_topology_unique_graph_delta": 0,
            "initial_max_abs_error": 1e-4,
            "same_shape_new_topology_max_abs_error": 1e-4,
            "changed_shape_max_abs_error": 1e-4,
            "compiled_over_eager_inference_median": 1.0,
        },
    }
    measurements = {
        "trusted_prepared_cache": {"prepared_cache_reuse": _timing_leaf()},
        "ragged_global": {
            "native_grouped_mm_inference": _timing_leaf(),
            "tiled_segmented_inference": _timing_leaf(),
            "tiled_segmented_training_fwd_bwd": _timing_leaf(),
        },
        "radius_ingestion": {
            "automatic_radius_direct_csr": _timing_leaf(),
            "explicit_coo_to_csr_reference": _timing_leaf(),
        },
        "triton_training_local": {
            "torch_complete_forward_backward": _timing_leaf(),
            "forced_triton_complete_forward_backward": _timing_leaf(),
        },
        "compiled_numerical_core": {
            "cold_first_execution_compile_included": _timing_leaf(count=1),
            "warm_steady_execution": {
                "compiled_same_shape_and_topology": _timing_leaf(),
                "eager_same_shape_and_topology": _timing_leaf(),
            },
            "same_shape_new_topology_single_execution": _timing_leaf(count=1),
            "changed_shape_single_execution": _timing_leaf(count=1),
        },
    }
    budget = {"target_peak_allocated_gib_lt": 16}
    timing = {"cuda_events": True}
    config = {"repeats": 7}
    plan = {
        "schema_version": 1,
        "experiment": "ela_cuda_completion_profile",
        "status": "schema_only",
        "source_manifest": {
            "path": "artifacts/manifest.json",
            "manifest_sha256": SHA,
            "combined_sha256": SHA,
            "file_count": 1,
            "verified_against_current_bytes": True,
        },
        "acceptance_contract": acceptance,
        "budget_contract": budget,
        "timing_contract": timing,
        "config": config,
    }
    lanes = {}
    for name in adjudicator.LANES:
        thresholds = {
            key: value
            for key, value in acceptance[name].items()
            if key != "peak_allocated_bytes_lt"
        }
        lanes[name] = {
            "status": "completed",
            "gate": {
                "passed": True,
                "thresholds": thresholds,
                "observed": observed[name],
            },
            "resource_gate": {
                "passed": True,
                "thresholds": {
                    "peak_allocated_bytes_lt": adjudicator.PEAK_ALLOCATED_LIMIT_BYTES,
                    "peak_allocated_gib_lt": 16,
                },
                "observed": {
                    "maximum_peak_allocated_bytes": 1024,
                    "maximum_peak_allocated_gib": 1024 / 1024**3,
                },
            },
            "measurements": measurements[name],
        }
    lanes["trusted_prepared_cache"].update(
        {
            "prepared_object_identity_reused": True,
            "immutable_trusted_prepared_admitted": True,
            "packed_template_identity_reused": True,
            "unsealed_dlpack_alias_invalidates_cache": True,
        }
    )
    lanes["ragged_global"].update(
        {
            "numerical_relative_l2_error": 0.01,
            "integrated_balanced_attention_relative_l2_error": 0.01,
            "integrated_balanced_attention_native_calls": 1,
            "dispatch_vs_direct_native_max_abs_error": 0.0,
        }
    )
    lanes["radius_ingestion"].update(
        {
            "topology_parity": {
                "equal_after_receiver_sender_canonicalization": True,
            },
            "output_parity": {
                "max_abs_error": 1e-6,
                "relative_l2_error": 1e-6,
            },
        }
    )
    dispatch = {"weighted_pair_gate_lanes": [[0, 1]]}
    lanes["triton_training_local"].update(
        {
            "dispatch_evidence": {
                "observed": dispatch,
                "expected": dict(dispatch),
            },
            "equivalence": {
                "output": {"max_abs": 1e-4, "relative_l2": 1e-4},
                "feature_gradient": {"max_abs": 1e-4, "relative_l2": 1e-4},
                "position_gradient": {"max_abs": 1e-4, "relative_l2": 1e-4},
                "parameter_gradient": {
                    "max_abs": 1e-4,
                    "relative_l2": 1e-4,
                    "candidate_only": [],
                    "reference_only": [],
                },
            },
        }
    )
    lanes["compiled_numerical_core"].update(
        {
            "fallback_status_final": "compiled_callable_retained",
            "warnings": {
                "cold": [],
                "warm": [],
                "same_shape_new_topology": [],
                "changed_shape": [],
            },
            "eager_parity": {
                "initial_max_abs_error": 1e-4,
                "same_shape_new_topology_max_abs_error": 1e-4,
                "changed_shape_max_abs_error": 1e-4,
            },
            "compiler_counter_snapshots": {
                "before": {},
                "after_cold": {"stats": {"unique_graphs": 1}},
                "after_warm": {"stats": {"unique_graphs": 1}},
                "after_topology": {"stats": {"unique_graphs": 1}},
                "after_shape": {"stats": {"unique_graphs": 2}},
            },
        }
    )
    report = {
        "schema_version": 1,
        "experiment": "ela_cuda_completion_profile",
        "status": "completed",
        "failures": [],
        "source_manifest": {
            "path": "artifacts/manifest.json",
            "manifest_sha256": SHA,
            "combined_sha256": SHA,
            "file_count": 1,
            "verified_against_current_bytes": True,
        },
        "device": {
            "actual": "cuda:0",
            "bfloat16_supported": True,
            "triton_available": True,
        },
        "acceptance_contract": acceptance,
        "budget_contract": budget,
        "timing_contract": timing,
        "config": config,
        "lanes": lanes,
    }
    return plan, report


def test_profiler_adjudication_requires_every_resource_gate() -> None:
    plan, report = _profiler_plan_and_report()
    assert (
        adjudicator._adjudicate_profiler(
            report,
            expected_source_sha256=SHA,
            expected_manifest_sha256=SHA,
            plan=plan,
        )
        == []
    )

    report["lanes"]["ragged_global"]["resource_gate"]["passed"] = False
    assert "G2 ragged_global resource gate" in adjudicator._adjudicate_profiler(
        report,
        expected_source_sha256=SHA,
        expected_manifest_sha256=SHA,
        plan=plan,
    )


def test_profiler_adjudication_recomputes_measurements_and_observed_gate() -> None:
    plan, report = _profiler_plan_and_report()
    report["lanes"]["trusted_prepared_cache"]["measurements"] = {}
    report["lanes"]["radius_ingestion"]["gate"]["observed"]["topology_equal"] = False

    failures = adjudicator._adjudicate_profiler(
        report,
        expected_source_sha256=SHA,
        expected_manifest_sha256=SHA,
        plan=plan,
    )
    assert "G2 trusted_prepared_cache resource gate" in failures
    assert "G2 radius_ingestion evidence binding" in failures


def test_profiler_adjudication_binds_latency_gates_to_raw_cuda_medians() -> None:
    plan, report = _profiler_plan_and_report()
    ragged = report["lanes"]["ragged_global"]
    ragged["measurements"]["native_grouped_mm_inference"] = _timing_leaf(
        milliseconds=2.0
    )
    compiled = report["lanes"]["compiled_numerical_core"]
    compiled["measurements"]["warm_steady_execution"][
        "compiled_same_shape_and_topology"
    ] = _timing_leaf(milliseconds=2.0)

    failures = adjudicator._adjudicate_profiler(
        report,
        expected_source_sha256=SHA,
        expected_manifest_sha256=SHA,
        plan=plan,
    )
    assert "G2 ragged_global evidence binding" in failures
    assert "G2 ragged_global observed gate" in failures
    assert "G2 compiled_numerical_core evidence binding" in failures
    assert "G2 compiled_numerical_core observed gate" in failures


@pytest.mark.parametrize(
    ("lane", "field"),
    [
        ("radius_ingestion", "output_parity"),
        ("triton_training_local", "equivalence"),
        ("compiled_numerical_core", "eager_parity"),
    ],
)
def test_profiler_adjudication_requires_independent_lane_evidence(
    lane: str,
    field: str,
) -> None:
    plan, report = _profiler_plan_and_report()
    del report["lanes"][lane][field]

    failures = adjudicator._adjudicate_profiler(
        report,
        expected_source_sha256=SHA,
        expected_manifest_sha256=SHA,
        plan=plan,
    )
    assert f"G2 {lane} evidence binding" in failures
    assert f"G2 {lane} observed gate" in failures


def test_pairing_adjudication_requires_exact_disabled_parameter_names() -> None:
    expected = ("full", "no-relation", "no-cg12", "no-multiscale")
    report = {
        "configuration": _configuration(
            expected,
            batch_size=2,
            steps=250,
            cutoff=6.0,
            learning_rate=0.001,
            weight_decay=0.0,
            model_seed=20260723,
            order_seed=20260723,
            split_seed=None,
        ),
        "pairing": _pairing(expected),
        "arms": {name: _arm(updates=250, split="train") for name in expected},
    }
    report["pairing"]["controls"]["no-cg12"]["disabled_lane_parameters"] = []

    failures = adjudicator._pairing_failures(
        report,
        job="G4",
        static_arms=expected,
    )
    assert "G4 no-cg12 state and schema pairing" in failures


def test_gpu_gate_adjudication_requires_actual_source_bound_receipt() -> None:
    report = {
        "schema_version": 1,
        "experiment": "ela_gpu_gate",
        "status": "passed",
        "failure": None,
        "exit_code": 0,
        "command": ["bash", "scripts/check.sh", "gpu"],
        "source_manifest": {
            "path": "artifacts/manifest.json",
            "manifest_sha256": SHA,
            "combined_sha256": SHA,
            "file_count": 1,
            "verified_against_current_bytes": True,
        },
    }
    assert (
        adjudicator._adjudicate_gpu_gate(
            report,
            expected_source_sha256=SHA,
            expected_manifest_sha256=SHA,
        )
        == []
    )

    report["exit_code"] = False
    assert "G1 exit code" in adjudicator._adjudicate_gpu_gate(
        report,
        expected_source_sha256=SHA,
        expected_manifest_sha256=SHA,
    )


def test_overfit_adjudication_compares_the_same_full_set() -> None:
    expected = ("full", "no-relation", "no-cg12", "no-multiscale")
    report = {
        "status": "completed",
        "task": "lba-overfit",
        "device": "cuda",
        "dtype": "float32",
        "model_family": "ELA_only",
        "legacy_models_present": False,
        "pairing": _pairing(expected),
        "configuration": _configuration(
            expected,
            batch_size=2,
            steps=250,
            cutoff=6.0,
            learning_rate=0.001,
            weight_decay=0.0,
            model_seed=20260723,
            order_seed=20260723,
            split_seed=None,
        ),
        "arms": {name: _arm(updates=250, split="train") for name in expected},
        "data": {
            "dataset": "vector-institute/atom3d-lba",
            "revision": "f93dd2d150a47c270f624620f84e07451a158705",
            "root": "data/atom3d_lba",
            "opened_splits": ["train"],
            "arrow_sha256": {name: SHA for name in adjudicator.LBA_TRAIN_FILES},
        },
        "split": {
            "kind": "frozen_train_only_capacity_subset",
            "indices": list(range(16)),
            "indices_sha256": (
                "b9ad7606160a067ebb4fb2935c415d51dc1dea3fb8aba28a42f5734a2f88e14a"
            ),
        },
        "topology": {
            "kind": "segment_balanced_knn",
            "cutoff_angstrom": 6.0,
            "intra_k": 16,
            "cross_k": 16,
            "relation_types": ["pocket-pocket", "ligand-ligand", "cross"],
            "graphs": 16,
            "directed_edges_with_self": 1024,
            "sample_identity_sha256": SHA,
            "edge_index_sha256": SHA,
            "edge_relation_sha256": SHA,
            "edge_topology_sha256": SHA,
            "joint_sha256": SHA,
        },
        "label_access": _lba_access(validation=False),
    }
    assert adjudicator._adjudicate_lba_overfit(report) == []

    report["arms"]["full"]["evaluation"]["normalized_mse"] = 2.0
    assert "G4 full matched-set loss reduction" in (
        adjudicator._adjudicate_lba_overfit(report)
    )


def test_id30_adjudication_requires_split_topology_and_frozen_count() -> None:
    expected = ("full", "no-relation", "no-cg12", "no-multiscale")
    report = {
        "status": "completed",
        "task": "lba-id30",
        "device": "cuda",
        "dtype": "float32",
        "model_family": "ELA_only",
        "legacy_models_present": False,
        "pairing": _pairing(expected),
        "configuration": _configuration(
            expected,
            batch_size=16,
            steps=220,
            cutoff=6.0,
            learning_rate=0.0003,
            weight_decay=0.01,
            model_seed=42,
            order_seed=42,
            split_seed=None,
        ),
        "arms": {
            name: _arm(updates=220, split="validation", count=466) for name in expected
        },
        "split": {
            "kind": "official_ID30_train_validation",
            "train_size": 3507,
            "validation_size": 466,
            "train_limited": False,
            "validation_limited": False,
        },
        "topology": {
            "train": {
                "sample_identity_sha256": SHA,
                "edge_topology_sha256": SHA,
            },
            "validation": {
                "sample_identity_sha256": SHA,
                "edge_topology_sha256": SHA,
            },
            "combined": {
                "directed_edges_with_self": 32_302_952,
                "split_receipts_sha256": SHA,
            },
            "frozen_identity_gate": {
                "passed": True,
                "expected": {
                    "train_graphs": 3507,
                    "validation_graphs": 466,
                    "directed_edges_with_self": 32_302_952,
                },
                "observed": {
                    "train_graphs": 3507,
                    "validation_graphs": 466,
                    "directed_edges_with_self": 32_302_952,
                },
            },
        },
        "data": {
            "dataset": "vector-institute/atom3d-lba",
            "revision": "f93dd2d150a47c270f624620f84e07451a158705",
            "root": "data/atom3d_lba",
            "opened_splits": ["train", "val"],
            "arrow_sha256": {
                **{name: SHA for name in adjudicator.LBA_TRAIN_FILES},
                adjudicator.LBA_VALIDATION_FILE: SHA,
            },
        },
        "label_access": _lba_access(validation=True),
    }
    assert adjudicator._adjudicate_lba_id30(report) == []

    report["topology"]["combined"]["directed_edges_with_self"] += 1
    assert "G5 split topology and frozen identity" in (
        adjudicator._adjudicate_lba_id30(report)
    )


def test_qm9_adjudication_requires_active_stagewise_coordinate_path() -> None:
    expected = ("full", "no-cg12", "no-multiscale", "stagewise")
    arms = {
        name: _arm(updates=100, split="validation", count=1000) for name in expected
    }
    arms["stagewise"]["initial_coordinate_state_sha256"] = "b" * 64
    arms["stagewise"]["final_coordinate_state_sha256"] = "c" * 64
    arms["stagewise"]["evaluation"]["coordinate_delta_mean"] = 0.01
    arms["stagewise"]["evaluation"]["coordinate_delta_max"] = 0.02
    report = {
        "status": "completed",
        "task": "qm9",
        "device": "cuda",
        "dtype": "float32",
        "model_family": "ELA_only",
        "legacy_models_present": False,
        "pairing": _pairing(expected[:-1], edge_types=0, stagewise=True),
        "configuration": _configuration(
            expected,
            batch_size=64,
            steps=100,
            cutoff=2.5,
            learning_rate=0.0003,
            weight_decay=0.01,
            model_seed=42,
            order_seed=42,
            split_seed=42,
            stagewise=True,
            edge_types=0,
        ),
        "arms": arms,
        "label_access": {
            "test_shard_opened": False,
            "test_indices_indexed": False,
            "test_labels_used": False,
            "test_evaluated": False,
            "processed_monolith_including_test_labels_loaded": True,
            "train_labels_accessed": True,
            "validation_labels_accessed": True,
        },
        "split": {
            "train_size": 110000,
            "validation_size_full": 10000,
            "validation_size_evaluated": 1000,
            "unused_test_size": 10000,
            "train_indices_sha256": SHA,
            "validation_indices_sha256": SHA,
            "unused_test_indices_sha256": SHA,
        },
        "data": {
            "root": "data/qm9",
            "file_sha256": adjudicator.QM9_FILE_HASHES,
        },
    }
    assert adjudicator._adjudicate_qm9(report) == []

    arms["stagewise"]["evaluation"]["coordinate_delta_max"] = 0.0
    assert "G3 stagewise final coordinate delta" in adjudicator._adjudicate_qm9(report)


def test_adjudication_missing_and_malformed_receipts_fail_closed(
    tmp_path: Path,
) -> None:
    (tmp_path / "gpu-completion-profile.json").write_text(
        '{"status": NaN}\n', encoding="utf-8"
    )
    result = adjudicator.adjudicate(
        tmp_path,
        expected_source_manifest_combined_sha256=SHA,
        expected_realdata_source_sha256=SHA,
    )
    assert result["status"] == "failed"
    assert "invalid receipt" in result["jobs"]["G2"]["failures"][0]
    assert result["input_sha256"]["G1"] is None


def test_adjudication_rejects_packet_and_report_command_mismatch(
    tmp_path: Path,
) -> None:
    (tmp_path / "source-manifest-pre-gpu.json").write_text(
        json.dumps({"combined_sha256": SHA}),
        encoding="utf-8",
    )
    (tmp_path / "gpu-completion-plan-v2.json").write_text("{}\n", encoding="utf-8")
    jobs = [{"id": name, "argv": []} for name in ("G1", "G2", "G3", "G4", "G5")]
    jobs.append({"id": "G6", "argv": ["wrong"]})
    (tmp_path / "gpu-job-packet.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_manifest": "source-manifest-pre-gpu.json",
                "jobs": jobs,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "qm9-screen.json").write_text(
        json.dumps({"command": ["wrong"]}),
        encoding="utf-8",
    )

    result = adjudicator.adjudicate(
        tmp_path,
        expected_source_manifest_combined_sha256=SHA,
        expected_realdata_source_sha256=SHA,
    )
    assert "G3 packet command" in result["jobs"]["G3"]["failures"]
    assert "G3 exact command" in result["jobs"]["G3"]["failures"]
    assert "G6 packet command" in result["jobs"]["G6"]["failures"]


def test_adjudication_rejects_duplicate_or_extra_packet_jobs(
    tmp_path: Path,
) -> None:
    (tmp_path / "source-manifest-pre-gpu.json").write_text(
        json.dumps({"combined_sha256": SHA}),
        encoding="utf-8",
    )
    (tmp_path / "gpu-completion-plan-v2.json").write_text("{}\n", encoding="utf-8")
    jobs = [{"id": name, "argv": []} for name in ("G1", "G2", "G3", "G4", "G5", "G6")]
    jobs.insert(2, {"id": "G3", "argv": ["tampered"]})
    (tmp_path / "gpu-job-packet.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_manifest": "source-manifest-pre-gpu.json",
                "jobs": jobs,
            }
        ),
        encoding="utf-8",
    )

    result = adjudicator.adjudicate(
        tmp_path,
        expected_source_manifest_combined_sha256=SHA,
        expected_realdata_source_sha256=SHA,
    )
    assert "packet job schema" in result["jobs"]["G1"]["failures"]
    assert "packet job schema" in result["jobs"]["G3"]["failures"]


def test_main_writes_failure_receipt_when_inputs_are_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "completion.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(_SCRIPT),
            str(tmp_path),
            str(output),
            "--expected-source-manifest-combined-sha256",
            SHA,
            "--expected-realdata-source-sha256",
            SHA,
        ],
    )
    assert adjudicator.main() == 2
    assert output.is_file()
    assert '"status": "failed"' in output.read_text(encoding="utf-8")
