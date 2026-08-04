#!/usr/bin/env python3
"""Fail-closed adjudication of the frozen ELA GPU and real-data packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


LANES = (
    "trusted_prepared_cache",
    "ragged_global",
    "radius_ingestion",
    "triton_training_local",
    "compiled_numerical_core",
)
PEAK_ALLOCATED_LIMIT_BYTES = 16 * 1024**3
GPU_GATE_COMMAND = ["bash", "scripts/check.sh", "gpu"]
REALDATA_JOB_ARGV = {
    "G3": [
        "uv",
        "run",
        "--locked",
        "--extra",
        "qm9",
        "python",
        "scripts/validate_realdata.py",
        "qm9",
        "artifacts/ela-completion-20260804/qm9-screen.json",
        "--device",
        "cuda",
        "--dtype",
        "float32",
        "--width",
        "64",
        "--depth",
        "3",
        "--batch-size",
        "64",
        "--steps",
        "100",
        "--cutoff",
        "2.5",
        "--learning-rate",
        "0.0003",
        "--weight-decay",
        "0.01",
        "--grad-clip",
        "1",
        "--model-seed",
        "42",
        "--order-seed",
        "42",
        "--threads",
        "1",
        "--data-root",
        "data/qm9",
        "--num-samples",
        "130000",
        "--train-size",
        "110000",
        "--val-size",
        "10000",
        "--validation-limit",
        "1000",
        "--split-seed",
        "42",
        "--arms",
        "full",
        "no-cg12",
        "no-multiscale",
        "--include-stagewise",
    ],
    "G4": [
        "uv",
        "run",
        "--locked",
        "--extra",
        "pdbbind",
        "python",
        "scripts/validate_realdata.py",
        "lba-overfit",
        "artifacts/ela-completion-20260804/lba-overfit-screen.json",
        "--device",
        "cuda",
        "--dtype",
        "float32",
        "--width",
        "64",
        "--depth",
        "3",
        "--batch-size",
        "2",
        "--steps",
        "250",
        "--cutoff",
        "6",
        "--learning-rate",
        "0.001",
        "--weight-decay",
        "0",
        "--grad-clip",
        "1",
        "--model-seed",
        "20260723",
        "--order-seed",
        "20260723",
        "--threads",
        "1",
        "--data-root",
        "data/atom3d_lba",
        "--arms",
        "full",
        "no-relation",
        "no-cg12",
        "no-multiscale",
    ],
    "G5": [
        "uv",
        "run",
        "--locked",
        "--extra",
        "pdbbind",
        "python",
        "scripts/validate_realdata.py",
        "lba-id30",
        "artifacts/ela-completion-20260804/lba-id30-screen.json",
        "--device",
        "cuda",
        "--dtype",
        "float32",
        "--width",
        "64",
        "--depth",
        "3",
        "--batch-size",
        "16",
        "--steps",
        "220",
        "--cutoff",
        "6",
        "--learning-rate",
        "0.0003",
        "--weight-decay",
        "0.01",
        "--grad-clip",
        "1",
        "--model-seed",
        "42",
        "--order-seed",
        "42",
        "--threads",
        "1",
        "--data-root",
        "data/atom3d_lba",
        "--train-limit",
        "0",
        "--validation-limit",
        "0",
        "--arms",
        "full",
        "no-relation",
        "no-cg12",
        "no-multiscale",
    ],
}


def expected_packet_argv(
    source_manifest_combined_sha256: str,
    realdata_source_sha256: str,
) -> dict[str, list[str]]:
    """Return the single canonical argv table for the frozen G1-G6 packet."""
    return {
        "G1": [
            "uv",
            "run",
            "--locked",
            "python",
            "scripts/run_gpu_gate.py",
            "--source-manifest",
            "artifacts/ela-completion-20260804/source-manifest-pre-gpu.json",
            "--output",
            "artifacts/ela-completion-20260804/gpu-gate-receipt.json",
        ],
        "G2": [
            "uv",
            "run",
            "--locked",
            "python",
            "scripts/profile_gpu_completion.py",
            "--source-manifest",
            "artifacts/ela-completion-20260804/source-manifest-pre-gpu.json",
            "--output",
            "artifacts/ela-completion-20260804/gpu-completion-profile.json",
        ],
        **{job: list(argv) for job, argv in REALDATA_JOB_ARGV.items()},
        "G6": [
            "uv",
            "run",
            "--locked",
            "python",
            "scripts/adjudicate_completion.py",
            "artifacts/ela-completion-20260804",
            "artifacts/ela-completion-20260804/completion-adjudication.json",
            "--expected-source-manifest-combined-sha256",
            source_manifest_combined_sha256,
            "--expected-realdata-source-sha256",
            realdata_source_sha256,
        ],
    }
QM9_FILE_HASHES = {
    "raw/gdb9.sdf": (
        "98c4e97d50ac549b8c9f0b2114b348a9a944718e17e50d9a724b729f1deaa28e"
    ),
    "raw/gdb9.sdf.csv": (
        "73a67793e3cfa9660f001278bd019c143f57e4785db537a01811cf2ce72aa7eb"
    ),
    "processed/data_v3.pt": (
        "9254af077d7bc651631bb56a3a689fb41004731b413bdd0ec8c6efa318229f83"
    ),
}
LBA_TRAIN_FILES = {
    "atom3d-lba-train-00000-of-00002.arrow",
    "atom3d-lba-train-00001-of-00002.arrow",
}
LBA_VALIDATION_FILE = "atom3d-lba-val.arrow"
MEASUREMENT_PATHS = {
    "trusted_prepared_cache": {
        ("prepared_cache_reuse",): "repeats",
    },
    "ragged_global": {
        ("native_grouped_mm_inference",): "repeats",
        ("tiled_segmented_inference",): "repeats",
        ("tiled_segmented_training_fwd_bwd",): "repeats",
    },
    "radius_ingestion": {
        ("automatic_radius_direct_csr",): "repeats",
        ("explicit_coo_to_csr_reference",): "repeats",
    },
    "triton_training_local": {
        ("torch_complete_forward_backward",): "repeats",
        ("forced_triton_complete_forward_backward",): "repeats",
    },
    "compiled_numerical_core": {
        ("cold_first_execution_compile_included",): "single",
        (
            "warm_steady_execution",
            "compiled_same_shape_and_topology",
        ): "repeats",
        (
            "warm_steady_execution",
            "eager_same_shape_and_topology",
        ): "repeats",
        ("same_shape_new_topology_single_execution",): "single",
        ("changed_shape_single_execution",): "single",
    },
}


def _load(path: Path) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_receipt(
    path: Path,
) -> tuple[Mapping[str, Any] | None, str | None, str | None]:
    try:
        digest = _file_sha256(path)
        return _load(path), digest, None
    except Exception as error:
        return None, None, f"{type(error).__name__}: {error}"


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _sha256_text(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require(failures: list[str], condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


def _maximum_peak_allocated_bytes(value: object) -> int | None:
    peaks: list[int] = []

    def visit(item: object) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if key == "peak_allocated_bytes":
                    if isinstance(child, bool) or not isinstance(child, int):
                        raise TypeError("peak_allocated_bytes must be an integer")
                    peaks.append(child)
                else:
                    visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return max(peaks) if peaks else None


def _percentile(samples: Sequence[float], quantile: float) -> float:
    ordered = sorted(samples)
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _timing_summary_valid(value: object, expected_count: int) -> bool:
    if not isinstance(value, Mapping):
        return False
    samples = value.get("samples_ms")
    if (
        not isinstance(samples, list)
        or len(samples) != expected_count
        or not all(
            _finite_number(sample) and float(sample) >= 0.0 for sample in samples
        )
    ):
        return False
    numeric = [float(sample) for sample in samples]
    expected = {
        "median_ms": sorted(numeric)[len(numeric) // 2]
        if len(numeric) % 2 == 1
        else 0.5
        * (sorted(numeric)[len(numeric) // 2 - 1] + sorted(numeric)[len(numeric) // 2]),
        "p95_ms": _percentile(numeric, 0.95),
        "min_ms": min(numeric),
        "max_ms": max(numeric),
    }
    return all(
        _finite_number(value.get(key))
        and math.isclose(float(value[key]), target, rel_tol=1e-12, abs_tol=1e-12)
        for key, target in expected.items()
    )


def _measurement_leaf_valid(value: object, expected_count: int) -> bool:
    return (
        isinstance(value, Mapping)
        and _timing_summary_valid(value.get("wall"), expected_count)
        and _timing_summary_valid(value.get("cuda_event"), expected_count)
        and isinstance(value.get("peak_allocated_bytes"), int)
        and not isinstance(value.get("peak_allocated_bytes"), bool)
        and value["peak_allocated_bytes"] >= 0
        and isinstance(value.get("peak_incremental_allocated_bytes"), int)
        and not isinstance(value.get("peak_incremental_allocated_bytes"), bool)
        and value["peak_incremental_allocated_bytes"] >= 0
    )


def _measurement_contract_valid(
    lane: str,
    measurements: object,
    *,
    repeats: int,
) -> bool:
    if not isinstance(measurements, Mapping):
        return False
    paths = MEASUREMENT_PATHS[lane]
    expected_top = {path[0] for path in paths}
    if set(measurements) != expected_top:
        return False
    if lane == "compiled_numerical_core":
        warm = measurements.get("warm_steady_execution")
        if not isinstance(warm, Mapping) or set(warm) != {
            "compiled_same_shape_and_topology",
            "eager_same_shape_and_topology",
        }:
            return False
    for path, count_kind in paths.items():
        value: object = measurements
        for key in path:
            if not isinstance(value, Mapping):
                return False
            value = value.get(key)
        count = repeats if count_kind == "repeats" else 1
        if not _measurement_leaf_valid(value, count):
            return False
    return True


def _mapping_at(value: object, *path: str) -> Mapping[str, Any] | None:
    current = value
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current if isinstance(current, Mapping) else None


def _cuda_median(measurements: object, *path: str) -> float | None:
    leaf = _mapping_at(measurements, *path)
    event = None if leaf is None else leaf.get("cuda_event")
    median = event.get("median_ms") if isinstance(event, Mapping) else None
    if not _finite_number(median) or float(median) <= 0.0:
        return None
    return float(median)


def _counter_unique_graphs(snapshot: object) -> int | None:
    if not isinstance(snapshot, Mapping):
        return None
    stats = snapshot.get("stats")
    if not isinstance(stats, Mapping):
        return None
    value = stats.get("unique_graphs")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _counter_delta(before: object, after: object) -> int | None:
    after_value = _counter_unique_graphs(after)
    if after_value is None:
        return None
    before_value = _counter_unique_graphs(before)
    return after_value - (0 if before_value is None else before_value)


def _lane_evidence_observed(
    name: str,
    lane: Mapping[str, Any],
    measurements: object,
) -> dict[str, Any] | None:
    """Rebuild one gate observation from independent raw receipt fields."""

    if name == "trusted_prepared_cache":
        return {
            key: lane.get(key)
            for key in (
                "prepared_object_identity_reused",
                "immutable_trusted_prepared_admitted",
                "packed_template_identity_reused",
                "unsealed_dlpack_alias_invalidates_cache",
            )
        }
    if name == "ragged_global":
        native = _cuda_median(measurements, "native_grouped_mm_inference")
        reference = _cuda_median(measurements, "tiled_segmented_inference")
        relative = lane.get("numerical_relative_l2_error")
        balanced = lane.get("integrated_balanced_attention_relative_l2_error")
        dispatch = lane.get("dispatch_vs_direct_native_max_abs_error")
        calls = lane.get("integrated_balanced_attention_native_calls")
        if (
            native is None
            or reference is None
            or not _finite_number(relative)
            or not _finite_number(balanced)
            or not _finite_number(dispatch)
            or isinstance(calls, bool)
            or not isinstance(calls, int)
        ):
            return None
        return {
            "transport_relative_l2": float(relative),
            "balanced_relative_l2": float(balanced),
            "native_over_segmented_inference_median": native / reference,
            "integrated_native_calls": calls,
            "dispatch_vs_direct_max_abs": float(dispatch),
        }
    if name == "radius_ingestion":
        topology = lane.get("topology_parity")
        output = lane.get("output_parity")
        if not isinstance(topology, Mapping) or not isinstance(output, Mapping):
            return None
        maximum = output.get("max_abs_error")
        relative = output.get("relative_l2_error")
        if not _finite_number(maximum) or not _finite_number(relative):
            return None
        return {
            "topology_equal": topology.get(
                "equal_after_receiver_sender_canonicalization"
            ),
            "output_max_abs_error": float(maximum),
            "output_relative_l2_error": float(relative),
        }
    if name == "triton_training_local":
        equivalence = lane.get("equivalence")
        dispatch = lane.get("dispatch_evidence")
        names = (
            "output",
            "feature_gradient",
            "position_gradient",
            "parameter_gradient",
        )
        if not isinstance(equivalence, Mapping) or not isinstance(dispatch, Mapping):
            return None
        entries = [equivalence.get(key) for key in names]
        if not all(isinstance(entry, Mapping) for entry in entries):
            return None
        relatives = [entry.get("relative_l2") for entry in entries]  # type: ignore[union-attr]
        absolutes = [entry.get("max_abs") for entry in entries]  # type: ignore[union-attr]
        parameter = equivalence.get("parameter_gradient")
        if (
            not all(_finite_number(value) for value in relatives)
            or not all(_finite_number(value) for value in absolutes)
            or not isinstance(parameter, Mapping)
            or not isinstance(parameter.get("candidate_only"), list)
            or not isinstance(parameter.get("reference_only"), list)
        ):
            return None
        return {
            "maximum_relative_l2": max(float(value) for value in relatives),
            "maximum_absolute_error": max(float(value) for value in absolutes),
            "complete_fused_dispatch_proven": (
                dispatch.get("observed") == dispatch.get("expected")
            ),
            "parameter_gradient_name_sets_equal": (
                parameter["candidate_only"] == [] and parameter["reference_only"] == []
            ),
        }
    if name == "compiled_numerical_core":
        steady = _mapping_at(measurements, "warm_steady_execution")
        compiled = _cuda_median(
            steady,
            "compiled_same_shape_and_topology",
        )
        eager = _cuda_median(steady, "eager_same_shape_and_topology")
        parity = lane.get("eager_parity")
        snapshots = lane.get("compiler_counter_snapshots")
        warnings_by_scenario = lane.get("warnings")
        if (
            compiled is None
            or eager is None
            or not isinstance(parity, Mapping)
            or not isinstance(snapshots, Mapping)
            or not isinstance(warnings_by_scenario, Mapping)
        ):
            return None
        errors = {
            key: parity.get(key)
            for key in (
                "initial_max_abs_error",
                "same_shape_new_topology_max_abs_error",
                "changed_shape_max_abs_error",
            )
        }
        warning_lists = tuple(warnings_by_scenario.values())
        if not all(_finite_number(value) for value in errors.values()) or not all(
            isinstance(values, list)
            and all(isinstance(message, str) for message in values)
            for values in warning_lists
        ):
            return None
        messages = [message for values in warning_lists for message in values]
        return {
            "fallback_occurred": lane.get("fallback_status_final")
            != "compiled_callable_retained",
            "fallback_warning_observed": any(
                "falling back" in message.lower() for message in messages
            ),
            "cold_unique_graph_delta": _counter_delta(
                snapshots.get("before"),
                snapshots.get("after_cold"),
            ),
            "warm_unique_graph_delta": _counter_delta(
                snapshots.get("after_cold"),
                snapshots.get("after_warm"),
            ),
            "same_shape_new_topology_unique_graph_delta": _counter_delta(
                snapshots.get("after_warm"),
                snapshots.get("after_topology"),
            ),
            **{key: float(value) for key, value in errors.items()},
            "compiled_over_eager_inference_median": compiled / eager,
        }
    return None


def _functional_gate_recomputed(name: str, observed: Mapping[str, Any]) -> bool:
    if name == "trusted_prepared_cache":
        return all(
            observed.get(key) is True
            for key in (
                "prepared_object_identity_reused",
                "immutable_trusted_prepared_admitted",
                "packed_template_identity_reused",
                "unsealed_dlpack_alias_invalidates_cache",
            )
        )
    if name == "ragged_global":
        return (
            _finite_number(observed.get("transport_relative_l2"))
            and float(observed["transport_relative_l2"]) <= 0.05
            and _finite_number(observed.get("balanced_relative_l2"))
            and float(observed["balanced_relative_l2"]) <= 0.05
            and _finite_number(observed.get("native_over_segmented_inference_median"))
            and float(observed["native_over_segmented_inference_median"]) <= 1.0
            and observed.get("integrated_native_calls") == 1
            and not isinstance(observed.get("integrated_native_calls"), bool)
            and _finite_number(observed.get("dispatch_vs_direct_max_abs"))
            and float(observed["dispatch_vs_direct_max_abs"]) == 0.0
        )
    if name == "radius_ingestion":
        return (
            observed.get("topology_equal") is True
            and _finite_number(observed.get("output_max_abs_error"))
            and float(observed["output_max_abs_error"]) <= 1e-5
            and _finite_number(observed.get("output_relative_l2_error"))
            and float(observed["output_relative_l2_error"]) <= 1e-5
        )
    if name == "triton_training_local":
        return (
            _finite_number(observed.get("maximum_relative_l2"))
            and float(observed["maximum_relative_l2"]) <= 1e-3
            and _finite_number(observed.get("maximum_absolute_error"))
            and float(observed["maximum_absolute_error"]) <= 5e-3
            and observed.get("complete_fused_dispatch_proven") is True
            and observed.get("parameter_gradient_name_sets_equal") is True
        )
    if name == "compiled_numerical_core":
        error_keys = (
            "initial_max_abs_error",
            "same_shape_new_topology_max_abs_error",
            "changed_shape_max_abs_error",
        )
        return (
            observed.get("fallback_occurred") is False
            and observed.get("fallback_warning_observed") is False
            and isinstance(observed.get("cold_unique_graph_delta"), int)
            and not isinstance(observed.get("cold_unique_graph_delta"), bool)
            and observed["cold_unique_graph_delta"] >= 1
            and observed.get("warm_unique_graph_delta") == 0
            and not isinstance(observed.get("warm_unique_graph_delta"), bool)
            and observed.get("same_shape_new_topology_unique_graph_delta") == 0
            and not isinstance(
                observed.get("same_shape_new_topology_unique_graph_delta"), bool
            )
            and all(
                _finite_number(observed.get(key)) and float(observed[key]) <= 1e-3
                for key in error_keys
            )
            and _finite_number(observed.get("compiled_over_eager_inference_median"))
            and float(observed["compiled_over_eager_inference_median"]) <= 1.0
        )
    return False


def _expected_disabled_parameters(
    arm: str,
    *,
    depth: int,
    edge_types: int,
) -> list[str]:
    if arm == "full":
        return []
    if arm == "no-relation":
        if edge_types == 0:
            return []
        suffixes = (
            "relation_score_bias",
            "relation_radial_scale",
            "relation_value_gate",
        )
    elif arm == "no-cg12":
        suffixes = (
            "tensor_closure.l1_l2_polar_out.weight",
            "tensor_closure.l1_l2_axial_out.weight",
            "tensor_closure.l1_l2_even_tensor_out.weight",
            "tensor_closure.l1_l2_odd_tensor_out.weight",
        )
    elif arm == "no-multiscale":
        suffixes = ("local_scale_score_mix", "local_scale_value_mix")
    else:
        raise ValueError(f"unsupported static arm: {arm}")
    return [
        f"core.blocks.{layer}.{suffix}" for layer in range(depth) for suffix in suffixes
    ]


def _source_binding_failures(
    report: Mapping[str, Any],
    *,
    job: str,
    expected_combined: str | None,
    expected_manifest: str | None,
) -> list[str]:
    failures: list[str] = []
    source = report.get("source_manifest")
    _require(
        failures,
        isinstance(source, Mapping)
        and source.get("verified_against_current_bytes") is True
        and _sha256_text(source.get("manifest_sha256"))
        and _sha256_text(source.get("combined_sha256"))
        and isinstance(source.get("file_count"), int)
        and not isinstance(source.get("file_count"), bool)
        and source["file_count"] > 0
        and isinstance(source.get("path"), str)
        and source.get("manifest_sha256") == expected_manifest
        and source.get("combined_sha256") == expected_combined,
        f"{job} source binding",
    )
    return failures


def _adjudicate_gpu_gate(
    report: Mapping[str, Any],
    *,
    expected_source_sha256: str | None,
    expected_manifest_sha256: str | None,
) -> list[str]:
    failures: list[str] = []
    _require(failures, report.get("schema_version") == 1, "G1 schema")
    _require(failures, report.get("experiment") == "ela_gpu_gate", "G1 experiment")
    _require(failures, report.get("status") == "passed", "G1 status")
    _require(failures, report.get("failure") is None, "G1 failure")
    _require(
        failures,
        not isinstance(report.get("exit_code"), bool) and report.get("exit_code") == 0,
        "G1 exit code",
    )
    _require(failures, report.get("command") == GPU_GATE_COMMAND, "G1 command")
    failures.extend(
        _source_binding_failures(
            report,
            job="G1",
            expected_combined=expected_source_sha256,
            expected_manifest=expected_manifest_sha256,
        )
    )
    return failures


def _adjudicate_profiler(
    report: Mapping[str, Any],
    *,
    expected_source_sha256: str | None = None,
    expected_manifest_sha256: str | None = None,
    plan: Mapping[str, Any] | None = None,
) -> list[str]:
    failures: list[str] = []
    _require(failures, report.get("schema_version") == 1, "G2 schema")
    _require(
        failures,
        report.get("experiment") == "ela_cuda_completion_profile",
        "G2 experiment",
    )
    _require(failures, report.get("status") == "completed", "G2 status")
    _require(failures, report.get("failures") == [], "G2 failures")
    failures.extend(
        _source_binding_failures(
            report,
            job="G2",
            expected_combined=expected_source_sha256,
            expected_manifest=expected_manifest_sha256,
        )
    )
    device = report.get("device")
    _require(
        failures,
        isinstance(device, Mapping)
        and isinstance(device.get("actual"), str)
        and device["actual"].startswith("cuda")
        and device.get("bfloat16_supported") is True
        and device.get("triton_available") is True,
        "G2 CUDA BF16 Triton device",
    )
    _require(
        failures,
        isinstance(plan, Mapping)
        and plan.get("schema_version") == 1
        and plan.get("experiment") == "ela_cuda_completion_profile"
        and plan.get("status") == "schema_only",
        "G2 frozen plan",
    )
    if isinstance(plan, Mapping):
        failures.extend(
            _source_binding_failures(
                plan,
                job="G2 plan",
                expected_combined=expected_source_sha256,
                expected_manifest=expected_manifest_sha256,
            )
        )
        for field in (
            "acceptance_contract",
            "budget_contract",
            "timing_contract",
            "config",
        ):
            _require(
                failures,
                report.get(field) == plan.get(field),
                f"G2 frozen {field}",
            )
    lanes = report.get("lanes")
    _require(
        failures,
        isinstance(lanes, Mapping) and tuple(lanes) == LANES,
        "G2 lane schema",
    )
    if isinstance(lanes, Mapping):
        acceptance = (
            plan.get("acceptance_contract") if isinstance(plan, Mapping) else None
        )
        config = plan.get("config") if isinstance(plan, Mapping) else None
        repeats = config.get("repeats") if isinstance(config, Mapping) else None
        for name in LANES:
            lane = lanes.get(name)
            _require(
                failures,
                isinstance(lane, Mapping) and lane.get("status") == "completed",
                f"G2 {name} status",
            )
            if not isinstance(lane, Mapping):
                continue
            gate = lane.get("gate")
            resource = lane.get("resource_gate")
            measurements = lane.get("measurements")
            lane_contract = (
                acceptance.get(name) if isinstance(acceptance, Mapping) else None
            )
            expected_thresholds = (
                {
                    key: value
                    for key, value in lane_contract.items()
                    if key != "peak_allocated_bytes_lt"
                }
                if isinstance(lane_contract, Mapping)
                else None
            )
            observed = gate.get("observed") if isinstance(gate, Mapping) else None
            evidence_observed = _lane_evidence_observed(
                name,
                lane,
                measurements,
            )
            _require(
                failures,
                isinstance(gate, Mapping)
                and gate.get("passed") is True
                and gate.get("thresholds") == expected_thresholds
                and isinstance(gate.get("observed"), Mapping)
                and bool(gate["observed"]),
                f"G2 {name} functional gate",
            )
            _require(
                failures,
                isinstance(observed, Mapping)
                and evidence_observed is not None
                and observed == evidence_observed,
                f"G2 {name} evidence binding",
            )
            _require(
                failures,
                evidence_observed is not None
                and _functional_gate_recomputed(name, evidence_observed),
                f"G2 {name} observed gate",
            )
            recorded_peak = _maximum_peak_allocated_bytes(measurements)
            _require(
                failures,
                isinstance(measurements, Mapping)
                and bool(measurements)
                and isinstance(repeats, int)
                and not isinstance(repeats, bool)
                and _measurement_contract_valid(
                    name,
                    measurements,
                    repeats=repeats,
                )
                and isinstance(resource, Mapping)
                and resource.get("passed") is True
                and resource.get("thresholds")
                == {
                    "peak_allocated_bytes_lt": PEAK_ALLOCATED_LIMIT_BYTES,
                    "peak_allocated_gib_lt": 16,
                }
                and isinstance(resource.get("observed"), Mapping)
                and resource["observed"].get("maximum_peak_allocated_bytes")
                == recorded_peak
                and isinstance(recorded_peak, int)
                and 0 < recorded_peak < PEAK_ALLOCATED_LIMIT_BYTES,
                f"G2 {name} resource gate",
            )
    return failures


def _arm_failures(
    report: Mapping[str, Any],
    *,
    job: str,
    expected: Sequence[str],
    updates: int,
    evaluation_split: str,
    evaluation_count: int,
) -> list[str]:
    failures: list[str] = []
    arms = report.get("arms")
    configuration = report.get("configuration")
    _require(
        failures,
        isinstance(arms, Mapping) and set(arms) == set(expected),
        f"{job} arm set",
    )
    _require(
        failures,
        isinstance(configuration, Mapping)
        and configuration.get("updates_per_arm") == updates
        and configuration.get("arm_execution_order") == list(expected)
        and configuration.get("paired_arms")
        == [name for name in expected if name != "stagewise"],
        f"{job} frozen configuration",
    )
    if not isinstance(arms, Mapping):
        return failures
    for name in expected:
        arm = arms.get(name)
        _require(
            failures,
            isinstance(arm, Mapping) and arm.get("status") == "completed",
            f"{job} {name} status",
        )
        if not isinstance(arm, Mapping):
            continue
        _require(
            failures,
            arm.get("updates_completed") == updates,
            f"{job} {name} update count",
        )
        _require(
            failures,
            arm.get("evaluation_split") == evaluation_split,
            f"{job} {name} evaluation split",
        )
        evaluation = arm.get("evaluation")
        _require(
            failures,
            isinstance(evaluation, Mapping)
            and evaluation.get("count") == evaluation_count
            and _finite_number(evaluation.get("mae"))
            and _finite_number(evaluation.get("rmse"))
            and _finite_number(evaluation.get("normalized_mse")),
            f"{job} {name} finite evaluation",
        )
        peak = arm.get("peak_cuda_memory_bytes")
        _require(
            failures,
            isinstance(peak, int)
            and not isinstance(peak, bool)
            and 0 < peak < PEAK_ALLOCATED_LIMIT_BYTES,
            f"{job} {name} peak allocation",
        )
    return failures


def _pairing_failures(
    report: Mapping[str, Any],
    *,
    job: str,
    static_arms: Sequence[str],
) -> list[str]:
    failures: list[str] = []
    pairing = report.get("pairing")
    arms = report.get("arms")
    configuration = report.get("configuration")
    _require(
        failures,
        isinstance(pairing, Mapping)
        and pairing.get("static_initial_predictions_identical") is True
        and _sha256_text(pairing.get("base_schema_sha256"))
        and _sha256_text(pairing.get("base_state_sha256")),
        f"{job} static pairing",
    )
    if (
        not isinstance(pairing, Mapping)
        or not isinstance(arms, Mapping)
        or not isinstance(configuration, Mapping)
    ):
        return failures
    depth = configuration.get("depth")
    edge_types = configuration.get("edge_types")
    if (
        isinstance(depth, bool)
        or not isinstance(depth, int)
        or isinstance(edge_types, bool)
        or not isinstance(edge_types, int)
    ):
        failures.append(f"{job} control configuration")
        return failures
    controls = pairing.get("controls")
    if not isinstance(controls, Mapping):
        failures.append(f"{job} pairing controls")
        return failures
    base_schema = pairing.get("base_schema_sha256")
    base_state = pairing.get("base_state_sha256")
    for name in static_arms:
        control = controls.get(name)
        arm = arms.get(name)
        _require(
            failures,
            isinstance(control, Mapping)
            and control.get("paired_schema") is True
            and control.get("base_schema_sha256") == base_schema
            and control.get("base_state_sha256") == base_state
            and control.get("initial_state_sha256") == base_state
            and control.get("disabled_lane_parameters")
            == _expected_disabled_parameters(
                name,
                depth=depth,
                edge_types=edge_types,
            )
            and isinstance(arm, Mapping)
            and arm.get("initial_state_sha256") == base_state,
            f"{job} {name} state and schema pairing",
        )
    return failures


def _configuration_failures(
    report: Mapping[str, Any],
    *,
    job: str,
    expected: Mapping[str, object],
) -> list[str]:
    failures: list[str] = []
    configuration = report.get("configuration")
    _require(failures, isinstance(configuration, Mapping), f"{job} configuration")
    if isinstance(configuration, Mapping):
        for key, value in expected.items():
            _require(
                failures,
                configuration.get(key) == value,
                f"{job} configuration {key}",
            )
    return failures


def _access_failures(report: Mapping[str, Any], *, job: str, lba: bool) -> list[str]:
    failures: list[str] = []
    access = report.get("label_access")
    _require(failures, isinstance(access, Mapping), f"{job} access receipt")
    if not isinstance(access, Mapping):
        return failures
    for key in (
        "test_shard_opened",
        "test_indices_indexed",
        "test_labels_used",
        "test_evaluated",
    ):
        _require(failures, access.get(key) is False, f"{job} {key}")
    if lba:
        _require(
            failures,
            access.get("test_labels_accessed") is False,
            f"{job} test_labels_accessed",
        )
        _require(
            failures,
            access.get("access_scope") == "this_run",
            f"{job} access scope",
        )
        _require(
            failures,
            access.get("test_label_storage_materialized_by_this_run") is False,
            f"{job} test storage this run",
        )
        _require(
            failures,
            access.get("historical_local_test_row_and_label_materialized") is True,
            f"{job} historical contamination",
        )
    return failures


def _adjudicate_qm9(report: Mapping[str, Any]) -> list[str]:
    expected = ("full", "no-cg12", "no-multiscale", "stagewise")
    failures: list[str] = []
    _require(failures, report.get("status") == "completed", "G3 status")
    _require(failures, report.get("task") == "qm9", "G3 task")
    _require(
        failures,
        report.get("device") == "cuda"
        and report.get("dtype") == "float32"
        and report.get("model_family") == "ELA_only"
        and report.get("legacy_models_present") is False,
        "G3 model and device contract",
    )
    failures.extend(
        _arm_failures(
            report,
            job="G3",
            expected=expected,
            updates=100,
            evaluation_split="validation",
            evaluation_count=1000,
        )
    )
    failures.extend(
        _pairing_failures(
            report,
            job="G3",
            static_arms=("full", "no-cg12", "no-multiscale"),
        )
    )
    failures.extend(
        _configuration_failures(
            report,
            job="G3",
            expected={
                "width": 64,
                "depth": 3,
                "cutoff": 2.5,
                "batch_size": 64,
                "updates_per_arm": 100,
                "learning_rate": 0.0003,
                "weight_decay": 0.01,
                "grad_clip": 1.0,
                "model_seed": 42,
                "order_seed": 42,
                "threads": 1,
                "split_seed": 42,
                "stagewise_functionality_arm": True,
            },
        )
    )
    pairing = report.get("pairing")
    if isinstance(pairing, Mapping):
        controls = pairing.get("controls")
        stagewise = controls.get("stagewise") if isinstance(controls, Mapping) else None
        _require(
            failures,
            isinstance(stagewise, Mapping)
            and stagewise.get("role")
            == "separate_stagewise_coordinate_functionality_arm",
            "G3 stagewise role",
        )
    arms = report.get("arms")
    stagewise_arm = arms.get("stagewise") if isinstance(arms, Mapping) else None
    if isinstance(stagewise_arm, Mapping):
        evaluation = stagewise_arm.get("evaluation")
        _require(
            failures,
            _sha256_text(stagewise_arm.get("initial_coordinate_state_sha256"))
            and _sha256_text(stagewise_arm.get("final_coordinate_state_sha256"))
            and stagewise_arm.get("initial_coordinate_state_sha256")
            != stagewise_arm.get("final_coordinate_state_sha256"),
            "G3 stagewise coordinate parameters changed",
        )
        _require(
            failures,
            isinstance(evaluation, Mapping)
            and _finite_number(evaluation.get("coordinate_delta_mean"))
            and _finite_number(evaluation.get("coordinate_delta_max"))
            and float(evaluation["coordinate_delta_max"]) > 0.0,
            "G3 stagewise final coordinate delta",
        )
    failures.extend(_access_failures(report, job="G3", lba=False))
    access = report.get("label_access")
    _require(
        failures,
        isinstance(access, Mapping)
        and access.get("processed_monolith_including_test_labels_loaded") is True,
        "G3 monolithic storage disclosure",
    )
    split = report.get("split")
    _require(
        failures,
        isinstance(split, Mapping)
        and split.get("train_size") == 110000
        and split.get("validation_size_full") == 10000
        and split.get("validation_size_evaluated") == 1000
        and split.get("unused_test_size") == 10000
        and _sha256_text(split.get("train_indices_sha256"))
        and _sha256_text(split.get("validation_indices_sha256"))
        and _sha256_text(split.get("unused_test_indices_sha256")),
        "G3 frozen split",
    )
    data = report.get("data")
    _require(
        failures,
        isinstance(data, Mapping)
        and data.get("root") == "data/qm9"
        and data.get("file_sha256") == QM9_FILE_HASHES,
        "G3 frozen data receipt",
    )
    _require(
        failures,
        isinstance(access, Mapping)
        and access.get("train_labels_accessed") is True
        and access.get("validation_labels_accessed") is True,
        "G3 train validation access",
    )
    return failures


def _adjudicate_lba_overfit(report: Mapping[str, Any]) -> list[str]:
    expected = ("full", "no-relation", "no-cg12", "no-multiscale")
    failures: list[str] = []
    _require(failures, report.get("status") == "completed", "G4 status")
    _require(failures, report.get("task") == "lba-overfit", "G4 task")
    _require(
        failures,
        report.get("device") == "cuda"
        and report.get("dtype") == "float32"
        and report.get("model_family") == "ELA_only"
        and report.get("legacy_models_present") is False,
        "G4 model and device contract",
    )
    failures.extend(
        _arm_failures(
            report,
            job="G4",
            expected=expected,
            updates=250,
            evaluation_split="train",
            evaluation_count=16,
        )
    )
    failures.extend(_pairing_failures(report, job="G4", static_arms=expected))
    failures.extend(
        _configuration_failures(
            report,
            job="G4",
            expected={
                "width": 64,
                "depth": 3,
                "cutoff": 6.0,
                "batch_size": 2,
                "updates_per_arm": 250,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "grad_clip": 1.0,
                "model_seed": 20260723,
                "order_seed": 20260723,
                "threads": 1,
                "split_seed": None,
                "stagewise_functionality_arm": False,
            },
        )
    )
    arms = report.get("arms")
    if isinstance(arms, Mapping):
        for name in expected:
            arm = arms.get(name)
            if not isinstance(arm, Mapping):
                continue
            initial = arm.get("initial_evaluation")
            final = arm.get("evaluation")
            _require(
                failures,
                isinstance(initial, Mapping)
                and isinstance(final, Mapping)
                and initial.get("count") == 16
                and final.get("count") == 16
                and _finite_number(initial.get("normalized_mse"))
                and _finite_number(final.get("normalized_mse"))
                and float(final["normalized_mse"]) < float(initial["normalized_mse"]),
                f"G4 {name} matched-set loss reduction",
            )
    failures.extend(_access_failures(report, job="G4", lba=True))
    access = report.get("label_access")
    _require(
        failures,
        isinstance(access, Mapping)
        and access.get("train_labels_accessed") is True
        and access.get("validation_labels_accessed") is False,
        "G4 train-only label access",
    )
    data = report.get("data")
    _require(
        failures,
        isinstance(data, Mapping)
        and data.get("dataset") == "vector-institute/atom3d-lba"
        and data.get("revision") == "f93dd2d150a47c270f624620f84e07451a158705"
        and data.get("root") == "data/atom3d_lba"
        and data.get("opened_splits") == ["train"]
        and isinstance(data.get("arrow_sha256"), Mapping)
        and set(data["arrow_sha256"]) == LBA_TRAIN_FILES
        and all(_sha256_text(value) for value in data["arrow_sha256"].values()),
        "G4 frozen data receipt",
    )
    split = report.get("split")
    _require(
        failures,
        isinstance(split, Mapping)
        and split.get("kind") == "frozen_train_only_capacity_subset"
        and split.get("indices") == list(range(16))
        and split.get("indices_sha256")
        == "b9ad7606160a067ebb4fb2935c415d51dc1dea3fb8aba28a42f5734a2f88e14a",
        "G4 frozen 16-complex split",
    )
    topology = report.get("topology")
    _require(
        failures,
        isinstance(topology, Mapping)
        and topology.get("kind") == "segment_balanced_knn"
        and topology.get("cutoff_angstrom") == 6.0
        and topology.get("intra_k") == 16
        and topology.get("cross_k") == 16
        and topology.get("relation_types")
        == ["pocket-pocket", "ligand-ligand", "cross"]
        and topology.get("graphs") == 16
        and isinstance(topology.get("directed_edges_with_self"), int)
        and topology["directed_edges_with_self"] > 0
        and all(
            _sha256_text(topology.get(key))
            for key in (
                "sample_identity_sha256",
                "edge_index_sha256",
                "edge_relation_sha256",
                "edge_topology_sha256",
                "joint_sha256",
            )
        ),
        "G4 topology receipt",
    )
    return failures


def _adjudicate_lba_id30(report: Mapping[str, Any]) -> list[str]:
    expected = ("full", "no-relation", "no-cg12", "no-multiscale")
    failures: list[str] = []
    _require(failures, report.get("status") == "completed", "G5 status")
    _require(failures, report.get("task") == "lba-id30", "G5 task")
    _require(
        failures,
        report.get("device") == "cuda"
        and report.get("dtype") == "float32"
        and report.get("model_family") == "ELA_only"
        and report.get("legacy_models_present") is False,
        "G5 model and device contract",
    )
    failures.extend(
        _arm_failures(
            report,
            job="G5",
            expected=expected,
            updates=220,
            evaluation_split="validation",
            evaluation_count=466,
        )
    )
    failures.extend(_pairing_failures(report, job="G5", static_arms=expected))
    failures.extend(
        _configuration_failures(
            report,
            job="G5",
            expected={
                "width": 64,
                "depth": 3,
                "cutoff": 6.0,
                "batch_size": 16,
                "updates_per_arm": 220,
                "learning_rate": 0.0003,
                "weight_decay": 0.01,
                "grad_clip": 1.0,
                "model_seed": 42,
                "order_seed": 42,
                "threads": 1,
                "split_seed": None,
                "stagewise_functionality_arm": False,
            },
        )
    )
    split = report.get("split")
    _require(
        failures,
        isinstance(split, Mapping)
        and split.get("kind") == "official_ID30_train_validation"
        and split.get("train_size") == 3507
        and split.get("validation_size") == 466
        and split.get("train_limited") is False
        and split.get("validation_limited") is False,
        "G5 frozen split",
    )
    topology = report.get("topology")
    train_topology = topology.get("train") if isinstance(topology, Mapping) else None
    validation_topology = (
        topology.get("validation") if isinstance(topology, Mapping) else None
    )
    combined_topology = (
        topology.get("combined") if isinstance(topology, Mapping) else None
    )
    _require(
        failures,
        isinstance(topology, Mapping)
        and isinstance(train_topology, Mapping)
        and isinstance(validation_topology, Mapping)
        and isinstance(combined_topology, Mapping)
        and isinstance(topology.get("frozen_identity_gate"), Mapping)
        and topology["frozen_identity_gate"].get("passed") is True
        and topology["frozen_identity_gate"].get("expected")
        == {
            "train_graphs": 3507,
            "validation_graphs": 466,
            "directed_edges_with_self": 32_302_952,
        }
        and topology["frozen_identity_gate"].get("observed")
        == topology["frozen_identity_gate"].get("expected")
        and combined_topology.get("directed_edges_with_self") == 32_302_952
        and _sha256_text(train_topology.get("sample_identity_sha256"))
        and _sha256_text(train_topology.get("edge_topology_sha256"))
        and _sha256_text(validation_topology.get("sample_identity_sha256"))
        and _sha256_text(validation_topology.get("edge_topology_sha256"))
        and _sha256_text(combined_topology.get("split_receipts_sha256")),
        "G5 split topology and frozen identity",
    )
    failures.extend(_access_failures(report, job="G5", lba=True))
    access = report.get("label_access")
    _require(
        failures,
        isinstance(access, Mapping)
        and access.get("train_labels_accessed") is True
        and access.get("validation_labels_accessed") is True,
        "G5 train validation label access",
    )
    data = report.get("data")
    _require(
        failures,
        isinstance(data, Mapping)
        and data.get("dataset") == "vector-institute/atom3d-lba"
        and data.get("revision") == "f93dd2d150a47c270f624620f84e07451a158705"
        and data.get("root") == "data/atom3d_lba"
        and data.get("opened_splits") == ["train", "val"]
        and isinstance(data.get("arrow_sha256"), Mapping)
        and set(data["arrow_sha256"]) == LBA_TRAIN_FILES | {LBA_VALIDATION_FILE}
        and all(_sha256_text(value) for value in data["arrow_sha256"].values()),
        "G5 frozen data receipt",
    )
    return failures


def adjudicate(
    run_dir: Path,
    *,
    expected_source_manifest_combined_sha256: str,
    expected_realdata_source_sha256: str,
) -> dict[str, Any]:
    paths = {
        "source": run_dir / "source-manifest-pre-gpu.json",
        "plan": run_dir / "gpu-completion-plan-v2.json",
        "packet": run_dir / "gpu-job-packet.json",
        "G1": run_dir / "gpu-gate-receipt.json",
        "G2": run_dir / "gpu-completion-profile.json",
        "G3": run_dir / "qm9-screen.json",
        "G4": run_dir / "lba-overfit-screen.json",
        "G5": run_dir / "lba-id30-screen.json",
    }
    reports: dict[str, Mapping[str, Any]] = {}
    input_sha256: dict[str, str | None] = {}
    load_failures: dict[str, str] = {}
    for name, path in paths.items():
        report, digest, error = _load_receipt(path)
        input_sha256[name] = digest
        if report is None:
            load_failures[name] = error or "unknown receipt error"
        else:
            reports[name] = report

    failures: dict[str, list[str]] = {
        job: [] for job in ("G1", "G2", "G3", "G4", "G5", "G6")
    }
    for job in ("G1", "G2", "G3", "G4", "G5"):
        if job in load_failures:
            failures[job].append(f"{job} invalid receipt: {load_failures[job]}")

    frozen_source = reports.get("source")
    frozen_source_sha256 = (
        frozen_source.get("combined_sha256")
        if isinstance(frozen_source, Mapping)
        else None
    )
    if (
        "source" in load_failures
        or not _sha256_text(frozen_source_sha256)
        or not _sha256_text(expected_source_manifest_combined_sha256)
        or frozen_source_sha256 != expected_source_manifest_combined_sha256
    ):
        message = load_failures.get("source", "invalid combined SHA-256")
        failures["G1"].append(f"source manifest invalid receipt: {message}")
        failures["G2"].append(f"source manifest invalid receipt: {message}")
        failures["G6"].append(f"source manifest invalid receipt: {message}")
    frozen_manifest_sha256 = input_sha256.get("source")
    for supporting_input, jobs_using_input in (
        ("plan", ("G2",)),
        ("packet", ("G1", "G2", "G3", "G4", "G5", "G6")),
    ):
        if supporting_input in load_failures:
            for job in jobs_using_input:
                failures[job].append(
                    f"{supporting_input} invalid receipt: {load_failures[supporting_input]}"
                )

    validators = {
        "G1": lambda report: _adjudicate_gpu_gate(
            report,
            expected_source_sha256=frozen_source_sha256,
            expected_manifest_sha256=frozen_manifest_sha256,
        ),
        "G2": lambda report: _adjudicate_profiler(
            report,
            expected_source_sha256=frozen_source_sha256,
            expected_manifest_sha256=frozen_manifest_sha256,
            plan=reports.get("plan"),
        ),
        "G3": _adjudicate_qm9,
        "G4": _adjudicate_lba_overfit,
        "G5": _adjudicate_lba_id30,
    }
    for job, validator in validators.items():
        report = reports.get(job)
        if report is None:
            continue
        try:
            failures[job].extend(validator(report))
        except Exception as error:
            failures[job].append(
                f"{job} invalid receipt: {type(error).__name__}: {error}"
            )

    g1_source = reports.get("G1", {}).get("source_manifest")
    g2_source = reports.get("G2", {}).get("source_manifest")
    if isinstance(g1_source, Mapping) and isinstance(g2_source, Mapping):
        for key in ("path", "manifest_sha256", "combined_sha256", "file_count"):
            if g1_source.get(key) != g2_source.get(key):
                failures["G1"].append(f"G1/G2 source mismatch: {key}")
                failures["G2"].append(f"G1/G2 source mismatch: {key}")

    packet = reports.get("packet")
    packet_jobs = packet.get("jobs") if isinstance(packet, Mapping) else None
    expected_job_ids = ("G1", "G2", "G3", "G4", "G5", "G6")
    packet_schema_valid = (
        isinstance(packet_jobs, list)
        and len(packet_jobs) == len(expected_job_ids)
        and all(isinstance(job, Mapping) for job in packet_jobs)
        and tuple(job.get("id") for job in packet_jobs) == expected_job_ids
    )
    packet_by_id = (
        {job["id"]: job for job in packet_jobs} if packet_schema_valid else {}
    )
    canonical_argv = expected_packet_argv(
        expected_source_manifest_combined_sha256,
        expected_realdata_source_sha256,
    )
    _require(
        failures["G1"],
        packet_schema_valid,
        "packet job schema",
    )
    for job in ("G2", "G3", "G4", "G5", "G6"):
        if "packet job schema" in failures["G1"]:
            failures[job].append("packet job schema")
    _require(
        failures["G6"],
        isinstance(packet, Mapping)
        and packet.get("schema_version") == 1
        and packet.get("source_manifest") == "source-manifest-pre-gpu.json",
        "packet metadata",
    )
    for job in ("G3", "G4", "G5"):
        packet_job = packet_by_id.get(job)
        argv = packet_job.get("argv") if isinstance(packet_job, Mapping) else None
        report_command = reports.get(job, {}).get("command")
        frozen_argv = canonical_argv[job]
        expected_command = frozen_argv[
            frozen_argv.index("scripts/validate_realdata.py") :
        ]
        _require(
            failures[job],
            argv == frozen_argv,
            f"{job} packet command",
        )
        _require(
            failures[job],
            report_command == expected_command,
            f"{job} exact command",
        )
    g1_packet = packet_by_id.get("G1")
    _require(
        failures["G1"],
        isinstance(g1_packet, Mapping)
        and g1_packet.get("argv") == canonical_argv["G1"],
        "G1 packet command",
    )
    g2_packet = packet_by_id.get("G2")
    _require(
        failures["G2"],
        isinstance(g2_packet, Mapping)
        and g2_packet.get("argv") == canonical_argv["G2"],
        "G2 packet command",
    )
    g6_packet = packet_by_id.get("G6")
    _require(
        failures["G6"],
        isinstance(g6_packet, Mapping)
        and g6_packet.get("argv") == canonical_argv["G6"],
        "G6 packet command",
    )
    g4_data = reports.get("G4", {}).get("data")
    g5_data = reports.get("G5", {}).get("data")
    if isinstance(g4_data, Mapping) and isinstance(g5_data, Mapping):
        g4_hashes = g4_data.get("arrow_sha256")
        g5_hashes = g5_data.get("arrow_sha256")
        _require(
            failures["G4"],
            isinstance(g4_hashes, Mapping)
            and isinstance(g5_hashes, Mapping)
            and all(
                g4_hashes.get(name) == g5_hashes.get(name) for name in LBA_TRAIN_FILES
            ),
            "G4/G5 train shard identity",
        )
        if "G4/G5 train shard identity" in failures["G4"]:
            failures["G5"].append("G4/G5 train shard identity")

    if not _sha256_text(expected_realdata_source_sha256):
        for job in ("G3", "G4", "G5"):
            failures[job].append(f"{job} invalid expected source hash")
    for job in ("G3", "G4", "G5"):
        report = reports.get(job)
        if (
            report is not None
            and report.get("source_sha256") != expected_realdata_source_sha256
        ):
            failures[job].append(f"{job} source hash")
    jobs = {}
    for job, messages in failures.items():
        outcome = "pass"
        if messages:
            outcome = (
                "invalid_receipt"
                if any(
                    "receipt" in message or "source" in message for message in messages
                )
                else "fail"
            )
        jobs[job] = {
            "passed": not messages,
            "outcome": outcome,
            "failures": messages,
        }
    return {
        "schema_version": 1,
        "status": "passed"
        if all(value["passed"] for value in jobs.values())
        else "failed",
        "jobs": jobs,
        "input_sha256": input_sha256,
        "frozen_source_manifest_combined_sha256": frozen_source_sha256,
        "expected_source_manifest_combined_sha256": (
            expected_source_manifest_combined_sha256
        ),
        "expected_realdata_source_sha256": expected_realdata_source_sha256,
        "interpretation": (
            "mechanical, numerical, resource, and bounded process evidence only; "
            "no accuracy-superiority claim"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--expected-source-manifest-combined-sha256",
        required=True,
    )
    parser.add_argument("--expected-realdata-source-sha256", required=True)
    args = parser.parse_args()
    try:
        result = adjudicate(
            args.run_dir,
            expected_source_manifest_combined_sha256=(
                args.expected_source_manifest_combined_sha256
            ),
            expected_realdata_source_sha256=args.expected_realdata_source_sha256,
        )
    except Exception as error:  # never lose the final fail-closed receipt
        result = {
            "schema_version": 1,
            "status": "failed",
            "jobs": {},
            "fatal_error": f"{type(error).__name__}: {error}",
            "interpretation": "invalid completion receipt",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
