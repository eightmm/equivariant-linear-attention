from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest
import torch


def _load_profiler() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts/profile_gpu_completion.py"
    spec = importlib.util.spec_from_file_location("profile_gpu_completion", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load GPU completion profiler")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROFILER = _load_profiler()


def test_timing_summary_retains_raw_samples_and_interpolated_p95() -> None:
    summary = PROFILER._timing_summary([5.0, 1.0, 3.0, 2.0, 4.0])

    assert summary == {
        "median_ms": 3.0,
        "p95_ms": pytest.approx(4.8),
        "min_ms": 1.0,
        "max_ms": 5.0,
        "samples_ms": [5.0, 1.0, 3.0, 2.0, 4.0],
    }


def test_topology_canonicalization_ignores_only_edge_order() -> None:
    first = torch.tensor([[2, 0, 1], [1, 2, 0]])
    reordered = first[:, torch.tensor([1, 2, 0])]
    different = torch.tensor([[2, 0, 1], [0, 2, 0]])

    actual = PROFILER._canonical_topology_codes(first, 3)
    torch.testing.assert_close(
        actual,
        PROFILER._canonical_topology_codes(reordered, 3),
    )
    assert not torch.equal(
        actual,
        PROFILER._canonical_topology_codes(different, 3),
    )


def test_parameter_gradient_error_reports_missing_and_numeric_error() -> None:
    reference = {
        "a": torch.tensor([1.0, 2.0]),
        "only_reference": torch.tensor([3.0]),
    }
    candidate = {
        "a": torch.tensor([1.0, 2.5]),
        "only_candidate": torch.tensor([4.0]),
    }

    error = PROFILER._parameter_gradient_error(candidate, reference)

    assert error["max_abs"] == pytest.approx(0.5)
    assert error["relative_l2"] == pytest.approx(0.5 / (5.0**0.5))
    assert error["compared_parameter_tensors"] == 1
    assert error["candidate_only"] == ["only_candidate"]
    assert error["reference_only"] == ["only_reference"]


def test_compiler_counter_delta_distinguishes_missing_from_zero() -> None:
    before: dict[str, dict[str, int]] = {}
    after = {"stats": {"unique_graphs": 2}}

    assert PROFILER._counter_value(before, "stats", "unique_graphs") is None
    assert (
        PROFILER._counter_delta(
            before,
            after,
            "stats",
            "unique_graphs",
        )
        == 2
    )
    assert (
        PROFILER._counter_delta(
            after,
            {"stats": {"unique_graphs": 2}},
            "stats",
            "unique_graphs",
        )
        == 0
    )
    assert PROFILER._counter_delta(after, {}, "stats", "unique_graphs") is None


def test_peak_allocation_gate_is_fail_closed_and_recursive() -> None:
    receipt = {
        "first": {"peak_allocated_bytes": 7},
        "nested": [{"peak_allocated_bytes": 11}],
    }
    assert PROFILER._maximum_recorded_peak_allocated_bytes(receipt) == 11
    assert PROFILER._maximum_recorded_peak_allocated_bytes({}) is None

    lane = {"measurements": receipt}
    assert PROFILER._attach_resource_gate(lane)
    assert lane["resource_gate"]["observed"]["maximum_peak_allocated_bytes"] == 11

    over_limit = {
        "measurements": {
            "sample": {"peak_allocated_bytes": PROFILER.PEAK_ALLOCATED_LIMIT_BYTES}
        }
    }
    assert not PROFILER._attach_resource_gate(over_limit)


def test_source_manifest_verification_binds_exact_current_bytes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sample.py"
    source.write_text("value = 1\n", encoding="utf-8")
    sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    size = source.stat().st_size
    combined = hashlib.sha256()
    for value in ("sample.py", str(size), sha256):
        combined.update(value.encode("utf-8"))
        combined.update(b"\0")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "combined_sha256": combined.hexdigest(),
                "files": [
                    {
                        "path": "sample.py",
                        "size_bytes": size,
                        "sha256": sha256,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    receipt = PROFILER._verify_source_manifest(manifest, tmp_path)
    assert receipt["combined_sha256"] == combined.hexdigest()
    assert receipt["verified_against_current_bytes"] is True

    source.write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source manifest mismatch"):
        PROFILER._verify_source_manifest(manifest, tmp_path)


def test_expected_triton_dispatch_covers_every_fused_family_per_layer() -> None:
    assert PROFILER._expected_triton_dispatch(2) == {
        "weighted_pair_gate_lanes": [[0, 1], [2, 3], [0, 1], [2, 3]],
        "tensor_pair_gate_lanes": [[4, 5], [4, 5]],
        "direction_triple_gate_lanes": [[6, 7, 8], [6, 7, 8]],
    }


def test_schema_only_cli_is_cpu_safe_and_declares_all_truthful_scopes(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "gpu-completion-schema.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/profile_gpu_completion.py",
            "--schema-only",
            "--output",
            str(output),
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert json.loads(result.stdout) == payload
    assert payload["schema_version"] == 1
    assert payload["experiment"] == "ela_cuda_completion_profile"
    assert payload["status"] == "schema_only"
    assert len(payload["git"]["sha"]) == 40
    assert payload["device"]["actual"] is None
    assert payload["budget_contract"] == {
        "target_wall_time_minutes_lt": 10,
        "target_peak_allocated_gib_lt": 16,
        "defaults_are_bounded": True,
        "claim": "hard per-lane timed peak-allocation acceptance gate",
    }
    assert tuple(payload["lanes"]) == PROFILER.LANE_NAMES
    assert all(lane["status"] == "planned" for lane in payload["lanes"].values())
    assert "public" in payload["lanes"]["trusted_prepared_cache"]["scope"]
    assert "private" in payload["lanes"]["ragged_global"]["scope"]
    assert "public" in payload["lanes"]["radius_ingestion"]["scope"]
    assert "private" in payload["lanes"]["triton_training_local"]["scope"]
    assert "private" in payload["lanes"]["compiled_numerical_core"]["scope"]
    assert payload["config"]["warmup"] == 3
    assert payload["config"]["repeats"] == 7
    assert sum(payload["config"]["ragged_counts"]) == 3834
    assert payload["timing_contract"]["peak_baseline"] == (
        "after optional lane setup and device synchronization"
    )
    assert payload["acceptance_contract"] == PROFILER._acceptance_contract()
    assert payload["acceptance_contract"]["ragged_global"] == {
        "transport_relative_l2_max": 0.05,
        "balanced_relative_l2_max": 0.05,
        "native_over_segmented_inference_median_max": 1.0,
        "integrated_native_calls_exact": 1,
        "dispatch_vs_direct_max_abs": 0.0,
        "peak_allocated_bytes_lt": 16 * 1024**3,
    }
    assert payload["acceptance_contract"]["compiled_numerical_core"][
        "maximum_absolute_error"
    ] == pytest.approx(1.0e-3)


def test_radius_receipt_thresholds_are_derived_from_frozen_contract() -> None:
    expected = {
        key: value
        for key, value in PROFILER._acceptance_contract()["radius_ingestion"].items()
        if key != "peak_allocated_bytes_lt"
    }

    assert PROFILER._functional_thresholds("radius_ingestion") == expected
    assert expected["latency_promotion_gate"] is None


def test_report_schema_rejects_missing_completion_lane() -> None:
    args = PROFILER._parser().parse_args(["--schema-only", "--output", "unused.json"])
    repository = Path(__file__).resolve().parents[1]
    report = PROFILER._base_report(
        args,
        repository=repository,
        schema_only=True,
    )
    del report["lanes"]["compiled_numerical_core"]

    with pytest.raises(ValueError, match="five ordered completion lanes"):
        PROFILER._validate_report(report)
