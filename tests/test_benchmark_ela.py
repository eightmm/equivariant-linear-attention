from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_benchmark_reports_public_ingestion_lanes(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "benchmark.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_ela.py",
            "--nodes",
            "8",
            "--degree",
            "2",
            "--input-irreps",
            "2x0e",
            "--output-irreps",
            "1x0e",
            "--width",
            "16",
            "--depth",
            "2",
            "--warmup",
            "0",
            "--repeats",
            "1",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--include-end-to-end",
            "--e2e-nodes",
            "8",
            "--e2e-degree",
            "2",
            "--e2e-cutoff",
            "1.5",
            "--output",
            str(output),
        ],
        check=True,
        cwd=repository,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 4
    assert payload["neighbor_discovery_included"] is False
    end_to_end = payload["end_to_end"]
    assert end_to_end["graph_ingestion_included"] is True
    assert end_to_end["neighbor_discovery_included"] is True
    assert end_to_end["neighbor_discovery_lanes"] == [
        "cold_automatic_radius",
        "moving_coordinate_execution",
    ]

    profiles = end_to_end["profiles"]
    assert set(profiles) == {
        "cold_automatic_radius",
        "prepared_cache_reuse",
        "cold_explicit_topology",
        "moving_coordinate_execution",
    }
    assert profiles["cold_automatic_radius"]["neighbor_discovery_included"]
    moving = profiles["moving_coordinate_execution"]
    assert moving["neighbor_discovery_included"]
    assert moving["coordinate_rebuild_included"]
    assert moving["coordinate_rebuilds_per_call"] == 1
    assert not profiles["prepared_cache_reuse"]["neighbor_discovery_included"]
    assert profiles["prepared_cache_reuse"]["prepared_cache_reused"]
    assert profiles["prepared_cache_reuse"]["packed_template_reused"]
    assert profiles["prepared_cache_reuse"]["immutable_storage_assumed"]
    assert not profiles["prepared_cache_reuse"]["content_revalidation_included"]
    assert not profiles["cold_explicit_topology"]["neighbor_discovery_included"]
    assert profiles["cold_explicit_topology"]["coo_to_csr_included"]
    for profile in profiles.values():
        assert profile["graph_ingestion_included"] is True
        assert profile["peak_allocated_bytes"] is None
        timing = profile["timing"]
        assert timing["min_ms"] <= timing["p50_ms"] <= timing["p95_ms"]
        assert timing["p95_ms"] <= timing["p99_ms"] <= timing["max_ms"]
