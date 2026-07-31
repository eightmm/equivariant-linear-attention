from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def test_resource_benchmark_is_strict_and_functionally_paired(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "overhead.json"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/benchmark_canonical_ela.py"),
            "--nodes",
            "8",
            "--degree",
            "2",
            "--width",
            "16",
            "--depth",
            "1",
            "--warmup",
            "0",
            "--repeats",
            "1",
            "--device",
            "cpu",
            "--dtype",
            "float64",
            "--expected-source-root",
            str(root / "src"),
            "--output",
            str(output),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(
        output.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"nonfinite JSON constant: {value}")
        ),
    )

    equivalence = payload["functional_equivalence"]
    assert equivalence["node_output_max_abs"] == 0.0
    assert equivalence["graph_output_max_abs"] == 0.0
    assert equivalence["feature_gradient_max_abs"] == 0.0
    assert equivalence["position_gradient_max_abs"] == 0.0
    assert equivalence["common_parameter_gradient_max_abs"] == 0.0
    assert equivalence["candidate_branch_gradients_finite"] is True
    assert equivalence["candidate_branch_gradients_nonzero"] is True
    assert payload["same_common_weights"] is True
    assert payload["source_verified"] is True
    assert payload["control"]["inference_peak_allocated_bytes"] is None
    assert len(payload["control"]["inference"]["samples_ms"]) == 1
