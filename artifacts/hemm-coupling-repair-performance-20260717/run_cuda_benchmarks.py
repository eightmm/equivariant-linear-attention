#!/usr/bin/env python3
"""Run fresh-process synchronized CUDA benchmarks and aggregate medians."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    records: list[dict[str, object]] = []
    started_all = time.perf_counter()
    for routing in ("ggg", "lgl"):
        for nodes_per_graph in (18, 29):
            command = [
                "uv",
                "run",
                "--locked",
                "python",
                "scripts/bench_attention.py",
                "--device",
                "cuda",
                "--dtype",
                "float32",
                "--graphs",
                "64",
                "--nodes-per-graph",
                str(nodes_per_graph),
                "--iters",
                "50",
                "--warmup",
                "20",
                "--routing",
                routing,
                "--memory-count",
                "1",
            ]
            for repeat in range(args.repeats):
                started = time.perf_counter()
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                wall_seconds = time.perf_counter() - started
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"benchmark failed ({routing}, {nodes_per_graph}, {repeat}): "
                        f"{completed.stderr[-2000:]}"
                    )
                rows = list(csv.DictReader(io.StringIO(completed.stdout)))
                if len(rows) != 2:
                    raise RuntimeError("benchmark must emit forward and forward_backward")
                for row in rows:
                    records.append(
                        {
                            "routing": routing,
                            "nodes_per_graph": nodes_per_graph,
                            "repeat": repeat,
                            "pass": row["pass"],
                            "milliseconds": float(row["ms"]),
                            "peak_memory_mib": float(row["peak_mem_mib"]),
                            "wall_seconds": wall_seconds,
                            "command": command,
                            "exit_code": completed.returncode,
                        }
                    )

    aggregates: list[dict[str, object]] = []
    for routing in ("ggg", "lgl"):
        for nodes_per_graph in (18, 29):
            for pass_name in ("forward", "forward_backward"):
                selected = [
                    row
                    for row in records
                    if row["routing"] == routing
                    and row["nodes_per_graph"] == nodes_per_graph
                    and row["pass"] == pass_name
                ]
                aggregates.append(
                    {
                        "routing": routing,
                        "graphs": 64,
                        "nodes_per_graph": nodes_per_graph,
                        "pass": pass_name,
                        "process_count": len(selected),
                        "median_process_mean_ms": statistics.median(
                            row["milliseconds"] for row in selected
                        ),
                        "min_process_mean_ms": min(
                            row["milliseconds"] for row in selected
                        ),
                        "max_process_mean_ms": max(
                            row["milliseconds"] for row in selected
                        ),
                        "median_peak_memory_mib": statistics.median(
                            row["peak_memory_mib"] for row in selected
                        ),
                    }
                )

    comparisons: list[dict[str, object]] = []
    for nodes_per_graph in (18, 29):
        for pass_name in ("forward", "forward_backward"):
            baseline = next(
                row
                for row in aggregates
                if row["routing"] == "ggg"
                and row["nodes_per_graph"] == nodes_per_graph
                and row["pass"] == pass_name
            )
            candidate = next(
                row
                for row in aggregates
                if row["routing"] == "lgl"
                and row["nodes_per_graph"] == nodes_per_graph
                and row["pass"] == pass_name
            )
            latency_ratio = (
                candidate["median_process_mean_ms"]
                / baseline["median_process_mean_ms"]
            )
            memory_ratio = (
                candidate["median_peak_memory_mib"]
                / baseline["median_peak_memory_mib"]
            )
            comparisons.append(
                {
                    "nodes_per_graph": nodes_per_graph,
                    "pass": pass_name,
                    "latency_ratio": latency_ratio,
                    "latency_increase_fraction": latency_ratio - 1.0,
                    "peak_memory_ratio": memory_ratio,
                    "peak_memory_increase_fraction": memory_ratio - 1.0,
                    "passes_20_percent_ceiling": latency_ratio <= 1.2
                    and memory_ratio <= 1.2,
                }
            )

    source = ROOT / "scripts" / "bench_attention.py"
    model_source = ROOT / "src" / "equivariant_attention" / "moment.py"
    result = {
        "schema_version": 1,
        "test_evaluated": False,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        ),
        "device": {
            "name": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
        "protocol": {
            "fresh_process_repeats": args.repeats,
            "warmup_iterations": 20,
            "measurement_iterations": 50,
            "graphs": 64,
            "nodes_per_graph": [18, 29],
            "dtype": "float32",
            "synchronized": True,
            "compile": False,
        },
        "bench_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "model_source_sha256": hashlib.sha256(
            model_source.read_bytes()
        ).hexdigest(),
        "records": records,
        "aggregates": aggregates,
        "comparisons": comparisons,
        "all_comparisons_within_20_percent": all(
            row["passes_20_percent_ceiling"] for row in comparisons
        ),
        "wall_seconds": time.perf_counter() - started_all,
    }
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
