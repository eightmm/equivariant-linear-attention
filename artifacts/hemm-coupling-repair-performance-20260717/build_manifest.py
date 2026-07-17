#!/usr/bin/env python3
"""Build the Codex Science provenance manifest for this run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[1]
RUN_ID = RUN_DIR.name


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: str, kind: str) -> dict[str, str]:
    return {"path": path, "kind": kind, "sha256": sha256(RUN_DIR / path)}


def main() -> None:
    saved = {
        "scope.md": "approved-plan",
        "compute-environment.json": "environment-record",
        "tdd-red.md": "test-record",
        "incumbent_counterfactual.py": "analysis-code",
        "incumbent-counterfactual.json": "counterfactual-output",
        "analyze_stage0.py": "analysis-code",
        "stage0-summary.json": "decision-table",
        "stage0-report.md": "scientific-report",
        "stage0-suite.json.gz": "compressed-raw-output",
        "stage0-rerun-provenance.json": "execution-receipt",
        "implementation-evidence.md": "verification-record",
        "run_cuda_benchmarks.py": "benchmark-code",
        "cuda-benchmarks.json": "unpinned-preliminary-benchmark-output",
        "cuda-benchmarks-vectorized-smoke.json": "smoke-benchmark-output",
        "cuda-benchmarks-vectorized.json": "benchmark-output",
        "qm9-smoke-lgl.json": "training-smoke-output",
        "run_qm9_paired.sh": "execution-code",
        "publish_qm9_ledger.py": "publication-code",
        "qm9-runs-public.jsonl": "execution-ledger",
        "qm9-runs-public-provenance.json": "publication-transform-receipt",
        "qm9/ggg-m1-seed41.json": "training-output",
        "qm9/lgl-m1-seed41.json": "training-output",
        "qm9/ggg-m1-seed42.json": "training-output",
        "qm9/lgl-m1-seed42.json": "training-output",
        "qm9/ggg-m1-seed43.json": "training-output",
        "qm9/lgl-m1-seed43.json": "training-output",
        "analyze_performance.py": "analysis-code",
        "build_manifest.py": "provenance-code",
        "performance-summary.json": "decision-table",
        "performance-report.md": "scientific-report",
        "performance-progress.md": "execution-narrative",
    }
    review_path = RUN_DIR / "independent-review.json"
    structural_review_path = RUN_DIR / "structural-review.json"
    if structural_review_path.exists():
        saved[structural_review_path.name] = "deterministic-review-output"
    if review_path.exists():
        saved[review_path.name] = "independent-review-receipt"
        review = json.loads(review_path.read_text(encoding="utf-8"))
        review_record = {
            "status": review["status"],
            "reviewer": review["reviewer"],
            "independent": review["independent"],
            "findings": review["findings"],
        }
    else:
        review_record = {"status": "pending", "findings": []}

    compute = json.loads(
        (RUN_DIR / "compute-environment.json").read_text(encoding="utf-8")
    )
    performance = json.loads(
        (RUN_DIR / "performance-summary.json").read_text(encoding="utf-8")
    )
    manifest = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "question": (
            "Why is the current HEMM pair gate inactive, can the registered "
            "coupling/router repair pass frozen Stage-0, and does M=1 lgl "
            "improve the registered QM9 gap validation/resource tradeoff over ggg?"
        ),
        "plan": [
            {"id": "scope", "status": "completed"},
            {"id": "counterfactual-diagnosis", "status": "completed"},
            {"id": "repair-and-tests", "status": "completed"},
            {"id": "stage0-matrix", "status": "completed"},
            {"id": "cuda-resource-gate", "status": "completed"},
            {"id": "paired-qm9-validation", "status": "completed"},
            {"id": "provenance-analysis", "status": "completed"},
        ],
        "inputs": [
            {
                "kind": "feedback",
                "url": "https://chatgpt.com/share/6a59db49-8b50-83ee-9691-07bb73a472f4",
                "accessed": "2026-07-17",
            },
            {
                "kind": "git-base",
                "commit": "591e7a241315f697a39c5354a48dd345639fed69",
            },
            {
                "kind": "qm9-local-data",
                "target": {"index": 4, "name": "gap", "unit": "eV"},
                "files": performance["data_identity"],
                "split_hashes": performance["split_hashes"],
                "test_evaluated": False,
            },
        ],
        "code": [
            {
                "path": "src/equivariant_attention/moment.py",
                "commit": "a8bda61868cf118b93b6a605001fb401c23f46c1",
                "sha256": sha256(ROOT / "src/equivariant_attention/moment.py"),
            },
            {
                "path": "scripts/probe_memory_activation.py",
                "commit": "a8bda61868cf118b93b6a605001fb401c23f46c1",
                "sha256": sha256(ROOT / "scripts/probe_memory_activation.py"),
            },
            {
                "path": "scripts/train_compare.py",
                "commit": "a8bda61868cf118b93b6a605001fb401c23f46c1",
                "sha256": sha256(ROOT / "scripts/train_compare.py"),
            },
        ],
        "executions": [
            {
                "command": "uv sync --locked --extra qm9",
                "exit_code": 0,
                "result": "locked PyG 2.8.0 and RDKit 2026.3.3 environment",
            },
            {
                "command": (
                    "uv run --locked python artifacts/hemm-coupling-repair-"
                    "performance-20260717/incumbent_counterfactual.py"
                ),
                "exit_code": 0,
                "output": "incumbent-counterfactual.json",
            },
            {
                "command": (
                    "uv run --locked python scripts/probe_memory_activation.py "
                    "--suite --memory-counts 4 8 --device cpu --dtype float64 "
                    "--hidden-dims 16 64 --seeds 401 402 403 --scenarios aligned "
                    "crossed spatial_only semantic_only --num-heads 4 --metrics-out "
                    "artifacts/hemm-coupling-repair-"
                    "performance-20260717/stage0-suite.json"
                ),
                "exit_code": 0,
                "output": "stage0-suite.json.gz",
                "receipt": "stage0-rerun-provenance.json",
            },
            {
                "command": (
                    "uv run --locked python artifacts/hemm-coupling-repair-"
                    "performance-20260717/analyze_stage0.py"
                ),
                "exit_code": 0,
                "output": "stage0-summary.json",
            },
            {
                "command": "scripts/check.sh fast",
                "exit_code": 0,
                "result": "235 passed; 89.20% coverage; CPU smoke passed",
            },
            {
                "command": "scripts/check.sh gpu",
                "exit_code": 0,
                "result": "bf16 and FP32 CUDA smoke passed",
            },
            {
                "command": (
                    "uv run --locked python artifacts/hemm-coupling-repair-"
                    "performance-20260717/run_cuda_benchmarks.py --repeats 5 "
                    "--out artifacts/hemm-coupling-repair-performance-20260717/"
                    "cuda-benchmarks-vectorized.json"
                ),
                "exit_code": 0,
                "output": "cuda-benchmarks-vectorized.json",
            },
            {
                "command": (
                    "uv run --locked python scripts/train_compare.py --dataset qm9 "
                    "--num-samples 512 --train-size 400 --val-size 100 --steps 2 "
                    "--routing lgl --memory-count 1 --skip-test-eval"
                ),
                "exit_code": 0,
                "output": "qm9-smoke-lgl.json",
            },
            {
                "command": (
                    "bash artifacts/hemm-coupling-repair-performance-20260717/"
                    "run_qm9_paired.sh"
                ),
                "exit_code": 0,
                "output": "qm9-runs.jsonl (local checkpoint evidence)",
            },
            {
                "command": (
                    "uv run --locked python artifacts/hemm-coupling-repair-"
                    "performance-20260717/publish_qm9_ledger.py"
                ),
                "exit_code": 0,
                "output": (
                    "qm9-runs-public.jsonl; "
                    "qm9-runs-public-provenance.json"
                ),
            },
            {
                "command": (
                    "uv run --locked python artifacts/hemm-coupling-repair-"
                    "performance-20260717/analyze_performance.py"
                ),
                "exit_code": 0,
                "output": "performance-summary.json",
            },
        ],
        "failed_attempts": [
            {
                "stage": "stage0-provenance",
                "outcome": (
                    "The first full suite omitted lane width/head metadata and was "
                    "excluded; the official suite was rerun with explicit metadata."
                ),
            },
            {
                "stage": "performance-analysis",
                "command": (
                    "uv run --locked python artifacts/hemm-coupling-repair-"
                    "performance-20260717/analyze_performance.py"
                ),
                "exit_code": 1,
                "outcome": (
                    "The first analysis assumed an eight-character ledger SHA; "
                    "it wrote no output and was rerun with exact prefix validation."
                ),
            },
        ],
        "environment": {
            **compute,
            "implementation_commit": "a8bda61868cf118b93b6a605001fb401c23f46c1",
            "qm9_model_seeds": [41, 42, 43],
            "split_seed": 42,
            "gpu_budget_minutes": 30,
            "gpu_accounting": performance["compute_envelope"],
        },
        "artifacts": [artifact(path, kind) for path, kind in saved.items()],
        "claims": [
            {
                "id": "claim-hemm-stage0-block",
                "text": (
                    "Neither the learned invariant router nor any registered "
                    "radial/identity/residual coupling admits M=4/M=8 across all "
                    "aligned Stage-0 lanes."
                ),
                "evidence": [
                    "stage0-summary.json",
                    "stage0-report.md",
                    "stage0-suite.json.gz",
                    "stage0-rerun-provenance.json",
                ],
            },
            {
                "id": "claim-local-vectorization",
                "text": (
                    "The exact-semantics vectorized same-graph candidate "
                    "implementation passes the registered final-source CUDA "
                    "resource gate; the earlier unpinned failure is diagnostic "
                    "rather than causal evidence."
                ),
                "evidence": [
                    "implementation-evidence.md",
                    "cuda-benchmarks-vectorized.json",
                ],
            },
            {
                "id": "claim-lgl-registered-probe",
                "text": (
                    "M=1 lgl passes the registered three-seed QM9 gap validation "
                    "and CUDA resource rules against matched M=1 ggg."
                ),
                "evidence": [
                    "performance-summary.json",
                    "performance-report.md",
                    "qm9-runs-public.jsonl",
                    "cuda-benchmarks-vectorized.json",
                ],
            },
            {
                "id": "claim-boundary",
                "text": (
                    "The performance result is validation-only on a random-row "
                    "warm split; no test, cold-molecule, EGNN, or default claim is made."
                ),
                "evidence": [
                    "performance-summary.json",
                    "qm9-runs-public.jsonl",
                    "qm9-runs-public-provenance.json",
                ],
            },
        ],
        "review": review_record,
    }
    (RUN_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
