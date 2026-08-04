#!/usr/bin/env python3
"""Run the frozen CUDA gate and always write a source-bound receipt."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import ModuleType
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
GPU_GATE_COMMAND = ("bash", "scripts/check.sh", "gpu")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _capture_module() -> ModuleType:
    path = ROOT / "scripts" / "capture_source_manifest.py"
    spec = importlib.util.spec_from_file_location("_ela_source_manifest", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load source-manifest implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _current_source_manifest() -> Mapping[str, Any]:
    module = _capture_module()
    manifest = module.build_manifest()
    if not isinstance(manifest, Mapping):
        raise TypeError("source-manifest builder returned a non-object")
    return manifest


def _verify_source_manifest(path: Path) -> dict[str, Any]:
    expected = _load_object(path)
    current = _current_source_manifest()
    for key in ("schema_version", "algorithm", "scope", "file_count", "files"):
        if expected.get(key) != current.get(key):
            raise RuntimeError(f"source manifest differs from current bytes: {key}")
    combined = expected.get("combined_sha256")
    if not isinstance(combined, str) or len(combined) != 64:
        raise ValueError("source manifest has no valid combined SHA-256")
    if combined != current.get("combined_sha256"):
        raise RuntimeError("source manifest combined SHA-256 differs from current bytes")
    return {
        "path": path.resolve().relative_to(ROOT).as_posix(),
        "manifest_sha256": _sha256(path),
        "combined_sha256": combined,
        "file_count": expected.get("file_count"),
        "verified_against_current_bytes": True,
    }


def _write_receipt(output: Path, receipt: Mapping[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def run_gate(
    source_manifest: Path,
    output: Path,
    *,
    command: Sequence[str] = GPU_GATE_COMMAND,
) -> tuple[dict[str, Any], int]:
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    source_receipt: Mapping[str, Any] | None = None
    gate_exit_code: int | None = None
    failure: str | None = None
    try:
        source_receipt = _verify_source_manifest(source_manifest.resolve())
        _write_receipt(
            output.resolve(),
            {
                "schema_version": 1,
                "experiment": "ela_gpu_gate",
                "status": "running",
                "command": list(command),
                "exit_code": None,
                "source_manifest": source_receipt,
                "started_at_utc": started_at,
                "elapsed_seconds": 0.0,
                "failure": None,
            },
        )
        environment = os.environ.copy()
        environment["UV_LOCKED"] = "1"
        completed = subprocess.run(
            tuple(command),
            cwd=ROOT,
            env=environment,
            check=False,
        )
        gate_exit_code = int(completed.returncode)
        if gate_exit_code != 0:
            failure = f"GPU gate exited with code {gate_exit_code}"
    except Exception as error:  # receipt must survive provenance/launch failures
        failure = f"{type(error).__name__}: {error}"
    receipt = {
        "schema_version": 1,
        "experiment": "ela_gpu_gate",
        "status": "passed" if failure is None else "failed",
        "command": list(command),
        "exit_code": gate_exit_code,
        "source_manifest": source_receipt,
        "started_at_utc": started_at,
        "elapsed_seconds": time.perf_counter() - started,
        "failure": failure,
    }
    _write_receipt(output.resolve(), receipt)
    return receipt, 0 if failure is None else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt, exit_code = run_gate(args.source_manifest, args.output)
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
