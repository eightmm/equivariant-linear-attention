#!/usr/bin/env python3
"""Capture the exact source bytes used by a claim-bearing local run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXACT_FILES = (
    "PROJECT.md",
    "pyproject.toml",
    "uv.lock",
    "scripts/ablate_architecture_lanes.py",
    "scripts/ablate_relation_transport.py",
    "scripts/benchmark_ela.py",
    "scripts/adjudicate_completion.py",
    "scripts/capture_source_manifest.py",
    "scripts/check.sh",
    "scripts/ml_smoke.py",
    "scripts/profile_gpu_completion.py",
    "scripts/run_completion_packet.py",
    "scripts/run_gpu_gate.py",
    "scripts/validate_realdata.py",
    "scripts/wheel_smoke.sh",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _source_paths() -> list[Path]:
    paths = {ROOT / name for name in EXACT_FILES}
    paths.update((ROOT / "src").rglob("*.py"))
    paths.update((ROOT / "tests").rglob("*.py"))
    missing = sorted(path.relative_to(ROOT).as_posix() for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError(f"source-manifest inputs are missing: {missing}")
    return sorted(paths, key=lambda path: path.relative_to(ROOT).as_posix())


def build_manifest() -> dict[str, object]:
    records: list[dict[str, object]] = []
    combined = hashlib.sha256()
    for path in _source_paths():
        relative = path.relative_to(ROOT).as_posix()
        size = path.stat().st_size
        sha256 = _sha256(path)
        records.append({"path": relative, "size_bytes": size, "sha256": sha256})
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(str(size).encode("ascii"))
        combined.update(b"\0")
        combined.update(sha256.encode("ascii"))
        combined.update(b"\0")
    return {
        "schema_version": 1,
        "algorithm": "sorted-path-size-sha256-v1",
        "git": {
            "head": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "dirty": bool(_git("status", "--porcelain")),
        },
        "scope": {
            "recursive": ["src/**/*.py", "tests/**/*.py"],
            "exact": list(EXACT_FILES),
        },
        "file_count": len(records),
        "combined_sha256": combined.hexdigest(),
        "files": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        output.relative_to(ROOT)
    except ValueError as error:
        raise ValueError("output must stay inside the repository") from error
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_manifest(), indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
