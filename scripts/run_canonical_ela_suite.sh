#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

MODE="${ELA_SUITE_MODE:-smoke}"
DEVICE="${ELA_SUITE_DEVICE:-$(uv run --locked python - <<'PY'
import torch
print("cuda" if torch.cuda.is_available() else "cpu")
PY
)}"
DTYPE="${ELA_SUITE_DTYPE:-$(if [[ "$DEVICE" == cuda* ]]; then echo bfloat16; else echo float64; fi)}"
RUN_DIR="${1:-artifacts/canonical-ela/$(date +%Y%m%d-%H%M%S)}"

case "$MODE" in
  smoke)
    NODES="${ELA_SUITE_NODES:-128}"
    DEGREE="${ELA_SUITE_DEGREE:-8}"
    WIDTH="${ELA_SUITE_WIDTH:-32}"
    DEPTH="${ELA_SUITE_DEPTH:-2}"
    WARMUP="${ELA_SUITE_WARMUP:-1}"
    REPEATS="${ELA_SUITE_REPEATS:-2}"
    REQUIRE_CLEAN="${ELA_SUITE_REQUIRE_CLEAN:-0}"
    ;;
  full)
    NODES="${ELA_SUITE_NODES:-4096}"
    DEGREE="${ELA_SUITE_DEGREE:-32}"
    WIDTH="${ELA_SUITE_WIDTH:-128}"
    DEPTH="${ELA_SUITE_DEPTH:-8}"
    WARMUP="${ELA_SUITE_WARMUP:-10}"
    REPEATS="${ELA_SUITE_REPEATS:-30}"
    REQUIRE_CLEAN="${ELA_SUITE_REQUIRE_CLEAN:-1}"
    ;;
  *)
    echo "ELA_SUITE_MODE must be smoke or full" >&2
    exit 2
    ;;
esac

if [[ -e "$RUN_DIR" ]]; then
  if [[ ! -d "$RUN_DIR" ]] || [[ -n "$(find "$RUN_DIR" -mindepth 1 -print -quit)" ]]; then
    echo "run directory must be new or empty: $RUN_DIR" >&2
    exit 2
  fi
else
  mkdir -p "$RUN_DIR"
fi

GIT_SHA="$(git rev-parse HEAD)"
GIT_STATUS="$(git status --porcelain)"
if [[ "$REQUIRE_CLEAN" == "1" && -n "$GIT_STATUS" ]]; then
  echo "full canonical suite requires a clean worktree" >&2
  exit 2
fi

uv run --locked python - "$REPO_ROOT" > "$RUN_DIR/source-provenance.txt" <<'PY'
from pathlib import Path
import sys

import equivariant_attention

repo_root = Path(sys.argv[1]).resolve()
expected = (repo_root / "src").resolve()
source_file = Path(equivariant_attention.__file__).resolve()
source_root = source_file.parents[1]
if source_root != expected:
    raise SystemExit(f"wrong source root: {source_root} != {expected}")
print(f"source_file={source_file}")
print(f"source_root={source_root}")
print(f"expected_source_root={expected}")
PY

{
  echo "mode=$MODE"
  echo "device=$DEVICE"
  echo "dtype=$DTYPE"
  echo "git_sha=$GIT_SHA"
  echo "git_dirty=$([[ -n "$GIT_STATUS" ]] && echo true || echo false)"
  uv run --locked python --version
  uv --version
  uv pip freeze
  nvidia-smi 2>/dev/null || true
} > "$RUN_DIR/environment.txt"

{
  git status --short
  git rev-parse HEAD
  git diff --stat
  git diff --check
} > "$RUN_DIR/git.txt"

uv run --locked pytest -q \
  tests/test_branch_fusion.py \
  tests/test_branch_fusion_zero_init.py \
  tests/test_canonical_api.py \
  tests/test_canonical_double_backward.py \
  tests/test_canonical_equivariance.py \
  tests/test_canonical_migration.py \
  tests/test_canonical_regression.py \
  tests/test_canonical_resource_benchmark.py \
  tests/test_canonical_resource_summary.py \
  tests/test_canonical_branch_fusion_downstream.py \
  tests/test_ela_context.py \
  tests/test_api_policy.py \
  2>&1 | tee "$RUN_DIR/focused-tests.log"

if [[ "$DEVICE" == cuda* ]]; then
  uv run --locked pytest -q tests/test_canonical_cuda.py \
    2>&1 | tee "$RUN_DIR/cuda-focused.log"
fi

if [[ "$MODE" == "full" ]]; then
  bash scripts/check.sh fast 2>&1 | tee "$RUN_DIR/fast-gate.log"
  if [[ "$DEVICE" == cuda* ]]; then
    bash scripts/check.sh gpu 2>&1 | tee "$RUN_DIR/gpu-gate.log"
  fi
fi

BENCHMARK_ARGS=(
  --nodes "$NODES"
  --degree "$DEGREE"
  --width "$WIDTH"
  --depth "$DEPTH"
  --warmup "$WARMUP"
  --repeats "$REPEATS"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --expected-source-root "$REPO_ROOT/src"
  --output "$RUN_DIR/overhead.json"
)
uv run --locked python scripts/benchmark_canonical_ela.py \
  "${BENCHMARK_ARGS[@]}" \
  2>&1 | tee "$RUN_DIR/benchmark.log"

uv run --locked python - "$RUN_DIR" "$MODE" "$DEVICE" "$DTYPE" "$GIT_SHA" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

run_dir = Path(sys.argv[1])
mode, device, dtype, git_sha = sys.argv[2:]

def reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")

overhead = json.loads(
    (run_dir / "overhead.json").read_text(encoding="utf-8"),
    parse_constant=reject_constant,
)
if overhead["git_sha"] != git_sha:
    raise SystemExit("benchmark git SHA does not match suite SHA")
if overhead["source_file"] != str(
    (Path.cwd() / "src/equivariant_attention/__init__.py").resolve()
):
    raise SystemExit("benchmark source file does not match current repository")

required = [
    "overhead.json",
    "environment.txt",
    "git.txt",
    "source-provenance.txt",
    "focused-tests.log",
    "benchmark.log",
]
commands = [
    "focused canonical pytest set",
    "benchmark_canonical_ela.py",
]
if device.startswith("cuda"):
    required.append("cuda-focused.log")
    commands.append("pytest tests/test_canonical_cuda.py")
if mode == "full":
    required.append("fast-gate.log")
    commands.append("bash scripts/check.sh fast")
    if device.startswith("cuda"):
        required.append("gpu-gate.log")
        commands.append("bash scripts/check.sh gpu")

files = {}
for name in required:
    path = run_dir / name
    if not path.is_file():
        raise SystemExit(f"missing required artifact: {name}")
    data = path.read_bytes()
    files[name] = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }

manifest = {
    "schema_version": 3,
    "suite": "canonical_ela",
    "status": "completed",
    "mode": mode,
    "device": device,
    "dtype": dtype,
    "git_sha": git_sha,
    "git_dirty": bool(overhead["git_dirty"]),
    "source_file": overhead["source_file"],
    "public_architecture": "ELA",
    "public_layer": "ELALayer",
    "benchmark_role": "functional_and_dtype_safety_not_resource_decision",
    "commands": commands,
    "files": files,
}
(run_dir / "manifest.json").write_text(
    json.dumps(manifest, indent=2, allow_nan=False),
    encoding="utf-8",
)
PY

printf 'Canonical ELA suite complete: %s\n' "$RUN_DIR"
