from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_gpu_gate.py"
_SPEC = importlib.util.spec_from_file_location("_ela_gpu_gate_runner", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
runner = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = runner
_SPEC.loader.exec_module(runner)
SHA = "a" * 64


def _source_receipt() -> dict[str, object]:
    return {
        "path": "artifacts/source.json",
        "manifest_sha256": SHA,
        "combined_sha256": SHA,
        "file_count": 1,
        "verified_against_current_bytes": True,
    }


def test_gpu_gate_runner_writes_passed_receipt_without_capturing_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "receipt.json"
    monkeypatch.setattr(runner, "_verify_source_manifest", lambda path: _source_receipt())

    def run(command, **kwargs):
        assert list(command) == ["bash", "scripts/check.sh", "gpu"]
        assert kwargs["check"] is False
        assert kwargs["env"]["UV_LOCKED"] == "1"
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, "run", run)
    receipt, exit_code = runner.run_gate(tmp_path / "source.json", output)
    assert exit_code == 0
    assert receipt["status"] == "passed"
    assert receipt["exit_code"] == 0
    assert output.is_file()


def test_gpu_gate_runner_preserves_failure_and_returns_two(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "receipt.json"
    monkeypatch.setattr(runner, "_verify_source_manifest", lambda path: _source_receipt())
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 17),
    )
    receipt, exit_code = runner.run_gate(tmp_path / "source.json", output)
    assert exit_code == 2
    assert receipt["status"] == "failed"
    assert receipt["exit_code"] == 17
    assert "code 17" in receipt["failure"]


def test_gpu_gate_runner_does_not_launch_on_source_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "receipt.json"

    def reject(path):
        raise RuntimeError("tampered")

    monkeypatch.setattr(runner, "_verify_source_manifest", reject)
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("launched")),
    )
    receipt, exit_code = runner.run_gate(tmp_path / "source.json", output)
    assert exit_code == 2
    assert receipt["status"] == "failed"
    assert receipt["exit_code"] is None
    assert "tampered" in receipt["failure"]
