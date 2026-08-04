from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_completion_packet.py"
_SPEC = importlib.util.spec_from_file_location("_ela_completion_packet_runner", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
runner = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = runner
_SPEC.loader.exec_module(runner)
SOURCE_SHA = "a" * 64
REALDATA_SHA = "b" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _job(job_id: str) -> dict[str, object]:
    return {
        "id": job_id,
        "label": f"job-{job_id}",
        "argv": runner._canonical_argv(SOURCE_SHA, REALDATA_SHA)[job_id],
    }


def _packet(tmp_path: Path) -> tuple[Path, dict[str, object], dict[str, object]]:
    source = {
        "combined_sha256": SOURCE_SHA,
        "file_count": 3,
    }
    source_path = tmp_path / "source-manifest-pre-gpu.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    packet = {
        "schema_version": 1,
        "source_manifest": source_path.name,
        "jobs": [_job(job_id) for job_id in runner.EXPECTED_JOB_IDS],
    }
    packet_path = tmp_path / "gpu-job-packet.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    identity = {
        "manifest_sha256": _sha256(source_path),
        "combined_sha256": source["combined_sha256"],
        "file_count": source["file_count"],
    }
    return packet_path, packet, identity


def _write_g1_receipt(tmp_path: Path, identity: dict[str, object]) -> None:
    source = {**identity, "verified_against_current_bytes": True}
    (tmp_path / "gpu-gate-receipt.json").write_text(
        json.dumps({"status": "passed", "exit_code": 0, "source_manifest": source}),
        encoding="utf-8",
    )


def _write_gpu_receipts(tmp_path: Path, identity: dict[str, object]) -> None:
    _write_g1_receipt(tmp_path, identity)
    source = {**identity, "verified_against_current_bytes": True}
    (tmp_path / "gpu-completion-profile.json").write_text(
        json.dumps({"status": "completed", "source_manifest": source}),
        encoding="utf-8",
    )


def _data_receipt(job: dict[str, object], *, status: str = "completed") -> dict[str, object]:
    argv = job["argv"]
    script_index = argv.index("scripts/validate_realdata.py")
    return {
        "status": status,
        "command": argv[script_index:],
        "source_sha256": REALDATA_SHA,
    }


def test_packet_requires_exact_digest_and_ordered_unique_jobs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    packet_path, packet, _ = _packet(tmp_path)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(
        runner,
        "_verify_source_current",
        lambda path: {"combined_sha256": SOURCE_SHA},
    )
    loaded = runner.load_frozen_packet(
        packet_path,
        expected_sha256=_sha256(packet_path),
    )
    assert [job["id"] for job in loaded["jobs"]] == list(runner.EXPECTED_JOB_IDS)

    packet["jobs"].append(_job("G6"))
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    with pytest.raises(runner.PacketError, match="ordered unique G1-G6"):
        runner.load_frozen_packet(packet_path, expected_sha256=_sha256(packet_path))
    with pytest.raises(runner.PacketError, match="SHA-256 mismatch"):
        runner.load_frozen_packet(packet_path, expected_sha256="0" * 64)


def test_packet_rejects_reordered_or_token_tampered_commands(
    tmp_path: Path,
    monkeypatch,
) -> None:
    packet_path, packet, _ = _packet(tmp_path)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(
        runner,
        "_verify_source_current",
        lambda path: {"combined_sha256": SOURCE_SHA},
    )

    packet["jobs"][0], packet["jobs"][1] = packet["jobs"][1], packet["jobs"][0]
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    with pytest.raises(runner.PacketError, match="ordered unique G1-G6"):
        runner.load_frozen_packet(packet_path, expected_sha256=_sha256(packet_path))

    _, packet, _ = _packet(tmp_path)
    packet["jobs"][1]["argv"][-1] = "tampered-output.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    with pytest.raises(runner.PacketError, match="G2 argv differs"):
        runner.load_frozen_packet(packet_path, expected_sha256=_sha256(packet_path))


def test_gpu_phase_halts_before_g2_when_g1_fails(tmp_path: Path) -> None:
    packet_path, packet, _ = _packet(tmp_path)
    submitted: list[str] = []

    def fail_g1(job: dict[str, object]) -> int:
        submitted.append(str(job["id"]))
        return 17

    with pytest.raises(runner.PacketError, match="G1 exited 17"):
        runner.execute_phase(
            packet_path,
            packet,
            phase="gpu",
            authorize_gpu=True,
            queue_job=fail_g1,
        )
    assert submitted == ["G1"]


def test_gpu_phase_requires_g1_receipt_before_submitting_g2(tmp_path: Path) -> None:
    packet_path, packet, _ = _packet(tmp_path)
    submitted: list[str] = []

    def no_receipt(job: dict[str, object]) -> int:
        submitted.append(str(job["id"]))
        return 0

    with pytest.raises(runner.PacketError, match="gpu-gate-receipt"):
        runner.execute_phase(
            packet_path,
            packet,
            phase="gpu",
            authorize_gpu=True,
            queue_job=no_receipt,
        )
    assert submitted == ["G1"]


def test_gpu_phase_requires_g2_terminal_receipt_even_if_wait_returns_zero(
    tmp_path: Path,
) -> None:
    packet_path, packet, identity = _packet(tmp_path)
    submitted: list[str] = []

    def missing_g2(job: dict[str, object]) -> int:
        submitted.append(str(job["id"]))
        if job["id"] == "G1":
            _write_g1_receipt(tmp_path, identity)
        return 0

    with pytest.raises(runner.PacketError, match="gpu-completion-profile"):
        runner.execute_phase(
            packet_path,
            packet,
            phase="gpu",
            authorize_gpu=True,
            queue_job=missing_g2,
        )
    assert submitted == ["G1", "G2"]


def test_data_phase_requires_explicit_authority_and_gpu_receipts(tmp_path: Path) -> None:
    packet_path, packet, identity = _packet(tmp_path)
    with pytest.raises(runner.PacketError, match="--authorize-data"):
        runner.execute_phase(packet_path, packet, phase="data")

    _write_gpu_receipts(tmp_path, identity)
    submitted: list[str] = []

    def complete(job: dict[str, object]) -> int:
        job_id = str(job["id"])
        submitted.append(job_id)
        (tmp_path / runner.RECEIPT_FILENAMES[job_id]).write_text(
            json.dumps(_data_receipt(job)),
            encoding="utf-8",
        )
        return 0

    result = runner.execute_phase(
        packet_path,
        packet,
        phase="data",
        authorize_data=True,
        queue_job=complete,
    )
    assert submitted == ["G3", "G4", "G5"]
    assert result["completed"] == submitted


def test_data_phase_stops_on_failed_terminal_receipt(tmp_path: Path) -> None:
    packet_path, packet, identity = _packet(tmp_path)
    _write_gpu_receipts(tmp_path, identity)
    submitted: list[str] = []

    def fail_g4_receipt(job: dict[str, object]) -> int:
        job_id = str(job["id"])
        submitted.append(job_id)
        status = "failed" if job_id == "G4" else "completed"
        (tmp_path / runner.RECEIPT_FILENAMES[job_id]).write_text(
            json.dumps(_data_receipt(job, status=status)),
            encoding="utf-8",
        )
        return 0

    with pytest.raises(runner.PacketError, match="G4 terminal receipt"):
        runner.execute_phase(
            packet_path,
            packet,
            phase="data",
            authorize_data=True,
            queue_job=fail_g4_receipt,
        )
    assert submitted == ["G3", "G4"]


def test_data_phase_rejects_stale_completed_receipt(tmp_path: Path) -> None:
    packet_path, packet, identity = _packet(tmp_path)
    _write_gpu_receipts(tmp_path, identity)
    g3 = packet["jobs"][2]
    (tmp_path / runner.RECEIPT_FILENAMES["G3"]).write_text(
        json.dumps(_data_receipt(g3)),
        encoding="utf-8",
    )
    submitted: list[str] = []

    with pytest.raises(runner.PacketError, match="G3 terminal receipt was not refreshed"):
        runner.execute_phase(
            packet_path,
            packet,
            phase="data",
            authorize_data=True,
            queue_job=lambda job: submitted.append(str(job["id"])) or 0,
        )
    assert submitted == ["G3"]


def test_dry_run_never_queues_or_requires_authority(tmp_path: Path) -> None:
    packet_path, packet, _ = _packet(tmp_path)
    result = runner.execute_phase(
        packet_path,
        packet,
        phase="data",
        dry_run=True,
        queue_job=lambda job: (_ for _ in ()).throw(AssertionError("queued")),
    )
    assert result["dry_run"] is True
    assert len(result["jobs"]) == 3


def test_finalize_runs_g6_directly_without_queue(tmp_path: Path) -> None:
    packet_path, packet, identity = _packet(tmp_path)
    _write_gpu_receipts(tmp_path, identity)
    direct: list[str] = []
    result = runner.execute_phase(
        packet_path,
        packet,
        phase="finalize",
        queue_job=lambda job: (_ for _ in ()).throw(AssertionError("queued")),
        run_direct=lambda job: direct.append(str(job["id"])) or 0,
    )
    assert direct == ["G6"]
    assert result["completed"] == ["G6"]


def test_queue_runner_passes_packet_argv_without_shell(monkeypatch) -> None:
    job = _job("G1")
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def run(command, **kwargs):
        calls.append((tuple(command), kwargs))
        if command[2] == "enqueue":
            return subprocess.CompletedProcess(command, 0, stdout="19\n", stderr="")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, "run", run)
    assert runner._queue_job(job) == 0
    assert calls[0][0][-len(job["argv"]) :] == tuple(job["argv"])
    assert calls[0][0][calls[0][0].index("--") + 1 :] == tuple(job["argv"])
    assert calls[0][1]["shell"] is False
    assert calls[1][0] == ("oms", "tsp-queue", "wait", "19")


def test_cli_refuses_execution_inside_pytest_before_loading_packet(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "guard")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_completion_packet.py",
            "gpu",
            str(tmp_path / "missing.json"),
            "--expected-packet-sha256",
            "0" * 64,
            "--authorize-gpu",
        ],
    )
    assert runner.main() == 2
