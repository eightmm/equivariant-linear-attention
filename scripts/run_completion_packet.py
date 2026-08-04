#!/usr/bin/env python3
"""Execute one explicitly authorized phase of the frozen completion packet."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
EXPECTED_JOB_IDS = ("G1", "G2", "G3", "G4", "G5", "G6")
PHASE_JOB_IDS = {
    "gpu": ("G1", "G2"),
    "data": ("G3", "G4", "G5"),
    "finalize": ("G6",),
}
EXPECTED_SCRIPTS = {
    "G1": "scripts/run_gpu_gate.py",
    "G2": "scripts/profile_gpu_completion.py",
    "G3": "scripts/validate_realdata.py",
    "G4": "scripts/validate_realdata.py",
    "G5": "scripts/validate_realdata.py",
    "G6": "scripts/adjudicate_completion.py",
}
RECEIPT_FILENAMES = {
    "G1": "gpu-gate-receipt.json",
    "G2": "gpu-completion-profile.json",
    "G3": "qm9-screen.json",
    "G4": "lba-overfit-screen.json",
    "G5": "lba-id30-screen.json",
}


class PacketError(RuntimeError):
    """The frozen packet or a prerequisite receipt is invalid."""


def _load_script_module(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    if spec is None or spec.loader is None:
        raise PacketError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _canonical_argv(source_sha256: str, realdata_sha256: str) -> Mapping[str, list[str]]:
    module = _load_script_module(
        "_ela_completion_packet_adjudicator",
        "adjudicate_completion.py",
    )
    value = module.expected_packet_argv(source_sha256, realdata_sha256)
    if not isinstance(value, Mapping):
        raise PacketError("canonical argv builder returned a non-object")
    return value


def _verify_source_current(path: Path) -> Mapping[str, Any]:
    module = _load_script_module("_ela_completion_packet_source", "run_gpu_gate.py")
    value = module._verify_source_manifest(path)
    if not isinstance(value, Mapping):
        raise PacketError("source verifier returned a non-object")
    return value


def _argument_after(argv: Sequence[str], flag: str) -> str:
    if argv.count(flag) != 1:
        raise PacketError(f"G6 must contain exactly one {flag}")
    index = argv.index(flag)
    if index + 1 >= len(argv):
        raise PacketError(f"G6 has no value for {flag}")
    return argv[index + 1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    except (OSError, ValueError, TypeError) as error:
        message = f"cannot load {path}: {type(error).__name__}: {error}"
        raise PacketError(message) from error
    if not isinstance(value, Mapping):
        raise PacketError(f"{path} must contain a JSON object")
    return value


def _validate_job_argv(job_id: str, argv: object) -> tuple[str, ...]:
    if (
        not isinstance(argv, list)
        or not argv
        or not all(isinstance(argument, str) and argument for argument in argv)
    ):
        raise PacketError(f"{job_id} argv must be a non-empty string list")
    expected_prefix = ("uv", "run", "--locked")
    command = tuple(argv)
    if command[:3] != expected_prefix or "python" not in command:
        raise PacketError(f"{job_id} argv has an unexpected launcher")
    script = EXPECTED_SCRIPTS[job_id]
    if script not in command or command.count(script) != 1:
        raise PacketError(f"{job_id} argv has an unexpected script")
    return command


def load_frozen_packet(path: Path, *, expected_sha256: str) -> Mapping[str, Any]:
    path = path.resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as error:
        raise PacketError("packet must stay inside the repository") from error
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise PacketError("expected packet SHA-256 must be 64 lowercase hex characters")
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise PacketError(
            f"packet SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    packet = _load_object(path)
    jobs = packet.get("jobs")
    if (
        packet.get("schema_version") != 1
        or packet.get("source_manifest") != "source-manifest-pre-gpu.json"
        or not isinstance(jobs, list)
        or len(jobs) != len(EXPECTED_JOB_IDS)
        or not all(isinstance(job, Mapping) for job in jobs)
        or tuple(job.get("id") for job in jobs) != EXPECTED_JOB_IDS
    ):
        raise PacketError("packet must contain the exact ordered unique G1-G6 job set")
    commands = {
        job_id: _validate_job_argv(job_id, job.get("argv"))
        for job_id, job in zip(EXPECTED_JOB_IDS, jobs, strict=True)
    }
    source_sha256 = _argument_after(
        commands["G6"],
        "--expected-source-manifest-combined-sha256",
    )
    realdata_sha256 = _argument_after(
        commands["G6"],
        "--expected-realdata-source-sha256",
    )
    canonical = _canonical_argv(source_sha256, realdata_sha256)
    for job_id in EXPECTED_JOB_IDS:
        if list(commands[job_id]) != canonical.get(job_id):
            raise PacketError(f"{job_id} argv differs from the canonical frozen command")
    verified_source = _verify_source_current(path.parent / "source-manifest-pre-gpu.json")
    if verified_source.get("combined_sha256") != source_sha256:
        raise PacketError("G6 source SHA-256 differs from the verified source manifest")
    return packet


def _source_identity(packet_path: Path, packet: Mapping[str, Any]) -> Mapping[str, Any]:
    source_name = packet.get("source_manifest")
    if not isinstance(source_name, str):
        raise PacketError("packet source manifest is missing")
    source_path = packet_path.parent / source_name
    source = _load_object(source_path)
    combined = source.get("combined_sha256")
    if not isinstance(combined, str) or len(combined) != 64:
        raise PacketError("source manifest combined SHA-256 is invalid")
    return {
        "manifest_sha256": _sha256(source_path),
        "combined_sha256": combined,
        "file_count": source.get("file_count"),
    }


def _require_gpu_prerequisites(
    packet_path: Path,
    packet: Mapping[str, Any],
    *,
    require_profiler: bool,
) -> None:
    expected = _source_identity(packet_path, packet)
    run_dir = packet_path.parent
    g1 = _load_object(run_dir / "gpu-gate-receipt.json")
    source = g1.get("source_manifest")
    if (
        g1.get("status") != "passed"
        or g1.get("exit_code") != 0
        or not isinstance(source, Mapping)
        or source.get("verified_against_current_bytes") is not True
        or any(source.get(key) != expected.get(key) for key in expected)
    ):
        raise PacketError("G1 did not produce a passing source-bound receipt")
    if not require_profiler:
        return
    g2 = _load_object(run_dir / "gpu-completion-profile.json")
    source = g2.get("source_manifest")
    if (
        g2.get("status") != "completed"
        or not isinstance(source, Mapping)
        or source.get("verified_against_current_bytes") is not True
        or any(source.get(key) != expected.get(key) for key in expected)
    ):
        raise PacketError("G2 did not produce a completed source-bound receipt")


def _receipt_path(packet_path: Path, job_id: str) -> Path:
    filename = RECEIPT_FILENAMES.get(job_id)
    if filename is None:
        raise PacketError(f"{job_id} has no declared terminal receipt")
    return packet_path.parent / filename


def _receipt_digest(packet_path: Path, job_id: str) -> str | None:
    path = _receipt_path(packet_path, job_id)
    if not path.exists():
        return None
    if not path.is_file():
        raise PacketError(f"{job_id} receipt path is not a file")
    return _sha256(path)


def _require_terminal_receipt(
    packet_path: Path,
    job: Mapping[str, Any],
    *,
    previous_digest: str | None,
    expected_realdata_sha256: str,
) -> None:
    job_id = str(job["id"])
    path = _receipt_path(packet_path, job_id)
    report = _load_object(path)
    current_digest = _sha256(path)
    if previous_digest is not None and current_digest == previous_digest:
        raise PacketError(f"{job_id} terminal receipt was not refreshed")
    expected_status = "passed" if job_id == "G1" else "completed"
    if report.get("status") != expected_status:
        raise PacketError(
            f"{job_id} terminal receipt status is not {expected_status}"
        )
    if job_id == "G1" and report.get("exit_code") != 0:
        raise PacketError("G1 terminal receipt has a nonzero exit code")
    if job_id in {"G3", "G4", "G5"}:
        argv = _validate_job_argv(job_id, job.get("argv"))
        script_index = argv.index("scripts/validate_realdata.py")
        if report.get("command") != list(argv[script_index:]):
            raise PacketError(f"{job_id} terminal receipt command mismatch")
        if report.get("source_sha256") != expected_realdata_sha256:
            raise PacketError(f"{job_id} terminal receipt source mismatch")


def _queue_job(job: Mapping[str, Any]) -> int:
    job_id = str(job["id"])
    label = str(job.get("label", job_id))
    argv = _validate_job_argv(job_id, job.get("argv"))
    enqueue = subprocess.run(
        (
            "oms",
            "tsp-queue",
            "enqueue",
            "--label",
            label,
            "--slots",
            "1",
            "--ledger-note",
            f"frozen completion packet {job_id}",
            "--",
            *argv,
        ),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    if enqueue.returncode != 0:
        raise PacketError(
            f"{job_id} enqueue failed: {enqueue.stderr.strip() or enqueue.stdout.strip()}"
        )
    queue_id = enqueue.stdout.strip()
    if not queue_id.isdigit():
        raise PacketError(f"{job_id} enqueue returned an invalid queue id")
    waited = subprocess.run(
        ("oms", "tsp-queue", "wait", queue_id),
        cwd=ROOT,
        check=False,
        shell=False,
    )
    return int(waited.returncode)


def _run_direct(job: Mapping[str, Any]) -> int:
    job_id = str(job["id"])
    argv = _validate_job_argv(job_id, job.get("argv"))
    completed = subprocess.run(argv, cwd=ROOT, check=False, shell=False)
    return int(completed.returncode)


def execute_phase(
    packet_path: Path,
    packet: Mapping[str, Any],
    *,
    phase: str,
    authorize_gpu: bool = False,
    authorize_data: bool = False,
    dry_run: bool = False,
    queue_job: Callable[[Mapping[str, Any]], int] = _queue_job,
    run_direct: Callable[[Mapping[str, Any]], int] = _run_direct,
) -> dict[str, object]:
    if phase not in PHASE_JOB_IDS:
        raise PacketError(f"unknown phase: {phase}")
    if not dry_run and phase == "gpu" and not authorize_gpu:
        raise PacketError("GPU phase requires --authorize-gpu")
    if not dry_run and phase == "data" and not authorize_data:
        raise PacketError("data phase requires --authorize-data")

    jobs = {str(job["id"]): job for job in packet["jobs"]}
    expected_realdata_sha256 = _argument_after(
        tuple(jobs["G6"]["argv"]),
        "--expected-realdata-source-sha256",
    )
    selected = PHASE_JOB_IDS[phase]
    if dry_run:
        return {
            "phase": phase,
            "dry_run": True,
            "jobs": [list(_validate_job_argv(job_id, jobs[job_id]["argv"])) for job_id in selected],
        }

    if phase == "data":
        _require_gpu_prerequisites(packet_path, packet, require_profiler=True)
    elif phase == "finalize":
        _require_gpu_prerequisites(packet_path, packet, require_profiler=True)

    completed: list[str] = []
    for job_id in selected:
        previous_digest = (
            _receipt_digest(packet_path, job_id) if job_id != "G6" else None
        )
        exit_code = (
            run_direct(jobs[job_id])
            if phase == "finalize"
            else queue_job(jobs[job_id])
        )
        if exit_code != 0:
            raise PacketError(
                f"{job_id} exited {exit_code}; later jobs were not submitted"
            )
        if job_id != "G6":
            _require_terminal_receipt(
                packet_path,
                jobs[job_id],
                previous_digest=previous_digest,
                expected_realdata_sha256=expected_realdata_sha256,
            )
        completed.append(job_id)
        if job_id == "G1":
            _require_gpu_prerequisites(packet_path, packet, require_profiler=False)
        elif job_id == "G2":
            _require_gpu_prerequisites(packet_path, packet, require_profiler=True)
    return {"phase": phase, "dry_run": False, "completed": completed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=tuple(PHASE_JOB_IDS))
    parser.add_argument("packet", type=Path)
    parser.add_argument("--expected-packet-sha256", required=True)
    parser.add_argument("--authorize-gpu", action="store_true")
    parser.add_argument("--authorize-data", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        if not args.dry_run and os.environ.get("PYTEST_CURRENT_TEST"):
            raise PacketError("execution is forbidden inside a pytest process")
        packet_path = args.packet.resolve()
        packet = load_frozen_packet(
            packet_path,
            expected_sha256=args.expected_packet_sha256,
        )
        result = execute_phase(
            packet_path,
            packet,
            phase=args.phase,
            authorize_gpu=args.authorize_gpu,
            authorize_data=args.authorize_data,
            dry_run=args.dry_run,
        )
    except Exception as error:
        print(json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}"}))
        return 2
    print(json.dumps({"status": "completed", **result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
