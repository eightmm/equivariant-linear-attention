from __future__ import annotations

import os
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from equivariant_attention.reproducibility import (
    configure_reproducibility,
    summarize_repeated_runs,
)


@pytest.fixture(autouse=True)
def _restore_torch_reproducibility_state() -> None:
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    yield
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark
    if workspace is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace


def test_strict_reproducibility_enables_and_records_every_runtime_control() -> None:
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)

    state = configure_reproducibility(seed=43, mode="strict")

    assert state == {
        "seed": 43,
        "mode": "strict",
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cublas_workspace_config": ":4096:8",
    }
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark


def test_seeded_reproducibility_preserves_legacy_nondeterministic_lane() -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True

    state = configure_reproducibility(seed=41, mode="seeded")

    assert state["mode"] == "seeded"
    assert state["deterministic_algorithms"] is False
    assert state["cudnn_deterministic"] is False
    assert not torch.are_deterministic_algorithms_enabled()


def _repeat_run(
    metric: float,
    *,
    final_hash: str,
    source_hash: str | None = None,
    mode: str = "seeded",
) -> dict[str, object]:
    source_hash = _sha("a") if source_hash is None else source_hash
    strict = mode == "strict"
    return {
        "dataset": "qm9",
        "source_sha256": source_hash,
        "initial_state_sha256": _sha("b"),
        "model_seed": 42,
        "split_hashes": {
            "train": _sha("c"),
            "validation": _sha("d"),
            "test": _sha("e"),
        },
        "data_identity": {"processed/data_v3.pt": _sha("f")},
        "run_config": {
            "dataset": "qm9",
            "model_seed": 42,
            "split_seed": 42,
            "determinism": mode,
            "device": "cuda",
        },
        "val_mae": metric,
        "final_state_sha256": final_hash,
        "reproducibility": {
            "seed": 42,
            "mode": mode,
            "deterministic_algorithms": strict,
            "deterministic_warn_only": False,
            "cudnn_deterministic": strict,
            "cudnn_benchmark": False,
            "cublas_workspace_config": ":4096:8" if strict else None,
        },
    }


def _sha(character: str) -> str:
    return character * 64


def test_repeat_gate_quantifies_noise_without_claiming_bitwise_identity() -> None:
    runs = [
        _repeat_run(value, final_hash=_sha(str(index + 1)))
        for index, value in enumerate([0.770, 0.772, 0.769, 0.771, 0.773])
    ]

    summary = summarize_repeated_runs(
        runs,
        metric_path="val_mae",
        max_metric_span=0.005,
        min_runs=5,
    )

    assert summary["run_count"] == 5
    assert summary["metric"]["span"] == pytest.approx(0.004)
    assert summary["metric"]["sample_std"] > 0.0
    assert summary["unique_final_state_count"] == 5
    assert summary["bitwise_reproducible"] is False
    assert summary["gate_mode"] == "metric_span"
    assert summary["gate_passed"] is True


def test_strict_repeat_gate_automatically_requires_identical_final_state() -> None:
    runs = [
        _repeat_run(0.77, final_hash=_sha("1"), mode="strict"),
        _repeat_run(0.77, final_hash=_sha("2"), mode="strict"),
        _repeat_run(0.77, final_hash=_sha("1"), mode="strict"),
    ]

    summary = summarize_repeated_runs(
        runs,
        metric_path="val_mae",
        max_metric_span=0.005,
        min_runs=3,
    )

    assert summary["metric"]["span"] == 0.0
    assert summary["bitwise_reproducible"] is False
    assert summary["gate_mode"] == "bitwise"
    assert summary["gate_passed"] is False


def test_repeat_gate_rejects_nonidentical_run_identity() -> None:
    runs = [
        _repeat_run(0.77, final_hash=_sha("1")),
        _repeat_run(0.77, final_hash=_sha("1"), source_hash=_sha("9")),
    ]

    with pytest.raises(ValueError, match="source_sha256"):
        summarize_repeated_runs(
            runs,
            metric_path="val_mae",
            max_metric_span=0.005,
            min_runs=2,
        )


@pytest.mark.parametrize("missing_field", ["source_sha256", "reproducibility"])
def test_repeat_gate_rejects_missing_required_identity(missing_field: str) -> None:
    first = _repeat_run(0.77, final_hash=_sha("1"))
    second = _repeat_run(0.77, final_hash=_sha("1"))
    first.pop(missing_field)
    second.pop(missing_field)

    with pytest.raises(ValueError, match=missing_field):
        summarize_repeated_runs(
            [first, second],
            metric_path="val_mae",
            max_metric_span=0.005,
            min_runs=2,
        )


def test_qm9_repeat_gate_requires_data_identity() -> None:
    runs = [
        _repeat_run(0.77, final_hash=_sha("1")),
        _repeat_run(0.77, final_hash=_sha("1")),
    ]
    for run in runs:
        run.pop("data_identity")

    with pytest.raises(ValueError, match="data_identity"):
        summarize_repeated_runs(
            runs,
            metric_path="val_mae",
            max_metric_span=0.005,
            min_runs=2,
        )


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("source_sha256", None),
        ("source_sha256", ""),
        ("source_sha256", "not-a-sha"),
        ("initial_state_sha256", {}),
        ("split_hashes", {}),
        ("run_config", {}),
        ("reproducibility", {}),
    ],
)
def test_repeat_gate_rejects_malformed_identity_values(
    field: str,
    invalid: object,
) -> None:
    runs = [
        _repeat_run(0.77, final_hash=_sha("1")),
        _repeat_run(0.77, final_hash=_sha("1")),
    ]
    for run in runs:
        run[field] = invalid

    with pytest.raises((TypeError, ValueError), match=field):
        summarize_repeated_runs(
            runs,
            metric_path="val_mae",
            max_metric_span=0.005,
            min_runs=2,
        )


def test_repeat_gate_rejects_expected_mode_mismatch() -> None:
    runs = [
        _repeat_run(0.77, final_hash=_sha("1")),
        _repeat_run(0.77, final_hash=_sha("1")),
    ]

    with pytest.raises(ValueError, match="expected_mode"):
        summarize_repeated_runs(
            runs,
            metric_path="val_mae",
            max_metric_span=0.005,
            min_runs=2,
            expected_mode="strict",
        )


def test_repeat_gate_cli_writes_machine_readable_verdict(tmp_path: Path) -> None:
    inputs = []
    for index, value in enumerate([0.770, 0.772, 0.769, 0.771, 0.773]):
        path = tmp_path / f"run-{index}.json"
        path.write_text(json.dumps(_repeat_run(value, final_hash=_sha(str(index + 1)))))
        inputs.append(path)
    output = tmp_path / "summary.json"
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "summarize_reproducibility_runs.py"
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--metric-path",
            "val_mae",
            "--max-metric-span",
            "0.005",
            "--min-runs",
            "5",
            "--output",
            str(output),
            *map(str, inputs),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(output.read_text())
    assert summary["gate_passed"] is True
    assert summary["input_paths"] == [str(path) for path in inputs]


def test_repeat_gate_cli_writes_failed_verdict_and_exits_one(tmp_path: Path) -> None:
    inputs = []
    for index, value in enumerate([0.770, 0.780]):
        path = tmp_path / f"run-{index}.json"
        path.write_text(json.dumps(_repeat_run(value, final_hash=_sha(str(index + 1)))))
        inputs.append(path)
    output = tmp_path / "summary.json"
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "summarize_reproducibility_runs.py"
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--max-metric-span",
            "0.005",
            "--min-runs",
            "2",
            "--output",
            str(output),
            *map(str, inputs),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert json.loads(output.read_text())["gate_passed"] is False


def test_repeat_gate_cli_identity_error_writes_no_verdict(tmp_path: Path) -> None:
    inputs = []
    for index in range(2):
        run = _repeat_run(0.77, final_hash=_sha("1"))
        run["source_sha256"] = None
        path = tmp_path / f"run-{index}.json"
        path.write_text(json.dumps(run))
        inputs.append(path)
    output = tmp_path / "summary.json"
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "summarize_reproducibility_runs.py"
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--max-metric-span",
            "0.005",
            "--min-runs",
            "2",
            "--output",
            str(output),
            *map(str, inputs),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert not output.exists()


def test_repeat_gate_cli_strict_hash_mismatch_exits_one(tmp_path: Path) -> None:
    inputs = []
    for index in range(2):
        path = tmp_path / f"run-{index}.json"
        path.write_text(
            json.dumps(
                _repeat_run(
                    0.77,
                    final_hash=_sha(str(index + 1)),
                    mode="strict",
                )
            )
        )
        inputs.append(path)
    output = tmp_path / "summary.json"
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "summarize_reproducibility_runs.py"
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--expected-mode",
            "strict",
            "--max-metric-span",
            "0.005",
            "--min-runs",
            "2",
            "--output",
            str(output),
            *map(str, inputs),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    summary = json.loads(output.read_text())
    assert summary["recorded_determinism_mode"] == "strict"
    assert summary["gate_mode"] == "bitwise"
    assert summary["gate_passed"] is False


def test_repeat_gate_cli_nonfinite_metric_writes_no_verdict(tmp_path: Path) -> None:
    inputs = []
    for index in range(2):
        run = _repeat_run(0.77, final_hash=_sha("1"))
        run["val_mae"] = float("nan")
        path = tmp_path / f"run-{index}.json"
        path.write_text(json.dumps(run))
        inputs.append(path)
    output = tmp_path / "summary.json"
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "summarize_reproducibility_runs.py"
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--max-metric-span",
            "0.005",
            "--min-runs",
            "2",
            "--output",
            str(output),
            *map(str, inputs),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert not output.exists()
