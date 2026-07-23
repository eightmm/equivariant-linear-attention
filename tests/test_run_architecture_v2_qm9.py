from __future__ import annotations

import json
import runpy
from pathlib import Path


def _symbols() -> dict[str, object]:
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "run_architecture_v2_qm9.py"
    )
    return runpy.run_path(script)


def _record(seed: int, val_mae: float) -> dict[str, object]:
    return {
        "model_seed": seed,
        "val_mae": val_mae,
        "elapsed_seconds": 1.0,
        "peak_cuda_memory_bytes": 100,
    }


def test_frozen_arms_and_commands_encode_only_registered_variants() -> None:
    symbols = _symbols()
    screen = symbols["screen_arms"](steps=500)
    confirmation = symbols["confirmation_arms"](
        "combined",
        seeds=range(41, 46),
        steps=2_000,
    )
    settings = symbols["ExecutionSettings"]()

    assert [arm["variant"] for arm in screen] == [
        "incumbent",
        "bounded",
        "tensor",
        "combined",
    ]
    assert {arm["model_seed"] for arm in screen} == {42}
    assert {arm["steps"] for arm in screen} == {500}
    assert len(confirmation) == 15
    assert {arm["role"] for arm in confirmation} == {
        "incumbent",
        "selected",
        "egnn",
    }
    assert {arm["model_seed"] for arm in confirmation} == set(range(41, 46))
    assert {arm["steps"] for arm in confirmation} == {2_000}

    expected = {
        "incumbent": ("unit", "0", False),
        "bounded": ("bounded", "0", False),
        "tensor": ("unit", "4", True),
        "combined": ("bounded", "4", True),
    }
    for arm in screen:
        command = symbols["build_train_command"](
            arm,
            Path("artifacts/metrics.json"),
            settings,
        )
        assert command[:5] == [
            "uv",
            "run",
            "--locked",
            "python",
            "scripts/train_compare.py",
        ]
        assert _argument(command, "--dataset") == "qm9"
        assert _argument(command, "--device") == "cuda"
        assert _argument(command, "--determinism") == "strict"
        assert _argument(command, "--amp-dtype") == "none"
        assert "--skip-test-eval" in command
        assert "--evaluate-test" not in command
        content, tensor_dim, tensor_kernel = expected[str(arm["variant"])]
        assert _argument(command, "--scalar-content-mode") == content
        assert _argument(command, "--hidden-tensor-dim") == tensor_dim
        assert ("--tensor-product-kernel" in command) is tensor_kernel

    egnn_command = symbols["build_train_command"](
        confirmation[2],
        Path("artifacts/egnn.json"),
        settings,
    )
    assert _argument(egnn_command, "--benchmark-model") == (
        "internal_static_egnn_baseline"
    )
    assert _argument(egnn_command, "--hidden-dim") == "91"
    assert "--tensor-product-kernel" not in egnn_command


def test_screen_admits_only_improving_guarded_candidate_and_selects_lowest() -> None:
    symbols = _symbols()
    arms = symbols["screen_arms"](steps=500)
    records = [
        _record(42, 0.600),
        _record(42, 0.595),
        _record(42, 0.585),
        _record(42, 0.579),
    ]

    decision = symbols["screen_decision"](arms, records)

    assert decision["confirmation_admitted"] is True
    assert decision["selected_variant"] == "combined"
    assert decision["candidates"]["bounded"]["eligible"] is False
    assert decision["candidates"]["tensor"]["eligible"] is True
    assert decision["candidates"]["combined"]["eligible"] is True

    records[2] = _record(42, 0.621)
    records[3] = _record(42, 0.591)
    rejected = symbols["screen_decision"](arms, records)
    assert rejected["confirmation_admitted"] is False
    assert rejected["selected_variant"] is None
    assert (
        rejected["candidates"]["tensor"]["criteria"][
            "candidate_minus_incumbent_at_most_0.020_eV"
        ]
        is False
    )


def test_confirmation_uses_frozen_c5_and_separate_egnn_rules() -> None:
    symbols = _symbols()
    seeds = tuple(range(41, 46))
    incumbent = [_record(seed, 0.600) for seed in seeds]
    candidate = [
        _record(41, 0.570),
        _record(42, 0.580),
        _record(43, 0.570),
        _record(44, 0.610),
        _record(45, 0.560),
    ]
    egnn = [
        _record(41, 0.580),
        _record(42, 0.590),
        _record(43, 0.560),
        _record(44, 0.620),
        _record(45, 0.570),
    ]

    c5 = symbols["confirmation_decision"](candidate, incumbent, seeds=seeds)
    competitiveness = symbols["egnn_competitiveness_decision"](
        candidate,
        egnn,
        seeds=seeds,
    )

    assert c5["passed"] is True
    assert c5["mean_improvement_eV"] >= 0.020
    assert c5["improving_pair_count"] == 4
    assert c5["worst_pair_improvement_eV"] >= -0.020
    assert competitiveness["passed"] is True
    assert (
        competitiveness["candidate_mean_val_mae_eV"]
        < (competitiveness["egnn_mean_val_mae_eV"])
    )
    assert competitiveness["candidate_paired_win_count"] >= 3

    candidate[3] = _record(44, 0.621)
    assert (
        symbols["confirmation_decision"](candidate, incumbent, seeds=seeds)["passed"]
        is False
    )


def test_atomic_json_write_replaces_complete_document(tmp_path: Path) -> None:
    symbols = _symbols()
    destination = tmp_path / "summary.json"
    destination.write_text('{"old": true}\n')

    symbols["atomic_write_json"](destination, {"status": "complete", "value": 1})

    assert json.loads(destination.read_text()) == {
        "status": "complete",
        "value": 1,
    }
    assert not list(tmp_path.glob(".*.tmp"))


def test_artifact_reference_accepts_output_outside_repository(
    tmp_path: Path,
) -> None:
    symbols = _symbols()
    repo_root = Path(__file__).resolve().parents[1]
    outside = tmp_path / "metrics.json"

    assert symbols["_artifact_reference"](outside, repo_root) == str(outside)
    assert (
        symbols["_artifact_reference"](
            repo_root / "artifacts" / "metrics.json",
            repo_root,
        )
        == "artifacts/metrics.json"
    )


def test_dry_run_marks_overrides_nonconfirmatory(capsys, tmp_path: Path) -> None:
    symbols = _symbols()

    result = symbols["main"](
        [
            str(tmp_path / "smoke"),
            "--dry-run",
            "--dataset",
            "synthetic",
            "--device",
            "cpu",
            "--screen-steps",
            "1",
            "--confirm-steps",
            "1",
            "--seeds",
            "41",
        ]
    )

    plan = json.loads(capsys.readouterr().out)
    assert result == 0
    assert plan["protocol"]["claims_admissible"] is False
    assert plan["protocol"]["mode"] == "smoke_or_exploratory"
    assert len(plan["screen_runs"]) == 4
    assert len(plan["confirmation_runs_if_admitted"]) == 3
    assert plan["test_evaluated"] is False
    assert plan["c6_large_complex_capacity"]["status"] == "not_evaluated"


def _argument(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]
