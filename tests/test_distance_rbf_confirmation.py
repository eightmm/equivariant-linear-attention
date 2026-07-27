"""Five-seed distance-RBF confirmation contracts."""

from __future__ import annotations

import json
from pathlib import Path
import runpy

import pytest


def _symbols() -> dict[str, object]:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_distance_rbf_confirmation.py"
    )
    return runpy.run_path(script)


def _record(
    seed: int,
    val_mae: float,
    *,
    latency: float = 0.01,
    memory: int = 1_000,
) -> dict[str, object]:
    return {
        "model_seed": seed,
        "val_mae": val_mae,
        "step_latency_median_seconds": latency,
        "peak_cuda_memory_bytes": memory,
    }


def test_commands_hold_features_split_budget_and_topology_fixed() -> None:
    symbols = _symbols()
    build_command = symbols["build_command"]
    for arm in symbols["ARMS"]:
        command = build_command(
            arm,
            seed=43,
            output=Path("metrics.json"),
            steps=2_000,
            device="cuda",
        )
        assert command[command.index("--model-seed") + 1] == "43"
        assert command[command.index("--split-seed") + 1] == "42"
        assert command[command.index("--steps") + 1] == "2000"
        assert "--evaluate-test" not in command
        assert "--skip-test-eval" in command
        if arm in {"incumbent", "distance_rbf"}:
            expected = "distance" if arm == "distance_rbf" else "squared"
            assert command[command.index("--local-rbf-spacing") + 1] == expected
            assert command[command.index("--local-cutoff") + 1] == "2.5"
            assert "--precompute-local-edges" in command
        elif arm == "egnn_matched":
            assert "--local-rbf-spacing" not in command
            assert command[command.index("--local-cutoff") + 1] == "2.5"
            assert "--precompute-local-edges" in command
        else:
            assert "--local-cutoff" not in command
            assert "--precompute-local-edges" not in command


def test_primary_decision_requires_every_registered_gate() -> None:
    symbols = _symbols()
    seeds = symbols["MODEL_SEEDS"]
    baseline = [_record(seed, 0.60) for seed in seeds]
    candidate = [
        _record(seed, value)
        for seed, value in zip(
            seeds,
            (0.58, 0.59, 0.59, 0.60, 0.61),
            strict=True,
        )
    ]

    decision = symbols["paired_promotion_decision"](candidate, baseline)

    assert decision["passed"] is False
    assert decision["mean_improvement_eV"] == pytest.approx(0.006)
    assert decision["criteria"]["mean_improvement_at_least_0.010_eV"] is False


def test_primary_decision_passes_a_material_consistent_gain() -> None:
    symbols = _symbols()
    seeds = symbols["MODEL_SEEDS"]
    baseline = [_record(seed, 0.62) for seed in seeds]
    candidate = [
        _record(seed, value)
        for seed, value in zip(
            seeds,
            (0.60, 0.60, 0.61, 0.61, 0.62),
            strict=True,
        )
    ]

    decision = symbols["paired_promotion_decision"](candidate, baseline)

    assert decision["passed"] is True
    assert decision["improving_seed_count"] == 4
    assert decision["worst_seed_improvement_eV"] == pytest.approx(0.0)


def test_dry_run_is_side_effect_free_and_keeps_test_closed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    symbols = _symbols()
    output = tmp_path / "confirmation"

    assert symbols["main"]([str(output), "--dry-run"]) == 0

    plan = json.loads(capsys.readouterr().out)
    assert plan["model_seeds"] == [41, 42, 43, 44, 45]
    assert plan["steps"] == 2_000
    assert plan["test_evaluated"] is False
    assert len(plan["commands"]) == 20
    assert not output.exists()
