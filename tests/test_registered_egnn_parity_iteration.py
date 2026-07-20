from __future__ import annotations

import json
import runpy
from pathlib import Path


def _symbols() -> dict[str, object]:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_registered_egnn_parity_iteration.py"
    )
    return runpy.run_path(script)


def _record(
    seed: int,
    val_mae: float,
    *,
    parameter_count: int = 153_000,
    elapsed_seconds: float = 10.0,
    peak_cuda_memory_bytes: int = 100,
) -> dict[str, object]:
    return {
        "model_seed": seed,
        "val_mae": val_mae,
        "parameter_count": parameter_count,
        "elapsed_seconds": elapsed_seconds,
        "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
    }


def test_iteration_builds_only_frozen_screen_and_confirmation_arms() -> None:
    symbols = _symbols()

    for candidate, expected_switch in (
        ("radial", "--learn-local-radial-gate"),
        ("pairwise", "--pairwise-local-content"),
        ("pairwise_zero_init", "--pairwise-local-content"),
    ):
        screen = symbols["registered_screen_arms"](candidate)
        confirmation = symbols["registered_confirmation_arms"](candidate)

        assert [arm["role"] for arm in screen] == ["attention_baseline", "candidate"]
        assert {arm["model_seed"] for arm in screen} == {42}
        assert {arm["steps"] for arm in screen} == {500}
        assert len(confirmation) == 10
        assert {arm["model_seed"] for arm in confirmation} == set(range(41, 46))
        assert {arm["steps"] for arm in confirmation} == {2_000}
        assert [arm["role"] for arm in confirmation[::2]] == ["candidate"] * 5
        assert [arm["role"] for arm in confirmation[1::2]] == ["egnn"] * 5

        for arm in [*screen, *confirmation]:
            command = symbols["build_train_command"](
                arm, Path("artifacts/study") / f"{arm['name']}.json"
            )
            assert command[:5] == [
                "uv",
                "run",
                "--locked",
                "python",
                "scripts/train_compare.py",
            ]
            assert "--dataset" in command and "qm9" in command
            assert "--split-seed" in command and "42" in command
            assert "--device" in command and "cuda" in command
            assert "--skip-test-eval" in command
            assert "--evaluate-test" not in command
            if arm["role"] == "candidate":
                assert expected_switch in command
                if candidate == "pairwise_zero_init":
                    assert "--pairwise-residual-scale-init" in command
                    assert "0.0" in command
            else:
                assert expected_switch not in command


def test_screen_requires_finite_nonregression_and_parameter_bound() -> None:
    symbols = _symbols()
    baseline = _record(42, 0.52, parameter_count=153_000)
    candidate = _record(42, 0.51, parameter_count=154_000)

    decision = symbols["screen_decision"](candidate, baseline)

    assert decision["confirmation_admitted"] is True
    assert decision["candidate_minus_baseline_eV"] == (
        candidate["val_mae"] - baseline["val_mae"]
    )
    assert decision["parameter_ratio"] < 1.05

    candidate["val_mae"] = 0.541
    assert not symbols["screen_decision"](candidate, baseline)[
        "confirmation_admitted"
    ]
    candidate["val_mae"] = 0.51
    candidate["parameter_count"] = 161_000
    assert not symbols["screen_decision"](candidate, baseline)[
        "confirmation_admitted"
    ]


def test_promotion_rule_is_absolute_and_paired_against_rerun_egnn() -> None:
    symbols = _symbols()
    egnn = [_record(seed, 0.409) for seed in range(41, 46)]
    candidate = [
        _record(41, 0.390),
        _record(42, 0.395),
        _record(43, 0.397),
        _record(44, 0.410),
        _record(45, 0.400),
    ]

    decision = symbols["promotion_decision"](candidate, egnn)

    assert decision["passed"] is True
    assert decision["candidate_mean_val_mae_eV"] <= 0.398932
    assert decision["improving_seed_count"] >= 3
    assert decision["worst_seed_improvement_eV"] >= -0.020

    candidate[0]["val_mae"] = 0.405
    assert not symbols["promotion_decision"](candidate, egnn)["passed"]


def test_iteration_limit_and_budget_are_frozen() -> None:
    symbols = _symbols()

    assert symbols["MAX_PACKET_GPU_SECONDS"] == 3_600
    assert symbols["MAX_ITERATIONS"] == 3
    assert symbols["PROMOTION_MAE_EV"] == 0.398932
    assert symbols["parse_args"](["out", "--candidate", "pairwise", "--iteration", "2"])
    assert symbols["parse_args"](
        ["out", "--candidate", "pairwise_zero_init", "--iteration", "3"]
    )


def test_completed_packet_is_ledgered_without_promotion_or_test_access() -> None:
    root = Path(__file__).resolve().parents[1]
    records = [
        json.loads(line)
        for line in (root / "docs" / "EXPERIMENTS.jsonl").read_text().splitlines()
    ]
    packet = [
        record
        for record in records
        if record.get("metrics", {}).get("packet_id") == "egnn-parity-20260720"
    ]

    assert [record["metrics"]["candidate"] for record in packet[-3:]] == [
        "radial",
        "pairwise",
        "pairwise_zero_init",
    ]
    assert all(record["exit"] == 0 for record in packet[-3:])
    assert all(record["metrics"]["test_evaluated"] is False for record in packet[-3:])
    assert all(record["metrics"]["promoted"] is False for record in packet[-3:])
    assert packet[-1]["metrics"]["packet_gpu_wall_seconds"] == 850.7427735850015
