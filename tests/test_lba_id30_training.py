from __future__ import annotations

from pathlib import Path
import runpy

import pytest
import torch

from equivariant_attention.benchmarking import GraphSample


def _symbols() -> dict[str, object]:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "train_lba_id30.py"
    )
    return runpy.run_path(script)


def _analysis_symbols() -> dict[str, object]:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "analyze_lba_id30.py"
    )
    return runpy.run_path(script)


def _sample(split: str, index: int, target: float) -> GraphSample:
    return GraphSample(
        node_feats=torch.zeros(2, 140),
        pos=torch.zeros(2, 3),
        target=torch.tensor([target]),
        sample_id=f"atom3d-lba:{split}:{index:07d}:digest{index}",
        readout_mask=torch.tensor([False, True]),
    )


def test_plan_uses_official_validation_and_has_no_test_switch() -> None:
    symbols = _symbols()
    args = symbols["parse_args"](["artifacts/lba", "--dry-run"])
    plan = symbols["_plan"](args)

    assert plan["split"] == "official_ID30_train_validation"
    assert plan["primary_metric"] == "best validation RMSE in pK"
    assert plan["arms"] == ["candidate", "incumbent", "egnn"]
    assert plan["test_evaluated"] is False
    assert not hasattr(args, "evaluate_test")


def test_plan_records_explicit_model_and_order_seeds() -> None:
    symbols = _symbols()
    args = symbols["parse_args"](
        [
            "artifacts/lba",
            "--model-seed",
            "41",
            "--order-seed",
            "1041",
            "--dry-run",
        ]
    )

    plan = symbols["_plan"](args)

    assert plan["model_seed"] == 41
    assert plan["order_seed"] == 1041


def test_explicit_model_seed_controls_initialization() -> None:
    symbols = _symbols()
    build_model = symbols["_build_model"]

    first = build_model("candidate", None, model_seed=41)
    repeated = build_model("candidate", None, model_seed=41)
    different = build_model("candidate", None, model_seed=42)

    assert all(
        torch.equal(first.state_dict()[key], repeated.state_dict()[key])
        for key in first.state_dict()
    )
    assert any(
        not torch.equal(first.state_dict()[key], different.state_dict()[key])
        for key in first.state_dict()
    )


def test_learning_rate_warms_up_and_cosine_decays() -> None:
    learning_rate = _symbols()["_learning_rate"]

    values = [
        learning_rate(
            step,
            total_steps=10,
            warmup_steps=2,
            base_lr=1e-3,
            min_lr_ratio=0.1,
        )
        for step in range(10)
    ]

    assert values[0] == pytest.approx(5e-4)
    assert values[1] == pytest.approx(1e-3)
    assert values[2] == pytest.approx(1e-3)
    assert values[-1] == pytest.approx(1e-4)
    assert all(left >= right for left, right in zip(values[1:], values[2:]))


def test_regression_metrics_include_tie_aware_spearman() -> None:
    metrics = _symbols()["_regression_metrics"](
        torch.tensor([1.0, 2.0, 2.0, 4.0]),
        torch.tensor([1.0, 3.0, 3.0, 5.0]),
    )

    assert metrics["mae_pK"] == pytest.approx(0.75)
    assert metrics["spearman"] == pytest.approx(1.0)
    assert metrics["count"] == 4


def test_split_validator_rejects_test_identity() -> None:
    validate = _symbols()["_validate_splits"]
    train = [_sample("train", 0, 5.0)]
    validation = [_sample("val", 0, 6.0)]

    validate(train, validation)
    with pytest.raises(ValueError, match="validation sample"):
        validate(train, [_sample("test", 0, 6.0)])


def test_comparison_uses_only_same_harness_for_registered_gate() -> None:
    comparison = _symbols()["_comparison"](
        [
            {
                "arm": "candidate",
                "best_validation": {"rmse_pK": 1.20},
            },
            {
                "arm": "incumbent",
                "best_validation": {"rmse_pK": 1.24},
            },
            {
                "arm": "egnn",
                "best_validation": {"rmse_pK": 1.18},
            },
        ]
    )

    assert comparison["registered_candidate_improvement_passed"] is True
    assert comparison["candidate_beats_egnn"] is False
    assert comparison["published_reference_is_same_harness"] is False


def test_paired_bootstrap_detects_uniformly_lower_candidate_error() -> None:
    summary = _analysis_symbols()["_paired_rmse_bootstrap"](
        torch.tensor([0.9, 2.1, 2.9, 4.1]),
        torch.tensor([0.5, 2.5, 2.5, 4.5]),
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
        replicates=1_000,
        seed=7,
    )

    assert summary["candidate_minus_baseline_rmse_pK"] < 0.0
    assert summary["ci95_high_pK"] < 0.0
    assert summary["probability_candidate_lower_rmse"] == pytest.approx(1.0)
