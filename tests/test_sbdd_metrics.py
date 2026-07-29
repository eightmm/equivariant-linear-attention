from __future__ import annotations

import pytest
import torch

from equivariant_attention.sbdd import (
    affinity_metrics,
    affinity_metrics_by_slice,
    pose_metrics,
    screening_metrics,
)


def test_affinity_metrics_include_error_correlation_ranking_uncertainty_calibration() -> (
    None
):
    target = torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float64)
    prediction = torch.tensor([5.0, 6.5, 6.5, 8.0], dtype=torch.float64)
    metrics = affinity_metrics(
        prediction,
        target,
        target_ids=("A", "A", "B", "B"),
        interval_lower=prediction - 0.75,
        interval_upper=prediction + 0.75,
    )

    assert metrics.mae == pytest.approx(0.25)
    assert metrics.rmse == pytest.approx(2**-1.5)
    assert metrics.pearson is not None
    assert metrics.spearman is not None
    assert metrics.within_target_spearman == pytest.approx(1.0)
    assert metrics.within_target_group_count == 2
    assert metrics.mae_standard_error > 0
    assert metrics.mae_ci95[0] <= metrics.mae <= metrics.mae_ci95[1]
    assert metrics.interval_coverage == pytest.approx(1.0)
    assert metrics.interval_mean_width == pytest.approx(1.5)


def test_affinity_slice_metrics_preserve_cold_axis_labels() -> None:
    slices = affinity_metrics_by_slice(
        torch.tensor([1.0, 2.0, 4.0, 5.0]),
        torch.tensor([1.5, 2.0, 3.0, 5.5]),
        slice_values=("warm", "cold-drug", "cold-drug", "warm"),
        slice_name="generalization_axis",
    )

    assert [(item.slice_name, item.slice_value) for item in slices] == [
        ("generalization_axis", "cold-drug"),
        ("generalization_axis", "warm"),
    ]
    assert all(item.metrics.count == 2 for item in slices)


def test_pose_metrics_report_topk_ranking_and_physical_auxiliaries() -> None:
    metrics = pose_metrics(
        scores=torch.tensor([3.0, 1.0, 2.0, 1.5, 0.0]),
        rmsd=torch.tensor([1.0, 4.0, 3.0, 1.5, 5.0]),
        group_ids=("a", "a", "a", "b", "b"),
        top_ks=(1, 2),
        success_rmsd=2.0,
        clash=torch.tensor([0.1, 1.0, 0.3, 0.2, 1.2]),
        strain=torch.tensor([0.2, 0.8, 0.4, 0.1, 1.0]),
        contact=torch.tensor([0.9, 0.2, 0.5, 0.8, 0.1]),
    )

    assert dict(metrics.topk_success) == {1: 1.0, 2: 1.0}
    assert metrics.within_pair_spearman == pytest.approx(1.0)
    assert metrics.top1_mean_rmsd == pytest.approx(1.25)
    assert metrics.top1_mean_clash == pytest.approx(0.15)
    assert metrics.top1_mean_strain == pytest.approx(0.15)
    assert metrics.top1_mean_contact == pytest.approx(0.85)


def test_pose_topk_success_is_fractional_at_a_score_tie_boundary() -> None:
    metrics = pose_metrics(
        scores=torch.zeros(3),
        rmsd=torch.tensor([1.0, 5.0, 6.0]),
        group_ids=("a", "a", "a"),
        top_ks=(1, 2),
        success_rmsd=2.0,
    )

    assert dict(metrics.topk_success) == pytest.approx({1: 1 / 3, 2: 2 / 3})
    assert metrics.top1_mean_rmsd == pytest.approx(4.0)


def test_screening_metrics_include_enrichment_bedroc_and_property_shortcut_audit() -> (
    None
):
    active = torch.tensor([1, 0, 1, 0, 0, 0], dtype=torch.bool)
    scores = torch.tensor([6.0, 2.0, 5.0, 1.0, 0.0, -1.0])
    property_scores = torch.tensor([0.6, 0.7, 0.5, 0.4, 0.3, 0.2])

    metrics = screening_metrics(
        scores,
        active,
        screen_ids=("screen-a",) * active.numel(),
        fractions=(0.5,),
        bedroc_alpha=20.0,
        property_only_scores=property_scores,
    )

    assert metrics.pr_auc == pytest.approx(1.0)
    assert dict(metrics.enrichment_factors)[0.5] == pytest.approx(2.0)
    assert dict(metrics.hit_rates)[0.5] == pytest.approx(2 / 3)
    assert metrics.bedroc == pytest.approx(1.0)
    assert metrics.shortcut_audit is not None
    assert metrics.shortcut_audit.model_pr_auc > metrics.shortcut_audit.property_pr_auc


def test_screening_metrics_are_tie_aware_at_enrichment_boundary() -> None:
    metrics = screening_metrics(
        scores=torch.tensor([2.0, 1.0, 1.0, 0.0]),
        active=torch.tensor([1, 1, 0, 0], dtype=torch.bool),
        screen_ids=("screen-a",) * 4,
        fractions=(0.5,),
    )

    # The second slot is shared equally by one active and one inactive tie.
    assert dict(metrics.hit_rates)[0.5] == pytest.approx(0.75)
    assert dict(metrics.enrichment_factors)[0.5] == pytest.approx(1.5)


def test_bedroc_stays_finite_for_very_strong_early_enrichment_weight() -> None:
    metrics = screening_metrics(
        scores=torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0]),
        active=torch.tensor([1, 1, 0, 0, 0], dtype=torch.bool),
        screen_ids=("screen-a",) * 5,
        bedroc_alpha=1e6,
    )

    assert metrics.bedroc == pytest.approx(1.0)


@pytest.mark.parametrize(
    "active",
    [
        torch.tensor([0, 0, 0], dtype=torch.bool),
        torch.tensor([1, 1, 1], dtype=torch.bool),
    ],
)
def test_screening_metrics_require_both_classes(active: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="positive and one negative"):
        screening_metrics(
            torch.arange(3.0),
            active,
            screen_ids=("screen-a",) * active.numel(),
        )


def test_screening_metrics_reject_flattening_multiple_screens() -> None:
    with pytest.raises(ValueError, match="exactly one screen"):
        screening_metrics(
            scores=torch.tensor([4.0, 3.0, 2.0, 1.0]),
            active=torch.tensor([1, 0, 1, 0], dtype=torch.bool),
            screen_ids=("screen-a", "screen-a", "screen-b", "screen-b"),
        )


@pytest.mark.parametrize(
    "screen_ids",
    [
        ("screen-a",),
        ("screen-a", "", "screen-a", "screen-a"),
    ],
)
def test_screening_metrics_validate_screen_identity(
    screen_ids: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="screen_ids"):
        screening_metrics(
            scores=torch.tensor([4.0, 3.0, 2.0, 1.0]),
            active=torch.tensor([1, 0, 1, 0], dtype=torch.bool),
            screen_ids=screen_ids,
        )
