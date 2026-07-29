"""Dependency-free SBDD metric primitives with explicit grouping semantics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch


@dataclass(frozen=True)
class AffinityMetrics:
    count: int
    mae: float
    rmse: float
    pearson: float | None
    spearman: float | None
    within_target_spearman: float | None
    within_target_group_count: int
    mae_standard_error: float
    mae_ci95: tuple[float, float]
    interval_coverage: float | None = None
    interval_mean_width: float | None = None


@dataclass(frozen=True)
class AffinitySlice:
    slice_name: str
    slice_value: str
    metrics: AffinityMetrics


@dataclass(frozen=True)
class PoseMetrics:
    group_count: int
    topk_success: tuple[tuple[int, float], ...]
    within_pair_spearman: float | None
    within_pair_group_count: int
    top1_mean_rmsd: float
    top1_mean_clash: float | None = None
    top1_mean_strain: float | None = None
    top1_mean_contact: float | None = None


@dataclass(frozen=True)
class PoseSlice:
    slice_name: str
    slice_value: str
    metrics: PoseMetrics


@dataclass(frozen=True)
class ScreeningShortcutAudit:
    model_pr_auc: float
    property_pr_auc: float
    pr_auc_delta: float


@dataclass(frozen=True)
class ScreeningMetrics:
    count: int
    positive_count: int
    pr_auc: float
    enrichment_factors: tuple[tuple[float, float], ...]
    hit_rates: tuple[tuple[float, float], ...]
    bedroc: float
    bedroc_alpha: float
    shortcut_audit: ScreeningShortcutAudit | None = None


def affinity_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    target_ids: Sequence[str] | None = None,
    interval_lower: torch.Tensor | None = None,
    interval_upper: torch.Tensor | None = None,
) -> AffinityMetrics:
    prediction = _finite_vector("prediction", prediction)
    target = _finite_vector("target", target)
    _same_length(prediction, target)
    count = prediction.numel()
    error = prediction - target
    absolute = error.abs()
    mae = float(absolute.mean())
    rmse = float(error.square().mean().sqrt())
    standard_error = (
        0.0 if count < 2 else float(absolute.std(unbiased=True) / math.sqrt(count))
    )
    ci = (max(0.0, mae - 1.96 * standard_error), mae + 1.96 * standard_error)
    pearson = _pearson(prediction, target)
    spearman = _spearman(prediction, target)
    within: list[float] = []
    if target_ids is not None:
        if len(target_ids) != count:
            raise ValueError("target_ids must match prediction length")
        for group in _groups(target_ids):
            if len(group) < 2:
                continue
            value = _spearman(prediction[group], target[group])
            if value is not None:
                within.append(value)
    coverage: float | None = None
    mean_width: float | None = None
    if (interval_lower is None) != (interval_upper is None):
        raise ValueError("interval_lower and interval_upper must be provided together")
    if interval_lower is not None and interval_upper is not None:
        lower = _finite_vector("interval_lower", interval_lower)
        upper = _finite_vector("interval_upper", interval_upper)
        _same_length(target, lower)
        _same_length(target, upper)
        if bool((lower > upper).any()):
            raise ValueError("prediction interval lower bound exceeds upper bound")
        coverage = float(((target >= lower) & (target <= upper)).double().mean())
        mean_width = float((upper - lower).mean())
    return AffinityMetrics(
        count=count,
        mae=mae,
        rmse=rmse,
        pearson=pearson,
        spearman=spearman,
        within_target_spearman=(
            None if not within else float(sum(within) / len(within))
        ),
        within_target_group_count=len(within),
        mae_standard_error=standard_error,
        mae_ci95=ci,
        interval_coverage=coverage,
        interval_mean_width=mean_width,
    )


def affinity_metrics_by_slice(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    slice_values: Sequence[str],
    slice_name: str,
) -> tuple[AffinitySlice, ...]:
    prediction = _finite_vector("prediction", prediction)
    target = _finite_vector("target", target)
    _same_length(prediction, target)
    if len(slice_values) != prediction.numel():
        raise ValueError("slice_values must match prediction length")
    if not slice_name.strip():
        raise ValueError("slice_name must be non-empty")
    result: list[AffinitySlice] = []
    for value in sorted(set(slice_values)):
        indices = [index for index, item in enumerate(slice_values) if item == value]
        result.append(
            AffinitySlice(
                slice_name=slice_name,
                slice_value=value,
                metrics=affinity_metrics(prediction[indices], target[indices]),
            )
        )
    return tuple(result)


def pose_metrics(
    scores: torch.Tensor,
    rmsd: torch.Tensor,
    *,
    group_ids: Sequence[str],
    top_ks: tuple[int, ...] = (1, 3, 5),
    success_rmsd: float = 2.0,
    clash: torch.Tensor | None = None,
    strain: torch.Tensor | None = None,
    contact: torch.Tensor | None = None,
) -> PoseMetrics:
    scores = _finite_vector("scores", scores)
    rmsd = _finite_vector("rmsd", rmsd)
    _same_length(scores, rmsd)
    if bool((rmsd < 0).any()):
        raise ValueError("rmsd must be nonnegative")
    if len(group_ids) != scores.numel():
        raise ValueError("group_ids must match score length")
    if not math.isfinite(success_rmsd) or success_rmsd < 0:
        raise ValueError("success_rmsd must be finite and nonnegative")
    if not top_ks or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in top_ks
    ):
        raise ValueError("top_ks must contain positive integers")
    top_ks = tuple(sorted(set(top_ks)))
    auxiliaries = {
        "clash": _optional_vector("clash", clash, scores),
        "strain": _optional_vector("strain", strain, scores),
        "contact": _optional_vector("contact", contact, scores),
    }
    grouped = _groups(group_ids)
    successes = {value: 0.0 for value in top_ks}
    correlations: list[float] = []
    top_rmsd: list[float] = []
    top_auxiliary: dict[str, list[float]] = {
        name: [] for name, value in auxiliaries.items() if value is not None
    }
    for indices in grouped:
        group_scores = scores[indices]
        group_rmsd = rmsd[indices]
        order = torch.argsort(group_scores, descending=True, stable=True)
        maximum = group_scores[order[0]]
        top_mask = group_scores == maximum
        top_rmsd.append(float(group_rmsd[top_mask].mean()))
        for name, values in auxiliaries.items():
            if values is not None:
                top_auxiliary[name].append(float(values[indices][top_mask].mean()))
        for k in top_ks:
            successes[k] += _tie_aware_topk_success(
                group_scores,
                group_rmsd <= success_rmsd,
                k=min(k, len(indices)),
            )
        correlation = _spearman(group_scores, -group_rmsd)
        if correlation is not None:
            correlations.append(correlation)
    group_count = len(grouped)
    return PoseMetrics(
        group_count=group_count,
        topk_success=tuple((k, successes[k] / group_count) for k in top_ks),
        within_pair_spearman=(
            None if not correlations else float(sum(correlations) / len(correlations))
        ),
        within_pair_group_count=len(correlations),
        top1_mean_rmsd=float(sum(top_rmsd) / len(top_rmsd)),
        top1_mean_clash=_optional_mean(top_auxiliary.get("clash")),
        top1_mean_strain=_optional_mean(top_auxiliary.get("strain")),
        top1_mean_contact=_optional_mean(top_auxiliary.get("contact")),
    )


def _tie_aware_topk_success(
    scores: torch.Tensor,
    successful: torch.Tensor,
    *,
    k: int,
) -> float:
    """Expected success under uniform selection inside the boundary tie."""

    order = torch.argsort(scores, descending=True, stable=True)
    boundary = scores[order[k - 1]]
    above = scores > boundary
    if bool(successful[above].any()):
        return 1.0
    tied = scores == boundary
    tied_count = int(tied.sum())
    tied_successes = int(successful[tied].sum())
    remaining = k - int(above.sum())
    if tied_successes == 0 or remaining == 0:
        return 0.0
    if remaining >= tied_count:
        return 1.0
    failures = tied_count - tied_successes
    if failures < remaining:
        return 1.0
    return 1.0 - math.comb(failures, remaining) / math.comb(
        tied_count,
        remaining,
    )


def pose_metrics_by_slice(
    scores: torch.Tensor,
    rmsd: torch.Tensor,
    *,
    group_ids: Sequence[str],
    slice_values: Sequence[str],
    slice_name: str,
    top_ks: tuple[int, ...] = (1, 3, 5),
    success_rmsd: float = 2.0,
) -> tuple[PoseSlice, ...]:
    scores = _finite_vector("scores", scores)
    rmsd = _finite_vector("rmsd", rmsd)
    _same_length(scores, rmsd)
    if len(group_ids) != scores.numel() or len(slice_values) != scores.numel():
        raise ValueError("group_ids and slice_values must match score length")
    # A pose group must have one slice label; otherwise a slice would split a pair.
    group_slice: dict[str, str] = {}
    for group, value in zip(group_ids, slice_values, strict=True):
        previous = group_slice.setdefault(group, value)
        if previous != value:
            raise ValueError("one pose group cannot span metric slices")
    result: list[PoseSlice] = []
    for value in sorted(set(slice_values)):
        indices = [index for index, item in enumerate(slice_values) if item == value]
        result.append(
            PoseSlice(
                slice_name=slice_name,
                slice_value=value,
                metrics=pose_metrics(
                    scores[indices],
                    rmsd[indices],
                    group_ids=tuple(group_ids[index] for index in indices),
                    top_ks=top_ks,
                    success_rmsd=success_rmsd,
                ),
            )
        )
    return tuple(result)


def screening_metrics(
    scores: torch.Tensor,
    active: torch.Tensor,
    *,
    screen_ids: Sequence[str],
    fractions: tuple[float, ...] = (0.01, 0.05, 0.1),
    bedroc_alpha: float = 20.0,
    property_only_scores: torch.Tensor | None = None,
) -> ScreeningMetrics:
    """Evaluate one screening universe; call separately for each screen.

    ``screen_ids`` must identify exactly one predeclared campaign/target/universe.
    This flat API intentionally rejects pooled rows from multiple screens because
    prevalence and candidate-set size make EF, hit rate, PR-AUC, and BEDROC
    incomparable across a silently flattened mixture.
    """

    scores = _finite_vector("scores", scores)
    if active.ndim != 1 or active.dtype != torch.bool:
        raise ValueError("active must be a one-dimensional boolean tensor")
    active = active.detach().cpu()
    _same_length(scores, active)
    _validate_single_screen(screen_ids, scores.numel())
    positive_count = int(active.sum())
    if positive_count == 0 or positive_count == active.numel():
        raise ValueError("screening metrics require one positive and one negative")
    if not fractions or any(
        not isinstance(value, (float, int))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not 0 < value <= 1
        for value in fractions
    ):
        raise ValueError("fractions must contain finite values in (0, 1]")
    fractions = tuple(sorted(set(float(value) for value in fractions)))
    if (
        isinstance(bedroc_alpha, bool)
        or not isinstance(bedroc_alpha, (float, int))
        or not math.isfinite(float(bedroc_alpha))
        or bedroc_alpha <= 0
    ):
        raise ValueError("bedroc_alpha must be finite and positive")
    pr_auc = _average_precision(scores, active)
    base_rate = positive_count / active.numel()
    hit_rates: list[tuple[float, float]] = []
    enrichment: list[tuple[float, float]] = []
    for fraction in fractions:
        hit_rate = _tie_aware_hit_rate(scores, active, fraction)
        hit_rates.append((fraction, hit_rate))
        enrichment.append((fraction, hit_rate / base_rate))
    shortcut: ScreeningShortcutAudit | None = None
    if property_only_scores is not None:
        property_scores = _finite_vector("property_only_scores", property_only_scores)
        _same_length(scores, property_scores)
        property_pr_auc = _average_precision(property_scores, active)
        shortcut = ScreeningShortcutAudit(
            model_pr_auc=pr_auc,
            property_pr_auc=property_pr_auc,
            pr_auc_delta=pr_auc - property_pr_auc,
        )
    return ScreeningMetrics(
        count=active.numel(),
        positive_count=positive_count,
        pr_auc=pr_auc,
        enrichment_factors=tuple(enrichment),
        hit_rates=tuple(hit_rates),
        bedroc=_bedroc(scores, active, float(bedroc_alpha)),
        bedroc_alpha=float(bedroc_alpha),
        shortcut_audit=shortcut,
    )


def _validate_single_screen(screen_ids: Sequence[str], count: int) -> None:
    if isinstance(screen_ids, (str, bytes)) or len(screen_ids) != count:
        raise ValueError("screen_ids must match score length")
    if any(not isinstance(value, str) or not value.strip() for value in screen_ids):
        raise ValueError("screen_ids must contain non-empty strings")
    if len(set(screen_ids)) != 1:
        raise ValueError(
            "flat screening_metrics accepts exactly one screen; "
            "evaluate campaigns separately"
        )


def _finite_vector(name: str, values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional tensor")
    if not torch.is_floating_point(values):
        raise ValueError(f"{name} must be floating point")
    values = values.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(values).all()):
        raise ValueError(f"{name} must be finite")
    return values


def _optional_vector(
    name: str,
    values: torch.Tensor | None,
    reference: torch.Tensor,
) -> torch.Tensor | None:
    if values is None:
        return None
    values = _finite_vector(name, values)
    _same_length(values, reference)
    return values


def _same_length(left: torch.Tensor, right: torch.Tensor) -> None:
    if left.numel() != right.numel():
        raise ValueError("metric inputs must have equal length")


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float | None:
    if left.numel() < 2:
        return None
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = left_centered.norm() * right_centered.norm()
    if denominator == 0:
        return None
    return float(torch.dot(left_centered, right_centered) / denominator)


def _spearman(left: torch.Tensor, right: torch.Tensor) -> float | None:
    return _pearson(_midranks(left), _midranks(right))


def _midranks(values: torch.Tensor, *, descending: bool = False) -> torch.Tensor:
    order = torch.argsort(values, descending=descending, stable=True)
    ranks = torch.empty_like(values, dtype=torch.float64)
    position = 0
    while position < order.numel():
        end = position + 1
        current = values[order[position]]
        while end < order.numel() and values[order[end]] == current:
            end += 1
        average = (position + 1 + end) / 2.0
        ranks[order[position:end]] = average
        position = end
    return ranks


def _groups(values: Sequence[str]) -> list[list[int]]:
    grouped: dict[str, list[int]] = {}
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value:
            raise ValueError("group identifiers must be non-empty strings")
        grouped.setdefault(value, []).append(index)
    return [grouped[value] for value in sorted(grouped)]


def _average_precision(scores: torch.Tensor, active: torch.Tensor) -> float:
    order = torch.argsort(scores, descending=True, stable=True)
    positives = int(active.sum())
    true_positive = 0
    seen = 0
    previous_recall = 0.0
    area = 0.0
    position = 0
    while position < order.numel():
        end = position + 1
        score = scores[order[position]]
        while end < order.numel() and scores[order[end]] == score:
            end += 1
        group = order[position:end]
        true_positive += int(active[group].sum())
        seen += end - position
        recall = true_positive / positives
        precision = true_positive / seen
        area += (recall - previous_recall) * precision
        previous_recall = recall
        position = end
    return area


def _tie_aware_hit_rate(
    scores: torch.Tensor,
    active: torch.Tensor,
    fraction: float,
) -> float:
    count = scores.numel()
    slots = max(1, math.ceil(fraction * count))
    order = torch.argsort(scores, descending=True, stable=True)
    boundary_score = scores[order[slots - 1]]
    above = scores > boundary_score
    tied = scores == boundary_score
    selected_above = int(above.sum())
    slots_in_tie = slots - selected_above
    expected_hits = float(active[above].sum())
    expected_hits += slots_in_tie * float(active[tied].double().mean())
    return expected_hits / slots


def _bedroc(scores: torch.Tensor, active: torch.Tensor, alpha: float) -> float:
    ranks = _midranks(scores, descending=True)
    active_ranks = ranks[active]
    count = scores.numel()
    positives = active_ranks.numel()
    # Shift every exponent by the best possible rank.  The shared positive
    # factor cancels in the normalization and avoids all-zero underflow for
    # large alpha.
    raw = float(torch.exp(-alpha * (active_ranks - 1) / count).mean())
    top = torch.arange(1, positives + 1, dtype=torch.float64)
    bottom = torch.arange(
        count - positives + 1,
        count + 1,
        dtype=torch.float64,
    )
    maximum = float(torch.exp(-alpha * (top - 1) / count).mean())
    minimum = float(torch.exp(-alpha * (bottom - 1) / count).mean())
    normalized = (raw - minimum) / (maximum - minimum)
    return min(1.0, max(0.0, normalized))


def _optional_mean(values: list[float] | None) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))
