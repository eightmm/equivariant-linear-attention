"""Grouped pose-ranking objectives; groups never interact in the loss."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import torch
import torch.nn.functional as F

from .schema import LabelDirection, LabelKind, LabelQualifier, ScientificLabel


@dataclass(frozen=True)
class PoseAuxiliaryPredictions:
    clash: torch.Tensor | None = None
    strain: torch.Tensor | None = None
    contact_logits: torch.Tensor | None = None


@dataclass(frozen=True)
class PoseAuxiliaryTargets:
    clash: torch.Tensor | None = None
    strain: torch.Tensor | None = None
    contact: torch.Tensor | None = None


@dataclass(frozen=True)
class GroupedPoseLoss:
    total: torch.Tensor
    ranking: torch.Tensor
    clash: torch.Tensor
    strain: torch.Tensor
    contact: torch.Tensor
    group_count: int


def censored_affinity_loss(
    prediction: torch.Tensor,
    labels: Sequence[ScientificLabel],
    *,
    prediction_direction: LabelDirection = LabelDirection.HIGHER_IS_STRONGER,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> torch.Tensor:
    """Squared distance to each exact value or admissible censoring interval.

    Lower-bound observations penalize only predictions below the bound;
    upper-bound observations penalize only predictions above it; interval
    observations have zero loss inside their interval. Labels remain in their
    original scientific unit and must share one unit and the declared
    prediction direction. The default matches :class:`AffinityHead`.
    """

    _validate_vector("prediction", prediction)
    if len(labels) != prediction.numel():
        raise ValueError("labels must contain one affinity label per prediction")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if not isinstance(prediction_direction, LabelDirection):
        raise TypeError("prediction_direction must be a LabelDirection")
    if not labels:
        raise ValueError("labels must be non-empty")
    units: set[str] = set()
    directions: set[LabelDirection] = set()
    terms: list[torch.Tensor] = []
    for value, label in zip(prediction.unbind(), labels, strict=True):
        if not isinstance(label, ScientificLabel):
            raise TypeError("labels must contain ScientificLabel values")
        if label.kind is not LabelKind.AFFINITY:
            raise ValueError("censored affinity loss accepts affinity labels only")
        units.add(label.unit.strip().casefold())
        directions.add(label.direction)
        if label.qualifier is LabelQualifier.EXACT:
            assert label.value is not None
            terms.append((value - float(label.value)).square())
        elif label.qualifier is LabelQualifier.LOWER_BOUND:
            assert label.lower is not None
            terms.append(F.relu(float(label.lower) - value).square())
        elif label.qualifier is LabelQualifier.UPPER_BOUND:
            assert label.upper is not None
            terms.append(F.relu(value - float(label.upper)).square())
        else:
            assert label.lower is not None
            assert label.upper is not None
            terms.append(
                F.relu(float(label.lower) - value).square()
                + F.relu(value - float(label.upper)).square()
            )
    if len(units) != 1:
        raise ValueError("all affinity labels must use the same scientific unit")
    if len(directions) != 1:
        raise ValueError("all affinity labels must use the same label direction")
    if directions != {prediction_direction}:
        raise ValueError(
            "affinity label direction must match the declared prediction direction"
        )
    loss = torch.stack(terms)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()


def grouped_pose_ranking_loss(
    scores: torch.Tensor,
    rmsd: torch.Tensor,
    *,
    group_ids: Sequence[str],
    method: Literal["listwise", "rmsd_aware"] = "listwise",
    temperature: float = 1.0,
    target_temperature: float = 1.0,
    auxiliary_predictions: PoseAuxiliaryPredictions | None = None,
    auxiliary_targets: PoseAuxiliaryTargets | None = None,
    auxiliary_weights: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> GroupedPoseLoss:
    """Return a mean-over-pairs loss plus explicit physical auxiliaries."""

    _validate_vector("scores", scores)
    _validate_vector("rmsd", rmsd)
    if scores.shape != rmsd.shape:
        raise ValueError("scores and rmsd must have equal shape")
    if scores.device != rmsd.device:
        raise ValueError("scores and rmsd must use the same device")
    if bool((rmsd < 0).any()):
        raise ValueError("rmsd must be nonnegative")
    if len(group_ids) != scores.numel():
        raise ValueError("group_ids must match score length")
    if method not in {"listwise", "rmsd_aware"}:
        raise ValueError("method must be 'listwise' or 'rmsd_aware'")
    for name, value in (
        ("temperature", temperature),
        ("target_temperature", target_temperature),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError(f"{name} must be finite and positive")
    if len(auxiliary_weights) != 3 or any(
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(float(value))
        or value < 0
        for value in auxiliary_weights
    ):
        raise ValueError(
            "auxiliary_weights must contain three finite nonnegative values"
        )

    groups = _group_indices(group_ids)
    if any(len(indices) < 2 for indices in groups):
        raise ValueError("every pose group must contain at least two poses")
    ranking_terms: list[torch.Tensor] = []
    for indices in groups:
        index = torch.tensor(indices, dtype=torch.long, device=scores.device)
        group_scores = scores[index]
        group_rmsd = rmsd[index]
        if method == "listwise":
            target_probability = torch.softmax(
                -group_rmsd.detach() / target_temperature,
                dim=0,
            )
            ranking_terms.append(
                -(
                    target_probability
                    * torch.log_softmax(
                        group_scores / temperature,
                        dim=0,
                    )
                ).sum()
            )
        else:
            ranking_terms.append(
                _rmsd_aware_pairwise(
                    group_scores,
                    group_rmsd,
                    temperature=float(temperature),
                )
            )
    ranking = torch.stack(ranking_terms).mean()
    zero = scores.sum() * 0.0
    clash = zero
    strain = zero
    contact = zero
    if (auxiliary_predictions is None) != (auxiliary_targets is None):
        raise ValueError(
            "auxiliary_predictions and auxiliary_targets must be provided together"
        )
    if auxiliary_predictions is not None and auxiliary_targets is not None:
        clash = _paired_auxiliary_loss(
            "clash",
            auxiliary_predictions.clash,
            auxiliary_targets.clash,
            scores,
            binary=False,
        )
        strain = _paired_auxiliary_loss(
            "strain",
            auxiliary_predictions.strain,
            auxiliary_targets.strain,
            scores,
            binary=False,
        )
        contact = _paired_auxiliary_loss(
            "contact",
            auxiliary_predictions.contact_logits,
            auxiliary_targets.contact,
            scores,
            binary=True,
        )
    total = (
        ranking
        + auxiliary_weights[0] * clash
        + auxiliary_weights[1] * strain
        + auxiliary_weights[2] * contact
    )
    return GroupedPoseLoss(
        total=total,
        ranking=ranking,
        clash=clash,
        strain=strain,
        contact=contact,
        group_count=len(groups),
    )


def _rmsd_aware_pairwise(
    scores: torch.Tensor,
    rmsd: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    terms: list[torch.Tensor] = []
    for left in range(scores.numel()):
        for right in range(left + 1, scores.numel()):
            difference = (rmsd[right] - rmsd[left]).detach()
            if difference == 0:
                continue
            # A positive difference means left is the better (lower-RMSD) pose.
            signed_score = torch.sign(difference) * (scores[left] - scores[right])
            weight = difference.abs().detach().clamp_min(1e-6)
            terms.append(F.softplus(-signed_score / temperature) * weight)
    if not terms:
        raise ValueError("rmsd-aware group requires at least two distinct RMSDs")
    normalizer = torch.stack(
        [
            (rmsd[right] - rmsd[left]).abs().detach().clamp_min(1e-6)
            for left in range(scores.numel())
            for right in range(left + 1, scores.numel())
            if rmsd[right] != rmsd[left]
        ]
    ).mean()
    return torch.stack(terms).mean() / normalizer


def _paired_auxiliary_loss(
    name: str,
    prediction: torch.Tensor | None,
    target: torch.Tensor | None,
    reference: torch.Tensor,
    *,
    binary: bool,
) -> torch.Tensor:
    if prediction is None and target is None:
        return reference.sum() * 0.0
    if prediction is None or target is None:
        raise ValueError(f"{name} prediction and target must be provided together")
    _validate_vector(f"{name} prediction", prediction)
    _validate_vector(f"{name} target", target)
    if prediction.shape != reference.shape or target.shape != reference.shape:
        raise ValueError(f"{name} prediction/target must match score shape")
    if prediction.device != reference.device or target.device != reference.device:
        raise ValueError(f"{name} tensors must use the score device")
    if binary:
        if bool(((target < 0) | (target > 1)).any()):
            raise ValueError("contact target must lie in [0, 1]")
        return F.binary_cross_entropy_with_logits(prediction, target)
    return F.mse_loss(prediction, target)


def _validate_vector(name: str, value: torch.Tensor) -> None:
    if value.ndim != 1 or value.numel() == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional tensor")
    if not torch.is_floating_point(value) or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite floating point")


def _group_indices(group_ids: Sequence[str]) -> list[list[int]]:
    groups: dict[str, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("group IDs must be non-empty strings")
        groups.setdefault(group_id, []).append(index)
    return [groups[group_id] for group_id in sorted(groups)]
