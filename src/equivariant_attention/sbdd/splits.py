"""Deterministic, label-blind SBDD split and leakage contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Mapping


class SplitMode(StrEnum):
    WARM_PAIR = "warm_pair"
    COLD_DRUG = "cold_drug"
    COLD_TARGET = "cold_target"
    COLD_BOTH = "cold_both"
    CAMPAIGN = "campaign"
    TEMPORAL = "temporal"


_SPLITS = ("train", "validation", "test")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class SplitRecord:
    """Only immutable IDs and provenance; labels are intentionally absent."""

    sample_id: str
    standardized_ligand_id: str
    protein_construct_id: str
    ligand_cluster_id: str
    protein_cluster_id: str
    pose_group_id: str
    replicate_group_id: str
    derived_view_group_id: str
    campaign_id: str | None = None
    observed_on: date | None = None

    def __post_init__(self) -> None:
        for name in (
            "sample_id",
            "standardized_ligand_id",
            "protein_construct_id",
            "ligand_cluster_id",
            "protein_cluster_id",
            "pose_group_id",
            "replicate_group_id",
            "derived_view_group_id",
        ):
            _text(name, getattr(self, name))
        if self.campaign_id is not None:
            _text("campaign_id", self.campaign_id)
        if self.observed_on is not None and not isinstance(self.observed_on, date):
            raise ValueError("observed_on must be datetime.date")


@dataclass(frozen=True)
class SplitPolicy:
    mode: SplitMode
    seed: int
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)
    validation_campaigns: tuple[str, ...] = ()
    test_campaigns: tuple[str, ...] = ()
    temporal_train_end: date | None = None
    temporal_validation_end: date | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, SplitMode):
            raise TypeError("mode must be a SplitMode")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if len(self.fractions) != 3:
            raise ValueError("fractions must contain train, validation, and test")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
            for value in self.fractions
        ):
            raise ValueError("split fractions must be finite and positive")
        if not math.isclose(sum(self.fractions), 1.0, abs_tol=1e-12):
            raise ValueError("split fractions must sum to one")
        for campaigns in (self.validation_campaigns, self.test_campaigns):
            if len(set(campaigns)) != len(campaigns):
                raise ValueError("campaign holdout IDs must be unique")
            for campaign in campaigns:
                _text("campaign holdout", campaign)
        if set(self.validation_campaigns) & set(self.test_campaigns):
            raise ValueError("validation and test campaigns must be disjoint")
        if self.mode is SplitMode.CAMPAIGN:
            if not self.validation_campaigns or not self.test_campaigns:
                raise ValueError(
                    "campaign split requires validation_campaigns and test_campaigns"
                )
        elif self.validation_campaigns or self.test_campaigns:
            raise ValueError("campaign holdouts are valid only for campaign mode")
        if self.mode is SplitMode.TEMPORAL:
            if self.temporal_train_end is None or self.temporal_validation_end is None:
                raise ValueError("temporal split requires both temporal cutoffs")
            if self.temporal_train_end >= self.temporal_validation_end:
                raise ValueError(
                    "temporal_train_end must precede temporal_validation_end"
                )
        elif (
            self.temporal_train_end is not None
            or self.temporal_validation_end is not None
        ):
            raise ValueError("temporal cutoffs are valid only for temporal mode")

    @property
    def sha256(self) -> str:
        payload = {
            "mode": self.mode.value,
            "seed": self.seed,
            "fractions": self.fractions,
            "validation_campaigns": sorted(self.validation_campaigns),
            "test_campaigns": sorted(self.test_campaigns),
            "temporal_train_end": (
                None
                if self.temporal_train_end is None
                else self.temporal_train_end.isoformat()
            ),
            "temporal_validation_end": (
                None
                if self.temporal_validation_end is None
                else self.temporal_validation_end.isoformat()
            ),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True, order=True)
class SplitAssignment:
    sample_id: str
    split: str

    def __post_init__(self) -> None:
        _text("sample_id", self.sample_id)
        if self.split not in _SPLITS:
            raise ValueError(f"split must be one of {_SPLITS}")


@dataclass(frozen=True)
class SplitResult:
    assignments: tuple[SplitAssignment, ...]
    policy_sha256: str
    membership_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.assignments, tuple):
            raise TypeError("assignments must be a tuple of SplitAssignment")
        membership: dict[str, str] = {}
        for assignment in self.assignments:
            if not isinstance(assignment, SplitAssignment):
                raise TypeError("assignments must contain only SplitAssignment")
            previous = membership.get(assignment.sample_id)
            if previous is not None:
                if previous != assignment.split:
                    raise ValueError(
                        f"{assignment.sample_id!r} has conflicting splits "
                        f"{previous!r} and {assignment.split!r}"
                    )
                raise ValueError("assignment sample_id values must be unique")
            membership[assignment.sample_id] = assignment.split
        _validate_sha256("policy sha256", self.policy_sha256)
        _validate_sha256("membership sha256", self.membership_sha256)
        expected = _membership_sha256(self.assignments, self.policy_sha256)
        if self.membership_sha256 != expected:
            raise ValueError("membership sha256 does not match assignments")

    @property
    def membership(self) -> Mapping[str, str]:
        return MappingProxyType(
            {assignment.sample_id: assignment.split for assignment in self.assignments}
        )

    def with_assignments(self, membership: Mapping[str, str]) -> SplitResult:
        assignments = tuple(
            sorted(
                (
                    SplitAssignment(sample_id=sample_id, split=split)
                    for sample_id, split in membership.items()
                ),
                key=lambda item: item.sample_id,
            )
        )
        return _make_result(assignments, self.policy_sha256)


@dataclass(frozen=True, order=True)
class LeakageFinding:
    key: str
    value: str
    splits: tuple[str, ...]


@dataclass(frozen=True)
class LeakageAudit:
    findings: tuple[LeakageFinding, ...]
    membership_sha256: str
    policy_sha256: str

    @property
    def clean(self) -> bool:
        return not self.findings


class _DisjointSet:
    def __init__(self, values: tuple[str, ...]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        self.parent[second] = first


def deterministic_split(
    records: tuple[SplitRecord, ...],
    policy: SplitPolicy,
) -> SplitResult:
    """Assign connected label-blind groups using a stable SHA-256 partition."""

    if not records:
        raise ValueError("at least one split record is required")
    ordered = tuple(sorted(records, key=lambda item: item.sample_id))
    _validate_cold_entity_cluster_consistency(ordered, policy.mode)
    sample_ids = tuple(record.sample_id for record in ordered)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_id values must be unique")
    dsu = _DisjointSet(sample_ids)
    grouping_fields = [
        "pose_group_id",
        "replicate_group_id",
        "derived_view_group_id",
    ]
    if policy.mode is SplitMode.WARM_PAIR:
        _union_by_composite(
            ordered,
            dsu,
            ("standardized_ligand_id", "protein_construct_id"),
        )
    elif policy.mode in {
        SplitMode.COLD_DRUG,
        SplitMode.COLD_TARGET,
        SplitMode.COLD_BOTH,
    }:
        grouping_fields.extend(_cold_grouping_fields(policy.mode))
    elif policy.mode is SplitMode.CAMPAIGN:
        grouping_fields.append("campaign_id")
    for field in grouping_fields:
        _union_by_field(ordered, dsu, field)

    components: dict[str, list[SplitRecord]] = {}
    for record in ordered:
        components.setdefault(dsu.find(record.sample_id), []).append(record)
    component_splits: dict[str, str] = {}
    for root, component in sorted(components.items()):
        if policy.mode is SplitMode.CAMPAIGN:
            desired = {_campaign_split(record, policy) for record in component}
            if len(desired) != 1:
                raise ValueError(
                    "one leakage-isolated group spans campaign split boundaries"
                )
            component_splits[root] = desired.pop()
        elif policy.mode is SplitMode.TEMPORAL:
            desired = {_temporal_split(record, policy) for record in component}
            if len(desired) != 1:
                raise ValueError("one leakage-isolated group spans a temporal boundary")
            component_splits[root] = desired.pop()
        else:
            identity = "\x1f".join(record.sample_id for record in component)
            component_splits[root] = _hashed_split(identity, policy)
    assignments = tuple(
        SplitAssignment(
            sample_id=record.sample_id,
            split=component_splits[dsu.find(record.sample_id)],
        )
        for record in ordered
    )
    return _make_result(assignments, policy.sha256)


def audit_split_leakage(
    records: tuple[SplitRecord, ...],
    result: SplitResult,
    *,
    policy: SplitPolicy,
) -> LeakageAudit:
    if result.policy_sha256 != policy.sha256:
        raise ValueError("split result and policy hashes disagree")
    _validate_cold_entity_cluster_consistency(records, policy.mode)
    canonical = _make_result(result.assignments, result.policy_sha256)
    if canonical.membership_sha256 != result.membership_sha256:
        raise ValueError("split membership hash does not match assignments")
    membership = result.membership
    record_ids = {record.sample_id for record in records}
    if record_ids != set(membership):
        raise ValueError("split membership must cover exactly the supplied sample IDs")
    fields = [
        "pose_group_id",
        "replicate_group_id",
        "derived_view_group_id",
    ]
    composite: tuple[str, ...] | None = None
    if policy.mode is SplitMode.WARM_PAIR:
        composite = ("standardized_ligand_id", "protein_construct_id")
    elif policy.mode in {
        SplitMode.COLD_DRUG,
        SplitMode.COLD_TARGET,
        SplitMode.COLD_BOTH,
    }:
        fields.extend(_cold_grouping_fields(policy.mode))
    elif policy.mode is SplitMode.CAMPAIGN:
        fields.append("campaign_id")
    findings: list[LeakageFinding] = []
    for field in fields:
        owners: dict[str, set[str]] = {}
        for record in records:
            value = getattr(record, field)
            if value is not None:
                owners.setdefault(value, set()).add(membership[record.sample_id])
        for value, splits in owners.items():
            if len(splits) > 1:
                findings.append(
                    LeakageFinding(
                        key=field,
                        value=value,
                        splits=tuple(sorted(splits)),
                    )
                )
    if composite is not None:
        owners = {}
        for record in records:
            value = "|".join(getattr(record, field) for field in composite)
            owners.setdefault(value, set()).add(membership[record.sample_id])
        for value, splits in owners.items():
            if len(splits) > 1:
                findings.append(
                    LeakageFinding(
                        key="pair_id",
                        value=value,
                        splits=tuple(sorted(splits)),
                    )
                )
    if policy.mode is SplitMode.CAMPAIGN:
        for record in records:
            expected = _campaign_split(record, policy)
            actual = membership[record.sample_id]
            if actual != expected:
                findings.append(
                    LeakageFinding(
                        key="campaign_policy",
                        value=record.sample_id,
                        splits=(actual, expected),
                    )
                )
    if policy.mode is SplitMode.TEMPORAL:
        for record in records:
            expected = _temporal_split(record, policy)
            actual = membership[record.sample_id]
            if actual != expected:
                findings.append(
                    LeakageFinding(
                        key="temporal_policy",
                        value=record.sample_id,
                        splits=(actual, expected),
                    )
                )
    return LeakageAudit(
        findings=tuple(sorted(findings)),
        membership_sha256=result.membership_sha256,
        policy_sha256=result.policy_sha256,
    )


def assert_no_split_leakage(
    records: tuple[SplitRecord, ...],
    result: SplitResult,
    *,
    policy: SplitPolicy,
) -> None:
    audit = audit_split_leakage(records, result, policy=policy)
    if not audit.clean:
        first = audit.findings[0]
        raise ValueError(
            "split leakage detected for "
            f"{first.key}={first.value!r} across {first.splits}"
        )


def _union_by_field(
    records: tuple[SplitRecord, ...],
    dsu: _DisjointSet,
    field: str,
) -> None:
    owner: dict[object, str] = {}
    for record in records:
        value = getattr(record, field)
        if value is None:
            continue
        previous = owner.setdefault(value, record.sample_id)
        dsu.union(previous, record.sample_id)


def _cold_grouping_fields(mode: SplitMode) -> tuple[str, ...]:
    if mode is SplitMode.COLD_DRUG:
        return ("standardized_ligand_id", "ligand_cluster_id")
    if mode is SplitMode.COLD_TARGET:
        return ("protein_construct_id", "protein_cluster_id")
    if mode is SplitMode.COLD_BOTH:
        return (
            "standardized_ligand_id",
            "ligand_cluster_id",
            "protein_construct_id",
            "protein_cluster_id",
        )
    raise ValueError("cold grouping fields require a cold split mode")


def _validate_cold_entity_cluster_consistency(
    records: tuple[SplitRecord, ...],
    mode: SplitMode,
) -> None:
    pairs: tuple[tuple[str, str], ...]
    if mode is SplitMode.COLD_DRUG:
        pairs = (("standardized_ligand_id", "ligand_cluster_id"),)
    elif mode is SplitMode.COLD_TARGET:
        pairs = (("protein_construct_id", "protein_cluster_id"),)
    elif mode is SplitMode.COLD_BOTH:
        pairs = (
            ("standardized_ligand_id", "ligand_cluster_id"),
            ("protein_construct_id", "protein_cluster_id"),
        )
    else:
        return
    for entity_field, cluster_field in pairs:
        cluster_by_entity: dict[str, str] = {}
        for record in records:
            entity = getattr(record, entity_field)
            cluster = getattr(record, cluster_field)
            previous = cluster_by_entity.setdefault(entity, cluster)
            if previous != cluster:
                raise ValueError(
                    f"{entity_field}={entity!r} maps to multiple "
                    f"{cluster_field} values"
                )


def _union_by_composite(
    records: tuple[SplitRecord, ...],
    dsu: _DisjointSet,
    fields: tuple[str, ...],
) -> None:
    owner: dict[tuple[object, ...], str] = {}
    for record in records:
        value = tuple(getattr(record, field) for field in fields)
        previous = owner.setdefault(value, record.sample_id)
        dsu.union(previous, record.sample_id)


def _hashed_split(identity: str, policy: SplitPolicy) -> str:
    digest = hashlib.sha256(f"{policy.seed}\x1e{identity}".encode("utf-8")).digest()
    score = int.from_bytes(digest[:8], "big") / 2**64
    train_end = policy.fractions[0]
    validation_end = train_end + policy.fractions[1]
    if score < train_end:
        return "train"
    if score < validation_end:
        return "validation"
    return "test"


def _campaign_split(record: SplitRecord, policy: SplitPolicy) -> str:
    if record.campaign_id is None:
        raise ValueError("campaign split requires campaign_id for every record")
    if record.campaign_id in policy.validation_campaigns:
        return "validation"
    if record.campaign_id in policy.test_campaigns:
        return "test"
    return "train"


def _temporal_split(record: SplitRecord, policy: SplitPolicy) -> str:
    if record.observed_on is None:
        raise ValueError("temporal split requires observed_on for every record")
    assert policy.temporal_train_end is not None
    assert policy.temporal_validation_end is not None
    if record.observed_on <= policy.temporal_train_end:
        return "train"
    if record.observed_on <= policy.temporal_validation_end:
        return "validation"
    return "test"


def _make_result(
    assignments: tuple[SplitAssignment, ...],
    policy_sha256: str,
) -> SplitResult:
    return SplitResult(
        assignments=assignments,
        policy_sha256=policy_sha256,
        membership_sha256=_membership_sha256(assignments, policy_sha256),
    )


def _membership_sha256(
    assignments: tuple[SplitAssignment, ...],
    policy_sha256: str,
) -> str:
    canonical = {
        "policy_sha256": policy_sha256,
        "assignments": [
            (assignment.sample_id, assignment.split) for assignment in assignments
        ],
    }
    return hashlib.sha256(
        json.dumps(canonical, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_sha256(name: str, value: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be 64 lowercase hex characters")
