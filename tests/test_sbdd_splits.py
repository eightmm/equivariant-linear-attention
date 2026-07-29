from __future__ import annotations

from datetime import date
import hashlib
import json

import pytest

from equivariant_attention.sbdd import (
    SplitAssignment,
    SplitMode,
    SplitPolicy,
    SplitRecord,
    SplitResult,
    assert_no_split_leakage,
    audit_split_leakage,
    deterministic_split,
)


def _record(
    index: int,
    *,
    ligand_cluster: str | None = None,
    protein_cluster: str | None = None,
    pose_group: str | None = None,
    replicate_group: str | None = None,
    campaign: str | None = None,
    observed: date | None = None,
) -> SplitRecord:
    pair = index // 2
    ligand_identity = pair % 6
    protein_identity = pair % 5
    return SplitRecord(
        sample_id=f"sample-{index:02d}",
        standardized_ligand_id=f"lig-{ligand_identity}",
        protein_construct_id=f"protein-{protein_identity}",
        ligand_cluster_id=(
            ligand_cluster or f"scaffold-{ligand_identity // 2}"
        ),
        protein_cluster_id=(
            protein_cluster or f"family-{protein_identity // 2}"
        ),
        pose_group_id=pose_group or f"poses-{pair}",
        replicate_group_id=replicate_group or f"rep-{pair}",
        derived_view_group_id=f"view-{pair}",
        campaign_id=campaign or f"campaign-{pair % 4}",
        observed_on=observed or date(2024 + pair % 3, 1, 1),
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


@pytest.mark.parametrize(
    "mode",
    [
        SplitMode.WARM_PAIR,
        SplitMode.COLD_DRUG,
        SplitMode.COLD_TARGET,
        SplitMode.COLD_BOTH,
    ],
)
def test_split_matrix_is_label_blind_deterministic_and_pose_isolated(
    mode: SplitMode,
) -> None:
    records = tuple(_record(index) for index in range(24))
    policy = SplitPolicy(mode=mode, seed=17)

    first = deterministic_split(records, policy)
    second = deterministic_split(tuple(reversed(records)), policy)

    assert first.membership_sha256 == second.membership_sha256
    assert first.assignments == second.assignments
    assert len(first.membership_sha256) == 64
    membership = first.membership
    for pair in range(12):
        assert (
            membership[f"sample-{2 * pair:02d}"]
            == membership[f"sample-{2 * pair + 1:02d}"]
        )
    assert audit_split_leakage(records, first, policy=policy).clean


def test_cold_both_prevents_ligand_and_protein_cluster_overlap() -> None:
    records = tuple(_record(index) for index in range(30))
    policy = SplitPolicy(mode=SplitMode.COLD_BOTH, seed=99)
    split = deterministic_split(records, policy)

    assert_no_split_leakage(records, split, policy=policy)
    for field in ("ligand_cluster_id", "protein_cluster_id"):
        owners: dict[str, set[str]] = {}
        for record in records:
            owners.setdefault(getattr(record, field), set()).add(
                split.membership[record.sample_id]
            )
        assert all(len(splits) == 1 for splits in owners.values())


@pytest.mark.parametrize(
    ("mode", "entity_field", "cluster_field"),
    [
        (
            SplitMode.COLD_DRUG,
            "standardized_ligand_id",
            "ligand_cluster_id",
        ),
        (
            SplitMode.COLD_TARGET,
            "protein_construct_id",
            "protein_cluster_id",
        ),
    ],
)
def test_cold_split_rejects_one_entity_drifting_between_clusters(
    mode: SplitMode,
    entity_field: str,
    cluster_field: str,
) -> None:
    first = _record(0)
    values = {
        "sample_id": "sample-drift",
        "standardized_ligand_id": first.standardized_ligand_id,
        "protein_construct_id": first.protein_construct_id,
        "ligand_cluster_id": first.ligand_cluster_id,
        "protein_cluster_id": first.protein_cluster_id,
        "pose_group_id": "pose-drift",
        "replicate_group_id": "rep-drift",
        "derived_view_group_id": "view-drift",
    }
    values[cluster_field] = f"{getattr(first, cluster_field)}-other"
    drifted = SplitRecord(**values)

    with pytest.raises(ValueError, match=f"{entity_field}.*multiple"):
        deterministic_split(
            (first, drifted),
            SplitPolicy(mode=mode, seed=3),
        )


def test_cold_entity_ids_are_audited_even_when_cluster_ids_differ() -> None:
    records = (_record(0), _record(2))
    policy = SplitPolicy(mode=SplitMode.COLD_DRUG, seed=3)
    split = deterministic_split(records, policy)
    tampered = split.with_assignments(
        {
            records[0].sample_id: "train",
            records[1].sample_id: "test",
        }
    )
    same_entity = (
        records[0],
        SplitRecord(
            sample_id=records[1].sample_id,
            standardized_ligand_id=records[0].standardized_ligand_id,
            protein_construct_id=records[1].protein_construct_id,
            ligand_cluster_id=records[0].ligand_cluster_id,
            protein_cluster_id=records[1].protein_cluster_id,
            pose_group_id=records[1].pose_group_id,
            replicate_group_id=records[1].replicate_group_id,
            derived_view_group_id=records[1].derived_view_group_id,
        ),
    )

    audit = audit_split_leakage(same_entity, tampered, policy=policy)
    assert any(
        finding.key == "standardized_ligand_id"
        for finding in audit.findings
    )


def test_leakage_audit_detects_manual_pose_and_replicate_leakage() -> None:
    records = (_record(0), _record(1))
    split = deterministic_split(
        records,
        SplitPolicy(mode=SplitMode.WARM_PAIR, seed=1),
    )
    tampered = split.with_assignments({"sample-00": "train", "sample-01": "test"})

    audit = audit_split_leakage(
        records,
        tampered,
        policy=SplitPolicy(mode=SplitMode.WARM_PAIR, seed=1),
    )
    assert not audit.clean
    assert {finding.key for finding in audit.findings} >= {
        "pose_group_id",
        "replicate_group_id",
    }
    with pytest.raises(ValueError, match="leakage"):
        assert_no_split_leakage(
            records,
            tampered,
            policy=SplitPolicy(mode=SplitMode.WARM_PAIR, seed=1),
        )


def test_campaign_holdout_uses_explicit_frozen_campaigns() -> None:
    records = tuple(_record(index) for index in range(20))
    policy = SplitPolicy(
        mode=SplitMode.CAMPAIGN,
        seed=4,
        validation_campaigns=("campaign-2",),
        test_campaigns=("campaign-3",),
    )
    split = deterministic_split(records, policy)

    for record in records:
        expected = (
            "validation"
            if record.campaign_id == "campaign-2"
            else "test"
            if record.campaign_id == "campaign-3"
            else "train"
        )
        assert split.membership[record.sample_id] == expected


def test_campaign_audit_rejects_membership_that_ignores_frozen_holdout() -> None:
    records = tuple(_record(index) for index in range(20))
    policy = SplitPolicy(
        mode=SplitMode.CAMPAIGN,
        seed=4,
        validation_campaigns=("campaign-2",),
        test_campaigns=("campaign-3",),
    )
    split = deterministic_split(records, policy)
    membership = dict(split.membership)
    held_out = next(record for record in records if record.campaign_id == "campaign-3")
    membership[held_out.sample_id] = "train"
    tampered = split.with_assignments(membership)

    audit = audit_split_leakage(records, tampered, policy=policy)

    assert any(item.key == "campaign_policy" for item in audit.findings)


def test_temporal_split_rejects_one_derived_group_crossing_time_boundary() -> None:
    records = (
        _record(
            0,
            pose_group="same",
            replicate_group="same",
            observed=date(2023, 1, 1),
        ),
        _record(
            1,
            pose_group="same",
            replicate_group="same",
            observed=date(2026, 1, 1),
        ),
    )
    policy = SplitPolicy(
        mode=SplitMode.TEMPORAL,
        seed=1,
        temporal_train_end=date(2024, 12, 31),
        temporal_validation_end=date(2025, 12, 31),
    )
    with pytest.raises(ValueError, match="temporal boundary"):
        deterministic_split(records, policy)


def test_split_rejects_duplicate_sample_ids() -> None:
    record = _record(0)
    with pytest.raises(ValueError, match="sample_id.*unique"):
        deterministic_split(
            (record, record),
            SplitPolicy(mode=SplitMode.COLD_DRUG, seed=1),
        )


def test_split_result_rejects_duplicate_sample_ids_before_membership_collapse() -> None:
    assignment = SplitAssignment(sample_id="sample-00", split="train")
    assignments = (assignment, assignment)
    policy_sha256 = "a" * 64

    with pytest.raises(ValueError, match="sample_id.*unique"):
        SplitResult(
            assignments=assignments,
            policy_sha256=policy_sha256,
            membership_sha256=_membership_sha256(assignments, policy_sha256),
        )


def test_split_result_rejects_one_sample_assigned_to_conflicting_splits() -> None:
    assignments = (
        SplitAssignment(sample_id="sample-00", split="train"),
        SplitAssignment(sample_id="sample-00", split="test"),
    )
    policy_sha256 = "a" * 64

    with pytest.raises(ValueError, match="sample-00.*conflicting splits"):
        SplitResult(
            assignments=assignments,
            policy_sha256=policy_sha256,
            membership_sha256=_membership_sha256(assignments, policy_sha256),
        )


@pytest.mark.parametrize(
    ("policy_sha256", "membership_sha256", "message"),
    [
        ("not-a-digest", "b" * 64, "policy sha256"),
        ("a" * 64, "not-a-digest", "membership sha256"),
    ],
)
def test_split_result_rejects_malformed_digests(
    policy_sha256: str,
    membership_sha256: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SplitResult(
            assignments=(SplitAssignment(sample_id="sample-00", split="train"),),
            policy_sha256=policy_sha256,
            membership_sha256=membership_sha256,
        )


def test_split_result_rejects_digest_that_does_not_authenticate_membership() -> None:
    with pytest.raises(ValueError, match="does not match assignments"):
        SplitResult(
            assignments=(SplitAssignment(sample_id="sample-00", split="train"),),
            policy_sha256="a" * 64,
            membership_sha256="b" * 64,
        )


def test_split_policy_rejects_string_mode() -> None:
    with pytest.raises(TypeError, match="SplitMode"):
        SplitPolicy(mode="cold_drug", seed=1)  # type: ignore[arg-type]
