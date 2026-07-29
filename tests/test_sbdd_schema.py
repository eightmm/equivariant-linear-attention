from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import math

import pytest
import torch

from equivariant_attention.sbdd import (
    AssayContext,
    ConformationSource,
    EntityIdentifiers,
    ForcePredictionPolicy,
    LabelDirection,
    LabelKind,
    LabelQualifier,
    LabelStandardizer,
    NegativeProvenance,
    PoseAnnotation,
    PoseClass,
    PoseClassThresholds,
    PredictionTask,
    PredictionUnit,
    ReplicateType,
    SBDDSampleContract,
    ScientificLabel,
    ScreeningOutcome,
    StructureIntake,
)


def _entities(*, pose_id: str | None = "pose-1") -> EntityIdentifiers:
    return EntityIdentifiers(
        sample_id="sample-1",
        ligand_id="lig-raw-1",
        standardized_ligand_id="lig-parent-1",
        protein_id="uniprot:P1",
        construct_id="P1:construct-a",
        complex_id="complex-1",
        assembly_id="assembly-1",
        pocket_id="pocket-1",
        pose_id=pose_id,
        ligand_cluster_id="scaffold-a",
        protein_cluster_id="family-a",
        pose_group_id="pair-1-poses",
        replicate_group_id="assay-replicates-1",
        derived_view_group_id="complex-1-views",
    )


def _structure(
    source: ConformationSource = ConformationSource.HOLO_BOUND,
) -> StructureIntake:
    return StructureIntake(
        structure_id="pdb:1ABC",
        conformation_source=source,
        coordinate_source="x-ray coordinates",
        assembly_id="assembly-1",
        pocket_definition="residues within 6 A of reference ligand",
        altloc_policy="highest occupancy; ties lexicographic",
        insertion_code_policy="preserve author residue key",
        nonstandard_residue_policy="retain with explicit residue token",
        missing_residue_policy="record and omit absent atoms",
        chain_break_policy="record TER and sequence discontinuity",
        sequence_alignment_id="sha256:alignment",
        experimental_method="x-ray diffraction",
        confidence_source="experimental resolution 1.8 A",
        cofactor_ids=("HEM:A:501",),
        water_ids=("HOH:A:901",),
        ion_ids=("ZN:A:700",),
        bound_ligand_id=(
            "lig-raw-1" if source is ConformationSource.HOLO_BOUND else None
        ),
    )


def _assay() -> AssayContext:
    return AssayContext(
        assay_id="assay-1",
        endpoint="pKd",
        unit="pK",
        species="Homo sapiens",
        method="radioligand displacement",
        conditions=(("temperature", "298 K"), ("pH", "7.4")),
    )


def _affinity_label() -> ScientificLabel:
    return ScientificLabel(
        kind=LabelKind.AFFINITY,
        qualifier=LabelQualifier.EXACT,
        direction=LabelDirection.HIGHER_IS_STRONGER,
        unit="pK",
        value=7.2,
        raw_value="7.2",
        assay=_assay(),
        replicate_type=ReplicateType.BIOLOGICAL,
        replicate_id="replicate-2",
    )


def test_affinity_contract_is_immutable_and_preserves_provenance() -> None:
    contract = SBDDSampleContract(
        task=PredictionTask.AFFINITY,
        prediction_unit=PredictionUnit.COMPLEX,
        entities=_entities(),
        structure=_structure(),
        label=_affinity_label(),
        source="PDBbind",
        source_snapshot="2020-refined",
        processing_version="sbdd-v1",
    )

    assert contract.structure.water_ids == ("HOH:A:901",)
    assert contract.label is not None
    assert contract.label.assay is not None
    assert contract.label.assay.conditions[1] == ("pH", "7.4")
    with pytest.raises(FrozenInstanceError):
        contract.source = "changed"  # type: ignore[misc]


def test_affinity_contract_rejects_docking_score_as_label() -> None:
    docking_score = ScientificLabel(
        kind=LabelKind.DOCKING_SCORE,
        qualifier=LabelQualifier.EXACT,
        direction=LabelDirection.LOWER_IS_STRONGER,
        unit="kcal/mol",
        value=-8.0,
        raw_value="-8.0",
    )
    with pytest.raises(ValueError, match="docking score.*affinity"):
        SBDDSampleContract(
            task=PredictionTask.AFFINITY,
            prediction_unit=PredictionUnit.COMPLEX,
            entities=_entities(),
            structure=_structure(),
            label=docking_score,
            source="dock",
            source_snapshot="run-1",
            processing_version="v1",
        )


@pytest.mark.parametrize(
    ("active", "provenance", "match"),
    [
        (False, NegativeProvenance.UNKNOWN, "unknown.*not a negative"),
        (False, NegativeProvenance.WRONG_POSE, "wrong pose.*not a screening"),
        (True, NegativeProvenance.MEASURED_INACTIVE, "active"),
        (None, NegativeProvenance.MEASURED_INACTIVE, "unknown outcome"),
    ],
)
def test_screening_outcome_never_turns_missing_or_wrong_pose_into_nonbinder(
    active: bool | None,
    provenance: NegativeProvenance,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        ScreeningOutcome(active=active, negative_provenance=provenance)


def test_screening_contract_keeps_measured_inactive_provenance() -> None:
    contract = SBDDSampleContract(
        task=PredictionTask.VIRTUAL_SCREENING,
        prediction_unit=PredictionUnit.PAIR,
        entities=_entities(pose_id=None),
        structure=_structure(ConformationSource.APO),
        screening_outcome=ScreeningOutcome(
            active=False,
            negative_provenance=NegativeProvenance.MEASURED_INACTIVE,
        ),
        source="prospective campaign",
        source_snapshot="2026-06-01",
        processing_version="screen-v2",
    )

    assert contract.screening_outcome is not None
    assert not contract.screening_outcome.active


def test_pose_classes_are_explicit_and_consistent_with_dataset_thresholds() -> None:
    thresholds = PoseClassThresholds(
        native_max_rmsd=2.0,
        near_native_max_rmsd=4.0,
    )
    pose = PoseAnnotation(
        pose_id="pose-1",
        pose_group_id="complex-1-poses",
        pose_class=PoseClass.NEAR_NATIVE,
        rmsd=3.0,
        rmsd_reference_id="crystal-pose-1",
        generation_method="frozen-docker-v2",
        thresholds=thresholds,
    )

    assert pose.pose_class is PoseClass.NEAR_NATIVE
    assert thresholds.classify(7.0) is PoseClass.WRONG_POSE
    with pytest.raises(ValueError, match="disagrees"):
        PoseAnnotation(
            pose_id="pose-2",
            pose_group_id="complex-1-poses",
            pose_class=PoseClass.NATIVE,
            rmsd=7.0,
            rmsd_reference_id="crystal-pose-1",
            generation_method="frozen-docker-v2",
            thresholds=thresholds,
        )


def test_scientific_label_requires_explicit_censoring_bounds() -> None:
    upper = ScientificLabel(
        kind=LabelKind.AFFINITY,
        qualifier=LabelQualifier.UPPER_BOUND,
        direction=LabelDirection.HIGHER_IS_STRONGER,
        unit="pK",
        upper=5.0,
        raw_value="<5",
        assay=_assay(),
    )
    assert upper.interval == (-math.inf, 5.0)
    with pytest.raises(ValueError, match="upper"):
        ScientificLabel(
            kind=LabelKind.AFFINITY,
            qualifier=LabelQualifier.UPPER_BOUND,
            direction=LabelDirection.HIGHER_IS_STRONGER,
            unit="pK",
            raw_value="<unknown",
        )


def test_label_standardizer_is_train_fitted_and_round_trips_original_units() -> None:
    train = torch.tensor([5.0, 7.0, 9.0], dtype=torch.float64)
    scaler = LabelStandardizer.fit_train(
        train,
        unit="pK",
        direction=LabelDirection.HIGHER_IS_STRONGER,
        training_membership_sha256="a" * 64,
    )
    transformed = scaler.transform(train)

    assert transformed.mean().item() == pytest.approx(0.0)
    assert torch.equal(scaler.inverse_transform(transformed), train)
    assert scaler.unit == "pK"
    with pytest.raises(ValueError, match="training membership"):
        LabelStandardizer.fit_train(
            train,
            unit="pK",
            direction=LabelDirection.HIGHER_IS_STRONGER,
            training_membership_sha256="not-a-hash",
        )


def test_potential_contract_defaults_to_conservative_force_policy() -> None:
    contract = SBDDSampleContract(
        task=PredictionTask.POTENTIAL,
        prediction_unit=PredictionUnit.SYSTEM,
        entities=_entities(pose_id=None),
        structure=_structure(ConformationSource.MD_FRAME),
        label=ScientificLabel(
            kind=LabelKind.ENERGY,
            qualifier=LabelQualifier.EXACT,
            direction=LabelDirection.LOWER_IS_STRONGER,
            unit="kcal/mol",
            value=-10.0,
            raw_value="-10.0",
        ),
        source="MD",
        source_snapshot="trajectory-sha256",
        processing_version="potential-v1",
    )

    assert contract.force_policy is ForcePredictionPolicy.CONSERVATIVE_NEGATIVE_GRADIENT


def test_scientific_contracts_reject_string_enum_bypasses() -> None:
    with pytest.raises(TypeError, match="ConformationSource"):
        replace(
            _structure(),
            conformation_source="holo_bound",  # type: ignore[arg-type]
            bound_ligand_id=None,
        )
    with pytest.raises(TypeError, match="LabelKind"):
        replace(
            _affinity_label(),
            kind="affinity",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="LabelQualifier"):
        replace(
            _affinity_label(),
            qualifier="exact",  # type: ignore[arg-type]
            value=None,
            lower=1.0,
            upper=2.0,
        )
    with pytest.raises(TypeError, match="PredictionTask"):
        SBDDSampleContract(
            task="affinity",  # type: ignore[arg-type]
            prediction_unit=PredictionUnit.COMPLEX,
            entities=_entities(),
            structure=_structure(),
            label=_affinity_label(),
            source="source",
            source_snapshot="snapshot",
            processing_version="version",
        )
    with pytest.raises(TypeError, match="NegativeProvenance"):
        ScreeningOutcome(
            active=False,
            negative_provenance="measured_inactive",  # type: ignore[arg-type]
        )
