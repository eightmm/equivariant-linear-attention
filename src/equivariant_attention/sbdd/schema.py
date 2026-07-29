"""Immutable scientific contracts for structure-based drug-design tasks.

This module deliberately owns biological vocabulary.  The generic equivariant
core remains domain agnostic and consumes only tensors, masks, and integer
relation/role identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
import re

import torch


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PredictionTask(StrEnum):
    AFFINITY = "affinity"
    POSE_RANKING = "pose_ranking"
    POSE_REFINEMENT = "pose_refinement"
    VIRTUAL_SCREENING = "virtual_screening"
    POTENTIAL = "potential"


class PredictionUnit(StrEnum):
    COMPLEX = "complex"
    POSE = "pose"
    PAIR = "pair"
    LIGAND_ATOM = "ligand_atom"
    SYSTEM = "system"


class ConformationSource(StrEnum):
    APO = "apo"
    HOLO_BOUND = "holo_bound"
    DOCKED = "docked"
    GENERATED = "generated"
    PREDICTED = "predicted"
    MD_FRAME = "md_frame"


class NegativeProvenance(StrEnum):
    MEASURED_INACTIVE = "measured_inactive"
    VERIFIED_NON_BINDER = "verified_non_binder"
    WRONG_POSE = "wrong_pose"
    PROPERTY_MATCHED_DECOY = "property_matched_decoy"
    RANDOM_PAIR = "random_pair"
    UNKNOWN = "unknown"


class PoseClass(StrEnum):
    NATIVE = "native"
    NEAR_NATIVE = "near_native"
    WRONG_POSE = "wrong_pose"


class LabelKind(StrEnum):
    AFFINITY = "affinity"
    DOCKING_SCORE = "docking_score"
    POSE_RMSD = "pose_rmsd"
    BINARY_ACTIVITY = "binary_activity"
    ENERGY = "energy"
    FORCE = "force"


class LabelDirection(StrEnum):
    HIGHER_IS_STRONGER = "higher_is_stronger"
    LOWER_IS_STRONGER = "lower_is_stronger"


class LabelQualifier(StrEnum):
    EXACT = "exact"
    LOWER_BOUND = "lower_bound"
    UPPER_BOUND = "upper_bound"
    INTERVAL = "interval"


class ReplicateType(StrEnum):
    TECHNICAL = "technical"
    BIOLOGICAL = "biological"
    INDEPENDENT_ASSAY = "independent_assay"
    UNKNOWN = "unknown"


class ForcePredictionPolicy(StrEnum):
    CONSERVATIVE_NEGATIVE_GRADIENT = "conservative_negative_gradient"
    DIRECT_AUXILIARY_ONLY = "direct_auxiliary_only"


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_optional_text(name: str, value: str | None) -> None:
    if value is not None:
        _require_text(name, value)


def _require_enum(
    name: str,
    value: object,
    enum_type: type[StrEnum],
) -> None:
    if not isinstance(value, enum_type):
        raise TypeError(f"{name} must be a {enum_type.__name__}")


@dataclass(frozen=True)
class EntityIdentifiers:
    """Stable IDs and label-blind grouping keys for one scientific sample."""

    sample_id: str
    ligand_id: str
    standardized_ligand_id: str
    protein_id: str
    construct_id: str
    complex_id: str
    assembly_id: str
    pocket_id: str
    pose_id: str | None
    ligand_cluster_id: str
    protein_cluster_id: str
    pose_group_id: str
    replicate_group_id: str
    derived_view_group_id: str
    campaign_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "sample_id",
            "ligand_id",
            "standardized_ligand_id",
            "protein_id",
            "construct_id",
            "complex_id",
            "assembly_id",
            "pocket_id",
            "ligand_cluster_id",
            "protein_cluster_id",
            "pose_group_id",
            "replicate_group_id",
            "derived_view_group_id",
        ):
            _require_text(name, getattr(self, name))
        _require_optional_text("pose_id", self.pose_id)
        _require_optional_text("campaign_id", self.campaign_id)


@dataclass(frozen=True)
class StructureIntake:
    """Receipt for structural decisions that otherwise create silent leakage."""

    structure_id: str
    conformation_source: ConformationSource
    coordinate_source: str
    assembly_id: str
    pocket_definition: str
    altloc_policy: str
    insertion_code_policy: str
    nonstandard_residue_policy: str
    missing_residue_policy: str
    chain_break_policy: str
    sequence_alignment_id: str
    experimental_method: str
    confidence_source: str | None = None
    cofactor_ids: tuple[str, ...] = ()
    water_ids: tuple[str, ...] = ()
    ion_ids: tuple[str, ...] = ()
    bound_ligand_id: str | None = None

    def __post_init__(self) -> None:
        _require_enum(
            "conformation_source",
            self.conformation_source,
            ConformationSource,
        )
        for name in (
            "structure_id",
            "coordinate_source",
            "assembly_id",
            "pocket_definition",
            "altloc_policy",
            "insertion_code_policy",
            "nonstandard_residue_policy",
            "missing_residue_policy",
            "chain_break_policy",
            "sequence_alignment_id",
            "experimental_method",
        ):
            _require_text(name, getattr(self, name))
        _require_optional_text("confidence_source", self.confidence_source)
        _require_optional_text("bound_ligand_id", self.bound_ligand_id)
        for collection_name in ("cofactor_ids", "water_ids", "ion_ids"):
            values = getattr(self, collection_name)
            if len(set(values)) != len(values):
                raise ValueError(f"{collection_name} must not contain duplicates")
            for value in values:
                _require_text(collection_name, value)
        if (
            self.conformation_source is ConformationSource.HOLO_BOUND
            and self.bound_ligand_id is None
        ):
            raise ValueError("holo-bound structure requires bound_ligand_id")
        if (
            self.conformation_source is ConformationSource.APO
            and self.bound_ligand_id is not None
        ):
            raise ValueError("apo structure cannot declare a bound ligand")


@dataclass(frozen=True)
class AssayContext:
    assay_id: str
    endpoint: str
    unit: str
    species: str
    method: str
    conditions: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for name in ("assay_id", "endpoint", "unit", "species", "method"):
            _require_text(name, getattr(self, name))
        keys: set[str] = set()
        for key, value in self.conditions:
            _require_text("condition key", key)
            _require_text("condition value", value)
            if key in keys:
                raise ValueError("assay condition keys must be unique")
            keys.add(key)


@dataclass(frozen=True)
class ScientificLabel:
    """Original-unit label including censoring and replicate semantics."""

    kind: LabelKind
    qualifier: LabelQualifier
    direction: LabelDirection
    unit: str
    raw_value: str
    value: float | None = None
    lower: float | None = None
    upper: float | None = None
    assay: AssayContext | None = None
    replicate_type: ReplicateType = ReplicateType.UNKNOWN
    replicate_id: str | None = None

    def __post_init__(self) -> None:
        _require_enum("kind", self.kind, LabelKind)
        _require_enum("qualifier", self.qualifier, LabelQualifier)
        _require_enum("direction", self.direction, LabelDirection)
        _require_enum("replicate_type", self.replicate_type, ReplicateType)
        if self.assay is not None and not isinstance(
            self.assay,
            AssayContext,
        ):
            raise TypeError("assay must be an AssayContext")
        _require_text("unit", self.unit)
        _require_text("raw_value", self.raw_value)
        _require_optional_text("replicate_id", self.replicate_id)
        for name in ("value", "lower", "upper"):
            number = getattr(self, name)
            if number is not None and not math.isfinite(float(number)):
                raise ValueError(f"{name} must be finite when provided")
        if self.qualifier is LabelQualifier.EXACT:
            if self.value is None:
                raise ValueError("exact label requires value")
            if self.lower is not None or self.upper is not None:
                raise ValueError("exact label cannot also declare interval bounds")
        elif self.qualifier is LabelQualifier.LOWER_BOUND:
            if self.lower is None:
                raise ValueError("lower-bound label requires lower")
            if self.value is not None or self.upper is not None:
                raise ValueError("lower-bound label accepts only lower")
        elif self.qualifier is LabelQualifier.UPPER_BOUND:
            if self.upper is None:
                raise ValueError("upper-bound label requires upper")
            if self.value is not None or self.lower is not None:
                raise ValueError("upper-bound label accepts only upper")
        else:
            if self.lower is None or self.upper is None:
                raise ValueError("interval label requires lower and upper")
            if self.lower > self.upper:
                raise ValueError("label interval must satisfy lower <= upper")
            if self.value is not None:
                raise ValueError("interval label cannot also declare exact value")
        if (
            self.assay is not None
            and self.assay.unit.strip().casefold() != self.unit.strip().casefold()
        ):
            raise ValueError("label and assay units must agree")

    @property
    def interval(self) -> tuple[float, float]:
        if self.qualifier is LabelQualifier.EXACT:
            assert self.value is not None
            return self.value, self.value
        if self.qualifier is LabelQualifier.LOWER_BOUND:
            assert self.lower is not None
            return self.lower, math.inf
        if self.qualifier is LabelQualifier.UPPER_BOUND:
            assert self.upper is not None
            return -math.inf, self.upper
        assert self.lower is not None and self.upper is not None
        return self.lower, self.upper


@dataclass(frozen=True)
class ScreeningOutcome:
    """A screening outcome where unknown means unknown, never negative."""

    active: bool | None
    negative_provenance: NegativeProvenance | None = None

    def __post_init__(self) -> None:
        if self.active is not None and not isinstance(self.active, bool):
            raise TypeError("active must be boolean or None")
        if self.negative_provenance is not None:
            _require_enum(
                "negative_provenance",
                self.negative_provenance,
                NegativeProvenance,
            )
        if self.active is True:
            if self.negative_provenance is not None:
                raise ValueError("active example cannot have negative provenance")
            return
        if self.active is None:
            if self.negative_provenance is not NegativeProvenance.UNKNOWN:
                raise ValueError(
                    "unknown outcome must carry unknown negative provenance"
                )
            return
        if self.negative_provenance is None:
            raise ValueError("inactive example requires negative provenance")
        if self.negative_provenance is NegativeProvenance.UNKNOWN:
            raise ValueError("unknown is not a negative or measured inactive")
        if self.negative_provenance is NegativeProvenance.WRONG_POSE:
            raise ValueError("wrong pose is not a screening non-binder")


@dataclass(frozen=True)
class PoseClassThresholds:
    """Dataset-declared RMSD thresholds for pose categories."""

    native_max_rmsd: float
    near_native_max_rmsd: float
    unit: str = "angstrom"

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.native_max_rmsd)
            or not math.isfinite(self.near_native_max_rmsd)
            or self.native_max_rmsd < 0
            or self.near_native_max_rmsd <= self.native_max_rmsd
        ):
            raise ValueError("pose thresholds require 0 <= native < near-native RMSD")
        _require_text("pose threshold unit", self.unit)

    def classify(self, rmsd: float) -> PoseClass:
        if not math.isfinite(rmsd) or rmsd < 0:
            raise ValueError("pose RMSD must be finite and nonnegative")
        if rmsd <= self.native_max_rmsd:
            return PoseClass.NATIVE
        if rmsd <= self.near_native_max_rmsd:
            return PoseClass.NEAR_NATIVE
        return PoseClass.WRONG_POSE


@dataclass(frozen=True)
class PoseAnnotation:
    pose_id: str
    pose_group_id: str
    pose_class: PoseClass
    rmsd: float
    rmsd_reference_id: str
    generation_method: str
    thresholds: PoseClassThresholds

    def __post_init__(self) -> None:
        _require_enum("pose_class", self.pose_class, PoseClass)
        if not isinstance(self.thresholds, PoseClassThresholds):
            raise TypeError("thresholds must be PoseClassThresholds")
        for name in (
            "pose_id",
            "pose_group_id",
            "rmsd_reference_id",
            "generation_method",
        ):
            _require_text(name, getattr(self, name))
        expected = self.thresholds.classify(self.rmsd)
        if self.pose_class is not expected:
            raise ValueError(
                f"pose_class {self.pose_class.value!r} disagrees with RMSD thresholds"
            )


@dataclass(frozen=True)
class LabelStandardizer:
    """Train-membership-bound affine transform with original-unit inverse."""

    mean: float
    scale: float
    unit: str
    direction: LabelDirection
    training_membership_sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.mean, bool)
            or not isinstance(self.mean, (int, float))
            or not math.isfinite(float(self.mean))
        ):
            raise ValueError("mean must be finite")
        if (
            isinstance(self.scale, bool)
            or not isinstance(self.scale, (int, float))
            or not math.isfinite(float(self.scale))
            or self.scale <= 0
        ):
            raise ValueError("scale must be finite and positive")
        _require_text("unit", self.unit)
        _require_enum("direction", self.direction, LabelDirection)
        if not _SHA256.fullmatch(self.training_membership_sha256):
            raise ValueError(
                "training membership sha256 must be 64 lowercase hex characters"
            )

    @classmethod
    def fit_train(
        cls,
        values: torch.Tensor,
        *,
        unit: str,
        direction: LabelDirection,
        training_membership_sha256: str,
    ) -> LabelStandardizer:
        if not _SHA256.fullmatch(training_membership_sha256):
            raise ValueError(
                "training membership sha256 must be 64 lowercase hex characters"
            )
        _require_text("unit", unit)
        if values.ndim != 1 or values.numel() < 2:
            raise ValueError(
                "train values must be one-dimensional with at least 2 rows"
            )
        if not torch.is_floating_point(values) or not bool(
            torch.isfinite(values).all()
        ):
            raise ValueError("train values must be finite floating point")
        mean = float(values.mean().item())
        scale = float(values.std(unbiased=True).item())
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(
                "train values must have positive finite sample standard deviation"
            )
        return cls(
            mean=mean,
            scale=scale,
            unit=unit,
            direction=direction,
            training_membership_sha256=training_membership_sha256,
        )

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        self._validate_values(values)
        return (values - self.mean) / self.scale

    def inverse_transform(self, values: torch.Tensor) -> torch.Tensor:
        self._validate_values(values)
        return values * self.scale + self.mean

    @staticmethod
    def _validate_values(values: torch.Tensor) -> None:
        if not torch.is_floating_point(values) or not bool(
            torch.isfinite(values).all()
        ):
            raise ValueError("label values must be finite floating point")


@dataclass(frozen=True)
class SBDDSampleContract:
    """Task-specific contract; it contains no model tensors."""

    task: PredictionTask
    prediction_unit: PredictionUnit
    entities: EntityIdentifiers
    structure: StructureIntake
    source: str
    source_snapshot: str
    processing_version: str
    label: ScientificLabel | None = None
    screening_outcome: ScreeningOutcome | None = None
    force_policy: ForcePredictionPolicy = (
        ForcePredictionPolicy.CONSERVATIVE_NEGATIVE_GRADIENT
    )

    def __post_init__(self) -> None:
        _require_enum("task", self.task, PredictionTask)
        _require_enum("prediction_unit", self.prediction_unit, PredictionUnit)
        _require_enum("force_policy", self.force_policy, ForcePredictionPolicy)
        if not isinstance(self.entities, EntityIdentifiers):
            raise TypeError("entities must be EntityIdentifiers")
        if not isinstance(self.structure, StructureIntake):
            raise TypeError("structure must be StructureIntake")
        if self.label is not None and not isinstance(
            self.label,
            ScientificLabel,
        ):
            raise TypeError("label must be ScientificLabel")
        if self.screening_outcome is not None and not isinstance(
            self.screening_outcome,
            ScreeningOutcome,
        ):
            raise TypeError("screening_outcome must be ScreeningOutcome")
        for name in ("source", "source_snapshot", "processing_version"):
            _require_text(name, getattr(self, name))
        if self.entities.assembly_id != self.structure.assembly_id:
            raise ValueError("entity and structure assembly IDs must agree")
        if self.task is PredictionTask.AFFINITY:
            if self.label is None:
                raise ValueError("affinity task requires an affinity label")
            if self.label.kind is LabelKind.DOCKING_SCORE:
                raise ValueError("docking score must not be used as an affinity label")
            if self.label.kind is not LabelKind.AFFINITY:
                raise ValueError("affinity task requires LabelKind.AFFINITY")
            if self.label.assay is None:
                raise ValueError("affinity label requires assay provenance")
            if self.screening_outcome is not None:
                raise ValueError("affinity and screening outcomes are separate tasks")
        elif self.task is PredictionTask.VIRTUAL_SCREENING:
            if self.screening_outcome is None:
                raise ValueError("virtual screening requires ScreeningOutcome")
            if self.label is not None:
                raise ValueError(
                    "virtual screening outcome is separate from affinity labels"
                )
        elif self.task is PredictionTask.POSE_RANKING:
            if self.entities.pose_id is None:
                raise ValueError("pose ranking requires pose_id")
            if self.label is None or self.label.kind is not LabelKind.POSE_RMSD:
                raise ValueError("pose ranking requires an RMSD label")
        elif self.task is PredictionTask.POSE_REFINEMENT:
            if self.entities.pose_id is None:
                raise ValueError("pose refinement requires pose_id")
        elif self.task is PredictionTask.POTENTIAL:
            if self.label is not None and self.label.kind not in {
                LabelKind.ENERGY,
                LabelKind.FORCE,
            }:
                raise ValueError("potential task accepts only energy or force labels")
            if self.force_policy is ForcePredictionPolicy.DIRECT_AUXILIARY_ONLY:
                raise ValueError(
                    "direct force is auxiliary-only and cannot be production force policy"
                )
