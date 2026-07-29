"""SBDD vocabulary and deterministic tensor adapters.

The generic model never imports this module.  Encoded outputs deliberately use
plain floating tensors and integer role identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch


class AtomRole(StrEnum):
    PROTEIN = "protein"
    LIGAND = "ligand"
    WATER = "water"
    METAL = "metal"
    COFACTOR = "cofactor"
    OTHER = "other"


class Hybridization(StrEnum):
    S = "s"
    SP = "sp"
    SP2 = "sp2"
    SP3 = "sp3"
    SP3D = "sp3d"
    SP3D2 = "sp3d2"
    UNKNOWN = "unknown"


class ProteinResidue(StrEnum):
    ALA = "ALA"
    ARG = "ARG"
    ASN = "ASN"
    ASP = "ASP"
    CYS = "CYS"
    GLN = "GLN"
    GLU = "GLU"
    GLY = "GLY"
    HIS = "HIS"
    ILE = "ILE"
    LEU = "LEU"
    LYS = "LYS"
    MET = "MET"
    PHE = "PHE"
    PRO = "PRO"
    SER = "SER"
    THR = "THR"
    TRP = "TRP"
    TYR = "TYR"
    VAL = "VAL"
    SEC = "SEC"
    PYL = "PYL"
    UNKNOWN = "UNK"


class BackboneRole(StrEnum):
    NONE = "none"
    N = "N"
    CA = "CA"
    C = "C"
    OXYGEN = "O"
    OXT = "OXT"
    SIDECHAIN = "sidechain"


class ProtonationState(StrEnum):
    PROTONATED = "protonated"
    DEPROTONATED = "deprotonated"
    NEUTRAL = "neutral"
    ZWITTERIONIC = "zwitterionic"
    UNKNOWN = "unknown"


class BondOrder(StrEnum):
    SINGLE = "single"
    DOUBLE = "double"
    TRIPLE = "triple"
    AROMATIC = "aromatic"
    DATIVE = "dative"
    UNKNOWN = "unknown"


class BondStereo(StrEnum):
    NONE = "none"
    E = "E"
    Z = "Z"
    UP = "up"
    DOWN = "down"
    UNKNOWN = "unknown"


def _require_nonempty(name: str, value: str | None) -> None:
    if value is None or not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_enum(
    name: str,
    value: object,
    enum_type: type[StrEnum],
) -> None:
    if not isinstance(value, enum_type):
        raise TypeError(f"{name} must be a {enum_type.__name__}")


@dataclass(frozen=True)
class SBDDAtomFeatures:
    atom_id: str
    atomic_number: int
    formal_charge: int
    donor: bool
    acceptor: bool
    aromatic: bool
    hybridization: Hybridization
    role: AtomRole
    residue: ProteinResidue | None = None
    residue_index: int | None = None
    atom_name: str | None = None
    chain_id: str | None = None
    backbone_role: BackboneRole = BackboneRole.NONE
    ligand_atom_id: str | None = None
    protonation: ProtonationState = ProtonationState.UNKNOWN

    def __post_init__(self) -> None:
        _require_enum("hybridization", self.hybridization, Hybridization)
        _require_enum("role", self.role, AtomRole)
        _require_enum("backbone_role", self.backbone_role, BackboneRole)
        _require_enum("protonation", self.protonation, ProtonationState)
        if self.residue is not None:
            _require_enum("residue", self.residue, ProteinResidue)
        _require_nonempty("atom_id", self.atom_id)
        if (
            isinstance(self.atomic_number, bool)
            or not isinstance(self.atomic_number, int)
            or not 1 <= self.atomic_number <= 118
        ):
            raise ValueError("atomic_number must be an integer in [1, 118]")
        if isinstance(self.formal_charge, bool) or not isinstance(
            self.formal_charge, int
        ):
            raise ValueError("formal_charge must be an integer")
        for name in ("donor", "acceptor", "aromatic"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if self.role is AtomRole.PROTEIN:
            if self.residue is None:
                raise ValueError("protein atom requires residue")
            if (
                isinstance(self.residue_index, bool)
                or not isinstance(self.residue_index, int)
                or self.residue_index < 0
            ):
                raise ValueError("protein atom requires nonnegative residue_index")
            _require_nonempty("protein atom_name", self.atom_name)
            _require_nonempty("protein chain_id", self.chain_id)
            if self.backbone_role is BackboneRole.NONE:
                raise ValueError("protein atom requires explicit backbone role")
            if self.ligand_atom_id is not None:
                raise ValueError("protein atom cannot declare ligand_atom_id")
        elif self.role is AtomRole.LIGAND:
            _require_nonempty("ligand_atom_id", self.ligand_atom_id)
            if self.residue is not None or self.residue_index is not None:
                raise ValueError("ligand atom cannot declare protein residue fields")
            if self.backbone_role is not BackboneRole.NONE:
                raise ValueError("ligand atom cannot declare a backbone role")
        else:
            if self.residue is not None or self.residue_index is not None:
                raise ValueError(
                    "non-protein atom cannot declare protein residue fields"
                )
            if self.ligand_atom_id is not None:
                raise ValueError("non-ligand atom cannot declare ligand_atom_id")


@dataclass(frozen=True)
class LigandBondFeatures:
    source_atom_id: str
    target_atom_id: str
    order: BondOrder
    stereo: BondStereo
    aromatic: bool = False
    conjugated: bool = False
    in_ring: bool = False

    def __post_init__(self) -> None:
        _require_enum("order", self.order, BondOrder)
        _require_enum("stereo", self.stereo, BondStereo)
        _require_nonempty("source_atom_id", self.source_atom_id)
        _require_nonempty("target_atom_id", self.target_atom_id)
        if self.source_atom_id == self.target_atom_id:
            raise ValueError("ligand bond cannot be a self bond")
        for name in ("aromatic", "conjugated", "in_ring"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if self.order is BondOrder.AROMATIC and not self.aromatic:
            raise ValueError("aromatic bond order requires aromatic=True")


@dataclass(frozen=True)
class EncodedAtoms:
    scalar_features: torch.Tensor
    role_ids: torch.Tensor
    residue_indices: torch.Tensor
    atom_ids: tuple[str, ...]
    atom_names: tuple[str | None, ...]
    chain_ids: tuple[str | None, ...]


@dataclass(frozen=True)
class EncodedLigandBonds:
    edge_index: torch.Tensor
    edge_features: torch.Tensor


@dataclass(frozen=True)
class SBDDFeatureSchema:
    """A versioned, ordered feature vocabulary with no learned state."""

    version: str
    roles: tuple[AtomRole, ...]
    hybridizations: tuple[Hybridization, ...]
    residues: tuple[ProteinResidue, ...]
    backbone_roles: tuple[BackboneRole, ...]
    protonation_states: tuple[ProtonationState, ...]
    bond_orders: tuple[BondOrder, ...]
    bond_stereos: tuple[BondStereo, ...]

    @classmethod
    def default(cls) -> SBDDFeatureSchema:
        return cls(
            version="sbdd-atom-v1",
            roles=tuple(AtomRole),
            hybridizations=tuple(Hybridization),
            residues=tuple(ProteinResidue),
            backbone_roles=tuple(BackboneRole),
            protonation_states=tuple(ProtonationState),
            bond_orders=tuple(BondOrder),
            bond_stereos=tuple(BondStereo),
        )

    def __post_init__(self) -> None:
        _require_nonempty("feature schema version", self.version)
        for name in (
            "roles",
            "hybridizations",
            "residues",
            "backbone_roles",
            "protonation_states",
            "bond_orders",
            "bond_stereos",
        ):
            values = getattr(self, name)
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{name} must be non-empty and unique")
        for name, values, enum_type in (
            ("roles", self.roles, AtomRole),
            ("hybridizations", self.hybridizations, Hybridization),
            ("residues", self.residues, ProteinResidue),
            ("backbone_roles", self.backbone_roles, BackboneRole),
            ("protonation_states", self.protonation_states, ProtonationState),
            ("bond_orders", self.bond_orders, BondOrder),
            ("bond_stereos", self.bond_stereos, BondStereo),
        ):
            if any(not isinstance(value, enum_type) for value in values):
                raise TypeError(
                    f"{name} must contain only {enum_type.__name__} values"
                )

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (
            "atomic_number_scaled",
            "formal_charge",
            "donor",
            "acceptor",
            "aromatic",
            *(f"hybridization={item.value}" for item in self.hybridizations),
            *(f"role={item.value}" for item in self.roles),
            *(f"residue={item.value}" for item in self.residues),
            *(f"backbone={item.value}" for item in self.backbone_roles),
            *(f"protonation={item.value}" for item in self.protonation_states),
        )

    @property
    def bond_feature_names(self) -> tuple[str, ...]:
        return (
            *(f"order={item.value}" for item in self.bond_orders),
            *(f"stereo={item.value}" for item in self.bond_stereos),
            "aromatic",
            "conjugated",
            "in_ring",
        )

    def encode_atoms(
        self,
        atoms: tuple[SBDDAtomFeatures, ...],
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> EncodedAtoms:
        if not atoms:
            raise ValueError("at least one atom is required")
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise TypeError("atom feature dtype must be floating point")
        atom_ids = tuple(atom.atom_id for atom in atoms)
        if len(set(atom_ids)) != len(atom_ids):
            raise ValueError("atom_id values must be unique")
        role_index = {value: index for index, value in enumerate(self.roles)}
        hybrid_index = {value: index for index, value in enumerate(self.hybridizations)}
        residue_index = {value: index for index, value in enumerate(self.residues)}
        backbone_index = {
            value: index for index, value in enumerate(self.backbone_roles)
        }
        protonation_index = {
            value: index for index, value in enumerate(self.protonation_states)
        }
        features = torch.zeros(
            (len(atoms), len(self.feature_names)),
            dtype=dtype,
            device=device,
        )
        roles = torch.empty(len(atoms), dtype=torch.long, device=device)
        hierarchy = torch.full(
            (len(atoms),),
            -1,
            dtype=torch.long,
            device=device,
        )
        hybrid_start = 5
        role_start = hybrid_start + len(self.hybridizations)
        residue_start = role_start + len(self.roles)
        backbone_start = residue_start + len(self.residues)
        protonation_start = backbone_start + len(self.backbone_roles)
        for row, atom in enumerate(atoms):
            if atom.role not in role_index:
                raise ValueError(f"role {atom.role!s} is absent from schema")
            if atom.hybridization not in hybrid_index:
                raise ValueError(
                    f"hybridization {atom.hybridization!s} is absent from schema"
                )
            features[row, :5] = torch.tensor(
                [
                    atom.atomic_number / 118.0,
                    float(atom.formal_charge),
                    float(atom.donor),
                    float(atom.acceptor),
                    float(atom.aromatic),
                ],
                dtype=dtype,
                device=device,
            )
            features[row, hybrid_start + hybrid_index[atom.hybridization]] = 1
            features[row, role_start + role_index[atom.role]] = 1
            roles[row] = role_index[atom.role]
            if atom.residue is not None:
                if atom.residue not in residue_index:
                    raise ValueError(f"residue {atom.residue!s} is absent from schema")
                features[row, residue_start + residue_index[atom.residue]] = 1
                assert atom.residue_index is not None
                hierarchy[row] = atom.residue_index
            features[row, backbone_start + backbone_index[atom.backbone_role]] = 1
            features[row, protonation_start + protonation_index[atom.protonation]] = 1
        return EncodedAtoms(
            scalar_features=features,
            role_ids=roles,
            residue_indices=hierarchy,
            atom_ids=atom_ids,
            atom_names=tuple(atom.atom_name for atom in atoms),
            chain_ids=tuple(atom.chain_id for atom in atoms),
        )

    def encode_ligand_bonds(
        self,
        bonds: tuple[LigandBondFeatures, ...],
        *,
        atom_ids: tuple[str, ...],
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> EncodedLigandBonds:
        if len(set(atom_ids)) != len(atom_ids):
            raise ValueError("atom_ids must be unique")
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise TypeError("bond feature dtype must be floating point")
        atom_index = {atom_id: index for index, atom_id in enumerate(atom_ids)}
        order_index = {value: index for index, value in enumerate(self.bond_orders)}
        stereo_index = {value: index for index, value in enumerate(self.bond_stereos)}
        edge_index = torch.empty((2, len(bonds)), dtype=torch.long, device=device)
        edge_features = torch.zeros(
            (len(bonds), len(self.bond_feature_names)),
            dtype=dtype,
            device=device,
        )
        stereo_start = len(self.bond_orders)
        flag_start = stereo_start + len(self.bond_stereos)
        seen: set[tuple[str, str]] = set()
        for row, bond in enumerate(bonds):
            if (
                bond.source_atom_id not in atom_index
                or bond.target_atom_id not in atom_index
            ):
                raise ValueError("ligand bond references unknown atom")
            key = tuple(sorted((bond.source_atom_id, bond.target_atom_id)))
            if key in seen:
                raise ValueError("duplicate undirected ligand bond")
            seen.add(key)
            edge_index[:, row] = torch.tensor(
                [
                    atom_index[bond.source_atom_id],
                    atom_index[bond.target_atom_id],
                ],
                dtype=torch.long,
                device=device,
            )
            edge_features[row, order_index[bond.order]] = 1
            edge_features[row, stereo_start + stereo_index[bond.stereo]] = 1
            edge_features[row, flag_start:] = torch.tensor(
                [bond.aromatic, bond.conjugated, bond.in_ring],
                dtype=dtype,
                device=device,
            )
        return EncodedLigandBonds(
            edge_index=edge_index,
            edge_features=edge_features,
        )
