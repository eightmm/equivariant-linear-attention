from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest
import torch

import equivariant_attention
from equivariant_attention.sbdd import (
    AtomRole,
    BackboneRole,
    BondOrder,
    BondStereo,
    Hybridization,
    LigandBondFeatures,
    ProteinResidue,
    ProtonationState,
    SBDDAtomFeatures,
    SBDDFeatureSchema,
)


def _protein_atom() -> SBDDAtomFeatures:
    return SBDDAtomFeatures(
        atom_id="A:42:CA",
        atomic_number=6,
        formal_charge=0,
        donor=False,
        acceptor=False,
        aromatic=False,
        hybridization=Hybridization.SP3,
        role=AtomRole.PROTEIN,
        residue=ProteinResidue.ALA,
        residue_index=42,
        atom_name="CA",
        chain_id="A",
        backbone_role=BackboneRole.CA,
    )


def _ligand_atom() -> SBDDAtomFeatures:
    return SBDDAtomFeatures(
        atom_id="L:7",
        atomic_number=7,
        formal_charge=1,
        donor=True,
        acceptor=False,
        aromatic=True,
        hybridization=Hybridization.SP2,
        role=AtomRole.LIGAND,
        ligand_atom_id="7",
        protonation=ProtonationState.PROTONATED,
    )


def test_feature_schema_encodes_required_atom_and_domain_fields_deterministically() -> (
    None
):
    schema = SBDDFeatureSchema.default()
    atoms = (_protein_atom(), _ligand_atom())

    first = schema.encode_atoms(atoms)
    second = schema.encode_atoms(atoms)

    assert schema.version == "sbdd-atom-v1"
    assert torch.equal(first.scalar_features, second.scalar_features)
    assert torch.equal(first.role_ids, torch.tensor([0, 1]))
    assert torch.equal(first.residue_indices, torch.tensor([42, -1]))
    assert first.atom_ids == ("A:42:CA", "L:7")
    assert first.scalar_features.shape == (2, len(schema.feature_names))
    for required in (
        "atomic_number_scaled",
        "formal_charge",
        "donor",
        "acceptor",
        "aromatic",
        "hybridization=sp2",
        "role=ligand",
        "residue=ALA",
        "backbone=CA",
        "protonation=protonated",
    ):
        assert required in schema.feature_names


def test_biological_vocabularies_are_immutable_and_live_under_sbdd_namespace() -> None:
    atom = _protein_atom()
    assert AtomRole.__module__.startswith("equivariant_attention.sbdd")
    assert ProteinResidue.__module__.startswith("equivariant_attention.sbdd")
    with pytest.raises(FrozenInstanceError):
        atom.atomic_number = 8  # type: ignore[misc]


def test_biological_vocabularies_are_not_reexported_by_generic_core_api() -> None:
    assert not hasattr(equivariant_attention, "AtomRole")
    assert not hasattr(equivariant_attention, "ProteinResidue")
    assert not hasattr(equivariant_attention, "SBDDFeatureSchema")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"role": AtomRole.PROTEIN, "residue": None},
            "protein.*residue",
        ),
        (
            {"role": AtomRole.LIGAND, "ligand_atom_id": None},
            "ligand_atom_id",
        ),
        (
            {"atomic_number": 0},
            "atomic_number",
        ),
    ],
)
def test_atom_feature_schema_fails_on_missing_scientific_fields(
    kwargs: dict[str, object],
    match: str,
) -> None:
    values: dict[str, object] = {
        "atom_id": "x",
        "atomic_number": 6,
        "formal_charge": 0,
        "donor": False,
        "acceptor": False,
        "aromatic": False,
        "hybridization": Hybridization.SP3,
        "role": AtomRole.PROTEIN,
        "residue": ProteinResidue.GLY,
        "residue_index": 1,
        "atom_name": "CA",
        "chain_id": "A",
        "backbone_role": BackboneRole.CA,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=match):
        SBDDAtomFeatures(**values)  # type: ignore[arg-type]


def test_ligand_bond_schema_preserves_order_stereo_and_protonation_context() -> None:
    schema = SBDDFeatureSchema.default()
    bonds = (
        LigandBondFeatures(
            source_atom_id="L:1",
            target_atom_id="L:2",
            order=BondOrder.DOUBLE,
            stereo=BondStereo.E,
            aromatic=False,
            conjugated=True,
            in_ring=False,
        ),
    )

    encoded = schema.encode_ligand_bonds(bonds, atom_ids=("L:1", "L:2"))

    assert encoded.edge_index.tolist() == [[0], [1]]
    assert encoded.edge_features.shape == (1, len(schema.bond_feature_names))
    assert (
        encoded.edge_features[0, schema.bond_feature_names.index("order=double")] == 1
    )
    assert encoded.edge_features[0, schema.bond_feature_names.index("stereo=E")] == 1


def test_bond_encoder_rejects_unknown_atom_references() -> None:
    schema = SBDDFeatureSchema.default()
    bond = LigandBondFeatures(
        source_atom_id="L:1",
        target_atom_id="missing",
        order=BondOrder.SINGLE,
        stereo=BondStereo.NONE,
    )
    with pytest.raises(ValueError, match="unknown atom"):
        schema.encode_ligand_bonds((bond,), atom_ids=("L:1",))


def test_feature_contracts_reject_string_enum_bypasses() -> None:
    with pytest.raises(TypeError, match="Hybridization"):
        replace(
            _protein_atom(),
            hybridization="sp3",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="AtomRole"):
        replace(
            _protein_atom(),
            role="protein",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="BondOrder"):
        LigandBondFeatures(
            source_atom_id="L:1",
            target_atom_id="L:2",
            order="single",  # type: ignore[arg-type]
            stereo=BondStereo.NONE,
        )
