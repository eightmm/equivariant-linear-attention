from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from equivariant_attention.benchmarking import collate_graphs
from equivariant_attention.pdbbind import (
    ATOM3D_LBA_REVISION,
    atom3d_lba_row_to_sample,
)
from equivariant_attention.sbdd import (
    AssayContext,
    AtomRole,
    BackboneRole,
    ConformationSource,
    EntityIdentifiers,
    Hybridization,
    LabelDirection,
    LabelKind,
    LabelQualifier,
    PredictionTask,
    PredictionUnit,
    ProteinResidue,
    SBDDAtomFeatures,
    SBDDRelationSchema,
    SBDDSampleContract,
    ScientificLabel,
    StructureIntake,
    adapt_preencoded_graph_sample,
    adapt_sbdd_affinity_complex,
    censored_affinity_loss,
)


def _assay() -> AssayContext:
    return AssayContext(
        assay_id="assay-1",
        endpoint="pKd",
        unit="pK",
        species="Homo sapiens",
        method="competition binding",
    )


def _label(
    *,
    value: float | None = 7.0,
    qualifier: LabelQualifier = LabelQualifier.EXACT,
    lower: float | None = None,
    upper: float | None = None,
    unit: str = "pK",
) -> ScientificLabel:
    return ScientificLabel(
        kind=LabelKind.AFFINITY,
        qualifier=qualifier,
        direction=LabelDirection.HIGHER_IS_STRONGER,
        unit=unit,
        raw_value=(
            str(value)
            if qualifier is LabelQualifier.EXACT
            else f"{qualifier.value}:{lower}:{upper}"
        ),
        value=value if qualifier is LabelQualifier.EXACT else None,
        lower=lower,
        upper=upper,
        assay=replace(_assay(), unit=unit),
    )


def _contract(label: ScientificLabel) -> SBDDSampleContract:
    return SBDDSampleContract(
        task=PredictionTask.AFFINITY,
        prediction_unit=PredictionUnit.COMPLEX,
        entities=EntityIdentifiers(
            sample_id="sample-1",
            ligand_id="ligand-raw-1",
            standardized_ligand_id="ligand-parent-1",
            protein_id="protein-1",
            construct_id="protein-1:construct-a",
            complex_id="complex-1",
            assembly_id="assembly-1",
            pocket_id="pocket-1",
            pose_id="pose-1",
            ligand_cluster_id="scaffold-1",
            protein_cluster_id="family-1",
            pose_group_id="pose-group-1",
            replicate_group_id="replicate-group-1",
            derived_view_group_id="view-group-1",
        ),
        structure=StructureIntake(
            structure_id="pdb:1ABC",
            conformation_source=ConformationSource.HOLO_BOUND,
            coordinate_source="x-ray coordinates",
            assembly_id="assembly-1",
            pocket_definition="all atoms within 6 A of the bound ligand",
            altloc_policy="highest occupancy",
            insertion_code_policy="preserve author residue key",
            nonstandard_residue_policy="retain explicit token",
            missing_residue_policy="record and omit",
            chain_break_policy="record TER",
            sequence_alignment_id="alignment-sha256",
            experimental_method="x-ray diffraction",
            bound_ligand_id="ligand-raw-1",
        ),
        label=label,
        source="PDBbind",
        source_snapshot="2020-refined",
        processing_version="adapter-test-v1",
    )


def _atoms() -> tuple[SBDDAtomFeatures, ...]:
    protein = {
        "atomic_number": 6,
        "formal_charge": 0,
        "donor": False,
        "acceptor": False,
        "aromatic": False,
        "hybridization": Hybridization.SP3,
        "role": AtomRole.PROTEIN,
        "residue": ProteinResidue.ALA,
        "residue_index": 42,
        "chain_id": "A",
    }
    return (
        SBDDAtomFeatures(
            atom_id="A:42:CA",
            atom_name="CA",
            backbone_role=BackboneRole.CA,
            **protein,
        ),
        SBDDAtomFeatures(
            atom_id="A:42:CB",
            atom_name="CB",
            backbone_role=BackboneRole.SIDECHAIN,
            **protein,
        ),
        SBDDAtomFeatures(
            atom_id="L:1",
            atomic_number=8,
            formal_charge=-1,
            donor=False,
            acceptor=True,
            aromatic=False,
            hybridization=Hybridization.SP2,
            role=AtomRole.LIGAND,
            ligand_atom_id="1",
        ),
    )


def _edge_index() -> torch.Tensor:
    # receiver <- sender, with one self edge per node.
    return torch.tensor(
        [[0, 1, 2, 0, 2], [0, 1, 2, 2, 0]],
        dtype=torch.long,
    )


def test_rich_adapter_is_label_blind_and_preserves_generic_annotations() -> None:
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.0, 2.0, 0.0]],
        dtype=torch.float64,
    )
    first = adapt_sbdd_affinity_complex(
        _contract(_label(value=7.0)),
        atoms=_atoms(),
        positions=positions,
        edge_index=_edge_index(),
    )
    relabeled = adapt_sbdd_affinity_complex(
        _contract(_label(value=4.5)),
        atoms=_atoms(),
        positions=positions,
        edge_index=_edge_index(),
    )

    assert first.sample.sample_id == relabeled.sample.sample_id
    assert first.receipt.sha256 == relabeled.receipt.sha256
    assert first.sample.target.item() == pytest.approx(7.0)
    assert relabeled.sample.target.item() == pytest.approx(4.5)
    assert first.receipt.identity_inputs == (
        "entities",
        "structure",
        "source_provenance",
        "feature_schema",
        "relation_schema",
        "node_identity",
        "geometry",
        "topology",
        "annotations",
    )
    assert first.receipt.receipt_version == "sbdd-graph-receipt-v2"
    assert first.sample.node_role_id is not None
    assert first.sample.node_role_id.tolist() == [0, 0, 1]
    assert first.sample.hierarchy_id is not None
    assert first.sample.hierarchy_id[0] == first.sample.hierarchy_id[1]
    assert first.sample.hierarchy_id[0] != first.sample.hierarchy_id[2]
    assert set(first.sample.hierarchy_id.tolist()) == {0, 1}
    assert first.sample.node_masks is not None
    assert first.sample.node_masks["protein"].tolist() == [True, True, False]
    assert first.sample.node_masks["ligand"].tolist() == [False, False, True]
    assert first.sample.node_masks["pocket"].tolist() == [True, True, False]
    assert first.sample.node_masks["interface"].tolist() == [True, False, True]
    assert first.sample.edge_relation_id is not None
    relations = SBDDRelationSchema.default()
    assert first.sample.edge_relation_id.tolist() == [
        relations.self_relation_id,
        relations.self_relation_id,
        relations.self_relation_id,
        relations.relation_id(AtomRole.PROTEIN, AtomRole.LIGAND),
        relations.relation_id(AtomRole.LIGAND, AtomRole.PROTEIN),
    ]
    with pytest.raises(TypeError):
        first.sample.node_masks["protein"] = torch.zeros(3, dtype=torch.bool)  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        first.receipt.source = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        first.receipt.as_mapping()["source"] = "changed"  # type: ignore[index]


def test_rich_adapter_identity_is_invariant_to_atom_and_edge_permutation() -> None:
    atoms = _atoms()
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.0, 2.0, 0.0]],
        dtype=torch.float64,
    )
    edge_index = _edge_index()
    first = adapt_sbdd_affinity_complex(
        _contract(_label()),
        atoms=atoms,
        positions=positions,
        edge_index=edge_index,
    )
    permutation = torch.tensor([2, 0, 1])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(3)
    permuted_edges = inverse[edge_index]
    edge_order = torch.tensor([4, 2, 0, 3, 1])
    second = adapt_sbdd_affinity_complex(
        _contract(_label()),
        atoms=tuple(atoms[index] for index in permutation.tolist()),
        positions=positions[permutation],
        edge_index=permuted_edges[:, edge_order],
    )

    assert first.sample.sample_id == second.sample.sample_id
    assert first.receipt.sha256 == second.receipt.sha256
    assert first.receipt.node_identity_policy == "stable-atom-id-v1"


def test_adapter_annotations_survive_generic_collation() -> None:
    first = adapt_sbdd_affinity_complex(
        _contract(_label(value=7.0)),
        atoms=_atoms(),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.0, 2.0, 0.0]]
        ),
        edge_index=_edge_index(),
    ).sample
    second = replace(
        adapt_sbdd_affinity_complex(
            _contract(_label(value=6.0)),
            atoms=_atoms(),
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.0, 2.2, 0.0]]
            ),
            edge_index=_edge_index(),
        ).sample,
        sample_id="second",
    )

    batch = collate_graphs((first, second))

    assert batch.node_role_id is not None
    assert batch.node_role_id.tolist() == [0, 0, 1, 0, 0, 1]
    assert batch.hierarchy_id is not None
    assert set(batch.hierarchy_id[:3].tolist()).isdisjoint(
        set(batch.hierarchy_id[3:].tolist())
    )
    assert batch.node_masks is not None
    assert batch.node_masks["ligand"].tolist() == [
        False,
        False,
        True,
        False,
        False,
        True,
    ]
    assert batch.edge_relation_id is not None
    assert batch.edge_relation_id.shape == (10,)


def test_preencoded_adapter_bridges_atom3d_lba_without_label_in_identity() -> None:
    row = {
        "input_ids": [6, 8],
        "coords": [[0.0, 0.0, 0.0], [1.7, 0.0, 0.0]],
        "labels": [7.2],
        "token_type_ids": [1, 2],
    }
    changed_label = {**row, "labels": [3.1]}
    raw = atom3d_lba_row_to_sample(
        row,
        split="train",
        row_index=0,
        revision=ATOM3D_LBA_REVISION,
    )
    raw_changed = atom3d_lba_row_to_sample(
        changed_label,
        split="train",
        row_index=0,
        revision=ATOM3D_LBA_REVISION,
    )
    edge_index = torch.tensor([[0, 1, 0, 1], [0, 1, 1, 0]])
    kwargs = {
        "edge_index": edge_index,
        "node_ids": ("atom3d:train:0:node-0", "atom3d:train:0:node-1"),
        "label_blind_entity_id": "atom3d-lba:train:0000000",
        "source": "vector-institute/atom3d-lba",
        "source_snapshot": ATOM3D_LBA_REVISION,
        "processing_version": "atom3d-lba-adapter-v1",
        "feature_schema_version": "atom3d-lba-token-v1",
    }

    first = adapt_preencoded_graph_sample(raw, **kwargs)
    second = adapt_preencoded_graph_sample(raw_changed, **kwargs)

    assert first.sample.sample_id == second.sample.sample_id
    assert first.receipt.sha256 == second.receipt.sha256
    assert first.sample.target.item() == pytest.approx(7.2)
    assert second.sample.target.item() == pytest.approx(3.1)
    assert first.sample.hierarchy_id is not None
    assert first.sample.hierarchy_id.tolist() == [0, 1]
    assert first.sample.node_masks is not None
    assert first.sample.node_masks["pocket"].tolist() == [True, False]
    assert first.sample.node_masks["interface"].tolist() == [True, True]
    assert first.sample.edge_relation_id is not None
    assert first.receipt.node_identity_policy == "caller-supplied-node-id-v1"


def test_preencoded_receipt_identifies_model_visible_readout_mask() -> None:
    row = {
        "input_ids": [6, 8],
        "coords": [[0.0, 0.0, 0.0], [1.7, 0.0, 0.0]],
        "labels": [7.2],
        "token_type_ids": [1, 2],
    }
    raw = atom3d_lba_row_to_sample(
        row,
        split="train",
        row_index=0,
        revision=ATOM3D_LBA_REVISION,
    )
    assert raw.readout_mask is not None
    changed_readout = replace(
        raw,
        readout_mask=~raw.readout_mask,
    )
    kwargs = {
        "edge_index": torch.tensor([[0, 1, 0, 1], [0, 1, 1, 0]]),
        "node_ids": ("atom3d:train:0:node-0", "atom3d:train:0:node-1"),
        "label_blind_entity_id": "atom3d-lba:train:0000000",
        "source": "vector-institute/atom3d-lba",
        "source_snapshot": ATOM3D_LBA_REVISION,
        "processing_version": "atom3d-lba-adapter-v1",
        "feature_schema_version": "atom3d-lba-token-v1",
    }

    first = adapt_preencoded_graph_sample(raw, **kwargs)
    second = adapt_preencoded_graph_sample(changed_readout, **kwargs)

    assert first.receipt.feature_sha256 == second.receipt.feature_sha256
    assert first.receipt.geometry_sha256 == second.receipt.geometry_sha256
    assert first.receipt.topology_sha256 == second.receipt.topology_sha256
    assert first.receipt.annotation_sha256 != second.receipt.annotation_sha256
    assert first.receipt.graph_identity_sha256 != second.receipt.graph_identity_sha256
    assert first.sample.sample_id != second.sample.sample_id


def test_preencoded_receipt_identifies_named_mask_semantics() -> None:
    row = {
        "input_ids": [6, 8],
        "coords": [[0.0, 0.0, 0.0], [1.7, 0.0, 0.0]],
        "labels": [7.2],
        "token_type_ids": [1, 2],
    }
    raw = atom3d_lba_row_to_sample(
        row,
        split="train",
        row_index=0,
        revision=ATOM3D_LBA_REVISION,
    )
    default_schema = SBDDRelationSchema.default()
    reordered_schema = SBDDRelationSchema(
        version=default_schema.version,
        roles=(
            AtomRole.LIGAND,
            AtomRole.PROTEIN,
            *default_schema.roles[2:],
        ),
    )
    kwargs = {
        "edge_index": torch.tensor([[0, 1, 0, 1], [0, 1, 1, 0]]),
        "node_ids": ("atom3d:train:0:node-0", "atom3d:train:0:node-1"),
        "label_blind_entity_id": "atom3d-lba:train:0000000",
        "source": "vector-institute/atom3d-lba",
        "source_snapshot": ATOM3D_LBA_REVISION,
        "processing_version": "atom3d-lba-adapter-v1",
        "feature_schema_version": "atom3d-lba-token-v1",
    }

    first = adapt_preencoded_graph_sample(
        raw,
        relation_schema=default_schema,
        **kwargs,
    )
    second = adapt_preencoded_graph_sample(
        raw,
        relation_schema=reordered_schema,
        **kwargs,
    )

    assert first.receipt.relation_schema_version == second.receipt.relation_schema_version
    assert first.receipt.topology_sha256 == second.receipt.topology_sha256
    assert first.receipt.annotation_sha256 != second.receipt.annotation_sha256
    assert first.receipt.graph_identity_sha256 != second.receipt.graph_identity_sha256


def test_censored_affinity_loss_respects_exact_bounds_and_intervals() -> None:
    prediction = torch.tensor(
        [7.0, 4.0, 6.0, 8.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    labels = (
        _label(value=5.0),
        _label(
            value=None,
            qualifier=LabelQualifier.LOWER_BOUND,
            lower=5.0,
        ),
        _label(
            value=None,
            qualifier=LabelQualifier.UPPER_BOUND,
            upper=5.0,
        ),
        _label(
            value=None,
            qualifier=LabelQualifier.INTERVAL,
            lower=5.0,
            upper=7.0,
        ),
    )

    per_item = censored_affinity_loss(prediction, labels, reduction="none")
    per_item.sum().backward()

    assert torch.equal(
        per_item,
        torch.tensor([4.0, 1.0, 1.0, 1.0], dtype=torch.float64),
    )
    assert prediction.grad is not None
    assert prediction.grad.tolist() == [4.0, -2.0, 2.0, 2.0]
    assert censored_affinity_loss(
        torch.tensor([6.0, 4.0, 6.0]),
        labels[1:],
    ).item() == 0.0


def test_censored_affinity_loss_rejects_incompatible_label_batches() -> None:
    with pytest.raises(ValueError, match="same scientific unit"):
        censored_affinity_loss(
            torch.tensor([1.0, 2.0]),
            (_label(value=1.0), _label(value=2.0, unit="nM")),
        )
    docking = replace(_label(value=1.0), kind=LabelKind.DOCKING_SCORE)
    with pytest.raises(ValueError, match="affinity"):
        censored_affinity_loss(torch.tensor([1.0]), (docking,))


def test_graph_adapter_refuses_to_flatten_censored_label_into_point_target() -> None:
    with pytest.raises(ValueError, match="censored_affinity_loss"):
        adapt_sbdd_affinity_complex(
            _contract(
                _label(
                    value=None,
                    qualifier=LabelQualifier.LOWER_BOUND,
                    lower=5.0,
                )
            ),
            atoms=_atoms(),
            positions=torch.zeros(3, 3),
            edge_index=_edge_index(),
        )
