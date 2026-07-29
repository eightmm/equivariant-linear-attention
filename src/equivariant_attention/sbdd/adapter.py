"""Label-blind adapters from SBDD records to the generic graph API.

The adapter owns all biological vocabulary. The generic model receives only
floating node features, coordinates, integer roles/relations, hierarchy IDs,
and named boolean masks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from types import MappingProxyType
from typing import Mapping, Sequence

import torch

from ..benchmarking import GraphSample
from .features import AtomRole, SBDDAtomFeatures, SBDDFeatureSchema
from .schema import (
    LabelQualifier,
    PredictionTask,
    PredictionUnit,
    SBDDSampleContract,
)


_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class SBDDRelationSchema:
    """Stable directed role-pair relation IDs for ``receiver <- sender`` edges."""

    version: str
    roles: tuple[AtomRole, ...]

    @classmethod
    def default(cls) -> SBDDRelationSchema:
        return cls(version="sbdd-directed-role-pair-v1", roles=tuple(AtomRole))

    def __post_init__(self) -> None:
        _require_text("relation schema version", self.version)
        if not self.roles or len(set(self.roles)) != len(self.roles):
            raise ValueError("relation roles must be non-empty and unique")
        if any(not isinstance(role, AtomRole) for role in self.roles):
            raise TypeError("relation roles must contain only AtomRole values")

    @property
    def self_relation_id(self) -> int:
        return 0

    @property
    def num_relations(self) -> int:
        return 1 + len(self.roles) ** 2

    @property
    def relation_names(self) -> tuple[str, ...]:
        return (
            "self",
            *(
                f"receiver={receiver.value}|sender={sender.value}"
                for receiver in self.roles
                for sender in self.roles
            ),
        )

    def relation_id(self, receiver: AtomRole, sender: AtomRole) -> int:
        if not isinstance(receiver, AtomRole) or not isinstance(
            sender,
            AtomRole,
        ):
            raise TypeError("receiver and sender must be AtomRole values")
        try:
            receiver_index = self.roles.index(receiver)
            sender_index = self.roles.index(sender)
        except ValueError as exc:
            raise ValueError("receiver and sender roles must belong to the schema") from exc
        return 1 + receiver_index * len(self.roles) + sender_index

    def encode(
        self,
        edge_index: torch.Tensor,
        node_role_id: torch.Tensor,
    ) -> torch.Tensor:
        edge_index = _validated_edge_index(
            edge_index,
            num_nodes=node_role_id.shape[0],
            device=node_role_id.device,
        )
        if node_role_id.dtype not in _INTEGER_DTYPES:
            raise TypeError("node_role_id must use an integer dtype")
        if node_role_id.ndim != 1:
            raise ValueError("node_role_id must have shape (N,)")
        roles = node_role_id.to(dtype=torch.long)
        if roles.numel() and bool(
            ((roles < 0) | (roles >= len(self.roles))).any().item()
        ):
            raise ValueError("node_role_id contains a role absent from the schema")
        receiver, sender = edge_index
        receiver_role = roles.index_select(0, receiver)
        sender_role = roles.index_select(0, sender)
        relation = 1 + receiver_role * len(self.roles) + sender_role
        return torch.where(receiver == sender, torch.zeros_like(relation), relation)


@dataclass(frozen=True)
class SBDDGraphReceipt:
    """Immutable, label-free provenance for one materialized generic graph.

    Version 2 is an additive correction to version 1: the v1 component
    digests for node identity, features, geometry, and topology keep their
    meanings, while ``annotation_sha256`` binds the model-visible
    ``GraphSample`` annotations that v1 omitted.  The aggregate graph identity
    expands accordingly, so v1 and v2 ``graph_identity_sha256`` values are
    intentionally not interchangeable.
    """

    receipt_version: str
    graph_identity_sha256: str
    entity_sample_id: str
    source: str
    source_snapshot: str
    processing_version: str
    structure_id: str
    coordinate_source: str
    feature_schema_version: str
    relation_schema_version: str
    hierarchy_policy: str
    node_identity_policy: str
    node_count: int
    edge_count: int
    node_identity_sha256: str
    feature_sha256: str
    geometry_sha256: str
    topology_sha256: str
    annotation_sha256: str
    identity_inputs: tuple[str, ...] = (
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

    def __post_init__(self) -> None:
        for name in (
            "receipt_version",
            "entity_sample_id",
            "source",
            "source_snapshot",
            "processing_version",
            "structure_id",
            "coordinate_source",
            "feature_schema_version",
            "relation_schema_version",
            "hierarchy_policy",
            "node_identity_policy",
        ):
            _require_text(name, getattr(self, name))
        for name in (
            "graph_identity_sha256",
            "node_identity_sha256",
            "feature_sha256",
            "geometry_sha256",
            "topology_sha256",
            "annotation_sha256",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if self.node_count <= 0:
            raise ValueError("node_count must be positive")
        if self.edge_count < self.node_count:
            raise ValueError("edge_count must include at least one self edge per node")

    @property
    def sha256(self) -> str:
        return _json_sha256(dict(self.as_mapping()))

    def as_mapping(self) -> Mapping[str, object]:
        return MappingProxyType(
            {field.name: getattr(self, field.name) for field in fields(self)}
        )


@dataclass(frozen=True)
class AdaptedSBDDGraph:
    sample: GraphSample
    receipt: SBDDGraphReceipt


def adapt_sbdd_affinity_complex(
    contract: SBDDSampleContract,
    *,
    atoms: tuple[SBDDAtomFeatures, ...],
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    feature_schema: SBDDFeatureSchema | None = None,
    relation_schema: SBDDRelationSchema | None = None,
    feature_dtype: torch.dtype = torch.float32,
) -> AdaptedSBDDGraph:
    """Encode one exact affinity complex without using its label in identity.

    Censored labels are intentionally refused: flattening a bound into
    ``GraphSample.target`` would silently turn it into an exact observation.
    Use :func:`censored_affinity_loss` with the original ``ScientificLabel``.
    """

    if not isinstance(contract, SBDDSampleContract):
        raise TypeError("contract must be an SBDDSampleContract")
    if contract.task is not PredictionTask.AFFINITY:
        raise ValueError("graph adapter currently accepts affinity contracts only")
    if contract.prediction_unit is not PredictionUnit.COMPLEX:
        raise ValueError("affinity graph adapter requires complex prediction units")
    label = contract.label
    if label is None:
        raise ValueError("affinity contract requires a label")
    if label.qualifier is not LabelQualifier.EXACT:
        raise ValueError(
            "censored labels cannot be flattened into GraphSample.target; "
            "use censored_affinity_loss"
        )
    assert label.value is not None
    if not atoms:
        raise ValueError("atoms must be non-empty")
    positions = _validated_positions(positions, num_nodes=len(atoms))
    feature_schema = (
        SBDDFeatureSchema.default() if feature_schema is None else feature_schema
    )
    relation_schema = (
        SBDDRelationSchema(
            version="sbdd-directed-role-pair-v1",
            roles=feature_schema.roles,
        )
        if relation_schema is None
        else relation_schema
    )
    if relation_schema.roles != feature_schema.roles:
        raise ValueError("feature and relation schemas must use identical role order")
    encoded = feature_schema.encode_atoms(
        atoms,
        dtype=feature_dtype,
        device=positions.device,
    )
    edge_index = _validated_edge_index(
        edge_index,
        num_nodes=len(atoms),
        device=positions.device,
    )
    relations = relation_schema.encode(edge_index, encoded.role_ids)
    hierarchy_id = _rich_hierarchy_ids(
        atoms,
        standardized_ligand_id=contract.entities.standardized_ligand_id,
        device=positions.device,
    )
    node_masks = _role_masks(
        encoded.role_ids,
        roles=feature_schema.roles,
        edge_index=edge_index,
    )
    ligand_mask = node_masks[AtomRole.LIGAND.value]
    if not bool(ligand_mask.any().item()):
        raise ValueError("affinity complex must contain at least one ligand atom")

    receipt = _rich_receipt(
        contract=contract,
        feature_schema=feature_schema,
        relation_schema=relation_schema,
        node_ids=encoded.atom_ids,
        node_features=encoded.scalar_features,
        positions=positions,
        edge_index=edge_index,
        edge_relation_id=relations,
        hierarchy_id=hierarchy_id,
        readout_mask=ligand_mask,
        node_role_id=encoded.role_ids,
        node_masks=node_masks,
    )
    target = encoded.scalar_features.new_tensor([float(label.value)])
    sample = GraphSample(
        node_feats=encoded.scalar_features,
        pos=positions,
        target=target,
        sample_id=f"sbdd:{receipt.graph_identity_sha256[:32]}",
        edge_index=edge_index,
        readout_mask=ligand_mask,
        edge_relation_id=relations,
        node_role_id=encoded.role_ids,
        hierarchy_id=hierarchy_id,
        node_masks=node_masks,
    )
    return AdaptedSBDDGraph(sample=sample, receipt=receipt)


def adapt_preencoded_graph_sample(
    sample: GraphSample,
    *,
    edge_index: torch.Tensor | None = None,
    node_ids: Sequence[str],
    label_blind_entity_id: str,
    source: str,
    source_snapshot: str,
    processing_version: str,
    feature_schema_version: str,
    relation_schema: SBDDRelationSchema | None = None,
) -> AdaptedSBDDGraph:
    """Bridge a label-blind, preencoded source such as ATOM3D-LBA.

    ``node_ids`` and ``label_blind_entity_id`` must derive from source identity
    or structure, never from the target. Stable node IDs make the receipt
    invariant to node and edge ordering.
    """

    if not isinstance(sample, GraphSample):
        raise TypeError("sample must be a GraphSample")
    for name, value in (
        ("label_blind_entity_id", label_blind_entity_id),
        ("source", source),
        ("source_snapshot", source_snapshot),
        ("processing_version", processing_version),
        ("feature_schema_version", feature_schema_version),
    ):
        _require_text(name, value)
    _validate_preencoded_sample(sample)
    node_ids = tuple(node_ids)
    _validate_node_ids(node_ids, num_nodes=sample.node_feats.shape[0])
    if sample.node_role_id is None:
        raise ValueError("preencoded SBDD sample requires node_role_id")
    if edge_index is not None and sample.edge_index is not None:
        raise ValueError("edge_index must be supplied either on sample or as argument")
    selected_edges = sample.edge_index if edge_index is None else edge_index
    if selected_edges is None:
        raise ValueError("typed SBDD relations require an explicit edge_index")
    selected_edges = _validated_edge_index(
        selected_edges,
        num_nodes=sample.node_feats.shape[0],
        device=sample.node_feats.device,
    )
    relation_schema = (
        SBDDRelationSchema.default()
        if relation_schema is None
        else relation_schema
    )
    role_ids = sample.node_role_id.to(dtype=torch.long)
    if role_ids.device != sample.node_feats.device:
        raise ValueError("node_role_id and node_feats must share one device")
    relations = relation_schema.encode(selected_edges, role_ids)
    node_masks = _role_masks(
        role_ids,
        roles=relation_schema.roles,
        edge_index=selected_edges,
    )
    ligand_mask = node_masks[AtomRole.LIGAND.value]
    if not bool(ligand_mask.any().item()):
        raise ValueError("preencoded SBDD sample must contain a ligand role")
    unique_roles = torch.unique(role_ids, sorted=True)
    hierarchy_id = torch.searchsorted(unique_roles, role_ids)
    readout_mask = (
        ligand_mask
        if sample.readout_mask is None
        else _validated_readout_mask(
            sample.readout_mask,
            num_nodes=sample.node_feats.shape[0],
            device=sample.node_feats.device,
        )
    )
    receipt = _preencoded_receipt(
        entity_sample_id=label_blind_entity_id,
        source=source,
        source_snapshot=source_snapshot,
        processing_version=processing_version,
        feature_schema_version=feature_schema_version,
        relation_schema=relation_schema,
        node_ids=node_ids,
        node_features=sample.node_feats,
        positions=sample.pos,
        edge_index=selected_edges,
        edge_relation_id=relations,
        hierarchy_id=hierarchy_id,
        readout_mask=readout_mask,
        node_role_id=role_ids,
        node_masks=node_masks,
    )
    adapted = GraphSample(
        node_feats=sample.node_feats,
        pos=sample.pos,
        target=sample.target,
        sample_id=f"sbdd:{receipt.graph_identity_sha256[:32]}",
        edge_index=selected_edges,
        readout_mask=readout_mask,
        edge_relation_id=relations,
        node_role_id=role_ids,
        hierarchy_id=hierarchy_id,
        node_masks=node_masks,
    )
    return AdaptedSBDDGraph(sample=adapted, receipt=receipt)


def _rich_hierarchy_ids(
    atoms: tuple[SBDDAtomFeatures, ...],
    *,
    standardized_ligand_id: str,
    device: torch.device,
) -> torch.Tensor:
    group_keys: list[str] = []
    for atom in atoms:
        if atom.role is AtomRole.PROTEIN:
            assert atom.chain_id is not None
            assert atom.residue_index is not None
            key = f"protein:{atom.chain_id}:{atom.residue_index:012d}"
        elif atom.role is AtomRole.LIGAND:
            key = f"ligand:{standardized_ligand_id}"
        else:
            key = f"{atom.role.value}:atom:{atom.atom_id}"
        group_keys.append(key)
    ordered = {key: index for index, key in enumerate(sorted(set(group_keys)))}
    return torch.tensor(
        [ordered[key] for key in group_keys],
        dtype=torch.long,
        device=device,
    )


def _role_masks(
    role_ids: torch.Tensor,
    *,
    roles: tuple[AtomRole, ...],
    edge_index: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    masks = {role.value: role_ids == index for index, role in enumerate(roles)}
    # For this adapter, represented protein nodes are the declared pocket.
    masks["pocket"] = masks[AtomRole.PROTEIN.value]
    receiver, sender = edge_index
    protein = masks[AtomRole.PROTEIN.value]
    ligand = masks[AtomRole.LIGAND.value]
    cross = (
        protein.index_select(0, receiver) & ligand.index_select(0, sender)
    ) | (
        ligand.index_select(0, receiver) & protein.index_select(0, sender)
    )
    interface = torch.zeros_like(protein)
    interface[receiver[cross]] = True
    interface[sender[cross]] = True
    masks["interface"] = interface
    return MappingProxyType(dict(sorted(masks.items())))


def _rich_receipt(
    *,
    contract: SBDDSampleContract,
    feature_schema: SBDDFeatureSchema,
    relation_schema: SBDDRelationSchema,
    node_ids: tuple[str, ...],
    node_features: torch.Tensor,
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    edge_relation_id: torch.Tensor,
    hierarchy_id: torch.Tensor,
    readout_mask: torch.Tensor,
    node_role_id: torch.Tensor,
    node_masks: Mapping[str, torch.Tensor],
) -> SBDDGraphReceipt:
    canonical = _canonical_graph_digests(
        node_ids=node_ids,
        node_features=node_features,
        positions=positions,
        edge_index=edge_index,
        edge_relation_id=edge_relation_id,
        hierarchy_id=hierarchy_id,
        readout_mask=readout_mask,
        node_role_id=node_role_id,
        node_masks=node_masks,
    )
    identity_payload = {
        "entities": asdict(contract.entities),
        "structure": asdict(contract.structure),
        "source_provenance": {
            "source": contract.source,
            "source_snapshot": contract.source_snapshot,
            "processing_version": contract.processing_version,
        },
        "feature_schema": feature_schema.version,
        "relation_schema": relation_schema.version,
        **canonical,
    }
    graph_identity = _json_sha256(identity_payload)
    return SBDDGraphReceipt(
        receipt_version="sbdd-graph-receipt-v2",
        graph_identity_sha256=graph_identity,
        entity_sample_id=contract.entities.sample_id,
        source=contract.source,
        source_snapshot=contract.source_snapshot,
        processing_version=contract.processing_version,
        structure_id=contract.structure.structure_id,
        coordinate_source=contract.structure.coordinate_source,
        feature_schema_version=feature_schema.version,
        relation_schema_version=relation_schema.version,
        hierarchy_policy="protein-residue-ligand-entity-v1",
        node_identity_policy="stable-atom-id-v1",
        node_count=node_features.shape[0],
        edge_count=edge_index.shape[1],
        node_identity_sha256=canonical["node_identity_sha256"],
        feature_sha256=canonical["feature_sha256"],
        geometry_sha256=canonical["geometry_sha256"],
        topology_sha256=canonical["topology_sha256"],
        annotation_sha256=canonical["annotation_sha256"],
    )


def _preencoded_receipt(
    *,
    entity_sample_id: str,
    source: str,
    source_snapshot: str,
    processing_version: str,
    feature_schema_version: str,
    relation_schema: SBDDRelationSchema,
    node_ids: tuple[str, ...],
    node_features: torch.Tensor,
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    edge_relation_id: torch.Tensor,
    hierarchy_id: torch.Tensor,
    readout_mask: torch.Tensor,
    node_role_id: torch.Tensor,
    node_masks: Mapping[str, torch.Tensor],
) -> SBDDGraphReceipt:
    canonical = _canonical_graph_digests(
        node_ids=node_ids,
        node_features=node_features,
        positions=positions,
        edge_index=edge_index,
        edge_relation_id=edge_relation_id,
        hierarchy_id=hierarchy_id,
        readout_mask=readout_mask,
        node_role_id=node_role_id,
        node_masks=node_masks,
    )
    identity_payload = {
        "entities": {"sample_id": entity_sample_id},
        "structure": {
            "structure_id": entity_sample_id,
            "coordinate_source": "preencoded GraphSample",
        },
        "source_provenance": {
            "source": source,
            "source_snapshot": source_snapshot,
            "processing_version": processing_version,
        },
        "feature_schema": feature_schema_version,
        "relation_schema": relation_schema.version,
        **canonical,
    }
    graph_identity = _json_sha256(identity_payload)
    return SBDDGraphReceipt(
        receipt_version="sbdd-graph-receipt-v2",
        graph_identity_sha256=graph_identity,
        entity_sample_id=entity_sample_id,
        source=source,
        source_snapshot=source_snapshot,
        processing_version=processing_version,
        structure_id=entity_sample_id,
        coordinate_source="preencoded GraphSample",
        feature_schema_version=feature_schema_version,
        relation_schema_version=relation_schema.version,
        hierarchy_policy="role-group-fallback-v1",
        node_identity_policy="caller-supplied-node-id-v1",
        node_count=node_features.shape[0],
        edge_count=edge_index.shape[1],
        node_identity_sha256=canonical["node_identity_sha256"],
        feature_sha256=canonical["feature_sha256"],
        geometry_sha256=canonical["geometry_sha256"],
        topology_sha256=canonical["topology_sha256"],
        annotation_sha256=canonical["annotation_sha256"],
    )


def _canonical_graph_digests(
    *,
    node_ids: tuple[str, ...],
    node_features: torch.Tensor,
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    edge_relation_id: torch.Tensor,
    hierarchy_id: torch.Tensor,
    readout_mask: torch.Tensor,
    node_role_id: torch.Tensor,
    node_masks: Mapping[str, torch.Tensor],
) -> dict[str, str]:
    _validate_node_ids(node_ids, num_nodes=node_features.shape[0])
    order = torch.tensor(
        sorted(range(len(node_ids)), key=node_ids.__getitem__),
        dtype=torch.long,
        device=node_features.device,
    )
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=order.device)
    canonical_edges = inverse.index_select(0, edge_index.reshape(-1)).reshape_as(
        edge_index
    )
    codes = canonical_edges[0] * len(node_ids) + canonical_edges[1]
    edge_order = torch.argsort(codes, stable=True)
    canonical_edges = canonical_edges.index_select(1, edge_order)
    canonical_relations = edge_relation_id.index_select(0, edge_order)
    canonical_hierarchy = hierarchy_id.index_select(0, order)
    canonical_annotations = {
        "readout_mask": _tensor_sha256(readout_mask.index_select(0, order)),
        "edge_relation_id": _tensor_sha256(canonical_relations),
        "node_role_id": _tensor_sha256(node_role_id.index_select(0, order)),
        "hierarchy_id": _tensor_sha256(canonical_hierarchy),
        "node_masks": {
            name: _tensor_sha256(mask.index_select(0, order))
            for name, mask in sorted(node_masks.items())
        },
    }
    return {
        "node_identity_sha256": _json_sha256(
            {"node_ids": [node_ids[index] for index in order.tolist()]}
        ),
        "feature_sha256": _tensor_sha256(node_features.index_select(0, order)),
        "geometry_sha256": _tensor_sha256(positions.index_select(0, order)),
        "topology_sha256": _tensor_sha256(
            canonical_edges,
            canonical_relations,
            canonical_hierarchy,
        ),
        "annotation_sha256": _json_sha256(canonical_annotations),
    }


def _validate_preencoded_sample(sample: GraphSample) -> None:
    if sample.node_feats.ndim != 2 or sample.node_feats.shape[0] == 0:
        raise ValueError("node_feats must have shape (N, F) with N > 0")
    if not torch.is_floating_point(sample.node_feats) or not bool(
        torch.isfinite(sample.node_feats).all()
    ):
        raise ValueError("node_feats must be finite floating point")
    _validated_positions(sample.pos, num_nodes=sample.node_feats.shape[0])
    if sample.pos.device != sample.node_feats.device:
        raise ValueError("node_feats and pos must share one device")
    if sample.target.numel() != 1 or not torch.is_floating_point(sample.target):
        raise ValueError("target must contain one floating scalar")
    if not bool(torch.isfinite(sample.target).all()):
        raise ValueError("target must be finite")


def _validated_readout_mask(
    readout_mask: torch.Tensor,
    *,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(readout_mask, torch.Tensor):
        raise TypeError("readout_mask must be a tensor")
    if readout_mask.dtype is not torch.bool:
        raise TypeError("readout_mask must use boolean dtype")
    if readout_mask.shape != (num_nodes,):
        raise ValueError(f"readout_mask must have shape ({num_nodes},)")
    if readout_mask.device != device:
        raise ValueError("readout_mask and node tensors must share one device")
    if not bool(readout_mask.any().item()):
        raise ValueError("readout_mask must select at least one node")
    return readout_mask


def _validated_positions(
    positions: torch.Tensor,
    *,
    num_nodes: int,
) -> torch.Tensor:
    if not isinstance(positions, torch.Tensor):
        raise TypeError("positions must be a tensor")
    if positions.shape != (num_nodes, 3):
        raise ValueError(f"positions must have shape ({num_nodes}, 3)")
    if not torch.is_floating_point(positions) or not bool(
        torch.isfinite(positions).all()
    ):
        raise ValueError("positions must be finite floating point")
    return positions


def _validated_edge_index(
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(edge_index, torch.Tensor):
        raise TypeError("edge_index must be a tensor")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E)")
    if edge_index.dtype not in _INTEGER_DTYPES:
        raise TypeError("edge_index must use an integer dtype")
    if edge_index.device != device:
        raise ValueError("edge_index and node tensors must share one device")
    result = edge_index.to(dtype=torch.long)
    if result.shape[1] < num_nodes:
        raise ValueError("edge_index must include at least one self edge per node")
    if result.numel() and bool(
        ((result < 0) | (result >= num_nodes)).any().item()
    ):
        raise ValueError("edge_index contains an out-of-range node")
    receiver, sender = result
    codes = receiver * num_nodes + sender
    if torch.unique(codes).numel() != codes.numel():
        raise ValueError("edge_index must not contain duplicate directed edges")
    self_nodes = receiver[receiver == sender]
    if (
        torch.unique(self_nodes).numel() != num_nodes
        or self_nodes.numel() != num_nodes
    ):
        raise ValueError("edge_index must contain exactly one self edge per node")
    return result


def _validate_node_ids(node_ids: tuple[str, ...], *, num_nodes: int) -> None:
    if len(node_ids) != num_nodes:
        raise ValueError("node_ids must contain one ID per node")
    for node_id in node_ids:
        _require_text("node ID", node_id)
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("node_ids must be unique")


def _tensor_sha256(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        contiguous = value.detach().cpu().contiguous()
        metadata = {
            "dtype": str(contiguous.dtype),
            "shape": tuple(contiguous.shape),
        }
        digest.update(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        )
        digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()
