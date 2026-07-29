from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

from .graph_layout import PackedGraphLayout, pack_graph_layout
from .multiscale import HierarchyAssignment
from .neighbors import PackedNeighborGraph


@dataclass(frozen=True)
class GraphSample:
    node_feats: torch.Tensor
    pos: torch.Tensor
    target: torch.Tensor
    sample_id: str
    edge_index: torch.Tensor | None = None
    readout_mask: torch.Tensor | None = None
    edge_relation_id: torch.Tensor | None = None
    node_role_id: torch.Tensor | None = None
    hierarchy_id: torch.Tensor | None = None
    node_masks: Mapping[str, torch.Tensor] | None = None


@dataclass(frozen=True)
class GraphBatch:
    node_feats: torch.Tensor
    pos: torch.Tensor
    batch: torch.Tensor
    target: torch.Tensor
    sample_ids: tuple[str, ...]
    edge_index: torch.Tensor | None = None
    edge_index_is_validated: bool = False
    readout_mask: torch.Tensor | None = None
    packed_neighbors: PackedNeighborGraph | None = None
    graph_layout: PackedGraphLayout | None = None
    edge_relation_id: torch.Tensor | None = None
    node_role_id: torch.Tensor | None = None
    hierarchy_id: torch.Tensor | None = None
    node_masks: Mapping[str, torch.Tensor] | None = None

    def to(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
        *,
        geometry_dtype: torch.dtype | None = None,
    ) -> GraphBatch:
        value_dtype = dtype if dtype is not None else self.node_feats.dtype
        if geometry_dtype is None:
            geometry_dtype = (
                torch.float64
                if torch.float64 in {value_dtype, self.pos.dtype}
                else torch.float32
            )
        if geometry_dtype not in {torch.float32, torch.float64}:
            raise TypeError("geometry_dtype must be float32 or float64")
        if self.graph_layout is not None:
            self.graph_layout.validate_batch(self.batch)
        moved_layout = (
            None
            if self.graph_layout is None
            else self.graph_layout.to(device)
        )
        moved_batch = (
            self.batch.to(device=device)
            if moved_layout is None
            else moved_layout.batch
        )
        return GraphBatch(
            node_feats=self.node_feats.to(device=device, dtype=value_dtype),
            pos=self.pos.to(device=device, dtype=geometry_dtype),
            batch=moved_batch,
            target=self.target.to(device=device, dtype=value_dtype),
            sample_ids=self.sample_ids,
            edge_index=(
                None
                if self.edge_index is None
                else self.edge_index.to(device=device, dtype=torch.long)
            ),
            edge_index_is_validated=self.edge_index_is_validated,
            readout_mask=(
                None
                if self.readout_mask is None
                else self.readout_mask.to(device=device)
            ),
            packed_neighbors=(
                None
                if self.packed_neighbors is None
                else self.packed_neighbors.to(device)
            ),
            graph_layout=moved_layout,
            edge_relation_id=(
                None
                if self.edge_relation_id is None
                else self.edge_relation_id.to(device=device, dtype=torch.long)
            ),
            node_role_id=(
                None
                if self.node_role_id is None
                else self.node_role_id.to(device=device, dtype=torch.long)
            ),
            hierarchy_id=(
                None
                if self.hierarchy_id is None
                else self.hierarchy_id.to(device=device, dtype=torch.long)
            ),
            node_masks=(
                None
                if self.node_masks is None
                else {
                    name: mask.to(device=device)
                    for name, mask in self.node_masks.items()
                }
            ),
        )

    def hierarchy_assignment(self) -> HierarchyAssignment:
        if self.hierarchy_id is None:
            raise ValueError("GraphBatch has no hierarchy_id annotation")
        return HierarchyAssignment(self.hierarchy_id, self.batch)


class SyntheticMoleculeDataset(Dataset[GraphSample]):
    """Small deterministic invariant regression task for benchmark smoke tests."""

    def __init__(
        self,
        num_samples: int,
        node_dim: int,
        min_nodes: int = 4,
        max_nodes: int = 9,
        seed: int = 0,
    ) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if node_dim <= 1:
            raise ValueError("node_dim must be greater than one")
        if min_nodes <= 0 or max_nodes < min_nodes:
            raise ValueError("node count bounds are invalid")
        self.samples = [
            _make_synthetic_sample(i, node_dim, min_nodes, max_nodes, seed)
            for i in range(num_samples)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> GraphSample:
        return self.samples[index]


def collate_graphs(samples: Sequence[GraphSample]) -> GraphBatch:
    if not samples:
        raise ValueError("at least one graph sample is required")
    node_feats = torch.cat([sample.node_feats for sample in samples], dim=0)
    pos = torch.cat([sample.pos for sample in samples], dim=0)
    target = torch.stack([sample.target.reshape(-1) for sample in samples], dim=0)
    batch = torch.cat(
        [
            torch.full((sample.node_feats.shape[0],), i, dtype=torch.long)
            for i, sample in enumerate(samples)
        ],
        dim=0,
    )
    edge_presence = [sample.edge_index is not None for sample in samples]
    if any(edge_presence) and not all(edge_presence):
        raise ValueError("all samples must either provide edge_index or omit it")
    edge_index = None
    edge_relation_id = None
    if all(edge_presence):
        offset_edges = []
        relation_presence = [
            sample.edge_relation_id is not None for sample in samples
        ]
        if any(relation_presence) and not all(relation_presence):
            raise ValueError(
                "all samples with edge_index must either provide "
                "edge_relation_id or omit it"
            )
        relation_parts = []
        node_offset = 0
        for sample in samples:
            sample_edges = _validated_sample_edge_index(sample)
            offset_edges.append(sample_edges + node_offset)
            if all(relation_presence):
                relation_parts.append(
                    _validated_sample_edge_relation_id(
                        sample,
                        num_edges=sample_edges.shape[1],
                    )
                )
            node_offset += sample.node_feats.shape[0]
        edge_index = torch.cat(offset_edges, dim=1)
        if relation_parts:
            edge_relation_id = torch.cat(relation_parts, dim=0)
    elif any(sample.edge_relation_id is not None for sample in samples):
        raise ValueError("edge_relation_id requires edge_index")
    role_presence = [sample.node_role_id is not None for sample in samples]
    if any(role_presence) and not all(role_presence):
        raise ValueError(
            "all samples must either provide node_role_id or omit it"
        )
    node_role_id = (
        torch.cat(
            [
                _validated_sample_node_index(
                    sample,
                    field_name="node_role_id",
                    require_contiguous=False,
                )[0]
                for sample in samples
            ],
            dim=0,
        )
        if all(role_presence)
        else None
    )
    hierarchy_presence = [sample.hierarchy_id is not None for sample in samples]
    if any(hierarchy_presence) and not all(hierarchy_presence):
        raise ValueError(
            "all samples must either provide hierarchy_id or omit it"
        )
    hierarchy_id = None
    if all(hierarchy_presence):
        hierarchy_parts = []
        coarse_offset = 0
        for sample in samples:
            sample_hierarchy, coarse_count = _validated_sample_node_index(
                sample,
                field_name="hierarchy_id",
                require_contiguous=True,
            )
            hierarchy_parts.append(sample_hierarchy + coarse_offset)
            coarse_offset += coarse_count
        hierarchy_id = torch.cat(hierarchy_parts, dim=0)
    mask_presence = [sample.node_masks is not None for sample in samples]
    if any(mask_presence) and not all(mask_presence):
        raise ValueError(
            "all samples must either provide node_masks or omit them"
        )
    node_masks = None
    if all(mask_presence):
        validated_masks = [
            _validated_sample_node_masks(sample) for sample in samples
        ]
        key_order = tuple(validated_masks[0])
        if any(tuple(masks) != key_order for masks in validated_masks[1:]):
            raise ValueError(
                "all sample node_masks must use the same sorted names"
            )
        node_masks = {
            name: torch.cat(
                [masks[name] for masks in validated_masks],
                dim=0,
            )
            for name in key_order
        }
    readout_presence = [sample.readout_mask is not None for sample in samples]
    if any(readout_presence) and not all(readout_presence):
        raise ValueError("all samples must either provide readout_mask or omit it")
    readout_mask = None
    if all(readout_presence):
        readout_mask = torch.cat(
            [_validated_sample_readout_mask(sample) for sample in samples],
            dim=0,
        )
    graph_counts = torch.tensor(
        [sample.node_feats.shape[0] for sample in samples],
        dtype=torch.long,
        device=batch.device,
    )
    graph_layout = pack_graph_layout(
        batch,
        graph_counts=graph_counts,
        assume_grouped=True,
    )
    return GraphBatch(
        node_feats=node_feats,
        pos=pos,
        batch=batch,
        target=target,
        sample_ids=tuple(sample.sample_id for sample in samples),
        edge_index=edge_index,
        edge_index_is_validated=edge_index is not None,
        readout_mask=readout_mask,
        graph_layout=graph_layout,
        edge_relation_id=edge_relation_id,
        node_role_id=node_role_id,
        hierarchy_id=hierarchy_id,
        node_masks=node_masks,
    )


def split_dataset(
    dataset: Dataset[GraphSample], train_size: int, val_size: int, seed: int
) -> tuple[list[int], list[int], list[int]]:
    n_samples = len(dataset)
    if train_size <= 0 or val_size <= 0:
        raise ValueError("train_size and val_size must be positive")
    if train_size + val_size >= n_samples:
        raise ValueError("train_size + val_size must leave at least one test sample")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(n_samples, generator=generator).tolist()
    train = order[:train_size]
    val = order[train_size : train_size + val_size]
    test = order[train_size + val_size :]
    return train, val, test


def load_qm9_samples(
    root: str | Path,
    target_index: int = 4,
    limit: int | None = None,
    *,
    local_cutoff: float | None = None,
) -> list[GraphSample]:
    try:
        from torch_geometric.datasets import QM9
    except ModuleNotFoundError as exc:
        raise ImportError(
            "QM9 loading requires optional dependencies: torch-geometric and rdkit"
        ) from exc

    dataset = QM9(root=str(root))
    if limit is not None:
        dataset = dataset[:limit]

    samples: list[GraphSample] = []
    for i, data in enumerate(dataset):
        node_feats = data.x.float()
        pos = data.pos.float()
        target = data.y[:, target_index].float().reshape(1)
        samples.append(
            GraphSample(
                node_feats=node_feats,
                pos=pos,
                target=target,
                sample_id=_qm9_sample_id(data, row_index=i),
                edge_index=(
                    None
                    if local_cutoff is None
                    else _radius_candidate_edge_index(pos, cutoff=local_cutoff)
                ),
            )
        )
    return samples


def _radius_candidate_edge_index(
    pos: torch.Tensor,
    *,
    cutoff: float,
) -> torch.Tensor:
    """Build row-0 receiver/row-1 sender radius candidates, including self."""
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("pos must have shape (N, 3)")
    if not torch.is_floating_point(pos):
        raise TypeError("pos must be floating point")
    if not isinstance(cutoff, (int, float)) or isinstance(cutoff, bool):
        raise TypeError("cutoff must be a real number")
    if not 0.0 < float(cutoff) < float("inf"):
        raise ValueError("cutoff must be finite and positive")
    displacement = pos[:, None, :] - pos[None, :, :]
    squared_distance = displacement.square().sum(dim=-1)
    receiver, sender = torch.nonzero(
        squared_distance < float(cutoff) ** 2,
        as_tuple=True,
    )
    return torch.stack([receiver, sender]).to(dtype=torch.long)


def _validated_sample_edge_index(sample: GraphSample) -> torch.Tensor:
    edge_index = sample.edge_index
    if edge_index is None:
        raise RuntimeError("sample edge_index unexpectedly missing")
    if not isinstance(edge_index, torch.Tensor):
        raise TypeError("sample edge_index must be a tensor")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("sample edge_index must have shape (2, E)")
    integer_dtypes = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
    if edge_index.dtype not in integer_dtypes:
        raise TypeError("sample edge_index must use an integer dtype")
    edge_index = edge_index.to(dtype=torch.long)
    if edge_index.numel():
        if bool((edge_index < 0).any().item()):
            raise ValueError("sample edge_index values must be nonnegative")
        if int(edge_index.max().item()) >= sample.node_feats.shape[0]:
            raise ValueError("sample edge_index values are out of range")
    receiver, sender = edge_index.unbind(dim=0)
    pair_codes = receiver * sample.node_feats.shape[0] + sender
    if torch.unique(pair_codes).numel() != pair_codes.numel():
        raise ValueError("sample edge_index must not contain duplicate directed edges")
    self_nodes = receiver[receiver == sender]
    has_self = torch.zeros(
        sample.node_feats.shape[0],
        dtype=torch.bool,
        device=edge_index.device,
    )
    has_self[self_nodes] = True
    if not bool(has_self.all().item()):
        raise ValueError("sample edge_index must contain a self edge for every node")
    return edge_index


def _validated_sample_edge_relation_id(
    sample: GraphSample,
    *,
    num_edges: int,
) -> torch.Tensor:
    relation_id = sample.edge_relation_id
    if relation_id is None:
        raise RuntimeError("edge_relation_id validation requires relation metadata")
    if not isinstance(relation_id, torch.Tensor):
        raise TypeError("sample edge_relation_id must be a tensor")
    integer_dtypes = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
    if relation_id.dtype not in integer_dtypes:
        raise TypeError("sample edge_relation_id must use an integer dtype")
    if relation_id.shape != (num_edges,):
        raise ValueError(
            "sample edge_relation_id must have one value per edge"
        )
    edge_index = sample.edge_index
    if edge_index is None:
        raise RuntimeError("edge_relation_id unexpectedly lacks edge_index")
    if relation_id.device != edge_index.device:
        raise ValueError(
            "sample edge_relation_id and edge_index must share one device"
        )
    relation_id = relation_id.to(dtype=torch.long)
    if relation_id.numel() and bool((relation_id < 0).any().item()):
        raise ValueError("sample edge_relation_id values must be nonnegative")
    return relation_id


def _validated_sample_node_index(
    sample: GraphSample,
    *,
    field_name: str,
    require_contiguous: bool,
) -> tuple[torch.Tensor, int]:
    value = getattr(sample, field_name)
    if value is None:
        raise RuntimeError(f"{field_name} validation requires metadata")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"sample {field_name} must be a tensor")
    integer_dtypes = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
    if value.dtype not in integer_dtypes:
        raise TypeError(f"sample {field_name} must use an integer dtype")
    if value.shape != (sample.node_feats.shape[0],):
        raise ValueError(f"sample {field_name} must have shape (N,)")
    if value.device != sample.node_feats.device:
        raise ValueError(
            f"sample {field_name} and node_feats must share one device"
        )
    value = value.to(dtype=torch.long)
    if value.numel() and bool((value < 0).any().item()):
        raise ValueError(f"sample {field_name} values must be nonnegative")
    count = int(value.max().item()) + 1 if value.numel() else 0
    if require_contiguous and value.numel():
        observed = torch.unique(value, sorted=True)
        expected = torch.arange(count, device=value.device)
        if not torch.equal(observed, expected):
            raise ValueError(
                f"sample {field_name} values must be contiguous from zero"
            )
    return value, count


def _validated_sample_node_masks(
    sample: GraphSample,
) -> dict[str, torch.Tensor]:
    masks = sample.node_masks
    if masks is None:
        raise RuntimeError("node_masks validation requires metadata")
    if not isinstance(masks, Mapping):
        raise TypeError("sample node_masks must be a mapping")
    if any(not isinstance(name, str) or not name for name in masks):
        raise ValueError("sample node_masks names must be nonempty strings")
    result: dict[str, torch.Tensor] = {}
    for name in sorted(masks):
        mask = masks[name]
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"sample node_masks[{name!r}] must be a tensor")
        if mask.dtype != torch.bool:
            raise TypeError(
                f"sample node_masks[{name!r}] must use boolean dtype"
            )
        if mask.shape != (sample.node_feats.shape[0],):
            raise ValueError(
                f"sample node_masks[{name!r}] must have shape (N,)"
            )
        if mask.device != sample.node_feats.device:
            raise ValueError(
                f"sample node_masks[{name!r}] and node_feats must share one device"
            )
        result[name] = mask
    return result


def _validated_sample_readout_mask(sample: GraphSample) -> torch.Tensor:
    readout_mask = sample.readout_mask
    if readout_mask is None:
        raise RuntimeError("readout_mask validation requires a mask")
    if readout_mask.dtype != torch.bool:
        raise TypeError("readout_mask must use boolean dtype")
    if readout_mask.shape != (sample.node_feats.shape[0],):
        raise ValueError(
            "readout_mask must have shape "
            f"({sample.node_feats.shape[0]},), got {tuple(readout_mask.shape)}"
        )
    if readout_mask.device != sample.node_feats.device:
        raise ValueError("readout_mask and node_feats must be on the same device")
    if not bool(readout_mask.any().item()):
        raise ValueError("readout_mask must select at least one node")
    return readout_mask


def _qm9_sample_id(data: object, *, row_index: int) -> str:
    raw_index = getattr(data, "idx", None)
    if raw_index is None:
        raw_index_text = "unknown"
    else:
        value = torch.as_tensor(raw_index)
        if value.numel() != 1:
            raise ValueError("QM9 data.idx must contain exactly one raw index")
        raw_index_text = str(int(value.item()))
    name = getattr(data, "name", None)
    name_text = "unknown" if name is None or not str(name).strip() else str(name)
    return (
        f"qm9-row-{row_index}-raw-index-{raw_index_text}-name-{name_text}"
    )


def _make_synthetic_sample(
    index: int, node_dim: int, min_nodes: int, max_nodes: int, seed: int
) -> GraphSample:
    generator = torch.Generator().manual_seed(seed + index * 1009)
    n_nodes = int(
        torch.randint(min_nodes, max_nodes + 1, (1,), generator=generator).item()
    )
    atom_ids = torch.randint(0, node_dim, (n_nodes,), generator=generator)
    node_feats = torch.nn.functional.one_hot(atom_ids, num_classes=node_dim).to(
        dtype=torch.float32
    )
    pos = torch.randn(n_nodes, 3, generator=generator)
    pos = pos - pos.mean(dim=0, keepdim=True)

    # The matrix-multiplication distance is thread-order dependent, so the
    # synthetic target would not be reproducible above 25 nodes.
    dist = torch.cdist(pos, pos, compute_mode="donot_use_mm_for_euclid_dist")
    charge = (atom_ids.float() + 1.0) / node_dim
    pair_weight = charge.unsqueeze(0) * charge.unsqueeze(1)
    upper = torch.triu(torch.ones_like(dist, dtype=torch.bool), diagonal=1)
    target = (torch.exp(-dist) * pair_weight)[upper].sum().reshape(1) / n_nodes
    return GraphSample(
        node_feats=node_feats, pos=pos, target=target, sample_id=f"synthetic-{index}"
    )
