from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

from .neighbors import PackedNeighborGraph


@dataclass(frozen=True)
class GraphSample:
    node_feats: torch.Tensor
    pos: torch.Tensor
    target: torch.Tensor
    sample_id: str
    edge_index: torch.Tensor | None = None
    readout_mask: torch.Tensor | None = None


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
        return GraphBatch(
            node_feats=self.node_feats.to(device=device, dtype=value_dtype),
            pos=self.pos.to(device=device, dtype=geometry_dtype),
            batch=self.batch.to(device=device),
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
        )


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
    if all(edge_presence):
        offset_edges = []
        node_offset = 0
        for sample in samples:
            sample_edges = _validated_sample_edge_index(sample)
            offset_edges.append(sample_edges + node_offset)
            node_offset += sample.node_feats.shape[0]
        edge_index = torch.cat(offset_edges, dim=1)
    readout_presence = [sample.readout_mask is not None for sample in samples]
    if any(readout_presence) and not all(readout_presence):
        raise ValueError("all samples must either provide readout_mask or omit it")
    readout_mask = None
    if all(readout_presence):
        readout_mask = torch.cat(
            [_validated_sample_readout_mask(sample) for sample in samples],
            dim=0,
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
