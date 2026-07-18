from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class GraphSample:
    node_feats: torch.Tensor
    pos: torch.Tensor
    target: torch.Tensor
    sample_id: str


@dataclass(frozen=True)
class GraphBatch:
    node_feats: torch.Tensor
    pos: torch.Tensor
    batch: torch.Tensor
    target: torch.Tensor
    sample_ids: tuple[str, ...]

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
    return GraphBatch(
        node_feats=node_feats,
        pos=pos,
        batch=batch,
        target=target,
        sample_ids=tuple(sample.sample_id for sample in samples),
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
    root: str | Path, target_index: int = 4, limit: int | None = None
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
            )
        )
    return samples


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

    dist = torch.cdist(pos, pos)
    charge = (atom_ids.float() + 1.0) / node_dim
    pair_weight = charge.unsqueeze(0) * charge.unsqueeze(1)
    upper = torch.triu(torch.ones_like(dist, dtype=torch.bool), diagonal=1)
    target = (torch.exp(-dist) * pair_weight)[upper].sum().reshape(1) / n_nodes
    return GraphSample(
        node_feats=node_feats, pos=pos, target=target, sample_id=f"synthetic-{index}"
    )
