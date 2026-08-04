#!/usr/bin/env python3
"""Bounded real-data validation for the canonical ELA public API.

This runner intentionally keeps dataset adapters outside the package.  Model
execution uses only ``ELA`` and ``ELAGraph`` from the public root.  The private
hooks below are used solely to disable zero-initialized lanes in paired
architecture ablations; they do not create another model or graph API.
"""

from __future__ import annotations

import os

# Deterministic CUDA GEMM must be configured before the first CUDA context.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Protocol, Sequence

import torch
import torch.nn.functional as F

from equivariant_linear_attention import ELA, ELAGraph
from equivariant_linear_attention.nn.multipoles import (
    _set_l1_l2_closure_enabled,
)


QM9_FILE_HASHES = {
    "raw/gdb9.sdf": (
        "98c4e97d50ac549b8c9f0b2114b348a9a944718e17e50d9a724b729f1deaa28e"
    ),
    "raw/gdb9.sdf.csv": (
        "73a67793e3cfa9660f001278bd019c143f57e4785db537a01811cf2ce72aa7eb"
    ),
    "processed/data_v3.pt": (
        "9254af077d7bc651631bb56a3a689fb41004731b413bdd0ec8c6efa318229f83"
    ),
}
QM9_TARGET_INDEX = 4
QM9_INPUT_DIM = 11
LBA_REVISION = "f93dd2d150a47c270f624620f84e07451a158705"
LBA_INPUT_DIM = 140
LBA_MAX_TOKEN = 137
LBA_RELATIONS = 3
LBA_ID30_TRAIN_SIZE = 3507
LBA_ID30_VALIDATION_SIZE = 466
LBA_ID30_DIRECTED_EDGES_WITH_SELF = 32302952
LBA_TRAIN_FILES = (
    "atom3d-lba-train-00000-of-00002.arrow",
    "atom3d-lba-train-00001-of-00002.arrow",
)
LBA_VALIDATION_FILE = "atom3d-lba-val.arrow"
STATIC_ARMS = ("full", "no-relation", "no-cg12", "no-multiscale")
_RELATION_SUFFIXES = (
    "relation_score_bias",
    "relation_radial_scale",
    "relation_value_gate",
)
_MULTISCALE_SUFFIXES = (
    "local_scale_score_mix",
    "local_scale_value_mix",
)
_CG12_TOKEN = "tensor_closure.l1_l2_"


class _GraphDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> ELAGraph: ...

    def targets(self, indices: Sequence[int]) -> torch.Tensor: ...


@dataclass(frozen=True)
class _TargetNormalizer:
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, target: torch.Tensor) -> _TargetNormalizer:
        values = target.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        if values.numel() == 0 or not bool(torch.isfinite(values).all().item()):
            raise ValueError("normalizer targets must be nonempty and finite")
        std = values.std(unbiased=False).clamp_min(1e-12)
        return cls(mean=values.mean(), std=std)

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.mean.to(value)) / self.std.to(value)

    def denormalize(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.std.to(value) + self.mean.to(value)

    def as_dict(self) -> dict[str, float]:
        return {"mean": float(self.mean.item()), "std": float(self.std.item())}


@dataclass
class _TaskData:
    train_dataset: _GraphDataset
    evaluation_dataset: _GraphDataset
    train_indices: tuple[int, ...]
    evaluation_indices: tuple[int, ...]
    input_dim: int
    edge_types: int
    prediction: str
    evaluation_split: str
    normalizer: _TargetNormalizer
    data: dict[str, Any]
    split: dict[str, Any]
    topology: dict[str, Any]
    access: dict[str, bool]
    limitations: list[str]


class _QM9Dataset:
    def __init__(self, root: Path) -> None:
        try:
            from torch_geometric.datasets import QM9
        except ModuleNotFoundError as exc:
            raise ImportError(
                "QM9 validation needs the locked optional dependencies; run "
                "with `uv run --locked --extra qm9 python ...`"
            ) from exc
        self.dataset = QM9(root=str(root))
        if len(self.dataset) < 130_000:
            raise ValueError("cached QM9 must contain at least 130,000 rows")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> ELAGraph:
        data = self.dataset[index]
        feature = data.x.to(dtype=torch.float32)
        position = data.pos.to(dtype=torch.float32)
        target = data.y.reshape(-1)[QM9_TARGET_INDEX].to(dtype=torch.float32)
        raw_index = int(data.idx.reshape(-1)[0].item())
        name = str(data.name)
        return ELAGraph(
            x=feature,
            pos=position,
            y=target,
            ids=(f"qm9-row-{index}-raw-index-{raw_index}-name-{name}",),
        )

    def targets(self, indices: Sequence[int]) -> torch.Tensor:
        storage = getattr(self.dataset, "_data", None)
        values = getattr(storage, "y", None)
        if isinstance(values, torch.Tensor) and values.ndim == 2:
            selection = torch.as_tensor(indices, dtype=torch.long)
            return values[selection, QM9_TARGET_INDEX].to(dtype=torch.float64)
        return torch.stack(
            [self[index].y.to(dtype=torch.float64) for index in indices]
        )


class _LBADataset:
    def __init__(
        self,
        rows: Any,
        *,
        split: str,
        limit: int | None = None,
        cutoff: float = 6.0,
        intra_k: int = 16,
        cross_k: int = 16,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("LBA adapter permits only train or val")
        available = len(rows)
        if limit is None:
            limit = available
        if limit <= 0 or limit > available:
            raise ValueError(f"invalid {split} limit {limit} for {available} rows")
        self.rows = rows
        self.split = split
        self.limit = limit
        self.cutoff = cutoff
        self.intra_k = intra_k
        self.cross_k = cross_k
        self._cache: list[ELAGraph | None] = [None] * limit

    def __len__(self) -> int:
        return self.limit

    def __getitem__(self, index: int) -> ELAGraph:
        if not 0 <= index < self.limit:
            raise IndexError(index)
        cached = self._cache[index]
        if cached is None:
            cached = _lba_row_to_graph(
                self.rows[index],
                split=self.split,
                row_index=index,
                cutoff=self.cutoff,
                intra_k=self.intra_k,
                cross_k=self.cross_k,
            )
            self._cache[index] = cached
        return cached

    def targets(self, indices: Sequence[int]) -> torch.Tensor:
        return torch.tensor(
            [float(self.rows[index]["labels"]) for index in indices],
            dtype=torch.float64,
        )


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _indices_sha256(indices: Sequence[int]) -> str:
    return hashlib.sha256(
        ",".join(str(index) for index in indices).encode("ascii")
    ).hexdigest()


def _update_tensor_hash(
    digest: Any,
    name: str,
    value: torch.Tensor,
) -> None:
    tensor = value.detach().to(device="cpu").contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        _update_tensor_hash(digest, name, value)
    return digest.hexdigest()


def _coordinate_state_sha256(model: torch.nn.Module) -> str | None:
    selected = [
        (name, value)
        for name, value in model.state_dict().items()
        if name.startswith("coordinate_head.")
        or name.startswith("coordinate_gate.")
    ]
    if not selected:
        return None
    digest = hashlib.sha256()
    for name, value in selected:
        _update_tensor_hash(digest, name, value)
    return digest.hexdigest()


def _schema_sha256(model: torch.nn.Module) -> str:
    schema = [
        (name, list(parameter.shape), str(parameter.dtype))
        for name, parameter in model.named_parameters()
    ]
    return hashlib.sha256(
        json.dumps(schema, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _prediction_sha256(value: torch.Tensor) -> str:
    digest = hashlib.sha256()
    _update_tensor_hash(digest, "prediction", value)
    return digest.hexdigest()


def _source_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "src").rglob("*.py"))
    paths.append(Path(__file__).resolve())
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_metadata() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"sha": None, "dirty": None}
    return {"sha": sha, "dirty": bool(status)}


def _lba_cache_root(root: Path) -> Path:
    return (
        root
        / "vector-institute___atom3d-lba"
        / "default"
        / "0.0.0"
        / LBA_REVISION
    )


def _lba_arrow_paths(root: Path, split: str) -> tuple[Path, ...]:
    """Resolve only explicitly admitted shards; never glob the cache."""

    if split == "train":
        names = LBA_TRAIN_FILES
    elif split == "val":
        names = (LBA_VALIDATION_FILE,)
    else:
        raise ValueError(
            "LBA real-data validation admits only train or val; the test "
            "shard is intentionally unreachable"
        )
    paths = tuple(_lba_cache_root(root) / name for name in names)
    missing = tuple(path for path in paths if not path.is_file())
    if missing:
        rendered = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"missing pinned LBA cache shard(s): {rendered}")
    return paths


def _open_lba_split(root: Path, split: str) -> tuple[Any, tuple[Path, ...]]:
    paths = _lba_arrow_paths(root, split)
    try:
        from datasets import Dataset, concatenate_datasets
    except ModuleNotFoundError as exc:
        raise ImportError(
            "LBA validation needs the locked optional dependencies; run with "
            "`uv run --locked --extra pdbbind python ...`"
        ) from exc
    parts = [Dataset.from_file(str(path)) for path in paths]
    rows = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    return rows, paths


def _row_identity(
    token: torch.Tensor,
    position: torch.Tensor,
    token_type: torch.Tensor,
) -> str:
    digest = hashlib.sha256()
    _update_tensor_hash(digest, "input_ids", token)
    _update_tensor_hash(digest, "coords", position)
    _update_tensor_hash(digest, "token_type_ids", token_type)
    return digest.hexdigest()[:16]


def _lba_row_to_graph(
    row: Any,
    *,
    split: str,
    row_index: int,
    cutoff: float = 6.0,
    intra_k: int = 16,
    cross_k: int = 16,
) -> ELAGraph:
    if split not in {"train", "val"}:
        raise ValueError("LBA row conversion permits only train or val")
    required = ("input_ids", "coords", "labels", "token_type_ids")
    missing = [name for name in required if name not in row]
    if missing:
        raise ValueError(f"LBA row is missing fields: {', '.join(missing)}")
    token = torch.as_tensor(row["input_ids"], dtype=torch.long)
    position64 = torch.as_tensor(row["coords"], dtype=torch.float64)
    token_type = torch.as_tensor(row["token_type_ids"], dtype=torch.long)
    if token.ndim != 1 or token_type.shape != token.shape:
        raise ValueError("LBA input_ids and token_type_ids must have shape (N,)")
    if position64.shape != (token.shape[0], 3):
        raise ValueError("LBA coords must have shape (N,3)")
    if not bool(torch.isfinite(position64).all().item()):
        raise ValueError("LBA coords must be finite")
    if token.numel() == 0 or bool(((token < 1) | (token > LBA_MAX_TOKEN)).any()):
        raise ValueError(f"LBA atom tokens must lie in [1,{LBA_MAX_TOKEN}]")
    if bool(((token_type < 0) | (token_type > 2)).any()):
        raise ValueError("LBA token_type_ids must contain only 0, 1, or 2")
    retained = token_type != 0
    retained_type = token_type[retained]
    if not bool((retained_type == 1).any().item()):
        raise ValueError("LBA row has no pocket atoms")
    if not bool((retained_type == 2).any().item()):
        raise ValueError("LBA row has no ligand atoms")
    label = torch.as_tensor(row["labels"], dtype=torch.float64)
    if label.numel() != 1 or not bool(torch.isfinite(label).all().item()):
        raise ValueError("LBA labels must contain one finite scalar")
    retained_token = token[retained]
    retained_position64 = position64[retained]
    ligand = retained_type == 2
    feature = torch.cat(
        (
            F.one_hot(retained_token, num_classes=LBA_MAX_TOKEN + 1),
            F.one_hot(retained_type - 1, num_classes=2),
        ),
        dim=-1,
    ).to(dtype=torch.float32)
    edge_index, edge_type = _segment_balanced_topology(
        retained_position64,
        ligand,
        intra_k=intra_k,
        cross_k=cross_k,
        cutoff=cutoff,
    )
    identity = _row_identity(retained_token, retained_position64, retained_type)
    return ELAGraph(
        x=feature,
        pos=retained_position64.to(dtype=torch.float32),
        edge_index=edge_index,
        edge_type=edge_type,
        y=label.reshape(()).to(dtype=torch.float32),
        ids=(f"atom3d-lba:{split}:{row_index:07d}:{identity}",),
    )


def _segment_balanced_topology(
    position: torch.Tensor,
    ligand: torch.Tensor,
    *,
    intra_k: int,
    cross_k: int,
    cutoff: float,
    chunk_nodes: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build deterministic public sender-to-receiver LBA candidates."""

    if position.ndim != 2 or position.shape[1] != 3 or position.shape[0] == 0:
        raise ValueError("position must have shape (N,3) with N > 0")
    if ligand.shape != (position.shape[0],) or ligand.dtype != torch.bool:
        raise ValueError("ligand must be boolean with shape (N,)")
    for name, value in (("intra_k", intra_k), ("cross_k", cross_k)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if not math.isfinite(float(cutoff)) or cutoff <= 0.0:
        raise ValueError("cutoff must be finite and positive")
    if chunk_nodes <= 0:
        raise ValueError("chunk_nodes must be positive")

    geometry = position.detach().to(device="cpu", dtype=torch.float64)
    segment = ligand.detach().to(device="cpu")
    nodes = geometry.shape[0]
    node_index = torch.arange(nodes, dtype=torch.long)
    cutoff_squared = float(cutoff) ** 2
    receivers: list[torch.Tensor] = []
    senders: list[torch.Tensor] = []
    for start in range(0, nodes, chunk_nodes):
        stop = min(start + chunk_nodes, nodes)
        rows = node_index[start:stop]
        displacement = geometry[start:stop, None, :] - geometry[None, :, :]
        squared = displacement.square().sum(dim=-1)
        self_edge = rows[:, None] == node_index[None, :]
        within = squared < cutoff_squared
        same = segment[start:stop, None] == segment[None, :]
        selections = [self_edge]
        for relation_mask, budget in ((same, intra_k), (~same, cross_k)):
            eligible = within & relation_mask & ~self_edge
            count = min(budget, nodes)
            if count == 0:
                selections.append(torch.zeros_like(eligible))
                continue
            ranked = torch.where(eligible, squared, math.inf)
            boundary = ranked.topk(count, dim=-1, largest=False).values[:, -1:]
            selections.append(eligible & (squared <= boundary))
        packed = torch.stack(selections, dim=1).reshape(stop - start, -1)
        row, column = packed.nonzero(as_tuple=True)
        receivers.append(rows[row])
        senders.append(column.remainder(nodes))

    receiver = torch.cat(receivers)
    sender = torch.cat(senders)
    source_ligand = segment[sender]
    target_ligand = segment[receiver]
    relation = torch.where(
        source_ligand != target_ligand,
        torch.full_like(sender, 2),
        torch.where(source_ligand, torch.ones_like(sender), torch.zeros_like(sender)),
    )
    # ELAGraph's public convention is source/sender first, target/receiver second.
    return torch.stack((sender, receiver)), relation


def _topology_manifest(datasets: Sequence[_GraphDataset]) -> dict[str, Any]:
    sample_digest = hashlib.sha256()
    edge_index_digest = hashlib.sha256()
    relation_digest = hashlib.sha256()
    edge_digest = hashlib.sha256()
    joint_digest = hashlib.sha256()
    edge_count = 0
    graph_count = 0
    for dataset in datasets:
        for index in range(len(dataset)):
            graph = dataset[index]
            if graph.edge_index is None or graph.edge_type is None:
                raise RuntimeError("LBA topology manifest requires typed edges")
            identifier = str(graph.ids[0]) if graph.ids is not None else str(index)
            encoded_identifier = identifier.encode("utf-8")
            sample_digest.update(encoded_identifier)
            joint_digest.update(encoded_identifier)
            _update_tensor_hash(edge_index_digest, "edge_index", graph.edge_index)
            _update_tensor_hash(relation_digest, "edge_type", graph.edge_type)
            _update_tensor_hash(edge_digest, "edge_index", graph.edge_index)
            _update_tensor_hash(edge_digest, "edge_type", graph.edge_type)
            _update_tensor_hash(joint_digest, "edge_index", graph.edge_index)
            _update_tensor_hash(joint_digest, "edge_type", graph.edge_type)
            edge_count += graph.edge_index.shape[1]
            graph_count += 1
    return {
        "sample_identity_sha256": sample_digest.hexdigest(),
        "edge_index_sha256": edge_index_digest.hexdigest(),
        "edge_relation_sha256": relation_digest.hexdigest(),
        "edge_topology_sha256": edge_digest.hexdigest(),
        "joint_sha256": joint_digest.hexdigest(),
        "graphs": graph_count,
        "directed_edges_with_self": edge_count,
    }


def _combine_topology_manifests(
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    payload = json.dumps(
        {"train": train, "validation": validation},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "split_receipts_sha256": hashlib.sha256(payload).hexdigest(),
        "graphs": int(train["graphs"]) + int(validation["graphs"]),
        "directed_edges_with_self": (
            int(train["directed_edges_with_self"])
            + int(validation["directed_edges_with_self"])
        ),
    }


def _lba_id30_identity_gate(
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    observed = {
        "train_graphs": int(train["graphs"]),
        "validation_graphs": int(validation["graphs"]),
        "directed_edges_with_self": (
            int(train["directed_edges_with_self"])
            + int(validation["directed_edges_with_self"])
        ),
    }
    expected = {
        "train_graphs": LBA_ID30_TRAIN_SIZE,
        "validation_graphs": LBA_ID30_VALIDATION_SIZE,
        "directed_edges_with_self": LBA_ID30_DIRECTED_EDGES_WITH_SELF,
    }
    passed = observed == expected
    gate = {"passed": passed, "expected": expected, "observed": observed}
    if not passed:
        raise ValueError(f"LBA ID30 frozen identity mismatch: {gate}")
    return gate


def _qm9_data(args: argparse.Namespace) -> _TaskData:
    root = args.data_root
    actual = {name: _file_sha256(root / name) for name in QM9_FILE_HASHES}
    mismatched = [name for name, value in actual.items() if value != QM9_FILE_HASHES[name]]
    if mismatched:
        raise ValueError(f"QM9 data identity mismatch: {', '.join(mismatched)}")
    dataset = _QM9Dataset(root)
    if args.train_size + args.val_size >= args.num_samples:
        raise ValueError("QM9 train+val sizes must leave an unused test partition")
    if args.num_samples > len(dataset):
        raise ValueError("QM9 num_samples exceeds the cached dataset")
    order = torch.randperm(
        args.num_samples,
        generator=torch.Generator().manual_seed(args.split_seed),
    ).tolist()
    train = tuple(order[: args.train_size])
    validation_full = tuple(
        order[args.train_size : args.train_size + args.val_size]
    )
    test = tuple(order[args.train_size + args.val_size :])
    validation = (
        validation_full
        if args.validation_limit == 0
        else validation_full[: args.validation_limit]
    )
    normalizer = _TargetNormalizer.fit(dataset.targets(train))
    return _TaskData(
        train_dataset=dataset,
        evaluation_dataset=dataset,
        train_indices=train,
        evaluation_indices=validation,
        input_dim=QM9_INPUT_DIM,
        edge_types=0,
        prediction="graph_mean",
        evaluation_split="validation",
        normalizer=normalizer,
        data={
            "dataset": "QM9",
            "root": str(root),
            "file_sha256": actual,
            "cached_rows": len(dataset),
            "target_index": QM9_TARGET_INDEX,
            "target_name": "gap",
            "target_unit": "eV",
            "node_features": "PyG 11D raw invariant atom features",
        },
        split={
            "kind": "seeded_random_row_warm_start",
            "seed": args.split_seed,
            "num_samples_boundary": args.num_samples,
            "train_size": len(train),
            "validation_size_full": len(validation_full),
            "validation_size_evaluated": len(validation),
            "unused_test_size": len(test),
            "train_indices_sha256": _indices_sha256(train),
            "validation_indices_sha256": _indices_sha256(validation_full),
            "evaluated_validation_indices_sha256": _indices_sha256(validation),
            "unused_test_indices_sha256": _indices_sha256(test),
        },
        topology={
            "kind": "automatic_radius",
            "cutoff_angstrom": args.cutoff,
            "self_edges": False,
            "graph_ingestion_in_step_latency": True,
        },
        access={
            "train_labels_accessed": True,
            "validation_labels_accessed": True,
            "test_shard_opened": False,
            "processed_monolith_including_test_labels_loaded": True,
            "test_labels_accessed": True,
            "test_indices_indexed": False,
            "test_labels_used": False,
            "test_evaluated": False,
        },
        limitations=[
            "random-row warm split, not scaffold or cold-entity generalization",
            "historical repository runs already accessed the test partition",
            "bounded update counts are architecture screens, not convergence claims",
        ],
    )


def _lba_overfit_data(args: argparse.Namespace) -> _TaskData:
    rows, paths = _open_lba_split(args.data_root, "train")
    dataset = _LBADataset(rows, split="train", limit=16, cutoff=args.cutoff)
    indices = tuple(range(len(dataset)))
    normalizer = _TargetNormalizer.fit(dataset.targets(indices))
    topology = {
        "kind": "segment_balanced_knn",
        "cutoff_angstrom": args.cutoff,
        "intra_k": 16,
        "cross_k": 16,
        "relation_types": ["pocket-pocket", "ligand-ligand", "cross"],
        **_topology_manifest((dataset,)),
        "graph_ingestion_in_step_latency": True,
    }
    return _TaskData(
        train_dataset=dataset,
        evaluation_dataset=dataset,
        train_indices=indices,
        evaluation_indices=indices,
        input_dim=LBA_INPUT_DIM,
        edge_types=LBA_RELATIONS,
        prediction="ligand_mean",
        evaluation_split="train",
        normalizer=normalizer,
        data={
            "dataset": "vector-institute/atom3d-lba",
            "revision": LBA_REVISION,
            "root": str(args.data_root),
            "opened_splits": ["train"],
            "arrow_sha256": {path.name: _file_sha256(path) for path in paths},
            "node_features": "138-way opaque token + pocket/ligand one-hot",
            "target": "affinity pK",
        },
        split={
            "kind": "frozen_train_only_capacity_subset",
            "indices": list(indices),
            "indices_sha256": _indices_sha256(indices),
        },
        topology=topology,
        access={
            "train_labels_accessed": True,
            "validation_labels_accessed": False,
            "test_shard_opened": False,
            "access_scope": "this_run",
            "test_label_storage_materialized_by_this_run": False,
            "historical_local_test_row_and_label_materialized": True,
            "test_labels_accessed": False,
            "test_indices_indexed": False,
            "test_labels_used": False,
            "test_evaluated": False,
        },
        limitations=[
            "train-only 16-complex overfit/capacity check",
            "no validation or generalization evidence",
            "bound pocket and ligand, not apo or pose-robust evaluation",
            "upstream cache metadata has no populated license field",
        ],
    )


def _lba_id30_data(args: argparse.Namespace) -> _TaskData:
    train_rows, train_paths = _open_lba_split(args.data_root, "train")
    validation_rows, validation_paths = _open_lba_split(args.data_root, "val")
    if args.train_limit == 0 and len(train_rows) != LBA_ID30_TRAIN_SIZE:
        raise ValueError(
            f"LBA ID30 train size mismatch: {len(train_rows)} != "
            f"{LBA_ID30_TRAIN_SIZE}"
        )
    if args.validation_limit == 0 and len(validation_rows) != LBA_ID30_VALIDATION_SIZE:
        raise ValueError(
            f"LBA ID30 validation size mismatch: {len(validation_rows)} != "
            f"{LBA_ID30_VALIDATION_SIZE}"
        )
    train_limit = None if args.train_limit == 0 else args.train_limit
    validation_limit = None if args.validation_limit == 0 else args.validation_limit
    train_dataset = _LBADataset(
        train_rows,
        split="train",
        limit=train_limit,
        cutoff=args.cutoff,
    )
    validation_dataset = _LBADataset(
        validation_rows,
        split="val",
        limit=validation_limit,
        cutoff=args.cutoff,
    )
    train = tuple(range(len(train_dataset)))
    validation = tuple(range(len(validation_dataset)))
    normalizer = _TargetNormalizer.fit(train_dataset.targets(train))
    train_topology = _topology_manifest((train_dataset,))
    validation_topology = _topology_manifest((validation_dataset,))
    combined_topology = _combine_topology_manifests(
        train_topology,
        validation_topology,
    )
    identity_gate = None
    if args.train_limit == 0 and args.validation_limit == 0:
        identity_gate = _lba_id30_identity_gate(
            train_topology,
            validation_topology,
        )
    topology = {
        "kind": "segment_balanced_knn",
        "cutoff_angstrom": args.cutoff,
        "intra_k": 16,
        "cross_k": 16,
        "relation_types": ["pocket-pocket", "ligand-ligand", "cross"],
        "train": train_topology,
        "validation": validation_topology,
        "combined": combined_topology,
        "frozen_identity_gate": identity_gate,
        "graph_ingestion_in_step_latency": True,
    }
    all_paths = (*train_paths, *validation_paths)
    return _TaskData(
        train_dataset=train_dataset,
        evaluation_dataset=validation_dataset,
        train_indices=train,
        evaluation_indices=validation,
        input_dim=LBA_INPUT_DIM,
        edge_types=LBA_RELATIONS,
        prediction="ligand_mean",
        evaluation_split="validation",
        normalizer=normalizer,
        data={
            "dataset": "vector-institute/atom3d-lba",
            "revision": LBA_REVISION,
            "root": str(args.data_root),
            "opened_splits": ["train", "val"],
            "arrow_sha256": {path.name: _file_sha256(path) for path in all_paths},
            "node_features": "138-way opaque token + pocket/ligand one-hot",
            "target": "affinity pK",
        },
        split={
            "kind": "official_ID30_train_validation",
            "train_size": len(train),
            "validation_size": len(validation),
            "train_indices_sha256": _indices_sha256(train),
            "validation_indices_sha256": _indices_sha256(validation),
            "train_limited": args.train_limit != 0,
            "validation_limited": args.validation_limit != 0,
        },
        topology=topology,
        access={
            "train_labels_accessed": True,
            "validation_labels_accessed": True,
            "test_shard_opened": False,
            "access_scope": "this_run",
            "test_label_storage_materialized_by_this_run": False,
            "historical_local_test_row_and_label_materialized": True,
            "test_labels_accessed": False,
            "test_indices_indexed": False,
            "test_labels_used": False,
            "test_evaluated": False,
        },
        limitations=[
            "validation has prior architecture-selection access",
            "bound-complex validation, not cold-target, apo, or pose robustness",
            "no test-shard path exists in this runner",
            "upstream cache metadata has no populated license field",
        ],
    )


def _configure_determinism(seed: int, threads: int) -> dict[str, Any]:
    if seed < 0 or threads <= 0:
        raise ValueError("seed must be nonnegative and threads positive")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {
        "seed": seed,
        "threads": threads,
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }


def _arm_parameter_names(model: ELA, arm: str) -> tuple[str, ...]:
    if arm == "no-relation":
        return tuple(
            name
            for name, _ in model.named_parameters()
            if name.endswith(_RELATION_SUFFIXES)
        )
    if arm == "no-cg12":
        return tuple(
            name for name, _ in model.named_parameters() if _CG12_TOKEN in name
        )
    if arm == "no-multiscale":
        return tuple(
            name
            for name, _ in model.named_parameters()
            if name.endswith(_MULTISCALE_SUFFIXES)
        )
    if arm == "full":
        return ()
    raise ValueError(f"unsupported paired arm: {arm}")


def _apply_arm_control(model: ELA, arm: str) -> tuple[str, ...]:
    names = _arm_parameter_names(model, arm)
    selected = set(names)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in selected:
                parameter.zero_()
                parameter.requires_grad_(False)
    if arm == "no-cg12":
        changed = _set_l1_l2_closure_enabled(model, False)
        if changed != model.config.depth:
            raise RuntimeError("CG12 ablation did not reach every ELA layer")
    elif arm == "no-multiscale":
        for layer in model.layers:
            layer._set_multiscale_local_enabled(False)
    return names


def _make_model(
    *,
    input_dim: int,
    edge_types: int,
    width: int,
    depth: int,
    cutoff: float,
    update_positions: bool,
) -> ELA:
    return ELA(
        f"{input_dim}x0e",
        "1x0e",
        width=width,
        depth=depth,
        cutoff=cutoff,
        edge_types=edge_types,
        update_positions=update_positions,
    )


def _build_arm_models(
    *,
    input_dim: int,
    edge_types: int,
    width: int,
    depth: int,
    cutoff: float,
    arms: Sequence[str],
    include_stagewise: bool,
    seed: int,
) -> tuple[dict[str, ELA], dict[str, Any]]:
    torch.manual_seed(seed)
    base = _make_model(
        input_dim=input_dim,
        edge_types=edge_types,
        width=width,
        depth=depth,
        cutoff=cutoff,
        update_positions=False,
    )
    base_state = copy.deepcopy(base.state_dict())
    base_schema = _schema_sha256(base)
    base_state_hash = _state_sha256(base)
    models: dict[str, ELA] = {}
    controls: dict[str, Any] = {}
    for arm in arms:
        model = _make_model(
            input_dim=input_dim,
            edge_types=edge_types,
            width=width,
            depth=depth,
            cutoff=cutoff,
            update_positions=False,
        )
        model.load_state_dict(base_state, strict=True)
        frozen = _apply_arm_control(model, arm)
        models[arm] = model
        controls[arm] = {
            "paired_schema": True,
            "base_schema_sha256": base_schema,
            "base_state_sha256": base_state_hash,
            "initial_state_sha256": _state_sha256(model),
            "disabled_lane_parameters": list(frozen),
        }
    if include_stagewise:
        torch.manual_seed(seed)
        stagewise = _make_model(
            input_dim=input_dim,
            edge_types=edge_types,
            width=width,
            depth=depth,
            cutoff=cutoff,
            update_positions=True,
        )
        common = {
            name: value
            for name, value in base_state.items()
            if name in stagewise.state_dict()
            and stagewise.state_dict()[name].shape == value.shape
        }
        missing, unexpected = stagewise.load_state_dict(common, strict=False)
        models["stagewise"] = stagewise
        controls["stagewise"] = {
            "paired_schema": False,
            "role": "separate_stagewise_coordinate_functionality_arm",
            "common_state_tensors_loaded": len(common),
            "stagewise_only_state_tensors": sorted(missing),
            "unexpected_state_tensors": sorted(unexpected),
            "initial_state_sha256": _state_sha256(stagewise),
        }
    return models, {
        "base_schema_sha256": base_schema,
        "base_state_sha256": base_state_hash,
        "controls": controls,
    }


def _ligand_mean(node: torch.Tensor, graph: ELAGraph) -> torch.Tensor:
    if node.ndim != 2:
        raise ValueError("node prediction must have shape (N,D)")
    if graph.x.shape[1] != LBA_INPUT_DIM:
        raise ValueError("ligand readout requires the canonical 140D LBA input")
    ligand = graph.x[:, -1] > 0.5
    batch = (
        torch.zeros(graph.num_nodes, device=node.device, dtype=torch.long)
        if graph.batch is None
        else graph.batch
    )
    graphs = graph.num_graphs
    count = torch.bincount(batch[ligand], minlength=graphs)
    if bool((count == 0).any().item()):
        raise ValueError("every LBA graph must retain at least one ligand atom")
    result = node.new_zeros((graphs, node.shape[1]))
    result.index_add_(0, batch[ligand], node[ligand])
    return result / count[:, None].to(dtype=node.dtype)


def _predict(model: ELA, graph: ELAGraph, kind: str) -> tuple[torch.Tensor, ELAGraph]:
    output = model(graph)
    if kind == "graph_mean":
        if output.graph_x is None:
            raise RuntimeError("ELA did not return graph_x")
        prediction = output.graph_x
    elif kind == "ligand_mean":
        prediction = _ligand_mean(output.x, graph)
    else:
        raise ValueError(f"unknown prediction kind: {kind}")
    if prediction.shape[1] != 1:
        raise RuntimeError("real-data runner expects one scalar output")
    return prediction[:, 0], output


def _batch(dataset: _GraphDataset, indices: Sequence[int]) -> ELAGraph:
    return ELAGraph.collate([dataset[index] for index in indices])


def _cyclic_indices(
    indices: Sequence[int],
    *,
    step: int,
    batch_size: int,
    seed: int,
) -> tuple[int, ...]:
    if not indices:
        raise ValueError("training indices must be nonempty")
    start = step * batch_size
    selected: list[int] = []
    while len(selected) < batch_size:
        epoch = start // len(indices)
        offset = start % len(indices)
        order = torch.randperm(
            len(indices),
            generator=torch.Generator().manual_seed(seed + epoch),
        ).tolist()
        take = min(batch_size - len(selected), len(indices) - offset)
        selected.extend(indices[order[position]] for position in range(offset, offset + take))
        start += take
    return tuple(selected)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


@torch.no_grad()
def _evaluate(
    model: ELA,
    dataset: _GraphDataset,
    indices: Sequence[int],
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    kind: str,
    normalizer: _TargetNormalizer,
) -> dict[str, float | int]:
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    coordinate_delta_norms: list[torch.Tensor] = []
    started = time.perf_counter()
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        graph = _batch(dataset, selected).to(
            device,
            dtype=dtype,
            non_blocking=device.type == "cuda",
        )
        prediction, output = _predict(model, graph, kind)
        if graph.y is None:
            raise RuntimeError("evaluation graph is missing targets")
        predictions.append(normalizer.denormalize(prediction).double().cpu())
        targets.append(graph.y.reshape(-1).double().cpu())
        if output.delta is not None:
            coordinate_delta_norms.append(
                output.delta.detach().double().norm(dim=-1).cpu()
            )
    _synchronize(device)
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    error = prediction - target
    normalized_error = error / float(normalizer.std.item())
    delta = (
        None
        if not coordinate_delta_norms
        else torch.cat(coordinate_delta_norms)
    )
    return {
        "count": target.numel(),
        "mae": float(error.abs().mean().item()),
        "rmse": float(error.square().mean().sqrt().item()),
        "normalized_mse": float(normalized_error.square().mean().item()),
        "coordinate_delta_mean": (
            None if delta is None else float(delta.mean().item())
        ),
        "coordinate_delta_max": (
            None if delta is None else float(delta.max().item())
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _run_arm(
    *,
    arm: str,
    model: ELA,
    task: _TaskData,
    steps: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    order_seed: int,
) -> dict[str, Any]:
    model = model.to(device=device, dtype=dtype)
    initial_state_hash = _state_sha256(model)
    initial_coordinate_state_hash = _coordinate_state_sha256(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    probe_indices = _cyclic_indices(
        task.train_indices,
        step=0,
        batch_size=batch_size,
        seed=order_seed,
    )
    probe = _batch(task.train_dataset, probe_indices).to(device, dtype=dtype)
    if probe.y is None:
        raise RuntimeError("training probe is missing targets")
    model.eval()
    with torch.no_grad():
        initial_prediction, initial_output = _predict(model, probe, task.prediction)
        initial_target = task.normalizer.normalize(probe.y.reshape(-1))
        initial_loss = (initial_prediction - initial_target).square().mean()
        initial_delta = (
            0.0
            if initial_output.delta is None
            else float(initial_output.delta.norm(dim=-1).max().item())
        )
        initial_prediction_hash = _prediction_sha256(initial_prediction)
    del initial_output, initial_prediction, initial_target, probe
    initial_evaluation = None
    if task.evaluation_split == "train":
        initial_evaluation = _evaluate(
            model,
            task.evaluation_dataset,
            task.evaluation_indices,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            kind=task.prediction,
            normalizer=task.normalizer,
        )
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    model.train()
    latencies: list[float] = []
    losses: list[float] = []
    pre_clip_norms: list[float] = []
    clipped = 0
    started = time.perf_counter()
    for step in range(steps):
        selected = _cyclic_indices(
            task.train_indices,
            step=step,
            batch_size=batch_size,
            seed=order_seed,
        )
        _synchronize(device)
        step_started = time.perf_counter()
        graph = _batch(task.train_dataset, selected).to(
            device,
            dtype=dtype,
            non_blocking=device.type == "cuda",
        )
        if graph.y is None:
            raise RuntimeError("training graph is missing targets")
        optimizer.zero_grad(set_to_none=True)
        prediction, _ = _predict(model, graph, task.prediction)
        target = task.normalizer.normalize(graph.y.reshape(-1))
        loss = (prediction - target).square().mean()
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(f"non-finite loss in arm {arm} at update {step + 1}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        norm_value = float(norm.detach().item())
        if not math.isfinite(norm_value):
            raise RuntimeError(f"non-finite gradient norm in arm {arm}")
        clipped += int(norm_value > grad_clip)
        pre_clip_norms.append(norm_value)
        optimizer.step()
        _synchronize(device)
        latencies.append(time.perf_counter() - step_started)
        losses.append(float(loss.detach().item()))

    evaluation = _evaluate(
        model,
        task.evaluation_dataset,
        task.evaluation_indices,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        kind=task.prediction,
        normalizer=task.normalizer,
    )
    window = latencies[min(10, len(latencies)) :]
    if not window:
        window = latencies
    peak = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    return {
        "arm": arm,
        "status": "completed",
        "updates_completed": steps,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "initial_state_sha256": initial_state_hash,
        "initial_coordinate_state_sha256": initial_coordinate_state_hash,
        "initial_prediction_sha256": initial_prediction_hash,
        "initial_probe_normalized_mse": float(initial_loss.item()),
        "initial_max_coordinate_delta": initial_delta,
        "initial_evaluation": initial_evaluation,
        "final_update_minibatch_normalized_mse": (
            losses[-1] if losses else float(initial_loss.item())
        ),
        "best_update_minibatch_normalized_mse": (
            min(losses, default=float(initial_loss.item()))
        ),
        "evaluation_split": task.evaluation_split,
        "evaluation": evaluation,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "grad_clip": grad_clip,
        },
        "clipped_updates": clipped,
        "clip_fraction": 0.0 if steps == 0 else clipped / steps,
        "mean_pre_clip_grad_norm": (
            0.0 if not pre_clip_norms else statistics.fmean(pre_clip_norms)
        ),
        "step_latency_scope": (
            "sample loading + ELAGraph.collate + device transfer + graph ingestion + "
            "forward + backward + optimizer"
        ),
        "step_latency_interpretation": (
            "observational within one sequential multi-arm process; do not treat as "
            "an unbiased architecture speed ranking"
        ),
        "step_latency_median_seconds": (
            None if not window else statistics.median(window)
        ),
        "step_latency_p90_seconds": _quantile(window, 0.90),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": peak,
        "final_state_sha256": _state_sha256(model),
        "final_coordinate_state_sha256": _coordinate_state_sha256(model),
    }


def _dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"unsupported dtype: {name}")


def _validate_args(args: argparse.Namespace) -> None:
    if args.width <= 0 or args.depth <= 0:
        raise ValueError("width and depth must be positive")
    if args.batch_size <= 0 or args.steps <= 0:
        raise ValueError("batch_size and steps must be positive")
    if args.learning_rate <= 0.0 or args.grad_clip <= 0.0:
        raise ValueError("learning_rate and grad_clip must be positive")
    if args.weight_decay < 0.0 or args.cutoff <= 0.0:
        raise ValueError("weight_decay must be nonnegative and cutoff positive")
    if args.model_seed < 0 or args.order_seed < 0 or args.threads <= 0:
        raise ValueError("seeds must be nonnegative and threads positive")
    if len(set(args.arms)) != len(args.arms):
        raise ValueError("arms must not contain duplicates")
    invalid = set(args.arms) - set(STATIC_ARMS)
    if invalid:
        raise ValueError(f"unknown arms: {sorted(invalid)}")
    if args.task == "qm9" and "no-relation" in args.arms:
        raise ValueError("no-relation is inapplicable to untyped QM9 radius graphs")
    if args.task == "qm9":
        if min(args.num_samples, args.train_size, args.val_size) <= 0:
            raise ValueError("QM9 sample and split sizes must be positive")
        if args.validation_limit < 0 or args.validation_limit > args.val_size:
            raise ValueError("QM9 validation_limit must lie in [0,val_size]")
        if args.split_seed < 0:
            raise ValueError("QM9 split_seed must be nonnegative")
    elif args.task == "lba-id30":
        if args.train_limit < 0 or args.validation_limit < 0:
            raise ValueError("LBA limits must be nonnegative")


def _write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _prediction_pairing_status(hashes: Sequence[str]) -> bool | None:
    if len(hashes) < 2:
        return None
    return len(set(hashes)) == 1


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--order-seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--include-stagewise", action="store_true")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate canonical ELA on bounded QM9 or ATOM3D-LBA data."
    )
    subparsers = parser.add_subparsers(dest="task", required=True)

    qm9 = subparsers.add_parser("qm9")
    _common_parser(qm9)
    qm9.set_defaults(
        data_root=Path("data/qm9"),
        cutoff=2.5,
        batch_size=64,
        arms=("full", "no-cg12", "no-multiscale"),
    )
    qm9.add_argument("--data-root", type=Path, default=Path("data/qm9"))
    qm9.add_argument("--num-samples", type=int, default=130_000)
    qm9.add_argument("--train-size", type=int, default=110_000)
    qm9.add_argument("--val-size", type=int, default=10_000)
    qm9.add_argument("--validation-limit", type=int, default=1_000)
    qm9.add_argument("--split-seed", type=int, default=42)
    qm9.add_argument("--arms", nargs="+", choices=STATIC_ARMS)

    overfit = subparsers.add_parser("lba-overfit")
    _common_parser(overfit)
    overfit.set_defaults(
        data_root=Path("data/atom3d_lba"),
        batch_size=2,
        steps=250,
        learning_rate=1e-3,
        weight_decay=0.0,
        model_seed=20260723,
        order_seed=20260723,
        arms=STATIC_ARMS,
    )
    overfit.add_argument(
        "--data-root", type=Path, default=Path("data/atom3d_lba")
    )
    overfit.add_argument("--arms", nargs="+", choices=STATIC_ARMS)

    id30 = subparsers.add_parser("lba-id30")
    _common_parser(id30)
    id30.set_defaults(
        data_root=Path("data/atom3d_lba"),
        batch_size=16,
        steps=220,
        arms=STATIC_ARMS,
    )
    id30.add_argument("--data-root", type=Path, default=Path("data/atom3d_lba"))
    id30.add_argument("--train-limit", type=int, default=0)
    id30.add_argument("--validation-limit", type=int, default=0)
    id30.add_argument("--arms", nargs="+", choices=STATIC_ARMS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    reproducibility = _configure_determinism(args.model_seed, args.threads)
    if args.task == "qm9":
        task = _qm9_data(args)
    elif args.task == "lba-overfit":
        task = _lba_overfit_data(args)
    else:
        task = _lba_id30_data(args)
    models, pairing = _build_arm_models(
        input_dim=task.input_dim,
        edge_types=task.edge_types,
        width=args.width,
        depth=args.depth,
        cutoff=args.cutoff,
        arms=args.arms,
        include_stagewise=args.include_stagewise,
        seed=args.model_seed,
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "task": args.task,
        "public_contract": "ELAGraph -> ELA -> ELAGraph",
        "model_family": "ELA_only",
        "legacy_models_present": False,
        "command": list(sys.argv if argv is None else [sys.argv[0], *argv]),
        "git": _git_metadata(),
        "source_sha256": _source_sha256(),
        "device": str(device),
        "dtype": args.dtype,
        "reproducibility": reproducibility,
        "data": task.data,
        "split": task.split,
        "topology": task.topology,
        "target_normalizer_train_only": task.normalizer.as_dict(),
        "label_access": task.access,
        "limitations": task.limitations,
        "configuration": {
            "input_irreps": f"{task.input_dim}x0e",
            "output_irreps": "1x0e",
            "width": args.width,
            "depth": args.depth,
            "cutoff": args.cutoff,
            "edge_types": task.edge_types,
            "batch_size": args.batch_size,
            "updates_per_arm": args.steps,
            "paired_arms": list(args.arms),
            "arm_execution_order": list(models),
            "stagewise_functionality_arm": args.include_stagewise,
            "prediction_readout": task.prediction,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "model_seed": args.model_seed,
            "order_seed": args.order_seed,
            "threads": args.threads,
            "split_seed": args.split_seed if args.task == "qm9" else None,
        },
        "pairing": pairing,
        "arms": {},
    }
    _write_result(args.output, result)
    initial_prediction_hashes: dict[str, str] = {}
    try:
        for arm, model in models.items():
            try:
                arm_result = _run_arm(
                    arm=arm,
                    model=model,
                    task=task,
                    steps=args.steps,
                    batch_size=args.batch_size,
                    device=device,
                    dtype=_dtype(args.dtype),
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    grad_clip=args.grad_clip,
                    order_seed=args.order_seed,
                )
            finally:
                model.to(device="cpu")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            result["arms"][arm] = arm_result
            if arm != "stagewise":
                initial_prediction_hashes[arm] = arm_result[
                    "initial_prediction_sha256"
                ]
            _write_result(args.output, result)
    except BaseException as exc:
        result["status"] = "failed"
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_result(args.output, result)
        raise
    result["pairing"]["static_initial_predictions_identical"] = (
        _prediction_pairing_status(tuple(initial_prediction_hashes.values()))
    )
    result["status"] = "completed"
    _write_result(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
