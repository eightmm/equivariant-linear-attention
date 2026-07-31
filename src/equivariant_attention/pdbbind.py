from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re

import torch
import torch.nn.functional as F

from .benchmarking import GraphSample


ATOM3D_LBA_REPO = "vector-institute/atom3d-lba"
ATOM3D_LBA_REVISION = "f93dd2d150a47c270f624620f84e07451a158705"
ATOM3D_LBA_MAX_ATOM_TOKEN = 137
ATOM3D_LBA_NODE_DIM = 140
_TRAIN_FILES = (
    "data/train-00000-of-00002.parquet",
    "data/train-00001-of-00002.parquet",
)
_REVISION = re.compile(r"^[0-9a-f]{40}$")
TOPOLOGY_CHUNK_NODES = 128


def _load_cached_atom3d_train(
    root: Path,
    *,
    revision: str,
) -> object | None:
    """Load the exact-revision Arrow cache without resolving a remote URI.

    ``datasets>=5`` resolves ``hf://`` data-file patterns against the Hub even
    when the processed Arrow shards already exist. That makes an otherwise
    reproducible offline packet fail before consulting its cache. The pinned
    dataset cache stores the revision in the directory name, so it is safe to
    prefer a complete two-shard train cache and retain the network loader as
    the fallback.
    """

    cache = (
        root
        / "vector-institute___atom3d-lba"
        / "default"
        / "0.0.0"
        / revision
    )
    shards = tuple(sorted(cache.glob("atom3d-lba-train-*.arrow")))
    if not shards:
        return None
    if len(shards) != len(_TRAIN_FILES):
        raise ValueError(
            "cached ATOM3D-LBA train split is incomplete: "
            f"expected {len(_TRAIN_FILES)} shards, found {len(shards)}"
        )
    from datasets import Dataset, concatenate_datasets

    return concatenate_datasets(
        [Dataset.from_file(str(shard)) for shard in shards]
    )


def topology_sha256(samples: Sequence[GraphSample]) -> str:
    """Canonical identity of a precomputed candidate list.

    Every runner must consume this one definition so a drifting candidate list
    cannot be mistaken for a matched topology across processes or packets.
    """
    digest = hashlib.sha256()
    for sample in samples:
        if sample.edge_index is None:
            raise ValueError("topology hash requires precomputed edges")
        digest.update(sample.sample_id.encode("utf-8"))
        digest.update(sample.edge_index.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def segment_balanced_knn_edge_index(
    pos: torch.Tensor,
    segment_mask: torch.Tensor,
    *,
    intra_k: int,
    cross_k: int,
    cutoff: float,
    chunk_nodes: int = TOPOLOGY_CHUNK_NODES,
) -> torch.Tensor:
    """Build self plus bounded same/cross-segment directed candidates.

    The retention test is the exact float64 squared displacement against the
    squared cutoff, so the candidate list is reproducible across processes and
    invariant under node permutation and BLAS thread count. A
    matrix-multiplication Euclidean distance is not: its float32 error grows
    with coordinate magnitude and depends on the reduction blocking.

    Translation invariance holds for the *stored* coordinates. Promoting inside
    this function cannot undo rounding that a translation already applied in the
    storage dtype, so float32 storage is invariant to at least a 1e3 Angstrom
    offset and float64 storage far beyond that.

    Exact ties at the neighbor boundary are all retained, so a receiver degree
    may exceed its budget. That keeps the selection permutation equivariant.
    """
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("pos must have shape (N, 3)")
    if pos.shape[0] == 0:
        raise ValueError("pos must contain at least one node")
    if not torch.is_floating_point(pos) or not bool(torch.isfinite(pos).all()):
        raise ValueError("pos must be finite and floating point")
    if segment_mask.dtype != torch.bool or segment_mask.shape != (pos.shape[0],):
        raise ValueError("segment_mask must be boolean with shape (N,)")
    if segment_mask.device != pos.device:
        raise ValueError("segment_mask and pos must use the same device")
    for name, value in (("intra_k", intra_k), ("cross_k", cross_k)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be nonnegative")
    if (
        isinstance(cutoff, bool)
        or not isinstance(cutoff, (int, float))
        or not math.isfinite(float(cutoff))
        or float(cutoff) <= 0.0
    ):
        raise ValueError("cutoff must be finite and positive")
    if isinstance(chunk_nodes, bool) or not isinstance(chunk_nodes, int):
        raise TypeError("chunk_nodes must be an integer")
    if chunk_nodes <= 0:
        raise ValueError("chunk_nodes must be positive")

    node_count = pos.shape[0]
    geometry = pos.detach().to(dtype=torch.float64)
    cutoff_squared = float(cutoff) * float(cutoff)
    node_index = torch.arange(node_count, device=pos.device)
    receivers: list[torch.Tensor] = []
    senders: list[torch.Tensor] = []
    for start in range(0, node_count, chunk_nodes):
        stop = min(start + chunk_nodes, node_count)
        rows = node_index[start:stop]
        displacement = geometry[start:stop].unsqueeze(1) - geometry.unsqueeze(0)
        squared = displacement.square().sum(dim=-1)
        self_edge = node_index.unsqueeze(0) == rows.unsqueeze(1)
        within = squared < cutoff_squared
        same_segment = segment_mask[start:stop].unsqueeze(1) == segment_mask.unsqueeze(0)
        selection = [self_edge]
        for relation_mask, maximum in (
            (same_segment, intra_k),
            (~same_segment, cross_k),
        ):
            eligible = within & relation_mask & ~self_edge
            budget = min(maximum, node_count)
            if budget == 0:
                selection.append(torch.zeros_like(eligible))
                continue
            ranked = torch.where(eligible, squared, squared.new_full((), math.inf))
            boundary = ranked.topk(budget, dim=-1, largest=False).values[:, -1:]
            selection.append(eligible & (squared <= boundary))
        # Row-major nonzero order is (self, intra, cross) then ascending sender.
        flat = torch.stack(selection, dim=1).reshape(stop - start, -1)
        row, column = flat.nonzero(as_tuple=True)
        receivers.append(rows[row])
        senders.append(column % node_count)
    return torch.stack([torch.cat(receivers), torch.cat(senders)])


def atom3d_lba_row_to_sample(
    row: Mapping[str, object],
    *,
    split: str,
    row_index: int,
    revision: str,
) -> GraphSample:
    _validate_train_location(split=split, row_index=row_index, revision=revision)
    return _atom3d_lba_row_to_sample(
        row,
        split=split,
        row_index=row_index,
        revision=revision,
    )


def _atom3d_lba_row_to_sample(
    row: Mapping[str, object],
    *,
    split: str,
    row_index: int,
    revision: str,
) -> GraphSample:
    required = ("input_ids", "coords", "labels", "token_type_ids")
    missing = [name for name in required if name not in row]
    if missing:
        raise ValueError(f"ATOM3D-LBA row is missing fields: {', '.join(missing)}")

    atom_tokens = torch.as_tensor(row["input_ids"], dtype=torch.long)
    coordinates = torch.as_tensor(row["coords"], dtype=torch.float64)
    token_types = torch.as_tensor(row["token_type_ids"], dtype=torch.long)
    label = torch.as_tensor(row["labels"], dtype=torch.float64)

    if atom_tokens.ndim != 1 or token_types.ndim != 1:
        raise ValueError("input_ids and token_type_ids must be one-dimensional")
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3)")
    lengths = {
        atom_tokens.shape[0],
        coordinates.shape[0],
        token_types.shape[0],
    }
    if len(lengths) != 1:
        raise ValueError("input_ids, coords, and token_type_ids need equal lengths")
    if label.numel() != 1:
        raise ValueError("labels must contain exactly one scalar")
    if not torch.isfinite(coordinates).all() or not torch.isfinite(label).all():
        raise ValueError("coords and labels must be finite")
    if bool(
        (
            (atom_tokens < 1)
            | (atom_tokens > ATOM3D_LBA_MAX_ATOM_TOKEN)
        ).any().item()
    ):
        raise ValueError(
            "atom token must lie between 1 and "
            f"{ATOM3D_LBA_MAX_ATOM_TOKEN}"
        )
    if bool(((token_types < 0) | (token_types > 2)).any().item()):
        raise ValueError("token_type_ids must contain only protein, pocket, or ligand")

    pocket = token_types == 1
    ligand = token_types == 2
    if not bool(pocket.any().item()):
        raise ValueError("ATOM3D-LBA row must contain at least one pocket atom")
    if not bool(ligand.any().item()):
        raise ValueError("ATOM3D-LBA row must contain at least one ligand atom")
    retained = pocket | ligand
    retained_atom_tokens = atom_tokens[retained]
    retained_coordinates = coordinates[retained]
    retained_types = token_types[retained]

    element_features = F.one_hot(
        retained_atom_tokens,
        num_classes=ATOM3D_LBA_MAX_ATOM_TOKEN + 1,
    ).to(dtype=torch.float32)
    segment_features = F.one_hot(
        retained_types - 1,
        num_classes=2,
    ).to(dtype=torch.float32)
    node_feats = torch.cat([element_features, segment_features], dim=-1)
    readout_mask = retained_types == 2
    target = label.reshape(1).to(dtype=torch.float32)
    pos = retained_coordinates.to(dtype=torch.float32)
    digest = _row_digest(
        retained_atom_tokens,
        retained_coordinates,
        retained_types,
    )
    sample_id = f"atom3d-lba:{split}:{row_index:07d}:{digest}"
    return GraphSample(
        node_feats=node_feats,
        pos=pos,
        target=target,
        sample_id=sample_id,
        readout_mask=readout_mask,
        node_role_id=retained_types - 1,
        node_masks={
            "ligand": ligand[retained],
            "pocket": pocket[retained],
        },
    )


def load_atom3d_lba_samples(
    root: str | Path,
    *,
    indices: Sequence[int],
    revision: str = ATOM3D_LBA_REVISION,
    split: str = "train",
) -> list[GraphSample]:
    validated_indices = _validate_indices(indices)
    _validate_train_location(split=split, row_index=0, revision=revision)
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise ImportError(
            "ATOM3D-LBA loading requires the optional 'pdbbind' dependencies"
        ) from exc

    root = Path(root)
    dataset = _load_cached_atom3d_train(root, revision=revision)
    if dataset is None:
        train_files = [
            (
                f"hf://datasets/{ATOM3D_LBA_REPO}@{revision}/"
                f"{relative_path}"
            )
            for relative_path in _TRAIN_FILES
        ]
        dataset = load_dataset(
            "parquet",
            data_files={"train": train_files},
            split=split,
            cache_dir=str(root),
        )
    if max(validated_indices) >= len(dataset):
        raise IndexError(
            f"ATOM3D-LBA {split} has {len(dataset)} rows; "
            f"requested index {max(validated_indices)}"
        )
    return [
        atom3d_lba_row_to_sample(
            dataset[index],
            split=split,
            row_index=index,
            revision=revision,
        )
        for index in validated_indices
    ]


def load_atom3d_lba_split_samples(
    root: str | Path,
    *,
    split: str,
    revision: str = ATOM3D_LBA_REVISION,
    indices: Sequence[int] | None = None,
) -> list[GraphSample]:
    """Load the official ID30 train or validation split from the pinned cache.

    The test split is deliberately inadmissible so architecture selection cannot
    accidentally consume its rows or labels. Target transforms must still be fit
    by the caller from the returned training samples only.
    """
    _validate_evaluation_location(split=split, row_index=0, revision=revision)
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise ImportError(
            "ATOM3D-LBA loading requires the optional 'pdbbind' dependencies"
        ) from exc

    dataset = load_dataset(
        ATOM3D_LBA_REPO,
        revision=revision,
        split=split,
        cache_dir=str(root),
    )
    selected_indices = (
        tuple(range(len(dataset)))
        if indices is None
        else _validate_indices(indices)
    )
    if max(selected_indices) >= len(dataset):
        raise IndexError(
            f"ATOM3D-LBA {split} has {len(dataset)} rows; "
            f"requested index {max(selected_indices)}"
        )
    return [
        _atom3d_lba_row_to_sample(
            dataset[index],
            split=split,
            row_index=index,
            revision=revision,
        )
        for index in selected_indices
    ]


def _validate_indices(indices: Sequence[int]) -> tuple[int, ...]:
    if isinstance(indices, (str, bytes)) or not isinstance(indices, Sequence):
        raise TypeError("indices must be a nonempty sequence of integers")
    values = tuple(indices)
    if not values:
        raise ValueError("indices must be nonempty")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in values):
        raise TypeError("indices must contain only integers")
    if any(index < 0 for index in values):
        raise ValueError("indices must be nonnegative")
    if len(set(values)) != len(values):
        raise ValueError("indices must be unique")
    return values


def _validate_train_location(*, split: str, row_index: int, revision: str) -> None:
    if split != "train":
        raise ValueError("the registered overfit loader permits only the train split")
    _validate_row_and_revision(row_index=row_index, revision=revision)


def _validate_evaluation_location(
    *,
    split: str,
    row_index: int,
    revision: str,
) -> None:
    if split not in {"train", "val"}:
        raise ValueError("the ID30 evaluation loader permits only train or val")
    _validate_row_and_revision(row_index=row_index, revision=revision)


def _validate_row_and_revision(*, row_index: int, revision: str) -> None:
    if isinstance(row_index, bool) or not isinstance(row_index, int) or row_index < 0:
        raise ValueError("row_index must be a nonnegative integer")
    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise ValueError("revision must be an immutable 40-character commit hash")


def _row_digest(
    atom_tokens: torch.Tensor,
    coordinates: torch.Tensor,
    token_types: torch.Tensor,
) -> str:
    """Return a label-blind canonical identity for one retained structure."""

    payload = {
        # Keep the frozen digest schema stable; values are treated as tokens.
        "atomic_numbers": atom_tokens.tolist(),
        "coords": coordinates.tolist(),
        "token_types": token_types.tolist(),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
