from __future__ import annotations

import math
import sys
from types import ModuleType

import pytest
import torch

from equivariant_attention.pdbbind import (
    ATOM3D_LBA_REPO,
    atom3d_lba_row_to_sample,
    load_atom3d_lba_samples,
    segment_balanced_knn_edge_index,
)


_REVISION = "f93dd2d150a47c270f624620f84e07451a158705"


def _row() -> dict[str, object]:
    return {
        "input_ids": [6, 8, 7, 6, 8],
        "coords": [
            [10.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 1.5, 0.2],
        ],
        "labels": 7.25,
        "token_type_ids": [0, 1, 1, 2, 2],
    }


def test_atom3d_row_keeps_pocket_ligand_alignment_and_ligand_readout() -> None:
    sample = atom3d_lba_row_to_sample(
        _row(),
        split="train",
        row_index=7,
        revision=_REVISION,
    )

    assert sample.node_feats.shape == (4, 140)
    assert torch.equal(
        sample.pos,
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, 1.5, 0.2],
            ]
        ),
    )
    assert sample.readout_mask is not None
    assert torch.equal(
        sample.readout_mask,
        torch.tensor([False, False, True, True]),
    )
    assert torch.equal(
        sample.node_feats[:, 138:],
        torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
        ),
    )
    assert sample.node_feats[0, 8].item() == 1.0
    assert sample.node_feats[1, 7].item() == 1.0
    assert sample.node_feats[2, 6].item() == 1.0
    assert sample.target.item() == pytest.approx(7.25)
    assert sample.sample_id.startswith("atom3d-lba:train:0000007:")
    assert len(sample.sample_id.rsplit(":", maxsplit=1)[-1]) == 16


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("input_ids", [6, 8], "equal lengths"),
        ("coords", [[0.0, 0.0]], "shape"),
        ("labels", math.inf, "finite"),
        ("token_type_ids", [0, 0, 0, 2, 2], "pocket"),
        ("token_type_ids", [0, 1, 1, 1, 1], "ligand"),
        ("input_ids", [6, 8, 138, 6, 8], "atom token"),
    ],
)
def test_atom3d_row_rejects_invalid_schema(
    field: str,
    value: object,
    match: str,
) -> None:
    row = _row()
    row[field] = value
    with pytest.raises((TypeError, ValueError), match=match):
        atom3d_lba_row_to_sample(
            row,
            split="train",
            row_index=0,
            revision=_REVISION,
        )


def test_loader_pins_repo_revision_split_indices_and_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pytest.TempPathFactory,
) -> None:
    calls: list[dict[str, object]] = []
    rows = [_row(), {**_row(), "labels": 6.5}, {**_row(), "labels": 8.0}]

    fake = ModuleType("datasets")

    def fake_load_dataset(repo: str, **kwargs: object) -> list[dict[str, object]]:
        calls.append({"repo": repo, **kwargs})
        return rows

    fake.load_dataset = fake_load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake)

    samples = load_atom3d_lba_samples(
        root=tmp_path,
        indices=(2, 0),
        revision=_REVISION,
    )

    assert calls == [
        {
            "repo": "parquet",
            "data_files": {
                "train": [
                    (
                        "hf://datasets/"
                        f"{ATOM3D_LBA_REPO}@{_REVISION}/"
                        "data/train-00000-of-00002.parquet"
                    ),
                    (
                        "hf://datasets/"
                        f"{ATOM3D_LBA_REPO}@{_REVISION}/"
                        "data/train-00001-of-00002.parquet"
                    ),
                ]
            },
            "split": "train",
            "cache_dir": str(tmp_path),
        }
    ]
    assert [sample.target.item() for sample in samples] == [8.0, 7.25]
    assert [sample.sample_id.split(":")[2] for sample in samples] == [
        "0000002",
        "0000000",
    ]


def test_loader_rejects_nontrain_split_and_duplicate_indices(tmp_path: object) -> None:
    with pytest.raises(ValueError, match="train"):
        load_atom3d_lba_samples(
            root=tmp_path,
            indices=(0,),
            revision=_REVISION,
            split="validation",
        )
    with pytest.raises(ValueError, match="unique"):
        load_atom3d_lba_samples(
            root=tmp_path,
            indices=(0, 0),
            revision=_REVISION,
        )


def test_segment_balanced_knn_uses_same_sparse_topology_under_permutation() -> None:
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [2.2, 0.1, 0.0],
            [0.2, 1.1, 0.0],
            [1.6, 1.4, 0.3],
        ],
        dtype=torch.float64,
    )
    ligand = torch.tensor([False, False, False, True, True])
    edges = segment_balanced_knn_edge_index(
        pos,
        ligand,
        intra_k=1,
        cross_k=1,
        cutoff=4.0,
    )

    receiver, sender = edges
    assert edges.shape == (2, 15)
    for node in range(5):
        selected = sender[receiver == node]
        assert node in selected
        nonself = selected[selected != node]
        assert (ligand[nonself] == ligand[node]).sum() == 1
        assert (ligand[nonself] != ligand[node]).sum() == 1

    permutation = torch.tensor([3, 0, 4, 2, 1])
    inverse = torch.argsort(permutation)
    permuted_edges = segment_balanced_knn_edge_index(
        pos[permutation],
        ligand[permutation],
        intra_k=1,
        cross_k=1,
        cutoff=4.0,
    )
    restored = permutation[permuted_edges]
    reference_codes = torch.sort(receiver * 5 + sender).values
    restored_codes = torch.sort(restored[0] * 5 + restored[1]).values
    assert torch.equal(restored_codes, reference_codes)
    assert torch.equal(inverse[permutation], torch.arange(5))


def test_segment_balanced_knn_keeps_self_and_respects_cutoff() -> None:
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [10.0, 0.0, 0.0]]
    )
    ligand = torch.tensor([False, False, True])
    edges = segment_balanced_knn_edge_index(
        pos,
        ligand,
        intra_k=2,
        cross_k=2,
        cutoff=1.0,
    )
    receiver, sender = edges

    assert {(int(i), int(j)) for i, j in edges.T} == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 2),
    }
    assert torch.equal(receiver[receiver == sender], torch.arange(3))
