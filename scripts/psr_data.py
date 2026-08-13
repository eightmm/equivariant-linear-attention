"""ATOM3D PSR (HuggingFace mirror) -> ELAGraph loading and one-time validation.

Source: ``vector-institute/atom3d-psr`` cached under ``data/atom3d_psr``.
Rows carry ``input_ids`` (atomic numbers, hydrogens included), ``coords``
(N x 3, float64), and ``labels`` with four scores. The mirror has no target
id column; ``python scripts/psr_data.py`` runs the validation pass that
establishes the label order, the element vocabulary, atom-count statistics,
and whether rows are target-contiguous.

Label order: ``[rmsd, gdt_ts, gdt_ha, tm]``. GDT_HA uses strictly tighter
thresholds than GDT_TS, so ``labels[2] <= labels[1]`` must hold on every row;
the dataset card's prose order (rmsd, tm, gdt_ts, gdt_ha) would instead force
``labels[3] <= labels[2]``, which the data violates. The validation pass
asserts this on the full dataset.

Validated on 2026-08-07 over all 44,214 rows: label order confirmed
(gdt_ha <= gdt_ts everywhere, rmsd/gdt_ts correlation -0.3..-0.5); atoms per
row mean ~2k, p95 ~4.5k, max 8,770; elements are H/C/N/O/S plus junk codes
(train: 19 x1, 125 x1, 132 x315; val: 132 x11; test: none) that fall into
the "other" bucket; the length probe found no detectable target-block
structure in row order, so per-target grouping is not recoverable and only
global -- not per-target -- ranking metrics are available on this mirror.

Requires ``datasets`` (not a project dependency):
    uv run --with "datasets>=3" python scripts/psr_data.py
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from equivariant_linear_attention import ELAGraph

LABEL_NAMES = ("rmsd", "gdt_ts", "gdt_ha", "tm")
TARGET_LABEL = "gdt_ts"

# Atomic numbers observed in the data; anything else falls into the last bucket.
ELEMENT_VOCAB = (1, 6, 7, 8, 16)
FEATURE_DIM = len(ELEMENT_VOCAB) + 1
INPUT_IRREPS = f"{FEATURE_DIM}x0e"

_LOOKUP = torch.full((max(ELEMENT_VOCAB) + 2,), len(ELEMENT_VOCAB))
for _slot, _number in enumerate(ELEMENT_VOCAB):
    _LOOKUP[_number] = _slot


def load_psr(data_root: str = "data/atom3d_psr"):
    from datasets import load_dataset

    return load_dataset("vector-institute/atom3d-psr", cache_dir=data_root)


def to_graph(row: dict, index: int | None = None) -> ELAGraph:
    numbers = torch.as_tensor(row["input_ids"], dtype=torch.long)
    slots = _LOOKUP[numbers.clamp(min=0, max=max(ELEMENT_VOCAB) + 1)]
    x = torch.nn.functional.one_hot(slots, FEATURE_DIM).to(torch.float32)
    labels = tuple(float(v) for v in row["labels"])
    target = labels[LABEL_NAMES.index(TARGET_LABEL)]
    return ELAGraph(
        x=x,
        pos=torch.as_tensor(row["coords"], dtype=torch.float32),
        y=torch.tensor([[target]], dtype=torch.float32),
        ids=((index, *labels),),
    )


def collate(rows: Sequence[dict], indices: Sequence[int] | None = None) -> ELAGraph:
    indices = range(len(rows)) if indices is None else indices
    return ELAGraph.collate(
        to_graph(row, index) for row, index in zip(rows, indices, strict=True)
    )


def _validate() -> None:
    import numpy as np
    import pyarrow.compute as pc

    dataset = load_psr()
    for split, ds in dataset.items():
        table = ds.data
        lengths = pc.list_value_length(table.column("input_ids")).to_numpy(
            zero_copy_only=False
        )
        labels = np.asarray(table.column("labels").to_pylist(), dtype=np.float64)
        elements = np.unique(
            np.concatenate([c.flatten().to_numpy() for c in table.column("input_ids").chunks])
        )

        assert labels.shape[1] == 4
        assert (labels[:, 2] <= labels[:, 1] + 1e-9).all(), "gdt_ha > gdt_ts somewhere"
        assert (labels[:, 0] >= 0.0).all()
        assert (labels[:, 1:] >= 0.0).all() and (labels[:, 1:] <= 1.0 + 1e-9).all()
        rmsd_gdt_corr = float(np.corrcoef(labels[:, 0], labels[:, 1])[0, 1])
        assert rmsd_gdt_corr < 0.0, "rmsd should anti-correlate with gdt_ts"

        unexpected = [int(z) for z in elements if z not in ELEMENT_VOCAB]
        quantiles = np.percentile(lengths, [50, 95]).astype(int)
        print(
            f"{split}: rows={len(ds)} atoms mean={lengths.mean():.0f}"
            f" median={quantiles[0]} p95={quantiles[1]} max={lengths.max()}"
            f" | rmsd/gdt_ts corr={rmsd_gdt_corr:.3f}"
            f" | elements={[int(z) for z in elements]} unexpected={unexpected}"
        )

        if split in {"train", "val"}:
            block = lengths[: len(lengths) - len(lengths) % 50].reshape(-1, 50)
            spread = np.median(block.max(axis=1) - block.min(axis=1))
            boundary = np.abs(np.diff(block.mean(axis=1)))
            print(
                f"{split}: contiguity probe - median within-block-of-50 length spread"
                f"={spread:.0f}, median boundary jump={np.median(boundary):.0f}"
            )

    from equivariant_linear_attention import ELA

    model = ELA(INPUT_IRREPS, "1x0e", width=32, depth=2)
    batch = collate([dataset["train"][i] for i in range(4)])
    output = model(batch)
    assert output.graph_x is not None and batch.y is not None
    loss = torch.nn.functional.mse_loss(output.graph_x, batch.y)
    loss.backward()
    print(
        f"smoke: batch nodes={batch.num_nodes} graphs={batch.num_graphs}"
        f" loss={float(loss.detach()):.4f} (forward+backward ok, cpu)"
    )


if __name__ == "__main__":
    _validate()
