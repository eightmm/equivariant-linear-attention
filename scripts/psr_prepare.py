"""ATOM3D PSR Zenodo split-by-year LMDB -> compact torch shards.

Reads the official ATOM3D LMDB splits (which carry the full atom table with
residue and atom names plus the target id, unlike the HuggingFace mirror)
and writes one ``.pt`` shard per split under ``data/atom3d_psr_prepared``:

    targets: list[str]            # CASP target id per decoy
    decoys: list[str]             # decoy id
    class_idx: list[int16 (N,)]   # 167-class (residue, heavy-atom) indices
    pos: list[float32 (N, 3)]
    scores: float32 (G, 4)        # rmsd, gdt_ts, gdt_ha, tm

Hydrogens are removed first; remaining atoms outside the 167 vocabulary
(OXT, non-standard residues, hetero atoms) are dropped and reported as a
heavy-atom drop rate.

Requires ``lmdb`` (not a project dependency):
    uv run --with lmdb python scripts/psr_prepare.py [--inspect]
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
from pathlib import Path

import torch
from psr_vocab import LABEL_NAMES, VOCAB

SPLITS = ("train", "val", "test")


def open_split(root: Path, split: str):
    import lmdb

    candidates = sorted(root.rglob(f"{split}")) + sorted(root.rglob(f"*{split}*"))
    for path in candidates:
        if (path / "data.mdb").exists():
            return lmdb.open(
                str(path), readonly=True, lock=False, readahead=False, subdir=True
            )
    raise FileNotFoundError(f"no LMDB directory for split {split!r} under {root}")


def read_entry(txn, index: int) -> dict:
    raw = txn.get(str(index).encode())
    if raw is None:
        raise KeyError(index)
    payload = gzip.GzipFile(fileobj=io.BytesIO(raw)).read()
    return json.loads(payload)


def inspect(root: Path) -> None:
    for split in SPLITS:
        env = open_split(root, split)
        with env.begin() as txn:
            count = int(txn.get(b"num_examples"))
            entry = read_entry(txn, 0)
            print(f"== {split}: num_examples={count}")
            for key, value in entry.items():
                text = json.dumps(value)[:200] if not isinstance(value, dict) else None
                if isinstance(value, dict):
                    inner = {
                        k: json.dumps(v)[:80] for k, v in list(value.items())[:12]
                    }
                    print(f"  {key}: dict {json.dumps(inner, indent=1)[:600]}")
                else:
                    print(f"  {key}: {text}")
        env.close()


def convert(root: Path, out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        env = open_split(root, split)
        targets: list[str] = []
        decoys: list[str] = []
        class_idx: list[torch.Tensor] = []
        positions: list[torch.Tensor] = []
        scores: list[list[float]] = []
        heavy_total = 0
        heavy_dropped = 0
        empty = 0
        with env.begin() as txn:
            count = int(txn.get(b"num_examples"))
            for index in range(count):
                entry = read_entry(txn, index)
                atoms = entry["atoms"]
                columns = atoms["columns"]
                rows = atoms["data"]
                col = {name: position for position, name in enumerate(columns)}
                slots: list[int] = []
                coords: list[list[float]] = []
                for row in rows:
                    element = str(row[col["element"]]).strip().upper()
                    if element == "H":
                        continue
                    heavy_total += 1
                    key = (
                        str(row[col["resname"]]).strip().upper(),
                        str(row[col["name"]]).strip().upper(),
                    )
                    slot = VOCAB.get(key)
                    if slot is None:
                        heavy_dropped += 1
                        continue
                    slots.append(slot)
                    coords.append(
                        [row[col["x"]], row[col["y"]], row[col["z"]]]
                    )
                if not slots:
                    empty += 1
                    continue
                score = entry["scores"]
                identifier = str(entry["id"]).split("'")
                targets.append(identifier[1])
                decoys.append(identifier[3])
                class_idx.append(torch.tensor(slots, dtype=torch.int16))
                positions.append(torch.tensor(coords, dtype=torch.float32))
                scores.append([float(score[name]) for name in LABEL_NAMES])
        env.close()
        torch.save(
            {
                "targets": targets,
                "decoys": decoys,
                "class_idx": class_idx,
                "pos": positions,
                "scores": torch.tensor(scores, dtype=torch.float32),
            },
            out_root / f"{split}.pt",
        )
        print(
            f"{split}: decoys={len(targets)} targets={len(set(targets))}"
            f" empty_skipped={empty} heavy_atoms={heavy_total}"
            f" dropped={heavy_dropped} ({heavy_dropped / max(heavy_total, 1):.4%})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/atom3d_psr_lmdb")
    parser.add_argument("--out", default="data/atom3d_psr_prepared")
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()
    if args.inspect:
        inspect(Path(args.root))
    else:
        convert(Path(args.root), Path(args.out))


if __name__ == "__main__":
    main()
