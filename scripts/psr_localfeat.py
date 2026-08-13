"""Derive contact-scale local invariants for the PSR shards.

For every atom, counts heavy-atom neighbors within 3.5 A and 5.0 A (self
excluded) and writes ``local_{split}.pt`` next to the shards: one int16
``(N, 2)`` tensor per decoy. These two scalars are the diagnostic probe for
the locality band-limit hypothesis — they inject exactly the contact-scale
signal the encoder's global channels attenuate, without touching the model.

    uv run python scripts/psr_localfeat.py
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

RADII = (3.5, 5.0)


def main() -> None:
    root = Path("data/atom3d_psr_prepared")
    for split in ("train", "val", "test"):
        shard = torch.load(root / f"{split}.pt", weights_only=False)
        started = time.monotonic()
        counts: list[torch.Tensor] = []
        for pos in shard["pos"]:
            distance = torch.cdist(pos, pos)
            per_radius = [
                (distance < radius).sum(dim=1).sub_(1) for radius in RADII
            ]
            counts.append(torch.stack(per_radius, dim=1).to(torch.int16))
        torch.save(counts, root / f"local_{split}.pt")
        sample = torch.cat([c.float() for c in counts[:200]])
        print(
            f"{split}: {len(counts)} decoys in {time.monotonic() - started:.0f}s;"
            f" mean counts {sample.mean(dim=0).tolist()}"
        )


if __name__ == "__main__":
    main()
