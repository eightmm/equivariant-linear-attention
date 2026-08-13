"""Throughput and VRAM probe for PSR training batch sizing.

Runs a few real fwd+bwd training steps at each candidate node budget and
reports seconds per step, atoms per second, and peak allocated VRAM, so the
screening budget can be chosen from measurement instead of guesswork:

    tsp uv run python scripts/psr_probe.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from psr_vocab import FEATURE_DIM
from train_psr import Split

from equivariant_linear_attention import ELA


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/atom3d_psr_prepared")
    parser.add_argument("--budgets", default="6000,12000,24000,48000,96000")
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    split = Split(Path(args.data_root) / "train.pt", limit=0)
    cuda = args.device.startswith("cuda")
    label_cols = [1]
    model = ELA(f"{FEATURE_DIM}x0e", "1x0e", width=args.width, depth=args.depth).to(
        args.device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for budget in (int(b) for b in args.budgets.split(",")):
        batches = split.batches(budget, shuffle=False)[: args.warmup + args.steps]
        if len(batches) < args.warmup + args.steps:
            print(json.dumps({"budget": budget, "skipped": "not enough batches"}))
            continue
        try:
            if cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            atoms = 0
            for index, indices in enumerate(batches):
                if index == args.warmup:
                    if cuda:
                        torch.cuda.synchronize()
                    started = time.monotonic()
                if index >= args.warmup:
                    atoms += sum(split.sizes[i] for i in indices)
                graph = split.graph(indices, label_cols).to(args.device)
                output = model(graph)
                assert output.graph_x is not None and graph.y is not None
                loss = torch.nn.functional.mse_loss(output.graph_x, graph.y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
            if cuda:
                torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            print(
                json.dumps(
                    {
                        "budget": budget,
                        "sec_per_step": round(elapsed / args.steps, 4),
                        "atoms_per_sec": round(atoms / elapsed),
                        "peak_gb": (
                            round(torch.cuda.max_memory_allocated() / 2**30, 2)
                            if cuda
                            else None
                        ),
                    }
                )
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(json.dumps({"budget": budget, "oom": True}))


if __name__ == "__main__":
    main()
