"""Speed/VRAM probe for training-time optimizations on PSR batches.

Measures eager fp32 against TF32, bf16 autocast, and torch.compile
(dynamic=True, since node counts vary per batch) on real PSR training
steps: seconds per step, atoms per second, peak VRAM, cold-start cost of
the first step, and the forward-output drift versus eager fp32.

    tsp uv run python scripts/psr_opt_probe.py --device cuda
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

CONFIGS = (
    {"name": "eager-fp32"},
    {"name": "eager-tf32", "tf32": True},
    {"name": "eager-bf16", "tf32": True, "amp": True},
    {"name": "compile-fp32", "compile": True},
    {"name": "compile-bf16", "tf32": True, "amp": True, "compile": True},
)


def build_model(args, device: str) -> ELA:
    torch.manual_seed(0)
    return ELA(
        f"{FEATURE_DIM}x0e", "1x0e", width=args.width, depth=args.depth
    ).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/atom3d_psr_prepared")
    parser.add_argument("--node-budget", type=int, default=12000)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    split = Split(Path(args.data_root) / "train.pt", limit=0)
    batches = split.batches(args.node_budget, shuffle=False)
    needed = 1 + args.warmup + args.steps
    if len(batches) < needed:
        raise SystemExit(f"need {needed} batches, have {len(batches)}")
    cuda = args.device.startswith("cuda")
    label_cols = [1]

    reference_graph = split.graph(batches[0], label_cols).to(args.device)
    reference = build_model(args, args.device)
    reference.eval()
    with torch.no_grad():
        expected = reference(reference_graph).graph_x
    assert expected is not None
    del reference

    for config in CONFIGS:
        torch.set_float32_matmul_precision(
            "high" if config.get("tf32") else "highest"
        )
        model = build_model(args, args.device)
        stepper = (
            torch.compile(model, dynamic=True) if config.get("compile") else model
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        autocast_enabled = bool(config.get("amp"))

        def run_forward(graph, stepper=stepper, amp=autocast_enabled):
            if amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    return stepper(graph)
            return stepper(graph)

        try:
            model.eval()
            with torch.no_grad():
                drift = float(
                    (run_forward(reference_graph).graph_x.float() - expected)
                    .abs()
                    .max()
                )
            model.train()
            if cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            cold_started = time.monotonic()
            cold = None
            atoms = 0
            for index, indices in enumerate(batches[:needed]):
                graph = split.graph(indices, label_cols).to(args.device)
                if index == 1 + args.warmup:
                    if cuda:
                        torch.cuda.synchronize()
                    started = time.monotonic()
                if index >= 1 + args.warmup:
                    atoms += sum(split.sizes[i] for i in indices)
                output = run_forward(graph)
                assert output.graph_x is not None and graph.y is not None
                loss = torch.nn.functional.mse_loss(
                    output.graph_x.float(), graph.y
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                if index == 0:
                    if cuda:
                        torch.cuda.synchronize()
                    cold = time.monotonic() - cold_started
            if cuda:
                torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            print(
                json.dumps(
                    {
                        "config": config["name"],
                        "sec_per_step": round(elapsed / args.steps, 4),
                        "atoms_per_sec": round(atoms / elapsed),
                        "peak_gb": (
                            round(torch.cuda.max_memory_allocated() / 2**30, 2)
                            if cuda
                            else None
                        ),
                        "first_step_sec": round(cold, 2) if cold else None,
                        "forward_drift_vs_fp32": drift,
                    }
                )
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(json.dumps({"config": config["name"], "oom": True}))
        except Exception as error:  # noqa: BLE001 - isolate compiler failures
            print(
                json.dumps(
                    {"config": config["name"], "error": repr(error)[:300]}
                )
            )
        del model, stepper, optimizer
        if cuda:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
