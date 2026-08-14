"""Train ELA on ATOM3D PSR (Zenodo split-by-year) prepared shards.

Reads the compact shards written by ``scripts/psr_prepare.py`` (torch-only,
no extra dependencies) and trains ``ELA("167x0e", ...)`` to predict decoy
quality scores from (residue, atom)-typed points and coordinates.

By default all four scores ``[rmsd, gdt_ts, gdt_ha, tm]`` are predicted
(rmsd as log1p); ``--labels gdt_ts`` trains the single canonical target.
Reported metrics are always computed on gdt_ts: global Pearson/Spearman and
the canonical mean per-target Spearman.

CPU smoke:
    uv run python scripts/train_psr.py --limit 32 --steps 2 --epochs 1
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from psr_vocab import ELEMENT_DIM, ELEMENT_OF_CLASS, FEATURE_DIM, LABEL_NAMES

from equivariant_linear_attention import ELA, ELAGraph

_ELEMENT_LOOKUP = torch.tensor(ELEMENT_OF_CLASS, dtype=torch.long)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/atom3d_psr_prepared")
    parser.add_argument("--out", default="outputs/psr")
    parser.add_argument("--labels", choices=("all", "gdt_ts"), default="all")
    parser.add_argument(
        "--features",
        choices=("resatom", "element", "element-local"),
        default="resatom",
        help="resatom: 167-class (residue, atom); element: C/N/O/S one-hot"
        " matching the ATOM3D/GVP baseline featurization; element-local adds"
        " neighbor counts at 3.5/5.0 A from scripts/psr_localfeat.py",
    )
    parser.add_argument(
        "--rank-weight",
        type=float,
        default=0.0,
        help="weight of the within-target pairwise ranking loss; enables"
        " target-grouped batching",
    )
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=500, help="warmup steps")
    parser.add_argument("--clip", type=float, default=10.0, help="grad-norm clip (0 = off)")
    parser.add_argument("--node-budget", type=int, default=12000)
    parser.add_argument(
        "--density-bandwidths",
        default="",
        help="comma-separated Angstrom bandwidths for the in-model node-linear"
        " soft neighbour-count channel; empty disables it",
    )
    parser.add_argument(
        "--num-local-charts",
        type=int,
        default=16,
        help="chart count of the local chart-recentered Mercer relation;"
        " 0 disables the sector and leaves absolute-scale features only",
    )
    parser.add_argument(
        "--local-points",
        type=int,
        default=0,
        help="neighbours per node for the NON-CANONICAL pointwise local-jet"
        " branch; 0 disables it and keeps the model edge-free. Above 0 the model"
        " builds a transient kNN support (inferred edges, gather/scatter, O(kN)"
        " memory, per-segment Python loop) and is a hard-cutoff upper-bound"
        " diagnostic only; see docs/LOCALITY_TRACK.md",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="graphs per split (0 = all)")
    parser.add_argument("--steps", type=int, default=0, help="steps per epoch (0 = all)")
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="evaluate the held-out test split (final runs only; screens stay val-only)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile(dynamic=True); probed 3.15x faster and ~10x less"
        " VRAM than eager fp32 with forward drift 2e-7 (cold start ~160s)",
    )
    return parser.parse_args()


LOCAL_SCALE = torch.tensor([10.0, 30.0])


class Split:
    def __init__(self, path: Path, limit: int, local: bool = False) -> None:
        shard = torch.load(path, weights_only=False)
        keep = slice(None) if limit <= 0 else slice(limit)
        self.targets: list[str] = shard["targets"][keep]
        self.class_idx: list[torch.Tensor] = shard["class_idx"][keep]
        self.pos: list[torch.Tensor] = shard["pos"][keep]
        self.scores: torch.Tensor = shard["scores"][keep]
        self.sizes = [int(t.shape[0]) for t in self.class_idx]
        self.local: list[torch.Tensor] | None = None
        if local:
            counts = torch.load(
                path.parent / f"local_{path.stem}.pt", weights_only=False
            )
            self.local = counts[keep]

    def __len__(self) -> int:
        return len(self.sizes)

    def batches(
        self, node_budget: int, shuffle: bool, by_target: bool = False
    ) -> list[list[int]]:
        if by_target:
            groups: dict[str, list[int]] = {}
            for index, target in enumerate(self.targets):
                groups.setdefault(target, []).append(index)
            keys = list(groups)
            if shuffle:
                random.shuffle(keys)
                for key in keys:
                    random.shuffle(groups[key])
            order = [index for key in keys for index in groups[key]]
        else:
            order = list(range(len(self)))
            if shuffle:
                random.shuffle(order)
        packed: list[list[int]] = []
        current: list[int] = []
        nodes = 0
        for index in order:
            if current and nodes + self.sizes[index] > node_budget:
                packed.append(current)
                current, nodes = [], 0
            current.append(index)
            nodes += self.sizes[index]
        if current:
            packed.append(current)
        return packed

    def graph(
        self, indices: list[int], label_cols: list[int], feature_dim: int = FEATURE_DIM
    ) -> ELAGraph:
        idx = torch.cat([self.class_idx[i].long() for i in indices])
        if feature_dim != FEATURE_DIM:
            idx = _ELEMENT_LOOKUP[idx]
            feature_dim = ELEMENT_DIM
        x = torch.nn.functional.one_hot(idx, feature_dim).to(torch.float32)
        if self.local is not None:
            counts = torch.cat([self.local[i] for i in indices]).float()
            x = torch.cat([x, counts / LOCAL_SCALE], dim=1)
        y = self.scores[indices][:, label_cols].clone()
        if 0 in label_cols:
            col = label_cols.index(0)
            y[:, col] = torch.log1p(y[:, col])
        return ELAGraph(
            x=x,
            pos=torch.cat([self.pos[i] for i in indices]),
            batch=torch.repeat_interleave(
                torch.arange(len(indices)),
                torch.tensor([self.sizes[i] for i in indices]),
            ),
            y=y,
        )


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    return float(a.dot(b) / denom) if denom > 0 else 0.0


def _ranks(values: torch.Tensor) -> torch.Tensor:
    order = values.argsort()
    ranks = torch.empty(len(values), dtype=torch.float64)
    ranks[order] = torch.arange(len(values), dtype=torch.float64)
    unique, inverse, counts = torch.unique(
        values, return_inverse=True, return_counts=True
    )
    sums = torch.zeros(len(unique), dtype=torch.float64).scatter_add_(0, inverse, ranks)
    return sums[inverse] / counts[inverse].to(torch.float64)


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    return _pearson(_ranks(a), _ranks(b))


@torch.no_grad()
def local_sector_state(model: ELA) -> dict[str, float]:
    """Tripwire for the local relation: is it still anchored and contact-scale?

    Chart collapse degrades the local sector into copies of the global Mercer
    and is otherwise indistinguishable from "locality does not help". The seed
    anchoring strength and the learned bandwidths are the two knobs training
    could use to walk away from locality, so both are logged every epoch.
    """

    scales: list[float] = []
    sigmas: list[float] = []
    for layer in model.layers:
        local = layer.relation.local
        if local is None:
            return {}
        scales.append(
            float(torch.nn.functional.softplus(local.raw_seed_scale).mean()) + 0.5
        )
        gamma = torch.nn.functional.softplus(local.raw_gamma) + 0.05
        sigmas.append(float((0.5 / gamma).sqrt().mean()) * model.config.length_scale)
    return {
        "local_seed_scale": round(sum(scales) / len(scales), 4),
        "local_sigma_a": round(sum(sigmas) / len(sigmas), 3),
    }


@torch.no_grad()
def evaluate(
    model: ELA,
    split: Split,
    label_cols: list[int],
    node_budget: int,
    device: str,
    feature_dim: int = FEATURE_DIM,
) -> dict[str, float]:
    model.eval()
    gdt_col = label_cols.index(LABEL_NAMES.index("gdt_ts"))
    predicted: list[torch.Tensor] = []
    for indices in split.batches(node_budget, shuffle=False):
        graph = split.graph(indices, label_cols, feature_dim).to(device)
        output = model(graph)
        assert output.graph_x is not None
        predicted.append(output.graph_x[:, gdt_col].double().cpu())
    prediction = torch.cat(predicted)
    truth = split.scores[:, LABEL_NAMES.index("gdt_ts")].double()

    per_target: list[float] = []
    by_target: dict[str, list[int]] = {}
    for row, target in enumerate(split.targets):
        by_target.setdefault(target, []).append(row)
    for rows in by_target.values():
        subset = torch.tensor(rows)
        if len(rows) >= 2 and truth[subset].std() > 0:
            per_target.append(_spearman(prediction[subset], truth[subset]))
    return {
        "pearson": _pearson(prediction, truth),
        "spearman": _spearman(prediction, truth),
        "per_target_spearman": sum(per_target) / max(len(per_target), 1),
        "targets": float(len(per_target)),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    root = Path(args.data_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    label_cols = (
        list(range(len(LABEL_NAMES)))
        if args.labels == "all"
        else [LABEL_NAMES.index("gdt_ts")]
    )
    use_local = args.features == "element-local"
    names = ("train", "val", "test") if args.eval_test else ("train", "val")
    splits = {
        name: Split(root / f"{name}.pt", args.limit, local=use_local)
        for name in names
    }
    print({name: len(split) for name, split in splits.items()})

    feature_dim = FEATURE_DIM if args.features == "resatom" else ELEMENT_DIM
    model = ELA(
        f"{feature_dim + 2 * use_local}x0e",
        f"{len(label_cols)}x0e",
        width=args.width,
        depth=args.depth,
        num_local_charts=args.num_local_charts,
        density_bandwidths=tuple(
            float(value) for value in args.density_bandwidths.split(",") if value
        ),
        local_points=args.local_points,
    ).to(args.device)
    runner = torch.compile(model, dynamic=True) if args.compile else model
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    steps_per_epoch = len(splits["train"].batches(args.node_budget, shuffle=False))
    if args.steps:
        steps_per_epoch = min(steps_per_epoch, args.steps)
    total_steps = max(args.epochs * steps_per_epoch, 1)

    def lr_scale(step: int) -> float:
        if step < args.warmup:
            return (step + 1) / max(args.warmup, 1)
        progress = (step - args.warmup) / max(total_steps - args.warmup, 1)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    history: list[dict] = []
    best = -math.inf

    cuda = args.device.startswith("cuda")
    for epoch in range(args.epochs):
        model.train()
        started = time.monotonic()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        losses: list[float] = []
        for step, indices in enumerate(
            splits["train"].batches(
                args.node_budget, shuffle=True, by_target=args.rank_weight > 0
            )
        ):
            if args.steps and step >= args.steps:
                break
            graph = splits["train"].graph(indices, label_cols, feature_dim).to(
                args.device
            )
            output = runner(graph)
            assert output.graph_x is not None and graph.y is not None
            loss = torch.nn.functional.mse_loss(output.graph_x, graph.y)
            if args.rank_weight > 0 and len(indices) > 1:
                col = label_cols.index(LABEL_NAMES.index("gdt_ts"))
                same = torch.tensor(
                    [
                        [splits["train"].targets[i] == splits["train"].targets[j]
                         for j in indices]
                        for i in indices
                    ],
                    device=args.device,
                ).triu(1)
                truth_gap = graph.y[:, col][:, None] - graph.y[:, col][None, :]
                pred_gap = (
                    output.graph_x[:, col][:, None] - output.graph_x[:, col][None, :]
                )
                pair = same & (truth_gap.abs() > 1e-4)
                if pair.any():
                    rank = torch.nn.functional.softplus(
                        -pred_gap[pair] * truth_gap[pair].sign() / 0.05
                    ).mean()
                    loss = loss + args.rank_weight * rank
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
        metrics = evaluate(
            runner,
            splits["val"],
            label_cols,
            args.node_budget,
            args.device,
            feature_dim,
        )
        record = {
            "epoch": epoch,
            "train_loss": sum(losses) / max(len(losses), 1),
            "lr": scheduler.get_last_lr()[0],
            "seconds": round(time.monotonic() - started, 1),
            **local_sector_state(model),
            **(
                {"peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
                if cuda
                else {}
            ),
            **{f"val_{k}": v for k, v in metrics.items()},
        }
        history.append(record)
        print(json.dumps(record))
        if metrics["per_target_spearman"] > best:
            best = metrics["per_target_spearman"]
            torch.save(model.state_dict(), out / "best.pt")
        (out / "history.json").write_text(json.dumps(history, indent=1))

    if args.eval_test:
        model.load_state_dict(torch.load(out / "best.pt", weights_only=True))
        final = {
            f"test_{k}": v
            for k, v in evaluate(
                runner,
                splits["test"],
                label_cols,
                args.node_budget,
                args.device,
                feature_dim,
            ).items()
        }
        print(json.dumps(final))
        (out / "test.json").write_text(json.dumps(final, indent=1))


if __name__ == "__main__":
    main()
