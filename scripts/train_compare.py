from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Sequence

import torch

from equivariant_attention.benchmarking import (
    GraphSample,
    SyntheticMoleculeDataset,
    collate_graphs,
    load_qm9_samples,
    split_dataset,
)
from equivariant_attention.training import build_regression_model, evaluate_regression, fit_target_normalizer, train_regression_step


QM9_DATA_HASHES = {
    "raw/gdb9.sdf": "98c4e97d50ac549b8c9f0b2114b348a9a944718e17e50d9a724b729f1deaa28e",
    "raw/gdb9.sdf.csv": "73a67793e3cfa9660f001278bd019c143f57e4785db537a01811cf2ce72aa7eb",
    "processed/data_v3.pt": "9254af077d7bc651631bb56a3a689fb41004731b413bdd0ec8c6efa318229f83",
}


def main() -> None:
    args = parse_args()
    split_seed = args.seed if args.split_seed is None else args.split_seed
    model_seed = args.seed if args.model_seed is None else args.model_seed
    device = torch.device(args.device)
    amp_dtype = _resolve_amp_dtype(args.amp_dtype)

    dataset = load_dataset(args)
    data_identity = _qm9_data_identity(args.data_root) if args.dataset == "qm9" else None
    train_size = args.train_size if args.train_size is not None else max(2, int(0.7 * len(dataset)))
    val_size = args.val_size if args.val_size is not None else max(1, int(0.15 * len(dataset)))
    train_idx, val_idx, test_idx = split_dataset(dataset, train_size=train_size, val_size=val_size, seed=split_seed)

    node_dim = dataset[0].node_feats.shape[1]
    run_started = time.perf_counter()
    torch.manual_seed(model_seed)
    model = build_regression_model(
        args.model,
        node_dim=node_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        moment_radial_trace=args.moment_radial_trace,
        moment_full_gram_invariants=args.moment_full_gram_invariants,
        moment_shifted_angular_kernel=args.moment_shifted_angular_kernel,
        moment_radial_distance_kernel=args.moment_radial_distance_kernel,
        moment_dynamic_moment_routing=args.moment_dynamic_moment_routing,
        moment_sinkhorn_iterations=args.moment_sinkhorn_iterations,
        moment_learnable_balance_exponent=args.moment_learnable_balance_exponent,
        moment_equivariant_ffn=args.moment_equivariant_ffn,
        moment_ffn_hidden_ratio=args.moment_ffn_hidden_ratio,
        moment_radial_distance_shift_init=args.moment_radial_distance_shift_init,
        moment_routing_hidden_dim=args.moment_routing_hidden_dim,
        moment_routing_delta_scale=args.moment_routing_delta_scale,
    ).to(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    normalizer = None if args.no_target_normalize else fit_target_normalizer(dataset[i] for i in train_idx)
    max_neighbors = _effective_max_neighbors(args.model, args.max_neighbors)

    final_loss = 0.0
    for step in range(args.steps):
        batch_indices = _cyclic_batch(train_idx, step, args.batch_size)
        batch = collate_graphs([dataset[i] for i in batch_indices], max_neighbors=max_neighbors)
        final_loss = train_regression_step(
            model,
            batch,
            optimizer,
            grad_clip=args.grad_clip,
            target_normalizer=normalizer,
            amp_dtype=amp_dtype,
        )

    val_batches = list(_iter_batches(dataset, val_idx, args.batch_size, max_neighbors))
    val_metrics = evaluate_regression(model, val_batches, target_normalizer=normalizer, amp_dtype=amp_dtype)
    elapsed_seconds = time.perf_counter() - run_started
    metrics = {
        "dataset": args.dataset,
        "model": args.model,
        "steps": args.steps,
        "train_loss": final_loss,
        "val_mae": val_metrics["mae"],
        "val_rmse": val_metrics["rmse"],
        "split_seed": split_seed,
        "split_kind": "seeded_random_row_warm_start",
        "split_hashes": {
            "train": _hash_indices(train_idx),
            "validation": _hash_indices(val_idx),
            "test": _hash_indices(test_idx),
        },
        "model_seed": model_seed,
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
        "target_normalized": normalizer is not None,
        "test_evaluated": not args.skip_test_eval,
        "amp_dtype": args.amp_dtype,
        "elapsed_seconds": elapsed_seconds,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "source_sha256": _source_hash(),
        "moment_features": {
            "radial_trace": args.moment_radial_trace,
            "full_gram_invariants": args.moment_full_gram_invariants,
            "shifted_angular_kernel": args.moment_shifted_angular_kernel,
            "radial_distance_kernel": args.moment_radial_distance_kernel,
            "radial_distance_shift_init": args.moment_radial_distance_shift_init,
            "dynamic_moment_routing": args.moment_dynamic_moment_routing,
            "sinkhorn_iterations": args.moment_sinkhorn_iterations,
            "routing_hidden_dim": args.moment_routing_hidden_dim,
            "routing_delta_scale": args.moment_routing_delta_scale,
            "learnable_balance_exponent": args.moment_learnable_balance_exponent,
            "equivariant_ffn": args.moment_equivariant_ffn,
            "ffn_hidden_ratio": args.moment_ffn_hidden_ratio,
        },
        "run_config": _run_config(
            args,
            split_seed=split_seed,
            model_seed=model_seed,
            effective_max_neighbors=max_neighbors,
        ),
    }
    if args.dataset == "qm9":
        metrics["target"] = _qm9_target_metadata(args.qm9_target_index)
        metrics["data_identity"] = data_identity
    if args.model == "moment_linear" and args.moment_equivariant_ffn:
        metrics["ffn_residual_scales"] = {
            "scalar": [float(layer.ffn_scalar_residual_scale.detach().cpu()) for layer in model.layers],
            "vector": [float(layer.ffn_vector_residual_scale.detach().cpu()) for layer in model.layers],
        }
    if args.model == "moment_linear" and args.moment_radial_distance_kernel:
        metrics["radial_distance_shifts"] = [
            (1.0 + torch.nn.functional.softplus(layer.raw_radial_distance_shift)).detach().cpu().tolist()
            for layer in model.layers
        ]
    if args.model == "moment_linear" and args.moment_dynamic_moment_routing:
        metrics["dynamic_routing_output_norms"] = [
            {
                "invariant": float(layer.routing_mlp[-1].weight.detach().float().norm().cpu()),
                "context": float(layer.routing_context.weight.detach().float().norm().cpu()),
            }
            for layer in model.layers
        ]
    if not args.skip_test_eval:
        test_batches = list(_iter_batches(dataset, test_idx, args.batch_size, max_neighbors))
        test_metrics = evaluate_regression(model, test_batches, target_normalizer=normalizer, amp_dtype=amp_dtype)
        metrics["test_mae"] = test_metrics["mae"]
        metrics["test_rmse"] = test_metrics["rmse"]
    if normalizer is not None:
        metrics["target_normalizer"] = normalizer.as_dict()
    text = json.dumps(metrics, indent=2, sort_keys=True)
    print(text)
    if args.metrics_out is not None:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(text + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare EGNN and equivariant-attention on small regression benchmarks.")
    parser.add_argument("--dataset", choices=["synthetic", "qm9"], default="synthetic")
    parser.add_argument(
        "--model",
        choices=["egnn", "rich_local", "rich_linear", "rich_linear_light", "moment_linear"],
        default="egnn",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/qm9"))
    parser.add_argument("--qm9-target-index", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--val-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-neighbors", type=int, default=None)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--model-seed", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--amp-dtype", choices=["none", "bf16"], default="none")
    parser.add_argument("--metrics-out", type=Path, default=None)
    parser.add_argument("--no-target-normalize", action="store_true")
    parser.add_argument("--skip-test-eval", action="store_true")
    parser.add_argument("--moment-radial-trace", action="store_true")
    parser.add_argument("--moment-full-gram-invariants", action="store_true")
    parser.add_argument("--moment-shifted-angular-kernel", action="store_true")
    parser.add_argument("--moment-radial-distance-kernel", action="store_true")
    parser.add_argument("--moment-radial-distance-shift-init", type=float, default=1.1)
    parser.add_argument("--moment-dynamic-moment-routing", action="store_true")
    parser.add_argument("--moment-sinkhorn-iterations", type=int, default=1)
    parser.add_argument("--moment-routing-hidden-dim", type=int, default=16)
    parser.add_argument("--moment-routing-delta-scale", type=float, default=0.25)
    parser.add_argument("--moment-learnable-balance-exponent", action="store_true")
    parser.add_argument(
        "--moment-equivariant-ffn",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--moment-ffn-hidden-ratio", type=float, default=2.0)
    return parser.parse_args(argv)


def load_dataset(args: argparse.Namespace) -> Sequence[GraphSample]:
    if args.dataset == "synthetic":
        return SyntheticMoleculeDataset(num_samples=args.num_samples, node_dim=8, seed=args.seed)
    return load_qm9_samples(args.data_root, target_index=args.qm9_target_index, limit=args.num_samples)


def _cyclic_batch(indices: Sequence[int], step: int, batch_size: int) -> list[int]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    offset = (step * batch_size) % len(indices)
    return [indices[(offset + i) % len(indices)] for i in range(batch_size)]


def _effective_max_neighbors(model_name: str, max_neighbors: int | None) -> int | None:
    if max_neighbors is not None:
        return max_neighbors
    return 0 if model_name.startswith("rich_linear") or model_name == "moment_linear" else None


def _resolve_amp_dtype(name: str) -> torch.dtype | None:
    if name == "none":
        return None
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"unknown amp dtype: {name}")


def _hash_indices(indices: Sequence[int]) -> str:
    canonical = ",".join(str(index) for index in indices).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def _source_hash() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "src").rglob("*.py"))
    paths.extend(
        path
        for path in [
            root / "scripts" / "train_compare.py",
            root / "PROJECT.md",
            root / "docs" / "LAYER_MATH.md",
            root / "docs" / "QM9_CONTRACT.md",
            root / "pyproject.toml",
            root / "uv.lock",
        ]
        if path.exists()
    )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _run_config(
    args: argparse.Namespace,
    *,
    split_seed: int,
    model_seed: int,
    effective_max_neighbors: int | None,
) -> dict[str, object]:
    return {
        "dataset": args.dataset,
        "data_root": str(args.data_root),
        "qm9_target_index": args.qm9_target_index,
        "num_samples": args.num_samples,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "batch_size": args.batch_size,
        "max_neighbors": args.max_neighbors,
        "effective_max_neighbors": effective_max_neighbors,
        "steps": args.steps,
        "model": args.model,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "split_seed": split_seed,
        "model_seed": model_seed,
        "device": args.device,
        "amp_dtype": args.amp_dtype,
        "target_normalized": not args.no_target_normalize,
        "test_evaluated": not args.skip_test_eval,
        "moment_radial_trace": args.moment_radial_trace,
        "moment_full_gram_invariants": args.moment_full_gram_invariants,
        "moment_shifted_angular_kernel": args.moment_shifted_angular_kernel,
        "moment_radial_distance_kernel": args.moment_radial_distance_kernel,
        "moment_radial_distance_shift_init": args.moment_radial_distance_shift_init,
        "moment_dynamic_moment_routing": args.moment_dynamic_moment_routing,
        "moment_sinkhorn_iterations": args.moment_sinkhorn_iterations,
        "moment_routing_hidden_dim": args.moment_routing_hidden_dim,
        "moment_routing_delta_scale": args.moment_routing_delta_scale,
        "moment_learnable_balance_exponent": args.moment_learnable_balance_exponent,
        "moment_equivariant_ffn": args.moment_equivariant_ffn,
        "moment_ffn_hidden_ratio": args.moment_ffn_hidden_ratio,
    }


def _qm9_data_identity(
    root: Path,
    expected: dict[str, str] = QM9_DATA_HASHES,
) -> dict[str, str]:
    actual = {relative: _hash_file(root / relative) for relative in expected}
    mismatches = [relative for relative, digest in actual.items() if digest != expected[relative]]
    if mismatches:
        joined = ", ".join(mismatches)
        raise ValueError(f"QM9 data identity mismatch: {joined}")
    return actual


def _hash_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"required data file not found: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _qm9_target_metadata(index: int) -> dict[str, str | int]:
    targets = (
        ("mu", "D"),
        ("alpha", "a0^3"),
        ("homo", "eV"),
        ("lumo", "eV"),
        ("gap", "eV"),
        ("r2", "a0^2"),
        ("zpve", "eV"),
        ("U0", "eV"),
        ("U", "eV"),
        ("H", "eV"),
        ("G", "eV"),
        ("Cv", "cal/(mol K)"),
        ("U0_atom", "eV"),
        ("U_atom", "eV"),
        ("H_atom", "eV"),
        ("G_atom", "eV"),
        ("A", "GHz"),
        ("B", "GHz"),
        ("C", "GHz"),
    )
    if not 0 <= index < len(targets):
        raise ValueError(f"QM9 target index must be between 0 and {len(targets) - 1}")
    name, unit = targets[index]
    return {"index": index, "name": name, "unit": unit}


def _iter_batches(
    dataset: Sequence[GraphSample],
    indices: Sequence[int],
    batch_size: int,
    max_neighbors: int | None,
):
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        yield collate_graphs([dataset[i] for i in chunk], max_neighbors=max_neighbors)


if __name__ == "__main__":
    main()
