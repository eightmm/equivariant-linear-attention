from __future__ import annotations

import argparse
import hashlib
import json
from math import sqrt
import time
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from equivariant_attention._egnn_baseline import (
    _DynamicEGNNBaseline,
    _StaticEGNNBaseline,
)
from equivariant_attention.moment import routing_head_counts as _routing_head_counts
from equivariant_attention.reproducibility import configure_reproducibility

from equivariant_attention.benchmarking import (
    GraphSample,
    SyntheticMoleculeDataset,
    collate_graphs,
    load_qm9_samples,
    split_dataset,
)
from equivariant_attention.diagnostics import (
    attention_weight_summary,
    dense_kernel_attention_summary,
    kernel_component_quantiles,
    kernel_parameter_summary,
    local_attention_summary,
    memory_assignment_summary,
    memory_center_summary,
    memory_pair_gate_summary,
    pair_gate_summary,
)
from equivariant_attention.training import (
    TargetNormalizer,
    build_regression_model,
    evaluate_regression,
    fit_target_normalizer,
    train_regression_step,
)


QM9_DATA_HASHES = {
    "raw/gdb9.sdf": "98c4e97d50ac549b8c9f0b2114b348a9a944718e17e50d9a724b729f1deaa28e",
    "raw/gdb9.sdf.csv": "73a67793e3cfa9660f001278bd019c143f57e4785db537a01811cf2ce72aa7eb",
    "processed/data_v3.pt": "9254af077d7bc651631bb56a3a689fb41004731b413bdd0ec8c6efa318229f83",
}

NODE_COUNT_STRATUM_UPPER_BOUNDS = (16, 32, 64, 128, 512, 2048)
_EGNN_BENCHMARK_MODEL_CHOICES = (
    "internal_static_egnn_baseline",
    "internal_dynamic_egnn_baseline",
)
_EGNN_BENCHMARK_MODELS = frozenset(_EGNN_BENCHMARK_MODEL_CHOICES)


def main() -> None:
    args = parse_args()
    split_seed = args.seed if args.split_seed is None else args.split_seed
    model_seed = args.seed if args.model_seed is None else args.model_seed
    reproducibility = configure_reproducibility(
        seed=model_seed,
        mode=args.determinism,
    )
    device = torch.device(args.device)
    amp_dtype = _resolve_amp_dtype(args.amp_dtype)

    dataset = load_dataset(args)
    data_identity = (
        _qm9_data_identity(args.data_root) if args.dataset == "qm9" else None
    )
    train_size = (
        args.train_size
        if args.train_size is not None
        else max(2, int(0.7 * len(dataset)))
    )
    val_size = (
        args.val_size if args.val_size is not None else max(1, int(0.15 * len(dataset)))
    )
    train_idx, val_idx, test_idx = split_dataset(
        dataset, train_size=train_size, val_size=val_size, seed=split_seed
    )

    node_dim = dataset[0].node_feats.shape[1]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    run_started = time.perf_counter()
    model = _build_benchmark_model(args, node_dim=node_dim).to(device=device)
    initial_state_hashes = _model_state_hashes(model)
    paired_base_initial_state_hashes = _paired_base_state_hashes(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    normalizer = (
        None
        if args.no_target_normalize
        else fit_target_normalizer(dataset[i] for i in train_idx)
    )
    final_loss = 0.0
    gradient_monitor: dict[str, float | int] = {}
    for step in range(args.steps):
        batch_indices = _cyclic_batch(train_idx, step, args.batch_size)
        batch = collate_graphs([dataset[i] for i in batch_indices])
        final_loss = train_regression_step(
            model,
            batch,
            optimizer,
            grad_clip=args.grad_clip,
            target_normalizer=normalizer,
            amp_dtype=amp_dtype,
            gradient_monitor=gradient_monitor,
        )

    val_batches = list(_iter_batches(dataset, val_idx, args.batch_size))
    val_metrics = evaluate_regression(
        model, val_batches, target_normalizer=normalizer, amp_dtype=amp_dtype
    )
    train_probe_indices = train_idx[: min(256, len(train_idx))]
    train_probe_metrics = evaluate_regression(
        model,
        _iter_batches(dataset, train_probe_indices, args.batch_size),
        target_normalizer=normalizer,
        amp_dtype=amp_dtype,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - run_started
    metrics = {
        "dataset": args.dataset,
        "model": args.benchmark_model,
        "steps": args.steps,
        "train_loss": final_loss,
        "train_probe": {
            "selection": "train_split_order_prefix",
            "sample_count": len(train_probe_indices),
            "batch_size": args.batch_size,
            **train_probe_metrics,
        },
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
        "test_evaluated": args.evaluate_test,
        "amp_dtype": args.amp_dtype,
        "reproducibility": reproducibility,
        "elapsed_seconds": elapsed_seconds,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "gradient_clipping": _gradient_clipping_summary(
            gradient_monitor, grad_clip=args.grad_clip
        ),
        **initial_state_hashes,
        "paired_base_initial_state_sha256": paired_base_initial_state_hashes[
            "initial_state_sha256"
        ],
        "paired_base_state_schema_sha256": paired_base_initial_state_hashes[
            "state_schema_sha256"
        ],
        "final_state_sha256": _model_state_hashes(model)["initial_state_sha256"],
        "initialization_hash_version": "canonical_state_dict_v1",
        "source_sha256": _source_hash(),
        "run_config": _run_config(
            args,
            split_seed=split_seed,
            model_seed=model_seed,
        ),
    }
    metrics["coordinate_diagnostics"] = _coordinate_update_diagnostics(
        model,
        dataset,
        val_idx,
        sample_count=min(32, len(val_idx)),
    )
    if args.dataset == "qm9":
        metrics["target"] = _qm9_target_metadata(args.qm9_target_index)
        metrics["data_identity"] = data_identity
    if args.benchmark_model == "factorized_moment":
        metrics["ffn_residual_scales"] = {
            "scalar": [
                float(layer.ffn_scalar_residual_scale.detach().cpu())
                for layer in model.layers
            ],
            "vector": [
                float(layer.ffn_vector_residual_scale.detach().cpu())
                for layer in model.layers
            ],
        }
        metrics["attention_residual_scales"] = {
            "scalar": [
                float(layer.scalar_residual_scale.detach().cpu())
                for layer in model.layers
            ],
            "vector": [
                float(layer.vector_residual_scale.detach().cpu())
                for layer in model.layers
            ],
        }
        beta = torch.cat(
            [
                layer.linear_kernel_max
                * torch.sigmoid(layer.raw_linear_kernel.detach())
                for layer in model.layers
            ]
        )
        gamma = torch.cat(
            [
                layer.vector_kernel_max
                * torch.sigmoid(layer.raw_vector_kernel.detach())
                for layer in model.layers
            ]
        )
        metrics["kernel_parameters"] = kernel_parameter_summary(beta, gamma)
        tensor_scales = [
            layer.tensor_kernel_max * torch.sigmoid(layer.raw_tensor_kernel.detach())
            for layer in model.layers
            if layer.raw_tensor_kernel is not None
        ]
        metrics["tensor_kernel_parameters"] = (
            _finite_parameter_summary(torch.cat(tensor_scales), name="eta")
            if tensor_scales
            else {"status": "not_applicable"}
        )
    else:
        coordinate_updates = args.benchmark_model == "internal_dynamic_egnn_baseline"
        metrics["baseline_details"] = {
            "name": args.benchmark_model,
            "official_reproduction": False,
            "coordinate_updates": coordinate_updates,
            "edge_topology": (
                "precomputed_radius_candidates_without_self"
                if args.precompute_local_edges
                else "same_graph_directed_complete_without_self"
            ),
            "distance_feature": "raw_squared_distance",
            "readout": "layernorm_node_linear_graph_mean",
        }
    metrics["gradient_norms"] = _gradient_norms(model)
    gradient_parameters = _gradient_parameter_diagnostics(model)
    gradient_parameters["measurement_point"] = (
        "after_final_training_backward_clip_and_optimizer_step"
    )
    metrics["gradient_parameters"] = gradient_parameters
    metrics["nonzero_gradient_parameter_count"] = gradient_parameters[
        "nonzero_gradient_parameter_count"
    ]
    metrics["coordinate_gradient_parameters"] = (
        _coordinate_gradient_parameter_diagnostics(model)
    )
    metrics["pairwise_local_gradient_parameters"] = (
        _named_gradient_parameter_diagnostics(model, "local_pairwise_content")
    )
    pairwise_module = getattr(model, "local_pairwise_content", None)
    metrics["pairwise_local_residual_scale"] = (
        float(pairwise_module.residual_scale.detach().cpu())
        if pairwise_module is not None
        else "not_applicable"
    )
    metrics["local_radial_gradient_parameters"] = _named_gradient_parameter_diagnostics(
        model, "local_radial"
    )
    metrics["node_count_strata"] = _node_count_strata_metrics(
        model,
        dataset,
        val_idx,
        batch_size=args.batch_size,
        normalizer=normalizer,
        amp_dtype=amp_dtype,
    )
    if args.benchmark_model in _EGNN_BENCHMARK_MODELS:
        metrics["bounded_diagnostics"] = {
            "schema_version": 1,
            "enabled": bool(args.bounded_diagnostics),
            "status": f"not_applicable_{args.benchmark_model}",
            "reason": "factorized attention diagnostics do not apply to EGNN messages",
        }
    else:
        metrics["bounded_diagnostics"] = (
            _bounded_model_diagnostics(
                model,
                dataset,
                val_idx,
                max_nodes=args.diagnostic_max_nodes,
                include_effective_rank=args.diagnostic_effective_rank,
                local_sample_count=args.diagnostic_sample_count,
            )
            if args.bounded_diagnostics
            else _disabled_bounded_diagnostics(args)
        )
    if args.evaluate_test:
        test_batches = list(_iter_batches(dataset, test_idx, args.batch_size))
        test_metrics = evaluate_regression(
            model, test_batches, target_normalizer=normalizer, amp_dtype=amp_dtype
        )
        metrics["test_mae"] = test_metrics["mae"]
        metrics["test_rmse"] = test_metrics["rmse"]
    if normalizer is not None:
        metrics["target_normalizer"] = normalizer.as_dict()
    text = json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False)
    print(text)
    if args.metrics_out is not None:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(text + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one registered model in the matched graph-regression harness."
    )
    parser.add_argument(
        "--benchmark-model",
        choices=["factorized_moment", *_EGNN_BENCHMARK_MODEL_CHOICES],
        default="factorized_moment",
    )
    parser.add_argument("--dataset", choices=["synthetic", "qm9"], default="synthetic")
    parser.add_argument("--data-root", type=Path, default=Path("data/qm9"))
    parser.add_argument("--qm9-target-index", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--val-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--hidden-tensor-dim", type=_nonnegative_int, default=0)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--linear-kernel-init", type=float, default=0.05)
    parser.add_argument(
        "--scalar-content-mode",
        choices=["unit", "bounded"],
        default="unit",
    )
    parser.add_argument("--tensor-product-kernel", action="store_true")
    parser.add_argument("--tensor-kernel-init", type=float, default=0.05)
    parser.add_argument("--tensor-kernel-max", type=float, default=1.0)
    parser.add_argument("--no-alignment-linear-term", action="store_true")
    parser.add_argument(
        "--no-linear-kernel",
        dest="no_alignment_linear_term",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--no-key-balancing", action="store_true")
    parser.add_argument(
        "--kernel-floor-mode",
        choices=["fixed", "inverse_graph_size"],
        default="fixed",
    )
    parser.add_argument(
        "--routing", choices=["ggg", "lgg", "ggl", "lgl", "lll"], default="ggg"
    )
    parser.add_argument(
        "--global-transport-mode",
        choices=["learned", "uniform", "none"],
        default="learned",
    )
    parser.add_argument("--local-cutoff", type=float, default=2.5)
    parser.add_argument("--num-rbf", type=int, default=16)
    parser.add_argument("--learn-local-radial-gate", action="store_true")
    parser.add_argument("--pairwise-local-content", action="store_true")
    parser.add_argument("--pairwise-residual-scale-init", type=float, default=0.1)
    parser.add_argument("--edge-conditioned-local-transport", action="store_true")
    parser.add_argument(
        "--edge-conditioned-local-sqrt-degree",
        action="store_true",
        help=(
            "divide edge-conditioned local receiver sums by the square root "
            "of incoming non-self candidate degree"
        ),
    )
    parser.add_argument("--precompute-local-edges", action="store_true")
    parser.add_argument("--memory-count", type=int, choices=[1, 4, 8], default=1)
    parser.add_argument("--memory-interaction", action="store_true")
    parser.add_argument("--memory-assignment-temperature", type=float, default=1.0)
    parser.add_argument("--memory-assignment-scale", type=float, default=2.5)
    parser.add_argument("--memory-interaction-cutoff", type=float, default=2.5)
    parser.add_argument("--radial-trace", action="store_true")
    parser.add_argument("--coordinate-updates", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--model-seed", type=int, default=None)
    parser.add_argument(
        "--determinism",
        choices=["seeded", "strict"],
        default="seeded",
        help=(
            "seeded preserves the historical lane; strict requests deterministic "
            "PyTorch algorithms and deterministic cuDNN controls"
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--amp-dtype", choices=["none", "bf16"], default="none")
    parser.add_argument("--metrics-out", type=Path, default=None)
    parser.add_argument("--no-target-normalize", action="store_true")
    parser.add_argument("--bounded-diagnostics", action="store_true")
    parser.add_argument("--diagnostic-max-nodes", type=_positive_int, default=128)
    parser.add_argument("--diagnostic-sample-count", type=_positive_int, default=32)
    parser.add_argument("--diagnostic-effective-rank", action="store_true")
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument("--evaluate-test", action="store_true")
    test_group.add_argument(
        "--skip-test-eval",
        dest="evaluate_test",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(evaluate_test=False)
    args = parser.parse_args(argv)
    if args.hidden_dim is None:
        args.hidden_dim = 91 if args.benchmark_model in _EGNN_BENCHMARK_MODELS else 64
    return args


def load_dataset(args: argparse.Namespace) -> Sequence[GraphSample]:
    if args.dataset == "synthetic":
        return SyntheticMoleculeDataset(
            num_samples=args.num_samples, node_dim=8, seed=args.seed
        )
    return load_qm9_samples(
        args.data_root,
        target_index=args.qm9_target_index,
        limit=args.num_samples,
        local_cutoff=(args.local_cutoff if args.precompute_local_edges else None),
    )


def _build_benchmark_model(
    args: argparse.Namespace,
    *,
    node_dim: int,
) -> torch.nn.Module:
    if args.benchmark_model in _EGNN_BENCHMARK_MODELS:
        incompatible = []
        defaults = {
            "routing": "ggg",
            "global_transport_mode": "learned",
            "num_heads": 4,
            "linear_kernel_init": 0.05,
            "kernel_floor_mode": "fixed",
            "local_cutoff": 2.5,
            "num_rbf": 16,
            "memory_count": 1,
            "memory_interaction": False,
            "memory_assignment_temperature": 1.0,
            "memory_assignment_scale": 2.5,
            "memory_interaction_cutoff": 2.5,
            "radial_trace": False,
            "learn_local_radial_gate": False,
            "pairwise_local_content": False,
            "pairwise_residual_scale_init": 0.1,
            "edge_conditioned_local_transport": False,
            "edge_conditioned_local_sqrt_degree": False,
            "hidden_tensor_dim": 0,
            "scalar_content_mode": "unit",
            "tensor_product_kernel": False,
            "tensor_kernel_init": 0.05,
            "tensor_kernel_max": 1.0,
            "no_alignment_linear_term": False,
            "no_key_balancing": False,
            "diagnostic_max_nodes": 128,
            "diagnostic_sample_count": 32,
            "diagnostic_effective_rank": False,
            "coordinate_updates": False,
        }
        for name, expected in defaults.items():
            if getattr(args, name) != expected:
                incompatible.append(name)
        if incompatible:
            names = ", ".join(sorted(incompatible))
            raise ValueError(
                "factorized-attention controls cannot be set for the internal "
                f"EGNN baseline: {names}"
            )
        baseline = (
            _DynamicEGNNBaseline
            if args.benchmark_model == "internal_dynamic_egnn_baseline"
            else _StaticEGNNBaseline
        )
        return baseline(
            node_dim=node_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
        )
    if args.benchmark_model != "factorized_moment":
        raise ValueError(f"unknown benchmark model: {args.benchmark_model}")
    local_head_counts = _routing_head_counts(
        args.routing,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )
    return build_regression_model(
        node_dim=node_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        linear_kernel_init=args.linear_kernel_init,
        use_alignment_linear_term=not args.no_alignment_linear_term,
        use_key_balancing=not args.no_key_balancing,
        kernel_floor_mode=args.kernel_floor_mode,
        local_head_counts=local_head_counts,
        global_transport_mode=args.global_transport_mode,
        local_cutoff=args.local_cutoff,
        num_rbf=args.num_rbf,
        learn_local_radial_gate=args.learn_local_radial_gate,
        use_pairwise_local_content=args.pairwise_local_content,
        pairwise_residual_scale_init=args.pairwise_residual_scale_init,
        use_edge_conditioned_local_transport=(args.edge_conditioned_local_transport),
        normalize_edge_conditioned_local_by_sqrt_degree=(
            args.edge_conditioned_local_sqrt_degree
        ),
        hidden_tensor_dim=args.hidden_tensor_dim,
        scalar_content_mode=args.scalar_content_mode,
        use_tensor_product_kernel=args.tensor_product_kernel,
        tensor_kernel_init=args.tensor_kernel_init,
        tensor_kernel_max=args.tensor_kernel_max,
        global_memory_count=args.memory_count,
        use_memory_interaction=args.memory_interaction,
        memory_assignment_temperature=args.memory_assignment_temperature,
        memory_assignment_scale=args.memory_assignment_scale,
        memory_interaction_cutoff=args.memory_interaction_cutoff,
        use_radial_trace=args.radial_trace,
        coordinate_updates=args.coordinate_updates,
    )


def _cyclic_batch(indices: Sequence[int], step: int, batch_size: int) -> list[int]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    offset = (step * batch_size) % len(indices)
    return [indices[(offset + i) % len(indices)] for i in range(batch_size)]


def _resolve_amp_dtype(name: str) -> torch.dtype | None:
    if name == "none":
        return None
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"unknown amp dtype: {name}")


def _gradient_norms(model: torch.nn.Module) -> dict[str, float]:
    def norm(parameters: Sequence[torch.nn.Parameter]) -> float:
        square_sum = sum(
            float(parameter.grad.detach().float().square().sum().cpu())
            for parameter in parameters
            if parameter.grad is not None
        )
        return sqrt(square_sum)

    all_parameters = list(model.parameters())
    beta_parameters = [
        layer.raw_linear_kernel
        for layer in model.layers
        if hasattr(layer, "raw_linear_kernel")
    ]
    gamma_parameters = [
        layer.raw_vector_kernel
        for layer in model.layers
        if hasattr(layer, "raw_vector_kernel")
    ]
    tensor_parameters = [
        layer.raw_tensor_kernel
        for layer in model.layers
        if getattr(layer, "raw_tensor_kernel", None) is not None
    ]
    summary = {"all": float(norm(all_parameters))}
    if beta_parameters:
        summary["beta_raw"] = float(norm(beta_parameters))
        summary["gamma_raw"] = float(norm(gamma_parameters))
    if tensor_parameters:
        summary["eta_raw"] = float(norm(tensor_parameters))
    return summary


def _finite_parameter_summary(
    value: torch.Tensor,
    *,
    name: str,
) -> dict[str, float]:
    value = value.detach().float()
    if value.numel() == 0 or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be a nonempty finite tensor")
    return {
        f"{name}.min": float(value.min().cpu()),
        f"{name}.mean": float(value.mean().cpu()),
        f"{name}.max": float(value.max().cpu()),
    }


def _gradient_parameter_diagnostics(model: torch.nn.Module) -> dict[str, int | str]:
    nonzero_count = 0
    nonfinite_count = 0
    with_gradient_count = 0
    tensor_count = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        tensor_count += 1
        gradient = parameter.grad.detach()
        finite = torch.isfinite(gradient)
        with_gradient_count += gradient.numel()
        nonzero_count += int(torch.count_nonzero((gradient != 0) & finite).item())
        nonfinite_count += int(torch.count_nonzero(~finite).item())
    return {
        "count_unit": "scalar_parameter_elements",
        "nonzero_definition": "finite_gradient_not_equal_zero",
        "nonzero_gradient_parameter_count": nonzero_count,
        "parameters_with_gradient_count": with_gradient_count,
        "nonfinite_gradient_parameter_count": nonfinite_count,
        "parameter_tensors_with_gradient_count": tensor_count,
    }


def _named_gradient_parameter_diagnostics(
    model: torch.nn.Module,
    name_fragment: str,
) -> dict[str, int | str]:
    named_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name_fragment in name
    ]
    trainable_parameters = [
        parameter for parameter in named_parameters if parameter.requires_grad
    ]
    with_gradient_count = 0
    nonzero_count = 0
    nonfinite_count = 0
    for parameter in trainable_parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        finite = torch.isfinite(gradient)
        with_gradient_count += gradient.numel()
        nonzero_count += int(torch.count_nonzero((gradient != 0) & finite).item())
        nonfinite_count += int(torch.count_nonzero(~finite).item())
    return {
        "count_unit": "scalar_parameter_elements",
        "parameter_count": sum(parameter.numel() for parameter in named_parameters),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in trainable_parameters
        ),
        "parameters_with_gradient_count": with_gradient_count,
        "nonzero_gradient_parameter_count": nonzero_count,
        "nonfinite_gradient_parameter_count": nonfinite_count,
    }


def _gradient_clipping_summary(
    monitor: dict[str, float | int],
    *,
    grad_clip: float | None,
) -> dict[str, float | int | str]:
    step_count = int(monitor.get("step_count", 0))
    if step_count <= 0:
        raise ValueError("gradient clipping summary requires at least one step")
    clipped_step_count = int(monitor.get("clipped_step_count", 0))
    return {
        "measurement_point": "before_clipping",
        "clip_threshold": "disabled" if grad_clip is None else float(grad_clip),
        "step_count": step_count,
        "clipped_step_count": clipped_step_count,
        "clip_fraction": clipped_step_count / step_count,
        "pre_clip_grad_norm_mean": float(monitor["pre_clip_grad_norm_sum"])
        / step_count,
        "pre_clip_grad_norm_max": float(monitor["pre_clip_grad_norm_max"]),
        "pre_clip_grad_norm_last": float(monitor["pre_clip_grad_norm_last"]),
    }


def _coordinate_gradient_parameter_diagnostics(
    model: torch.nn.Module,
) -> dict[str, int | str]:
    coordinate_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if "coordinate_updaters" in name
    ]
    with_gradient_count = 0
    nonzero_count = 0
    nonfinite_count = 0
    for parameter in coordinate_parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        finite = torch.isfinite(gradient)
        with_gradient_count += gradient.numel()
        nonzero_count += int(torch.count_nonzero((gradient != 0) & finite).item())
        nonfinite_count += int(torch.count_nonzero(~finite).item())
    return {
        "count_unit": "scalar_parameter_elements",
        "parameter_count": sum(
            parameter.numel() for parameter in coordinate_parameters
        ),
        "parameters_with_gradient_count": with_gradient_count,
        "nonzero_gradient_parameter_count": nonzero_count,
        "nonfinite_gradient_parameter_count": nonfinite_count,
    }


def _coordinate_update_diagnostics(
    model: torch.nn.Module,
    dataset: Sequence[GraphSample],
    validation_indices: Sequence[int],
    *,
    sample_count: int,
) -> dict[str, object]:
    if sample_count <= 0 or not validation_indices:
        raise ValueError("coordinate diagnostics require validation samples")
    selected_indices = list(validation_indices[:sample_count])
    graph_batch = collate_graphs([dataset[index] for index in selected_indices])
    parameter = next(model.parameters())
    graph_batch = graph_batch.to(device=parameter.device, dtype=parameter.dtype)
    captured_steps: list[torch.Tensor] = []
    handles = [
        updater.register_forward_hook(
            lambda _module, _inputs, output: captured_steps.append(output.detach())
        )
        for updater in getattr(model, "coordinate_updaters", ())
    ]
    training_states = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            output = model(
                graph_batch.node_feats,
                graph_batch.pos,
                batch=graph_batch.batch,
            )
    finally:
        for handle in handles:
            handle.remove()
        for module, training in training_states:
            module.training = training

    enabled = "node_positions" in output
    updated = output.get("node_positions", graph_batch.pos)
    displacement = updated - graph_batch.pos
    displacement_norm = torch.linalg.vector_norm(displacement.float(), dim=-1)
    graph_counts = torch.bincount(graph_batch.batch)
    input_centers = _batched_coordinate_means(
        graph_batch.pos,
        graph_batch.batch,
        graph_counts,
    )
    output_centers = _batched_coordinate_means(
        updated,
        graph_batch.batch,
        graph_counts,
    )
    centroid_drift = torch.linalg.vector_norm(
        (output_centers - input_centers).float(), dim=-1
    )
    layer_summaries = []
    for layer_index, step in enumerate(captured_steps):
        step_norm = torch.linalg.vector_norm(step.float(), dim=-1)
        step_centers = _batched_coordinate_means(
            step,
            graph_batch.batch,
            graph_counts,
        )
        layer_summaries.append(
            {
                "layer_index": layer_index,
                "step_rms_angstrom": float(step_norm.square().mean().sqrt().cpu()),
                "step_max_angstrom": float(step_norm.max().cpu()),
                "centroid_drift_max_angstrom": float(
                    torch.linalg.vector_norm(step_centers.float(), dim=-1).max().cpu()
                ),
            }
        )
    displacement_max = float(displacement_norm.max().cpu())
    layer_active = any(layer["step_max_angstrom"] > 0.0 for layer in layer_summaries)
    return {
        "schema_version": 1,
        "enabled": enabled,
        "active": enabled and layer_active,
        "sample_count": len(selected_indices),
        "node_count": int(graph_batch.pos.shape[0]),
        "selection": "validation_split_order_prefix",
        "registered_step_max_angstrom": 0.25,
        "displacement_rms_angstrom": float(
            displacement_norm.square().mean().sqrt().cpu()
        ),
        "displacement_max_angstrom": displacement_max,
        "centroid_drift_max_angstrom": float(centroid_drift.max().cpu()),
        "layers": layer_summaries,
        "excluded_from_elapsed_seconds": True,
        "excluded_from_peak_cuda_memory_bytes": True,
    }


def _batched_coordinate_means(
    value: torch.Tensor,
    batch: torch.Tensor,
    graph_counts: torch.Tensor,
) -> torch.Tensor:
    num_graphs = int(graph_counts.numel())
    summed = value.new_zeros((num_graphs, value.shape[-1])).index_add(0, batch, value)
    return summed / graph_counts.to(dtype=value.dtype).unsqueeze(-1)


def _model_state_hashes(model: torch.nn.Module) -> dict[str, str]:
    return _state_hashes(model)


def _paired_base_state_hashes(model: torch.nn.Module) -> dict[str, str]:
    return _state_hashes(
        model,
        excluded_prefixes=(
            "coordinate_updaters.",
            "local_pairwise_content.",
            "vector_out.",
            "tensor_out.",
        ),
    )


def _state_hashes(
    model: torch.nn.Module,
    *,
    excluded_prefixes: tuple[str, ...] = (),
) -> dict[str, str]:
    state_digest = hashlib.sha256()
    schema_digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if name.startswith(excluded_prefixes):
            continue
        metadata = json.dumps(
            [name, str(tensor.dtype), list(tensor.shape)],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
        schema_digest.update(len(metadata).to_bytes(8, "big"))
        schema_digest.update(metadata)
        state_digest.update(len(metadata).to_bytes(8, "big"))
        state_digest.update(metadata)
        contiguous = tensor.detach().cpu().contiguous().reshape(-1)
        raw = contiguous.view(torch.uint8).numpy().tobytes()
        state_digest.update(len(raw).to_bytes(8, "big"))
        state_digest.update(raw)
    return {
        "initial_state_sha256": state_digest.hexdigest(),
        "state_schema_sha256": schema_digest.hexdigest(),
    }


def _stratify_indices_by_node_count(
    dataset: Sequence[GraphSample],
    indices: Sequence[int],
) -> dict[str, list[int]]:
    strata: dict[str, list[int]] = {}
    for index in indices:
        node_count = int(dataset[index].node_feats.shape[0])
        if node_count <= 0:
            raise ValueError("graph samples must contain at least one node")
        lower = 1
        label = ""
        for upper in NODE_COUNT_STRATUM_UPPER_BOUNDS:
            if node_count <= upper:
                label = f"{lower}-{upper}"
                break
            lower = upper + 1
        if not label:
            label = f"{lower}+"
        strata.setdefault(label, []).append(index)
    return strata


def _node_count_strata_metrics(
    model: torch.nn.Module,
    dataset: Sequence[GraphSample],
    indices: Sequence[int],
    *,
    batch_size: int,
    normalizer: TargetNormalizer | None,
    amp_dtype: torch.dtype | None,
) -> dict[str, object]:
    metrics: dict[str, dict[str, float | int]] = {}
    for label, stratum_indices in _stratify_indices_by_node_count(
        dataset, indices
    ).items():
        evaluated = evaluate_regression(
            model,
            _iter_batches(dataset, stratum_indices, batch_size),
            target_normalizer=normalizer,
            amp_dtype=amp_dtype,
        )
        counts = [int(dataset[index].node_feats.shape[0]) for index in stratum_indices]
        metrics[label] = {
            "sample_count": len(stratum_indices),
            "observed_min_nodes": min(counts),
            "observed_max_nodes": max(counts),
            "mae": float(evaluated["mae"]),
            "rmse": float(evaluated["rmse"]),
        }
    return {
        "schema_version": 1,
        "split": "validation",
        "registered_upper_bounds": list(NODE_COUNT_STRATUM_UPPER_BOUNDS),
        "excluded_from_elapsed_seconds": True,
        "excluded_from_peak_cuda_memory_bytes": True,
        "metrics": metrics,
    }


def _disabled_bounded_diagnostics(args: argparse.Namespace) -> dict[str, object]:
    return {
        "schema_version": 1,
        "enabled": False,
        "status": "disabled",
        "effective_rank": {
            "enabled": bool(args.diagnostic_effective_rank),
            "max_matrix_size": int(args.diagnostic_max_nodes),
        },
        "local_validation_sample_count": int(args.diagnostic_sample_count),
        "instrumentation": {
            "connected": [],
            "unconnected": [
                "runtime_query_key_activations",
                "attention_weights",
                "memory_assignments_and_coupling",
            ],
        },
    }


def _bounded_model_diagnostics(
    model: torch.nn.Module,
    dataset: Sequence[GraphSample],
    validation_indices: Sequence[int],
    *,
    max_nodes: int,
    include_effective_rank: bool,
    local_sample_count: int = 32,
) -> dict[str, object]:
    """Instrument one bounded validation graph and one global attention head."""

    from equivariant_attention.moment import (
        _bounded_irrep,
        _bounded_kernel_scale,
        _memory_assignments_and_coupling,
        _positive_scalar_features,
        _stable_vector_norm,
        _tensor_product_features,
        _unit_ball,
    )

    if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes <= 0:
        raise ValueError("max_nodes must be a positive integer")
    selected_local_indices = _select_bounded_validation_indices(
        dataset,
        validation_indices,
        max_nodes=max_nodes,
        sample_count=local_sample_count,
    )
    common: dict[str, object] = {
        "schema_version": 1,
        "enabled": True,
        "effective_rank": {
            "enabled": include_effective_rank,
            "max_matrix_size": max_nodes,
        },
        "excluded_from_elapsed_seconds": True,
        "excluded_from_peak_cuda_memory_bytes": True,
    }
    if not selected_local_indices:
        return {
            **common,
            "status": "skipped_no_eligible_validation_graph",
            "instrumentation": {
                "connected": [],
                "unconnected": [
                    "runtime_query_key_activations",
                    "attention_weights",
                    "memory_assignments_and_coupling",
                ],
            },
        }

    dataset_index = selected_local_indices[0]
    sample_id = dataset[dataset_index].sample_id
    graph_batch = collate_graphs([dataset[dataset_index]])
    parameter = next(model.parameters())
    graph_batch = graph_batch.to(device=parameter.device, dtype=parameter.dtype)
    global_layers = [
        (index, layer)
        for index, layer in enumerate(model.layers)
        if layer.global_head_count > 0
    ]
    global_transport_mode = model.config.global_transport_mode
    if not global_layers or global_transport_mode == "none":
        started = time.perf_counter()
        local_attention = _bounded_local_attention_diagnostics(
            model,
            graph_batch,
            dataset_index=dataset_index,
        )
        if local_attention["status"] == "ok":
            local_attention["validation_distribution"] = (
                _bounded_local_validation_diagnostics(
                    model,
                    dataset,
                    selected_local_indices,
                )
            )
        diagnostic_seconds = time.perf_counter() - started
        global_status = (
            "skipped_no_global_attention_head"
            if not global_layers
            else "disabled_no_global_transport"
        )
        return {
            **common,
            "status": "ok",
            "elapsed_seconds": float(diagnostic_seconds),
            "batch": {
                "split": "validation",
                "dataset_index": int(dataset_index),
                "sample_id": sample_id,
                "node_count": int(graph_batch.node_feats.shape[0]),
            },
            "instrumentation": {
                "global_transport_mode": global_transport_mode,
                "connected": [
                    "selected_trained_local_attention_weights",
                    "all_local_layers_and_heads_on_bounded_validation_sample",
                ]
                if local_attention["status"] == "ok"
                else [],
                "unconnected": [
                    "global_kernel_components",
                    "global_attention_weights",
                    "memory_assignments_and_coupling",
                ],
            },
            "kernel_attention": {
                "status": global_status,
                "transport_mode": global_transport_mode,
            },
            "local_attention": local_attention,
            "memory": {
                "status": "not_applicable",
                "transport_connected": False,
            },
        }

    layer_index, layer = global_layers[0]
    captured: dict[str, torch.Tensor | int] = {}

    def capture_layer_input(
        _module: torch.nn.Module,
        inputs: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        captured["scalars"] = inputs[0].detach()
        captured["vectors"] = inputs[1].detach()
        captured["global_pos"] = inputs[2].detach()
        captured["batch"] = inputs[4].detach()
        captured["num_graphs"] = int(inputs[5])
        persistent_tensor = kwargs.get("persistent_tensor")
        if isinstance(persistent_tensor, torch.Tensor):
            captured["persistent_tensor"] = persistent_tensor.detach()

    training_states = [(module, module.training) for module in model.modules()]
    handle = layer.register_forward_pre_hook(
        capture_layer_input,
        with_kwargs=True,
    )
    if parameter.device.type == "cuda":
        torch.cuda.synchronize(parameter.device)
    started = time.perf_counter()
    try:
        model.eval()
        with torch.no_grad():
            model(graph_batch.node_feats, graph_batch.pos, batch=graph_batch.batch)
    finally:
        handle.remove()
        for module, training in training_states:
            module.training = training
    if not captured:
        raise RuntimeError("bounded diagnostic layer hook did not execute")

    with torch.no_grad():
        scalars = captured["scalars"]
        vectors = captured["vectors"]
        global_pos = captured["global_pos"]
        batch = captured["batch"]
        num_graphs = int(captured["num_graphs"])
        if num_graphs != 1:
            raise RuntimeError("bounded diagnostics require one collated graph")
        node_count = scalars.shape[0]
        normalized_scalars = layer.norm(scalars)
        bounded_vectors = _bounded_irrep(vectors, layer.eps)
        query_content = _positive_scalar_features(
            layer.query_scalar(normalized_scalars).reshape(
                node_count, layer.num_heads, layer.head_dim
            ),
            layer.eps,
            mode=layer.scalar_content_mode,
        )
        key_content = _positive_scalar_features(
            layer.key_scalar(normalized_scalars).reshape(
                node_count, layer.num_heads, layer.head_dim
            ),
            layer.eps,
            mode=layer.scalar_content_mode,
        )
        query_scalar = query_content
        key_scalar = key_content
        if layer.tensor_kernel_query is not None:
            persistent_tensor = captured.get("persistent_tensor")
            if not isinstance(persistent_tensor, torch.Tensor):
                raise RuntimeError(
                    "tensor-kernel diagnostics require persistent tensor input"
                )
            tensor_scale = _bounded_kernel_scale(
                layer.raw_tensor_kernel,
                layer.tensor_kernel_max,
            )
            tensor_query, tensor_key = _tensor_product_features(
                layer.tensor_kernel_query(persistent_tensor),
                layer.tensor_kernel_key(persistent_tensor),
                tensor_scale,
                eps=layer.eps,
            )
            query_scalar = torch.cat([query_scalar, tensor_query], dim=-1)
            key_scalar = torch.cat([key_scalar, tensor_key], dim=-1)
        query_vector = _unit_ball(
            layer.query_vector(bounded_vectors)
            * torch.tanh(layer.query_vector_gate(normalized_scalars)).unsqueeze(-1),
            layer.eps,
        )
        key_vector = _unit_ball(
            layer.key_vector(bounded_vectors)
            * torch.tanh(layer.key_vector_gate(normalized_scalars)).unsqueeze(-1),
            layer.eps,
        )

        global_heads = slice(layer.local_head_count, layer.num_heads)
        head_index = layer.local_head_count
        beta = layer.linear_kernel_max * torch.sigmoid(
            layer.raw_linear_kernel[head_index]
        )
        gamma = layer.vector_kernel_max * torch.sigmoid(
            layer.raw_vector_kernel[head_index]
        )
        pair_gate = None
        memory: dict[str, object]
        if layer.use_memory_interaction and layer.global_memory_count > 1:
            router_latent = torch.tanh(
                layer.memory_router_out(
                    F.silu(layer.memory_router_in(key_content[:, global_heads]))
                )
            )
            router_latent = router_latent / _stable_vector_norm(
                router_latent
            ).clamp_min(torch.finfo(router_latent.dtype).tiny)
            assignment, coupling, centers = _memory_assignments_and_coupling(
                key_content[:, global_heads],
                global_pos,
                batch,
                num_graphs=num_graphs,
                memory_count=layer.global_memory_count,
                temperature=layer.memory_assignment_temperature,
                assignment_scale=layer.memory_assignment_scale,
                interaction_cutoff=layer.memory_interaction_cutoff,
                interact=True,
                router_latent=router_latent,
            )
            global_head_offset = head_index - layer.local_head_count
            head_assignment = assignment[:, global_head_offset]
            head_coupling = coupling[0, global_head_offset]
            pair_gate = torch.einsum(
                "im,mn,jn->ij",
                head_assignment,
                head_coupling,
                head_assignment,
            )
            memory = {
                "status": "active",
                "transport_connected": True,
                "assignment_scope": "selected_graph_and_head",
                "assignment": memory_assignment_summary(
                    assignment[:, global_head_offset : global_head_offset + 1]
                ),
                "coupling": kernel_component_quantiles(
                    {"coupling": head_coupling}, quantiles=(0.0, 0.5, 1.0)
                ),
                "pair_gate": pair_gate_summary(pair_gate),
                "all_head_activation": memory_pair_gate_summary(
                    assignment,
                    coupling[0],
                ),
                "centers": memory_center_summary(
                    centers[0],
                    interaction_cutoff=layer.memory_interaction_cutoff,
                ),
                "centers_finite": bool(torch.isfinite(centers).all().item()),
            }
            assignment_source = "shared_invariant_router_exact_recompute"
        elif layer.use_memory_interaction:
            assignment = key_content.new_ones((node_count, layer.global_head_count, 1))
            analytic_pair_gate = key_content.new_ones((node_count, node_count))
            memory = {
                "status": "exact_single_memory_bypass",
                "transport_connected": False,
                "assignment": memory_assignment_summary(assignment),
                "pair_gate": pair_gate_summary(analytic_pair_gate),
            }
            assignment_source = "analytic_single_memory_identity"
        else:
            memory = {
                "status": "disabled",
                "transport_connected": False,
                "memory_count": int(layer.global_memory_count),
            }
            assignment_source = "not_used"

        kernel_attention = dense_kernel_attention_summary(
            query_scalar[:, head_index],
            key_scalar[:, head_index],
            query_vector[:, head_index],
            key_vector[:, head_index],
            beta=beta,
            gamma=gamma,
            kernel_floor=layer.kernel_floor,
            kernel_floor_mode=layer.kernel_floor_mode,
            graph_size=node_count,
            alignment_linear_term=layer.use_alignment_linear_term,
            balanced=layer.use_key_balancing,
            pair_gate=pair_gate,
            include_effective_rank=include_effective_rank,
            max_nodes=max_nodes,
        )
        if global_transport_mode == "uniform":
            uniform_weights = torch.ones(
                node_count,
                node_count,
                dtype=torch.float64,
                device=scalars.device,
            )
            kernel_attention = {
                "status": "exact_uniform_transport",
                "transport_mode": "uniform",
                **{
                    f"attention.{name}": value
                    for name, value in attention_weight_summary(
                        uniform_weights,
                        include_effective_rank=include_effective_rank,
                        effective_rank_max_size=max_nodes,
                    ).items()
                },
            }
            memory = {
                "status": "not_applicable_uniform_transport",
                "transport_connected": False,
            }
            assignment_source = "not_applicable_uniform_transport"

    if parameter.device.type == "cuda":
        torch.cuda.synchronize(parameter.device)
    local_attention = _bounded_local_attention_diagnostics(
        model,
        graph_batch,
        dataset_index=dataset_index,
    )
    if local_attention["status"] == "ok":
        local_attention["validation_distribution"] = (
            _bounded_local_validation_diagnostics(
                model,
                dataset,
                selected_local_indices,
            )
        )
    diagnostic_seconds = time.perf_counter() - started
    return {
        **common,
        "status": "ok",
        "elapsed_seconds": float(diagnostic_seconds),
        "batch": {
            "split": "validation",
            "dataset_index": int(dataset_index),
            "sample_id": sample_id,
            "node_count": int(node_count),
        },
        "instrumentation": {
            "global_transport_mode": global_transport_mode,
            "activation_source": "exact_recompute_from_captured_layer_input",
            "assignment_source": assignment_source,
            "layer_index": int(layer_index),
            "head_index": int(head_index),
            "connected": (
                [
                    "selected_runtime_layer_input",
                    "exact_uniform_global_attention_matrix",
                ]
                if global_transport_mode == "uniform"
                else [
                    "selected_runtime_layer_input",
                    "selected_trained_query_key_projections",
                    "selected_trained_beta_gamma",
                    "configured_floor_alignment_and_balancing",
                    "selected_global_attention_matrix",
                ]
            )
            + (
                [
                    "selected_trained_local_attention_weights",
                    "all_local_layers_and_heads_on_bounded_validation_sample",
                ]
                if local_attention["status"] == "ok"
                else []
            ),
            "unconnected": [
                "value_transport_and_residual_updates",
                "validation_graphs_outside_bounded_sample",
                "amp_execution_path",
                *(
                    []
                    if local_attention["status"] == "ok"
                    else ["local_attention_weights"]
                ),
            ],
        },
        "kernel_attention": kernel_attention,
        "local_attention": local_attention,
        "memory": memory,
    }


def _bounded_local_attention_diagnostics(
    model: torch.nn.Module,
    graph_batch: object,
    *,
    dataset_index: int,
) -> dict[str, object]:
    """Recompute every trained local layer on one already bounded graph."""
    local_layers = [
        (index, layer)
        for index, layer in enumerate(model.layers)
        if layer.local_head_count > 0
    ]
    if not local_layers:
        return {
            "status": "skipped_no_local_attention_head",
            "transport_connected": False,
        }
    parameter = next(model.parameters())
    captured: dict[int, dict[str, object]] = {}

    def capture_layer_input(
        layer_index: int,
        inputs: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        captured[layer_index] = {
            "scalars": inputs[0].detach(),
            "vectors": inputs[1].detach(),
            "raw_pos": inputs[3].detach(),
            "batch": inputs[4].detach(),
            "num_graphs": int(inputs[5]),
            "local_geometry": tuple(value.detach() for value in inputs[7]),
        }
        persistent_tensor = kwargs.get("persistent_tensor")
        if isinstance(persistent_tensor, torch.Tensor):
            captured[layer_index]["persistent_tensor"] = persistent_tensor.detach()

    training_states = [(module, module.training) for module in model.modules()]
    handles = [
        layer.register_forward_pre_hook(
            lambda _module, inputs, kwargs, index=layer_index: capture_layer_input(
                index, inputs, kwargs
            ),
            with_kwargs=True,
        )
        for layer_index, layer in local_layers
    ]
    try:
        model.eval()
        with torch.no_grad():
            model(graph_batch.node_feats, graph_batch.pos, batch=graph_batch.batch)
    finally:
        for handle in handles:
            handle.remove()
        for module, training in training_states:
            module.training = training
    if len(captured) != len(local_layers):
        raise RuntimeError("bounded local diagnostic layer hook did not execute")

    with torch.no_grad():
        layer_results = [
            _local_layer_attention_diagnostics(
                layer,
                captured[layer_index],
                layer_index=layer_index,
            )
            for layer_index, layer in local_layers
        ]
    first = layer_results[0]
    return {
        "status": "ok",
        "transport_connected": True,
        "activation_source": "exact_recompute_from_captured_layer_input",
        "dataset_index": int(dataset_index),
        "layer_index": first["layer_index"],
        "head_indices": first["head_indices"],
        "summary": first["summary"],
        "layers": layer_results,
        "model_device": str(parameter.device),
    }


def _local_layer_attention_diagnostics(
    layer: torch.nn.Module,
    captured: dict[str, object],
    *,
    layer_index: int,
) -> dict[str, object]:
    from equivariant_attention.moment import (
        _bounded_irrep,
        _bounded_kernel_scale,
        _local_attention_weights,
        _positive_scalar_features,
        _tensor_product_features,
        _unit_ball,
    )

    scalars = captured["scalars"]
    vectors = captured["vectors"]
    raw_pos = captured["raw_pos"]
    batch = captured["batch"]
    num_graphs = int(captured["num_graphs"])
    local_geometry = captured["local_geometry"]
    node_count = scalars.shape[0]
    normalized_scalars = layer.norm(scalars)
    bounded_vectors = _bounded_irrep(vectors, layer.eps)
    query_scalar = _positive_scalar_features(
        layer.query_scalar(normalized_scalars).reshape(
            node_count, layer.num_heads, layer.head_dim
        ),
        layer.eps,
        mode=layer.scalar_content_mode,
    )
    key_scalar = _positive_scalar_features(
        layer.key_scalar(normalized_scalars).reshape(
            node_count, layer.num_heads, layer.head_dim
        ),
        layer.eps,
        mode=layer.scalar_content_mode,
    )
    if layer.tensor_kernel_query is not None:
        persistent_tensor = captured.get("persistent_tensor")
        if not isinstance(persistent_tensor, torch.Tensor):
            raise RuntimeError(
                "tensor-kernel diagnostics require persistent tensor input"
            )
        tensor_query, tensor_key = _tensor_product_features(
            layer.tensor_kernel_query(persistent_tensor),
            layer.tensor_kernel_key(persistent_tensor),
            _bounded_kernel_scale(
                layer.raw_tensor_kernel,
                layer.tensor_kernel_max,
            ),
            eps=layer.eps,
        )
        query_scalar = torch.cat([query_scalar, tensor_query], dim=-1)
        key_scalar = torch.cat([key_scalar, tensor_key], dim=-1)
    query_vector = _unit_ball(
        layer.query_vector(bounded_vectors)
        * torch.tanh(layer.query_vector_gate(normalized_scalars)).unsqueeze(-1),
        layer.eps,
    )
    key_vector = _unit_ball(
        layer.key_vector(bounded_vectors)
        * torch.tanh(layer.key_vector_gate(normalized_scalars)).unsqueeze(-1),
        layer.eps,
    )
    alignment_scale = _bounded_kernel_scale(
        layer.raw_linear_kernel, layer.linear_kernel_max
    )
    alignment_dot_scale = (
        alignment_scale
        if layer.use_alignment_linear_term
        else torch.zeros_like(alignment_scale)
    )
    kernel_scale = _bounded_kernel_scale(
        layer.raw_vector_kernel, layer.vector_kernel_max
    )
    local = slice(0, layer.local_head_count)
    receiver, sender, weights, _, squared_distance = _local_attention_weights(
        query_scalar[:, local],
        key_scalar[:, local],
        query_vector[:, local],
        key_vector[:, local],
        kernel_scale[local],
        raw_pos,
        batch,
        num_graphs=num_graphs,
        balanced=layer.use_key_balancing,
        alignment_scale=alignment_scale[local],
        alignment_dot_scale=alignment_dot_scale[local],
        kernel_floor=layer.kernel_floor,
        cutoff=layer.local_cutoff,
        num_rbf=layer.num_rbf,
        radial_weight=layer.local_radial_weight[local],
        radial_bias=layer.local_radial_bias[local],
        local_geometry=local_geometry,
    )
    summary = local_attention_summary(
        receiver,
        sender,
        weights,
        squared_distance,
        num_nodes=node_count,
    )
    heads = [
        {
            "head_index": head_index,
            "summary": local_attention_summary(
                receiver,
                sender,
                weights[:, head_index : head_index + 1],
                squared_distance,
                num_nodes=node_count,
            ),
        }
        for head_index in range(layer.local_head_count)
    ]
    return {
        "layer_index": int(layer_index),
        "head_indices": list(range(layer.local_head_count)),
        "summary": summary,
        "heads": heads,
    }


def _select_bounded_validation_indices(
    dataset: Sequence[GraphSample],
    validation_indices: Sequence[int],
    *,
    max_nodes: int,
    sample_count: int,
) -> list[int]:
    for name, value in (("max_nodes", max_nodes), ("sample_count", sample_count)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    eligible = sorted(
        (
            index
            for index in validation_indices
            if int(dataset[index].node_feats.shape[0]) <= max_nodes
        ),
        key=lambda index: (int(dataset[index].node_feats.shape[0]), index),
    )
    if len(eligible) <= sample_count:
        return eligible
    if sample_count == 1:
        return [eligible[0]]
    positions = [
        position * (len(eligible) - 1) // (sample_count - 1)
        for position in range(sample_count)
    ]
    return [eligible[position] for position in positions]


def _bounded_local_validation_diagnostics(
    model: torch.nn.Module,
    dataset: Sequence[GraphSample],
    dataset_indices: Sequence[int],
) -> dict[str, object]:
    parameter = next(model.parameters())
    observations: dict[int, list[dict[str, object]]] = {}
    node_counts: list[int] = []
    for dataset_index in dataset_indices:
        graph_batch = collate_graphs([dataset[dataset_index]]).to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        node_counts.append(int(graph_batch.node_feats.shape[0]))
        result = _bounded_local_attention_diagnostics(
            model,
            graph_batch,
            dataset_index=dataset_index,
        )
        if result["status"] != "ok":
            raise RuntimeError("selected model unexpectedly has no local diagnostics")
        for layer in result["layers"]:
            observations.setdefault(layer["layer_index"], []).append(layer)

    layers = []
    for layer_index in sorted(observations):
        layer_observations = observations[layer_index]
        head_indices = layer_observations[0]["head_indices"]
        layers.append(
            {
                "layer_index": int(layer_index),
                "sample_count": len(layer_observations),
                "aggregate": _aggregate_local_summaries(
                    [observation["summary"] for observation in layer_observations]
                ),
                "heads": [
                    {
                        "head_index": int(head_index),
                        "aggregate": _aggregate_local_summaries(
                            [
                                observation["heads"][head_index]["summary"]
                                for observation in layer_observations
                            ]
                        ),
                    }
                    for head_index in head_indices
                ],
            }
        )
    return {
        "scope": "deterministic_bounded_validation_sample",
        "selection": {
            "dataset_indices": [int(index) for index in dataset_indices],
            "sample_count": len(dataset_indices),
            "node_count.min": min(node_counts),
            "node_count.max": max(node_counts),
        },
        "layers": layers,
    }


def _aggregate_local_summaries(
    summaries: Sequence[dict[str, object]],
) -> dict[str, float]:
    if not summaries:
        raise ValueError("at least one local summary is required")
    numeric_keys = sorted(
        key
        for key in summaries[0]
        if all(
            isinstance(summary.get(key), (int, float))
            and not isinstance(summary.get(key), bool)
            for summary in summaries
        )
    )
    aggregate: dict[str, float] = {}
    for key in numeric_keys:
        values = [float(summary[key]) for summary in summaries]
        aggregate[f"{key}.sample_min"] = min(values)
        aggregate[f"{key}.sample_mean"] = sum(values) / len(values)
        aggregate[f"{key}.sample_max"] = max(values)
    return aggregate


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a nonnegative integer")
    return parsed


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
            root / "scripts" / "run_architecture_v2_qm9.py",
            root / "scripts" / "run_registered_coordinate_study.py",
            root / "scripts" / "run_registered_egnn_parity_iteration.py",
            root
            / "artifacts"
            / "architecture-v2-positive-tensor-20260723"
            / "scope.md",
            root
            / "artifacts"
            / "architecture-v2-positive-tensor-20260723"
            / "initialization-preserving-followup.md",
            root / "artifacts" / "dynamic-coordinate-egnn-20260719" / "scope.md",
            root / "artifacts" / "egnn-parity-20260720" / "scope.md",
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


def _kernel_formula(
    args: argparse.Namespace,
    inverse_positive_baseline: bool,
) -> str:
    content = "a_dot_b"
    if args.tensor_product_kernel:
        content += " + eta*(1+u)"
    if inverse_positive_baseline:
        return f"{content} + (c + beta*(1 + delta*t))/N_g + gamma*t^2"
    return f"{content} + c + beta*(1 + delta*t) + gamma*t^2"


def _run_config(
    args: argparse.Namespace,
    *,
    split_seed: int,
    model_seed: int,
) -> dict[str, object]:
    if args.benchmark_model in _EGNN_BENCHMARK_MODELS:
        coordinate_updates = args.benchmark_model == "internal_dynamic_egnn_baseline"
        return {
            "dataset": args.dataset,
            "data_root": str(args.data_root),
            "qm9_target_index": args.qm9_target_index,
            "num_samples": args.num_samples,
            "train_size": args.train_size,
            "val_size": args.val_size,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "model": args.benchmark_model,
            "comparison_role": "internal_same_harness_baseline",
            "official_reproduction": False,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_heads": "not_applicable",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "split_seed": split_seed,
            "model_seed": model_seed,
            "dataset_seed": args.seed,
            "determinism": args.determinism,
            "device": args.device,
            "amp_dtype": args.amp_dtype,
            "target_normalized": not args.no_target_normalize,
            "test_evaluated": args.evaluate_test,
            "coordinate_updates": coordinate_updates,
            "coordinate_update_count": (
                args.num_layers - 1 if coordinate_updates else 0
            ),
            "coordinate_update_max_step_angstrom": (
                0.25 if coordinate_updates else "not_applicable"
            ),
            "coordinate_update_formula": (
                "centered_bounded_relative_vectors_times_invariant_edge_scalars"
                if coordinate_updates
                else "not_applicable"
            ),
            "geometry_recomputed_per_layer": coordinate_updates,
            "edge_topology": (
                "precomputed_radius_candidates_without_self"
                if args.precompute_local_edges
                else "same_graph_directed_complete_without_self"
            ),
            "precompute_local_edges": args.precompute_local_edges,
            "distance_feature": "raw_squared_distance",
            "edge_gate": "learned_sigmoid",
            "aggregation": "sum",
            "node_update": "residual_two_layer_silu_mlp",
            "readout": "layernorm_node_linear_graph_mean",
            "output_initialization": "zero",
            "global_transport_mode": "not_applicable",
            "global_transport_executed": False,
            "global_geometry_executed": False,
            "global_attention_formula": "not_applicable",
            "routing": "not_applicable",
            "local_head_counts": [],
            "bounded_diagnostics": args.bounded_diagnostics,
            "diagnostic_max_nodes": args.diagnostic_max_nodes,
            "diagnostic_sample_count": args.diagnostic_sample_count,
            "diagnostic_effective_rank": args.diagnostic_effective_rank,
        }
    inverse_positive_baseline = args.kernel_floor_mode == "inverse_graph_size"
    local_head_counts = _routing_head_counts(
        args.routing,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )
    has_global_heads = any(count < args.num_heads for count in local_head_counts)
    global_formula = (
        {
            "learned": "factorized_learned_kernel",
            "uniform": "exact_graph_mean",
            "none": "no_global_transport",
        }[args.global_transport_mode]
        if has_global_heads
        else "not_applicable_no_global_heads"
    )
    global_key_balancing = (
        ("disabled" if args.no_key_balancing else "one_cycle")
        if has_global_heads and args.global_transport_mode == "learned"
        else "not_applicable"
    )
    global_transport_executed = (
        has_global_heads and args.global_transport_mode != "none"
    )
    if args.memory_interaction and args.memory_count > 1:
        hemm_status = "stage0_blocked_experimental"
    elif args.memory_interaction:
        hemm_status = "exact_single_memory_bypass"
    else:
        hemm_status = "disabled"
    return {
        "dataset": args.dataset,
        "data_root": str(args.data_root),
        "qm9_target_index": args.qm9_target_index,
        "num_samples": args.num_samples,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "model": args.benchmark_model,
        "comparison_role": "equivariant_attention_candidate",
        "official_reproduction": False,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "split_seed": split_seed,
        "model_seed": model_seed,
        "dataset_seed": args.seed,
        "determinism": args.determinism,
        "device": args.device,
        "amp_dtype": args.amp_dtype,
        "target_normalized": not args.no_target_normalize,
        "test_evaluated": args.evaluate_test,
        "coordinate_updates": args.coordinate_updates,
        "coordinate_update_count": (
            args.num_layers - 1 if args.coordinate_updates else 0
        ),
        "coordinate_update_max_step_angstrom": (
            0.25 if args.coordinate_updates else "not_applicable"
        ),
        "coordinate_update_formula": (
            "centered_bounded_invariant_gated_polar_vector_state"
            if args.coordinate_updates
            else "not_applicable"
        ),
        "geometry_recomputed_per_layer": args.coordinate_updates,
        "attention": "factorized_moment",
        "kernel_version": (
            4 if args.scalar_content_mode != "unit" or args.tensor_product_kernel else 3
        ),
        "balance_cycles": 0 if args.no_key_balancing else 1,
        "key_balancing": not args.no_key_balancing,
        "alignment_linear_term": not args.no_alignment_linear_term,
        "alignment_constant_retained": True,
        "linear_kernel_init": args.linear_kernel_init,
        "hidden_tensor_dim": args.hidden_tensor_dim,
        "scalar_content_mode": args.scalar_content_mode,
        "scalar_content_norm_bound": (
            2.0 if args.scalar_content_mode == "bounded" else 1.0
        ),
        "tensor_product_kernel": args.tensor_product_kernel,
        "tensor_product_kernel_formula": (
            "eta*(1+frobenius_dot(unit_ball_2e_query,unit_ball_2e_key))"
            if args.tensor_product_kernel
            else "not_applicable"
        ),
        "tensor_kernel_init": args.tensor_kernel_init,
        "tensor_kernel_max": args.tensor_kernel_max,
        "kernel_floor_mode": args.kernel_floor_mode,
        "kernel_scaling_formula_version": "positive_baseline_v1",
        "graph_size_scaled_positive_baseline": inverse_positive_baseline,
        "kernel_formula": _kernel_formula(args, inverse_positive_baseline),
        "graph_size_scaled_terms": (
            ["kernel_floor", "alignment_constant", "alignment_linear"]
            if inverse_positive_baseline
            else []
        ),
        "graph_size_unscaled_terms": [
            "content",
            "alignment_quadratic",
            *(["shifted_tensor_product"] if args.tensor_product_kernel else []),
        ],
        "routing": args.routing,
        "global_transport_mode": args.global_transport_mode,
        "global_transport_executed": global_transport_executed,
        "global_geometry_executed": global_transport_executed,
        "global_attention_formula": global_formula,
        "global_key_balancing": global_key_balancing,
        "local_head_counts": list(local_head_counts),
        "local_cutoff": args.local_cutoff,
        "num_rbf": args.num_rbf,
        "learn_local_radial_gate": args.learn_local_radial_gate,
        "edge_conditioned_local_transport": (args.edge_conditioned_local_transport),
        "edge_conditioned_local_aggregation": (
            "cutoff_sum_over_sqrt_receiver_degree"
            if args.edge_conditioned_local_sqrt_degree
            else (
                "cutoff_sum"
                if args.edge_conditioned_local_transport
                else "not_applicable"
            )
        ),
        "precompute_local_edges": args.precompute_local_edges,
        "local_candidate_builder": (
            "load_time_dense_radius_scan"
            if args.precompute_local_edges
            else "forward_same_graph_cartesian_fallback"
        ),
        "pairwise_local_content": args.pairwise_local_content,
        "pairwise_residual_scale_init": args.pairwise_residual_scale_init,
        "pairwise_local_formula": (
            "shared_receiver_sender_rbf_mlp_plus_degree_and_cutoff_mass"
            if args.pairwise_local_content
            else "not_applicable"
        ),
        "pairwise_local_aggregation": (
            "cutoff_sum_over_sqrt_degree"
            if args.pairwise_local_content
            else "not_applicable"
        ),
        "memory_count": args.memory_count,
        "memory_interaction": args.memory_interaction,
        "hemm_admission_status": hemm_status,
        "memory_assignment_temperature": args.memory_assignment_temperature,
        "memory_assignment_scale": args.memory_assignment_scale,
        "memory_interaction_cutoff": args.memory_interaction_cutoff,
        "radial_trace": args.radial_trace,
        "bounded_diagnostics": args.bounded_diagnostics,
        "diagnostic_max_nodes": args.diagnostic_max_nodes,
        "diagnostic_sample_count": args.diagnostic_sample_count,
        "diagnostic_effective_rank": args.diagnostic_effective_rank,
        "ffn_hidden_ratio": 2.0,
    }


def _qm9_data_identity(
    root: Path,
    expected: dict[str, str] = QM9_DATA_HASHES,
) -> dict[str, str]:
    actual = {relative: _hash_file(root / relative) for relative in expected}
    mismatches = [
        relative for relative, digest in actual.items() if digest != expected[relative]
    ]
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
):
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        yield collate_graphs([dataset[i] for i in chunk])


if __name__ == "__main__":
    main()
