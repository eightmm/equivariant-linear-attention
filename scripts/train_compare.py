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

from equivariant_attention.benchmarking import (
    GraphSample,
    SyntheticMoleculeDataset,
    collate_graphs,
    load_qm9_samples,
    split_dataset,
)
from equivariant_attention.diagnostics import (
    dense_kernel_attention_summary,
    kernel_component_quantiles,
    kernel_parameter_summary,
    memory_assignment_summary,
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


def main() -> None:
    args = parse_args()
    split_seed = args.seed if args.split_seed is None else args.split_seed
    model_seed = args.seed if args.model_seed is None else args.model_seed
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
    torch.manual_seed(model_seed)
    local_head_counts = _routing_head_counts(
        args.routing,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )
    model = build_regression_model(
        node_dim=node_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        linear_kernel_init=args.linear_kernel_init,
        use_alignment_linear_term=not args.no_alignment_linear_term,
        use_key_balancing=not args.no_key_balancing,
        kernel_floor_mode=args.kernel_floor_mode,
        local_head_counts=local_head_counts,
        local_cutoff=args.local_cutoff,
        num_rbf=args.num_rbf,
        learn_local_radial_gate=args.learn_local_radial_gate,
        global_memory_count=args.memory_count,
        use_memory_interaction=args.memory_interaction,
        memory_assignment_temperature=args.memory_assignment_temperature,
        memory_assignment_scale=args.memory_assignment_scale,
        memory_interaction_cutoff=args.memory_interaction_cutoff,
        use_radial_trace=args.radial_trace,
    ).to(device=device)
    initial_state_hashes = _model_state_hashes(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    normalizer = (
        None
        if args.no_target_normalize
        else fit_target_normalizer(dataset[i] for i in train_idx)
    )
    final_loss = 0.0
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
        )

    val_batches = list(_iter_batches(dataset, val_idx, args.batch_size))
    val_metrics = evaluate_regression(
        model, val_batches, target_normalizer=normalizer, amp_dtype=amp_dtype
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - run_started
    metrics = {
        "dataset": args.dataset,
        "model": "factorized_moment",
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
        "test_evaluated": args.evaluate_test,
        "amp_dtype": args.amp_dtype,
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
        **initial_state_hashes,
        "initialization_hash_version": "canonical_state_dict_v1",
        "source_sha256": _source_hash(),
        "run_config": _run_config(
            args,
            split_seed=split_seed,
            model_seed=model_seed,
        ),
    }
    if args.dataset == "qm9":
        metrics["target"] = _qm9_target_metadata(args.qm9_target_index)
        metrics["data_identity"] = data_identity
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
            float(layer.scalar_residual_scale.detach().cpu()) for layer in model.layers
        ],
        "vector": [
            float(layer.vector_residual_scale.detach().cpu()) for layer in model.layers
        ],
    }
    beta = torch.cat(
        [
            layer.linear_kernel_max * torch.sigmoid(layer.raw_linear_kernel.detach())
            for layer in model.layers
        ]
    )
    gamma = torch.cat(
        [
            layer.vector_kernel_max * torch.sigmoid(layer.raw_vector_kernel.detach())
            for layer in model.layers
        ]
    )
    metrics["kernel_parameters"] = kernel_parameter_summary(beta, gamma)
    metrics["gradient_norms"] = _gradient_norms(model)
    gradient_parameters = _gradient_parameter_diagnostics(model)
    gradient_parameters["measurement_point"] = (
        "after_final_training_backward_clip_and_optimizer_step"
    )
    metrics["gradient_parameters"] = gradient_parameters
    metrics["nonzero_gradient_parameter_count"] = gradient_parameters[
        "nonzero_gradient_parameter_count"
    ]
    metrics["node_count_strata"] = _node_count_strata_metrics(
        model,
        dataset,
        val_idx,
        batch_size=args.batch_size,
        normalizer=normalizer,
        amp_dtype=amp_dtype,
    )
    metrics["bounded_diagnostics"] = (
        _bounded_model_diagnostics(
            model,
            dataset,
            val_idx,
            max_nodes=args.diagnostic_max_nodes,
            include_effective_rank=args.diagnostic_effective_rank,
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
        description="Train the factorized-moment equivariant attention model."
    )
    parser.add_argument("--dataset", choices=["synthetic", "qm9"], default="synthetic")
    parser.add_argument("--data-root", type=Path, default=Path("data/qm9"))
    parser.add_argument("--qm9-target-index", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--val-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--linear-kernel-init", type=float, default=0.05)
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
    parser.add_argument("--routing", choices=["ggg", "lgl", "lll"], default="ggg")
    parser.add_argument("--local-cutoff", type=float, default=2.5)
    parser.add_argument("--num-rbf", type=int, default=16)
    parser.add_argument("--learn-local-radial-gate", action="store_true")
    parser.add_argument("--memory-count", type=int, choices=[1, 4, 8], default=1)
    parser.add_argument("--memory-interaction", action="store_true")
    parser.add_argument("--memory-assignment-temperature", type=float, default=1.0)
    parser.add_argument("--memory-assignment-scale", type=float, default=2.5)
    parser.add_argument("--memory-interaction-cutoff", type=float, default=2.5)
    parser.add_argument("--radial-trace", action="store_true")
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
    parser.add_argument("--bounded-diagnostics", action="store_true")
    parser.add_argument("--diagnostic-max-nodes", type=_positive_int, default=128)
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
    return parser.parse_args(argv)


def load_dataset(args: argparse.Namespace) -> Sequence[GraphSample]:
    if args.dataset == "synthetic":
        return SyntheticMoleculeDataset(
            num_samples=args.num_samples, node_dim=8, seed=args.seed
        )
    return load_qm9_samples(
        args.data_root, target_index=args.qm9_target_index, limit=args.num_samples
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


def _routing_head_counts(
    routing: str,
    *,
    num_layers: int,
    num_heads: int,
) -> tuple[int, ...]:
    if routing == "ggg":
        return (0,) * num_layers
    if num_layers != 3:
        raise ValueError("lgl and lll routing presets require exactly three layers")
    if routing == "lgl":
        return (num_heads, 0, num_heads)
    if routing == "lll":
        return (num_heads, num_heads, num_heads)
    raise ValueError(f"unknown routing preset: {routing}")


def _gradient_norms(model: torch.nn.Module) -> dict[str, float]:
    def norm(parameters: Sequence[torch.nn.Parameter]) -> float:
        square_sum = sum(
            float(parameter.grad.detach().float().square().sum().cpu())
            for parameter in parameters
            if parameter.grad is not None
        )
        return sqrt(square_sum)

    all_parameters = list(model.parameters())
    beta_parameters = [layer.raw_linear_kernel for layer in model.layers]
    gamma_parameters = [layer.raw_vector_kernel for layer in model.layers]
    return {
        "all": float(norm(all_parameters)),
        "beta_raw": float(norm(beta_parameters)),
        "gamma_raw": float(norm(gamma_parameters)),
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


def _model_state_hashes(model: torch.nn.Module) -> dict[str, str]:
    state_digest = hashlib.sha256()
    schema_digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
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
) -> dict[str, object]:
    """Instrument one bounded validation graph and one global attention head."""

    from equivariant_attention.moment import (
        _bounded_irrep,
        _memory_assignments_and_coupling,
        _normalize_positive_features,
        _unit_ball,
    )

    if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes <= 0:
        raise ValueError("max_nodes must be a positive integer")
    eligible = [
        index
        for index in validation_indices
        if int(dataset[index].node_feats.shape[0]) <= max_nodes
    ]
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
    if not eligible:
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

    global_layers = [
        (index, layer)
        for index, layer in enumerate(model.layers)
        if layer.global_head_count > 0
    ]
    if not global_layers:
        return {
            **common,
            "status": "skipped_no_global_attention_head",
            "instrumentation": {
                "connected": [],
                "unconnected": [
                    "global_kernel_components",
                    "global_attention_weights",
                    "memory_assignments_and_coupling",
                ],
            },
        }

    dataset_index = eligible[0]
    layer_index, layer = global_layers[0]
    graph_batch = collate_graphs([dataset[dataset_index]])
    parameter = next(model.parameters())
    graph_batch = graph_batch.to(device=parameter.device, dtype=parameter.dtype)
    captured: dict[str, torch.Tensor | int] = {}

    def capture_layer_input(
        _module: torch.nn.Module,
        inputs: tuple[object, ...],
    ) -> None:
        captured["scalars"] = inputs[0].detach()
        captured["vectors"] = inputs[1].detach()
        captured["global_pos"] = inputs[2].detach()
        captured["batch"] = inputs[4].detach()
        captured["num_graphs"] = int(inputs[5])

    training_states = [(module, module.training) for module in model.modules()]
    handle = layer.register_forward_pre_hook(capture_layer_input)
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
        query_scalar = _normalize_positive_features(
            F.elu(
                layer.query_scalar(normalized_scalars).reshape(
                    node_count, layer.num_heads, layer.head_dim
                )
            )
            + 1.0,
            layer.eps,
        )
        key_scalar = _normalize_positive_features(
            F.elu(
                layer.key_scalar(normalized_scalars).reshape(
                    node_count, layer.num_heads, layer.head_dim
                )
            )
            + 1.0,
            layer.eps,
        )
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
            assignment, coupling, centers = _memory_assignments_and_coupling(
                key_scalar[:, global_heads],
                global_pos,
                batch,
                num_graphs=num_graphs,
                memory_count=layer.global_memory_count,
                temperature=layer.memory_assignment_temperature,
                assignment_scale=layer.memory_assignment_scale,
                interaction_cutoff=layer.memory_interaction_cutoff,
                interact=True,
            )
            head_assignment = assignment[:, 0]
            head_coupling = coupling[0, 0]
            pair_gate = torch.einsum(
                "im,mn,jn->ij",
                head_assignment,
                head_coupling,
                head_assignment,
            )
            memory = {
                "status": "active",
                "transport_connected": True,
                "assignment_scope": "all_global_heads_in_selected_layer",
                "assignment": memory_assignment_summary(assignment),
                "coupling": kernel_component_quantiles(
                    {"coupling": coupling}, quantiles=(0.0, 0.5, 1.0)
                ),
                "centers_finite": bool(torch.isfinite(centers).all().item()),
            }
            assignment_source = "exact_model_helper_recompute"
        elif layer.use_memory_interaction:
            assignment = key_scalar.new_ones((node_count, layer.global_head_count, 1))
            memory = {
                "status": "exact_single_memory_bypass",
                "transport_connected": False,
                "assignment": memory_assignment_summary(assignment),
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

    if parameter.device.type == "cuda":
        torch.cuda.synchronize(parameter.device)
    diagnostic_seconds = time.perf_counter() - started
    return {
        **common,
        "status": "ok",
        "elapsed_seconds": float(diagnostic_seconds),
        "batch": {
            "split": "validation",
            "dataset_index": int(dataset_index),
            "node_count": int(node_count),
        },
        "instrumentation": {
            "activation_source": "exact_recompute_from_captured_layer_input",
            "assignment_source": assignment_source,
            "layer_index": int(layer_index),
            "head_index": int(head_index),
            "connected": [
                "selected_runtime_layer_input",
                "selected_trained_query_key_projections",
                "selected_trained_beta_gamma",
                "configured_floor_alignment_and_balancing",
                "selected_global_attention_matrix",
            ],
            "unconnected": [
                "all_other_layers_and_heads",
                "local_attention_weights",
                "value_transport_and_residual_updates",
                "full_validation_distribution",
                "amp_execution_path",
            ],
        },
        "kernel_attention": kernel_attention,
        "memory": memory,
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
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
) -> dict[str, object]:
    inverse_positive_baseline = args.kernel_floor_mode == "inverse_graph_size"
    return {
        "dataset": args.dataset,
        "data_root": str(args.data_root),
        "qm9_target_index": args.qm9_target_index,
        "num_samples": args.num_samples,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "model": "factorized_moment",
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
        "test_evaluated": args.evaluate_test,
        "attention": "factorized_moment",
        "kernel_version": 3,
        "balance_cycles": 0 if args.no_key_balancing else 1,
        "key_balancing": not args.no_key_balancing,
        "alignment_linear_term": not args.no_alignment_linear_term,
        "alignment_constant_retained": True,
        "linear_kernel_init": args.linear_kernel_init,
        "kernel_floor_mode": args.kernel_floor_mode,
        "kernel_scaling_formula_version": "positive_baseline_v1",
        "graph_size_scaled_positive_baseline": inverse_positive_baseline,
        "kernel_formula": (
            "a_dot_b + (c + beta*(1 + delta*t))/N_g + gamma*t^2"
            if inverse_positive_baseline
            else "a_dot_b + c + beta*(1 + delta*t) + gamma*t^2"
        ),
        "graph_size_scaled_terms": (
            ["kernel_floor", "alignment_constant", "alignment_linear"]
            if inverse_positive_baseline
            else []
        ),
        "graph_size_unscaled_terms": ["content", "alignment_quadratic"],
        "routing": args.routing,
        "local_head_counts": list(
            _routing_head_counts(
                args.routing,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
            )
        ),
        "local_cutoff": args.local_cutoff,
        "num_rbf": args.num_rbf,
        "learn_local_radial_gate": args.learn_local_radial_gate,
        "memory_count": args.memory_count,
        "memory_interaction": args.memory_interaction,
        "memory_assignment_temperature": args.memory_assignment_temperature,
        "memory_assignment_scale": args.memory_assignment_scale,
        "memory_interaction_cutoff": args.memory_interaction_cutoff,
        "radial_trace": args.radial_trace,
        "bounded_diagnostics": args.bounded_diagnostics,
        "diagnostic_max_nodes": args.diagnostic_max_nodes,
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
