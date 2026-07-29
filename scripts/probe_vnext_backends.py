from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from equivariant_attention import EquivariantAttention, EquivariantAttentionConfig
from equivariant_attention import moment


def _timed_forward_backward(
    function: object,
    *,
    warmup: int,
    repeats: int,
    model: torch.nn.Module | None = None,
) -> float:
    samples: list[float] = []
    for index in range(warmup + repeats):
        if model is not None:
            model.zero_grad(set_to_none=True)
        start = perf_counter()
        output = function()  # type: ignore[operator]
        if isinstance(output, tuple):
            loss, differentiable = output
            if model is None:
                torch.autograd.grad(loss, differentiable)
            else:
                loss.backward()
        elapsed = perf_counter() - start
        if index >= warmup:
            samples.append(elapsed)
    if model is not None:
        model.zero_grad(set_to_none=True)
    return median(samples)


def _global_probe(
    *,
    graphs: int,
    nodes_per_graph: int,
    warmup: int,
    repeats: int,
) -> dict[str, float | int]:
    generator = torch.Generator().manual_seed(20260729)
    nodes, heads, content, values = graphs * nodes_per_graph, 4, 16, 37
    counts = torch.full((graphs,), nodes_per_graph, dtype=torch.long)
    batch = torch.repeat_interleave(torch.arange(graphs), counts)
    base = (
        torch.rand(nodes, heads, content, generator=generator),
        torch.rand(nodes, heads, content, generator=generator),
        0.2 * torch.randn(nodes, heads, 3, generator=generator),
        0.2 * torch.randn(nodes, heads, 3, generator=generator),
        torch.tensor([0.1, 0.2, 0.3, 0.4]),
        torch.randn(nodes, heads, values, generator=generator),
        0.2 * torch.rand(nodes, heads, 40, generator=generator),
        0.2 * torch.rand(nodes, heads, 40, generator=generator),
    )

    def run(backend: str) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        differentiable = tuple(value.detach().requires_grad_(True) for value in base)
        query, key, query_vector, key_vector, scale, value, spatial_q, spatial_k = (
            differentiable
        )
        output = moment._factorized_moment_attention(
            query,
            key,
            query_vector,
            key_vector,
            scale,
            value,
            batch,
            num_graphs=graphs,
            balanced=True,
            alignment_scale=torch.full((heads,), 0.1),
            alignment_dot_scale=torch.full((heads,), 0.08),
            graph_counts=counts,
            spatial_features=spatial_q,
            spatial_key_features=spatial_k,
            reduction_backend=backend,
        )
        return output.square().mean(), differentiable

    with torch.no_grad():
        reference_inputs = tuple(value.clone() for value in base)
        query, key, query_vector, key_vector, scale, value, spatial_q, spatial_k = (
            reference_inputs
        )
        common = dict(
            num_graphs=graphs,
            balanced=True,
            alignment_scale=torch.full((heads,), 0.1),
            alignment_dot_scale=torch.full((heads,), 0.08),
            graph_counts=counts,
            spatial_features=spatial_q,
            spatial_key_features=spatial_k,
        )
        outer = moment._factorized_moment_attention(
            query,
            key,
            query_vector,
            key_vector,
            scale,
            value,
            batch,
            reduction_backend="outer_scatter",
            **common,
        )
        gemm = moment._factorized_moment_attention(
            query,
            key,
            query_vector,
            key_vector,
            scale,
            value,
            batch,
            reduction_backend="feature_gemm",
            **common,
        )
    outer_seconds = _timed_forward_backward(
        lambda: run("outer_scatter"),
        warmup=warmup,
        repeats=repeats,
    )
    gemm_seconds = _timed_forward_backward(
        lambda: run("feature_gemm"),
        warmup=warmup,
        repeats=repeats,
    )
    return {
        "nodes": nodes,
        "heads": heads,
        "spatial_feature_width": 40,
        "value_width": values,
        "forward_max_abs_error": float((gemm - outer).abs().max()),
        "timing_scope": "forward plus gradients with respect to every tensor input",
        "outer_forward_input_backward_seconds": outer_seconds,
        "feature_gemm_forward_input_backward_seconds": gemm_seconds,
        "feature_gemm_over_outer_ratio": gemm_seconds / outer_seconds,
    }


def _local_probe(
    *,
    nodes: int,
    degree: int,
    warmup: int,
    repeats: int,
) -> dict[str, float | int]:
    generator = torch.Generator().manual_seed(20260730)
    receiver = torch.arange(nodes).repeat_interleave(degree)
    values = torch.randn(
        receiver.numel(),
        8,
        generator=generator,
    )
    row_ptr = moment._receiver_csr_row_ptr(receiver, nodes)

    def run(segment: bool) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        value = values.detach().requires_grad_(True)
        output = moment._receiver_sum(
            receiver,
            nodes,
            value,
            offsets=row_ptr if segment else None,
        )[0]
        return output.square().mean(), (value,)

    with torch.no_grad():
        index_output = moment._receiver_sum(
            receiver,
            nodes,
            values,
        )[0]
        segment_output = moment._receiver_sum(
            receiver,
            nodes,
            values,
            offsets=row_ptr,
        )[0]
    index_seconds = _timed_forward_backward(
        lambda: run(False),
        warmup=warmup,
        repeats=repeats,
    )
    segment_seconds = _timed_forward_backward(
        lambda: run(True),
        warmup=warmup,
        repeats=repeats,
    )
    return {
        "nodes": nodes,
        "degree": degree,
        "edges": receiver.numel(),
        "row_ptr_dtype_bits": torch.iinfo(row_ptr.dtype).bits,
        "measurement_scope": "prepacked reduction only; excludes CSR construction",
        "forward_max_abs_error": float(
            (segment_output - index_output).abs().max()
        ),
        "index_add_forward_input_backward_seconds": index_seconds,
        "segment_csr_forward_input_backward_seconds": segment_seconds,
        "segment_over_index_add_ratio": segment_seconds / index_seconds,
    }


def _model_state_bytes(model: torch.nn.Module) -> int:
    return sum(value.numel() * value.element_size() for value in model.state_dict().values())


def _saved_tensor_payload_bytes(
    function: object,
    model: torch.nn.Module,
) -> int:
    saved_bytes = 0

    def record(value: torch.Tensor) -> torch.Tensor:
        nonlocal saved_bytes
        saved_bytes += value.numel() * value.element_size()
        return value

    model.zero_grad(set_to_none=True)
    with torch.autograd.graph.saved_tensors_hooks(record, lambda value: value):
        loss, _differentiable = function()  # type: ignore[operator]
        loss.backward()
    model.zero_grad(set_to_none=True)
    return saved_bytes


def _hybrid_probe(
    *,
    warmup: int,
    repeats: int,
) -> dict[str, float | int]:
    common = dict(
        node_dim=8,
        hidden_irreps="64x0e + 4x1o",
        output_irreps="1x0e",
        num_layers=3,
        num_heads=4,
        local_head_counts=(0, 0, 0),
        local_cutoff=10.0,
        global_reduction_backend="feature_gemm",
    )
    torch.manual_seed(20260731)
    baseline = EquivariantAttention(EquivariantAttentionConfig(**common))
    torch.manual_seed(20260731)
    candidate = EquivariantAttention(
        EquivariantAttentionConfig(
            **common,
            use_sparse_low_rank_local_residual=True,
            local_residual_rank=4,
            local_residual_layers=(0, 2),
            local_reduction_backend="segment_csr",
        )
    )
    torch.manual_seed(20260731)
    gated_lgl = EquivariantAttention(
        EquivariantAttentionConfig(
            **{
                **common,
                "local_head_counts": (4, 0, 4),
            },
            use_gated_local_transport=True,
        )
    )
    counts = torch.full((8,), 32, dtype=torch.long)
    batch = torch.repeat_interleave(torch.arange(8), counts)
    nodes = int(counts.sum())
    receiver = torch.arange(nodes).repeat_interleave(8)
    graph_start = (receiver.reshape(nodes, 8) // 32) * 32
    sender = graph_start + (
        receiver.reshape(nodes, 8)
        - graph_start
        + torch.arange(8).reshape(1, 8)
    ) % 32
    edge_index = torch.stack([receiver, sender.reshape(-1)])
    generator = torch.Generator().manual_seed(20260732)
    node_feats = torch.randn(nodes, 8, generator=generator)
    pos = 0.05 * torch.randn(nodes, 3, generator=generator)

    def run(
        model: EquivariantAttention,
        *,
        edges: torch.Tensor | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        features = node_feats.detach().requires_grad_(True)
        coordinates = pos.detach().requires_grad_(True)
        output = model(
            features,
            coordinates,
            batch=batch,
            edge_index=edges,
        )
        return output["graph_scalars"].square().mean(), (features, coordinates)

    baseline_seconds = _timed_forward_backward(
        lambda: run(baseline, edges=None),
        warmup=warmup,
        repeats=repeats,
        model=baseline,
    )
    candidate_seconds = _timed_forward_backward(
        lambda: run(candidate, edges=edge_index),
        warmup=warmup,
        repeats=repeats,
        model=candidate,
    )
    gated_seconds = _timed_forward_backward(
        lambda: run(gated_lgl, edges=edge_index),
        warmup=warmup,
        repeats=repeats,
        model=gated_lgl,
    )
    with torch.no_grad():
        baseline_output = baseline(node_feats, pos, batch=batch)
        candidate_output = candidate(
            node_feats,
            pos,
            batch=batch,
            edge_index=edge_index,
        )
    baseline_bytes = _model_state_bytes(baseline)
    candidate_bytes = _model_state_bytes(candidate)
    baseline_saved_bytes = _saved_tensor_payload_bytes(
        lambda: run(baseline, edges=None),
        baseline,
    )
    candidate_saved_bytes = _saved_tensor_payload_bytes(
        lambda: run(candidate, edges=edge_index),
        candidate,
    )
    gated_saved_bytes = _saved_tensor_payload_bytes(
        lambda: run(gated_lgl, edges=edge_index),
        gated_lgl,
    )
    finite_differences = [
        (candidate_output[name] - baseline_output[name]).abs().max()
        for name in baseline_output
        if baseline_output[name].numel()
    ]
    return {
        "nodes": nodes,
        "edges": edge_index.shape[1],
        "active_sparse_layers": 2,
        "local_rank": 4,
        "timing_scope": "forward plus full parameter and input backward; excludes optimizer",
        "zero_init_forward_max_abs_error": float(
            torch.stack(finite_differences).max()
        ),
        "baseline_forward_parameter_backward_seconds": baseline_seconds,
        "sparse_residual_forward_parameter_backward_seconds": candidate_seconds,
        "sparse_residual_over_baseline_ratio": (
            candidate_seconds / baseline_seconds
        ),
        "gated_lgl_forward_parameter_backward_seconds": gated_seconds,
        "sparse_residual_over_gated_lgl_ratio": (
            candidate_seconds / gated_seconds
        ),
        "baseline_state_bytes": baseline_bytes,
        "sparse_residual_state_bytes": candidate_bytes,
        "state_bytes_ratio": candidate_bytes / baseline_bytes,
        "gated_lgl_state_bytes": _model_state_bytes(gated_lgl),
        "baseline_saved_tensor_payload_bytes": baseline_saved_bytes,
        "sparse_residual_saved_tensor_payload_bytes": candidate_saved_bytes,
        "gated_lgl_saved_tensor_payload_bytes": gated_saved_bytes,
        "sparse_saved_payload_over_baseline_ratio": (
            candidate_saved_bytes / baseline_saved_bytes
        ),
        "sparse_saved_payload_over_gated_lgl_ratio": (
            candidate_saved_bytes / gated_saved_bytes
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats <= 0 or args.threads <= 0:
        raise ValueError("warmup, repeats, and threads must be positive as applicable")
    torch.set_num_threads(args.threads)
    result = {
        "environment": {
            "torch_version": torch.__version__,
            "device": "cpu",
            "threads": args.threads,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "global": _global_probe(
            graphs=16,
            nodes_per_graph=18,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
        "local": _local_probe(
            nodes=4096,
            degree=32,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
        "hybrid": _hybrid_probe(
            warmup=args.warmup,
            repeats=args.repeats,
        ),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
