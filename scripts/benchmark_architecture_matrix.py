#!/usr/bin/env python3
"""Bounded architecture/resource matrix for the generic 3D model.

This runner is deliberately a mechanics benchmark, not a model-quality
experiment.  It constructs deterministic simple directed candidate graphs
with exactly ``E = kN`` supplied edges and one self edge per node, builds every
model through the public structured configuration boundary, and records
forward plus optimizer-inclusive train step resources. Synthetic graph
construction is measured separately and is explicitly excluded from both
model timings.

Large grids are opt-in.  For example, the K04 grid can be requested with::

    uv run python scripts/benchmark_architecture_matrix.py \
      --nodes 128,512,2048,8192 \
      --degrees 4,8,16,32,64,128 \
      --variants uniform,skew,ragged \
      --device cuda --output artifacts/architecture-matrix.json

No row in this artifact is an accuracy, convergence, or architecture-
superiority claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F

from equivariant_attention.benchmarking import GraphBatch
from equivariant_attention.config import ArchitectureConfig
from equivariant_attention.execution import resolve_execution_metadata
from equivariant_attention.graph_layout import pack_graph_layout
from equivariant_attention.irreps import CartesianIrreps
from equivariant_attention.neighbors import build_receiver_csr
from equivariant_attention.training import (
    build_regression_model,
    predict_graph_scalar,
)


SCHEMA = "equivariant_attention.architecture_resource_matrix"
SCHEMA_VERSION = 1
REQUIRED_ARMS = (
    "legacy_lgl",
    "deep_global",
    "h3_r4",
    "h4_r2",
    "h6_r2",
    "workspace_l3",
)
OPTIONAL_ARMS = ("standard",)
GRAPH_VARIANTS = ("uniform", "skew", "ragged")
DEFAULT_NODES = (16,)
DEFAULT_DEGREES = (2,)
DEFAULT_VARIANTS = GRAPH_VARIANTS


@dataclass(frozen=True, slots=True)
class ArmDefinition:
    """Static architecture identity independent of width and backend policy."""

    name: str
    profile: str
    depth: int
    local_rank: int | None
    edge_consumption: str
    description: str


@dataclass(frozen=True, slots=True)
class SyntheticCase:
    """One deterministic CPU graph and its construction receipt."""

    batch: GraphBatch
    metadata: dict[str, object]
    construction_ms: float


class TopologyInfeasibleError(ValueError):
    """The requested exact simple directed topology cannot exist."""


ARM_DEFINITIONS = {
    "legacy_lgl": ArmDefinition(
        name="legacy_lgl",
        profile="minimal",
        depth=3,
        local_rank=None,
        edge_consumption="coo_local_heads",
        description=(
            "legacy three-block gated/grouped LGL compatibility baseline"
        ),
    ),
    "deep_global": ArmDefinition(
        name="deep_global",
        profile="minimal",
        depth=6,
        local_rank=None,
        edge_consumption="none",
        description="six homogeneous exact-global blocks without a local lane",
    ),
    "h3_r4": ArmDefinition(
        name="h3_r4",
        profile="minimal",
        depth=3,
        local_rank=4,
        edge_consumption="sparse_residual",
        description="three exact-global blocks, each with rank-4 sparse residual",
    ),
    "h4_r2": ArmDefinition(
        name="h4_r2",
        profile="minimal",
        depth=4,
        local_rank=2,
        edge_consumption="sparse_residual",
        description="four exact-global blocks, each with rank-2 sparse residual",
    ),
    "h6_r2": ArmDefinition(
        name="h6_r2",
        profile="minimal",
        depth=6,
        local_rank=2,
        edge_consumption="sparse_residual",
        description="six exact-global blocks, each with rank-2 sparse residual",
    ),
    "workspace_l3": ArmDefinition(
        name="workspace_l3",
        profile="high_order",
        depth=3,
        local_rank=None,
        edge_consumption="coo_transient_workspace",
        description=(
            "low-l persistent carrier with nonpersistent aggregate/project l=3"
        ),
    ),
    "standard": ArmDefinition(
        name="standard",
        profile="standard",
        depth=3,
        local_rank=None,
        edge_consumption="none",
        description="standard persistent 0e/1o/2e structured profile",
    ),
}


def build_arm_config(
    name: str,
    *,
    node_dim: int,
    width: int,
    num_heads: int,
    global_backend: str = "auto",
    local_backend: str = "auto",
    geometry_cache_mode: str = "auto",
    workspace_channels: int = 1,
) -> ArchitectureConfig:
    """Build one named arm through the public structured configuration API."""

    try:
        definition = ARM_DEFINITIONS[name]
    except KeyError as error:
        choices = ", ".join(ARM_DEFINITIONS)
        raise ValueError(f"unknown arm {name!r}; expected one of: {choices}") from error
    config = ArchitectureConfig.for_profile(
        definition.profile,
        node_dim=node_dim,
        width=width,
        num_heads=num_heads,
        num_layers=definition.depth,
    )
    config = replace(
        config,
        global_transport=replace(
            config.global_transport,
            reduction_backend=global_backend,
        ),
        neighbor=replace(
            config.neighbor,
            geometry_cache_mode=geometry_cache_mode,
            provider_kind="precomputed",
        ),
    )
    if name == "legacy_lgl":
        config = replace(
            config,
            local=replace(
                config.local,
                local_head_counts=(num_heads, 0, num_heads),
                use_gated_local_transport=True,
                use_grouped_invariant_normalization=True,
                requested_backend="materialized",
            ),
        )
    elif definition.local_rank is not None:
        config = replace(
            config,
            local=replace(
                config.local,
                local_head_counts=(0,) * definition.depth,
                use_sparse_low_rank_local_residual=True,
                local_residual_rank=definition.local_rank,
                residual_stride=1,
                requested_backend=local_backend,
            ),
        )
    elif name == "workspace_l3":
        config = replace(
            config,
            representation=replace(
                config.representation,
                transient_workspace_channels=workspace_channels,
                transient_workspace_layers=tuple(range(definition.depth)),
            ),
            local=replace(config.local, requested_backend="materialized"),
        )
    return config


def build_architecture_arms(
    names: Sequence[str] = REQUIRED_ARMS,
    *,
    node_dim: int,
    widths: dict[str, int] | None = None,
    width: int = 16,
    num_heads: int = 4,
    global_backend: str = "auto",
    local_backend: str = "auto",
    geometry_cache_mode: str = "auto",
    workspace_channels: int = 1,
) -> dict[str, ArchitectureConfig]:
    """Return the ordered named architecture matrix."""

    _validate_arm_names(names)
    resolved_widths = {} if widths is None else widths
    return {
        name: build_arm_config(
            name,
            node_dim=node_dim,
            width=resolved_widths.get(name, width),
            num_heads=num_heads,
            global_backend=global_backend,
            local_backend=local_backend,
            geometry_cache_mode=geometry_cache_mode,
            workspace_channels=workspace_channels,
        )
        for name in names
    }


def make_synthetic_case(
    num_nodes: int,
    *,
    degree: int,
    variant: str,
    node_dim: int,
    seed: int,
) -> SyntheticCase:
    """Create a deterministic CPU case with exactly ``degree * num_nodes`` edges."""

    _positive_integer("num_nodes", num_nodes)
    _positive_integer("degree", degree)
    _positive_integer("node_dim", node_dim)
    if variant not in GRAPH_VARIANTS:
        raise ValueError(f"variant must be one of: {', '.join(GRAPH_VARIANTS)}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    started = time.perf_counter()
    (
        graph_sizes,
        graph_edge_counts,
        topology_plan,
    ) = _topology_plan(num_nodes, degree=degree, variant=variant)
    batch_index = torch.repeat_interleave(
        torch.arange(len(graph_sizes), dtype=torch.long),
        torch.tensor(graph_sizes, dtype=torch.long),
    )
    edge_parts = []
    node_offset = 0
    for graph_size, graph_edge_count in zip(
        graph_sizes,
        graph_edge_counts,
        strict=True,
    ):
        if variant == "skew":
            receiver_degrees = _skew_receiver_degrees(
                graph_size,
                edge_count=graph_edge_count,
            )
        else:
            receiver_degrees = _balanced_receiver_degrees(
                graph_size,
                edge_count=graph_edge_count,
            )
        edge_parts.append(
            _graph_edges(
                graph_size,
                receiver_degrees=receiver_degrees,
                node_offset=node_offset,
            )
        )
        node_offset += graph_size
    edge_index = torch.cat(edge_parts, dim=1)
    expected_edges = num_nodes * degree
    _assert_topology_postconditions(
        edge_index,
        batch_index=batch_index,
        num_nodes=num_nodes,
        expected_edges=expected_edges,
    )

    generator = torch.Generator(device="cpu").manual_seed(seed)
    node_feats = torch.randn(
        num_nodes,
        node_dim,
        generator=generator,
        dtype=torch.float32,
    )
    # All nonself distances lie below the default 2.5 cutoff.
    pos = torch.rand(
        num_nodes,
        3,
        generator=generator,
        dtype=torch.float32,
    ) - 0.5
    target = torch.randn(
        len(graph_sizes),
        1,
        generator=generator,
        dtype=torch.float32,
    )
    layout = pack_graph_layout(
        batch_index,
        graph_counts=torch.tensor(graph_sizes, dtype=torch.long),
        assume_grouped=True,
    )
    # The flag is set only after _assert_topology_postconditions validates the
    # full public edge contract.
    batch = GraphBatch(
        node_feats=node_feats,
        pos=pos,
        batch=batch_index,
        target=target,
        sample_ids=tuple(
            f"synthetic-{variant}-{seed}-{index}"
            for index in range(len(graph_sizes))
        ),
        edge_index=edge_index,
        edge_index_is_validated=True,
        graph_layout=layout,
    )
    receiver_degree = torch.bincount(edge_index[0], minlength=num_nodes)
    mean_degree = float(expected_edges) / float(num_nodes)
    self_edges = int((edge_index[0] == edge_index[1]).sum().item())
    metadata: dict[str, object] = {
        "variant": variant,
        "nodes": num_nodes,
        "requested_degree": degree,
        "expected_edges": expected_edges,
        "supplied_edges": int(edge_index.shape[1]),
        "exact_e_equals_k_n": edge_index.shape[1] == expected_edges,
        "graph_count": len(graph_sizes),
        "graph_sizes": list(graph_sizes),
        "within_graph_edges": True,
        "receiver_mean_degree": mean_degree,
        "receiver_min_degree": int(receiver_degree.min().item()),
        "receiver_max_degree": int(receiver_degree.max().item()),
        "receiver_degree_skew": (
            float(receiver_degree.max().item()) / mean_degree
        ),
        "self_edge_count": self_edges,
        "self_edge_per_node": self_edges == num_nodes,
        "unique_directed_edges": True,
        "simple_directed_graph_with_self": True,
        "unique_edge_capacity": sum(size * size for size in graph_sizes),
        "edge_index_sha256": _tensor_sha256(edge_index),
        "graph_layout_structure": layout.structure,
        **topology_plan,
    }
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return SyntheticCase(
        batch=batch,
        metadata=metadata,
        construction_ms=elapsed_ms,
    )


def resolve_parameter_matched_widths(
    names: Sequence[str],
    *,
    node_dim: int,
    reference_width: int,
    num_heads: int,
    maximum_width: int,
    global_backend: str,
    local_backend: str,
    geometry_cache_mode: str,
    workspace_channels: int,
) -> tuple[dict[str, int], dict[str, object]]:
    """Select the nearest bounded width to the legacy parameter count.

    This is a discrete search, not a guarantee.  The receipt explicitly marks
    arms that do not reach the one-percent parameter window.
    """

    _validate_arm_names(names)
    _positive_integer("reference_width", reference_width)
    _positive_integer("maximum_width", maximum_width)
    if maximum_width < num_heads:
        raise ValueError("maximum_width must be at least num_heads")
    candidates = tuple(range(num_heads, maximum_width + 1, num_heads))
    if not candidates:
        raise ValueError("parameter matching has no admissible widths")

    reference_config = build_arm_config(
        "legacy_lgl",
        node_dim=node_dim,
        width=reference_width,
        num_heads=num_heads,
        global_backend=global_backend,
        local_backend=local_backend,
        geometry_cache_mode=geometry_cache_mode,
        workspace_channels=workspace_channels,
    )
    reference_parameters = _count_parameters(
        build_regression_model(
            node_dim,
            architecture_config=reference_config,
        )
    )
    widths: dict[str, int] = {}
    search: dict[str, object] = {}
    for name in names:
        if name == "legacy_lgl":
            selected_width = reference_width
            selected_parameters = reference_parameters
        else:
            candidates_with_counts = []
            for candidate_width in candidates:
                config = build_arm_config(
                    name,
                    node_dim=node_dim,
                    width=candidate_width,
                    num_heads=num_heads,
                    global_backend=global_backend,
                    local_backend=local_backend,
                    geometry_cache_mode=geometry_cache_mode,
                    workspace_channels=workspace_channels,
                )
                parameter_count = _count_parameters(
                    build_regression_model(
                        node_dim,
                        architecture_config=config,
                    )
                )
                candidates_with_counts.append(
                    (candidate_width, parameter_count)
                )
            selected_width, selected_parameters = min(
                candidates_with_counts,
                key=lambda item: (
                    abs(item[1] - reference_parameters),
                    item[0],
                ),
            )
        ratio = selected_parameters / reference_parameters
        widths[name] = selected_width
        search[name] = {
            "selected_width": selected_width,
            "parameter_count": selected_parameters,
            "ratio_to_legacy_target": ratio,
            "within_one_percent": abs(ratio - 1.0) <= 0.01,
        }
    return widths, {
        "enabled": True,
        "reference_arm": "legacy_lgl",
        "reference_width": reference_width,
        "reference_parameter_count": reference_parameters,
        "candidate_widths": list(candidates),
        "selection": search,
        "guaranteed_match": False,
    }


def run_architecture_matrix(
    *,
    nodes: Sequence[int] = DEFAULT_NODES,
    degrees: Sequence[int] = DEFAULT_DEGREES,
    variants: Sequence[str] = DEFAULT_VARIANTS,
    arms: Sequence[str] = REQUIRED_ARMS,
    node_dim: int = 8,
    width: int = 16,
    num_heads: int = 4,
    workspace_channels: int = 1,
    parameter_match: bool = True,
    parameter_search_max_width: int | None = None,
    global_backend: str = "auto",
    local_backend: str = "auto",
    geometry_cache_mode: str = "auto",
    device: str = "cpu",
    dtype: str = "float32",
    warmup: int = 1,
    repeats: int = 2,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    seed: int = 20260729,
    max_wall_seconds: float = 120.0,
    threads: int | None = 1,
) -> dict[str, object]:
    """Execute a bounded resource matrix and return strict-JSON-safe data."""

    nodes = tuple(nodes)
    degrees = tuple(degrees)
    variants = tuple(variants)
    arms = tuple(arms)
    _validate_run_request(
        nodes=nodes,
        degrees=degrees,
        variants=variants,
        arms=arms,
        node_dim=node_dim,
        width=width,
        num_heads=num_heads,
        workspace_channels=workspace_channels,
        warmup=warmup,
        repeats=repeats,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        max_wall_seconds=max_wall_seconds,
        threads=threads,
    )
    resolved_device = _resolve_device(device)
    resolved_dtype = _resolve_dtype(dtype)
    benchmark_started = time.perf_counter()
    if width % num_heads:
        raise ValueError("width must be divisible by num_heads")
    search_maximum = (
        max(width * 2, width + 4 * num_heads)
        if parameter_search_max_width is None
        else parameter_search_max_width
    )
    if parameter_match:
        widths, parameter_match_receipt = resolve_parameter_matched_widths(
            arms,
            node_dim=node_dim,
            reference_width=width,
            num_heads=num_heads,
            maximum_width=search_maximum,
            global_backend=global_backend,
            local_backend=local_backend,
            geometry_cache_mode=geometry_cache_mode,
            workspace_channels=workspace_channels,
        )
    else:
        widths = {name: width for name in arms}
        parameter_match_receipt = {
            "enabled": False,
            "reference_arm": "legacy_lgl",
            "reference_width": width,
            "candidate_widths": [],
            "selection": {},
            "guaranteed_match": False,
        }
    configs = build_architecture_arms(
        arms,
        node_dim=node_dim,
        widths=widths,
        width=width,
        num_heads=num_heads,
        global_backend=global_backend,
        local_backend=local_backend,
        geometry_cache_mode=geometry_cache_mode,
        workspace_channels=workspace_channels,
    )
    arm_receipts = _arm_receipts(
        configs,
        widths=widths,
        legacy_reference_width=width,
    )

    previous_threads = torch.get_num_threads()
    if threads is not None:
        torch.set_num_threads(threads)
    rows: list[dict[str, object]] = []
    try:
        case_index = 0
        for num_nodes in nodes:
            for degree in degrees:
                for variant in variants:
                    case_seed = seed + case_index
                    case_index += 1
                    try:
                        case = make_synthetic_case(
                            num_nodes,
                            degree=degree,
                            variant=variant,
                            node_dim=node_dim,
                            seed=case_seed,
                        )
                    except TopologyInfeasibleError as error:
                        rows.extend(
                            _infeasible_topology_row(
                                name=name,
                                num_nodes=num_nodes,
                                degree=degree,
                                variant=variant,
                                reason=str(error),
                            )
                            for name in arms
                        )
                        continue
                    for arm_index, name in enumerate(arms):
                        elapsed = time.perf_counter() - benchmark_started
                        if elapsed >= max_wall_seconds:
                            rows.append(
                                _skipped_row(
                                    name=name,
                                    case=case,
                                    status="skipped_time_budget",
                                    reason=(
                                        "global wall budget exhausted before "
                                        "model construction"
                                    ),
                                )
                            )
                            continue
                        config = configs[name]
                        row_seed = seed + 10_000 * case_index + arm_index
                        row = _run_one_row(
                            name=name,
                            config=config,
                            case=case,
                            device=resolved_device,
                            dtype=resolved_dtype,
                            warmup=warmup,
                            repeats=repeats,
                            learning_rate=learning_rate,
                            weight_decay=weight_decay,
                            seed=row_seed,
                        )
                        rows.append(row)
    finally:
        if threads is not None:
            torch.set_num_threads(previous_threads)

    comparisons = _relative_comparisons(rows)
    result: dict[str, object] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "request": {
            "nodes": list(nodes),
            "degrees": list(degrees),
            "variants": list(variants),
            "arms": list(arms),
            "node_dim": node_dim,
            "reference_width": width,
            "num_heads": num_heads,
            "workspace_channels": workspace_channels,
            "global_backend": global_backend,
            "local_backend": local_backend,
            "geometry_cache_mode": geometry_cache_mode,
            "device": str(resolved_device),
            "dtype": _dtype_name(resolved_dtype),
            "warmup": warmup,
            "repeats": repeats,
            "optimizer": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "seed": seed,
            "max_wall_seconds": max_wall_seconds,
            "threads": threads,
        },
        "environment": _environment_receipt(resolved_device),
        "parameter_match": parameter_match_receipt,
        "arm_receipts": arm_receipts,
        "rows": rows,
        "relative_to_legacy_lgl": comparisons,
        "elapsed_seconds": time.perf_counter() - benchmark_started,
        "claim_boundary": {
            "accuracy_evaluated": False,
            "convergence_evaluated": False,
            "resource_matching_enforced": False,
            "parameter_matching_is_discrete_and_reported": True,
            "train_step_matching_is_reported_not_enforced": True,
            "synthetic_candidate_edges_are_not_neighbor_discovery": True,
            "architecture_superiority_claimed": False,
        },
    }
    # Fail here rather than emit non-standard NaN/Infinity JSON.
    json.dumps(result, allow_nan=False, sort_keys=True)
    return result


def _run_one_row(
    *,
    name: str,
    config: ArchitectureConfig,
    case: SyntheticCase,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> dict[str, object]:
    definition = ARM_DEFINITIONS[name]
    model = None
    optimizer = None
    prepared = None
    try:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        preparation_started = time.perf_counter()
        prepared, neighbor_representation = _prepare_batch(
            case.batch,
            edge_consumption=definition.edge_consumption,
            local_backend=config.local.requested_backend,
            device=device,
            dtype=dtype,
        )
        _synchronize(device)
        preparation_ms = (time.perf_counter() - preparation_started) * 1000.0
        model = build_regression_model(
            config.node_dim,
            architecture_config=config,
        ).to(device=device, dtype=dtype)

        parameter_count = _count_parameters(model)
        trainable_parameter_count = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        model_bytes = _model_bytes(model)
        execution = _execution_receipt(
            config=config,
            batch=prepared,
            edge_count=int(case.metadata["supplied_edges"]),
            device=device,
            dtype=dtype,
            sparse_local_executed=definition.edge_consumption
            in {"coo_local_heads", "sparse_residual"},
            transient_workspace_executed=(
                definition.edge_consumption == "coo_transient_workspace"
            ),
        )
        forward = _measure_forward(
            model,
            prepared,
            device=device,
            warmup=warmup,
            repeats=repeats,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        train_step = _measure_train_step(
            model,
            prepared,
            optimizer,
            device=device,
            warmup=warmup,
            repeats=repeats,
        )
        return {
            "status": "completed",
            "arm": name,
            "nodes": case.metadata["nodes"],
            "degree": case.metadata["requested_degree"],
            "edges": case.metadata["supplied_edges"],
            "variant": case.metadata["variant"],
            "graph_identity_sha256": case.metadata["edge_index_sha256"],
            "edge_consumption": definition.edge_consumption,
            "neighbor_representation": neighbor_representation,
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_parameter_count,
            **model_bytes,
            "optimizer_state_bytes": _optimizer_state_bytes(optimizer),
            "input_state_bytes": _batch_tensor_bytes(prepared),
            "forward": forward,
            "train_step": train_step,
            "execution": execution,
            "graph": case.metadata,
            "graph_construction": {
                "synthetic_generation_shared_across_arms": True,
                "synthetic_generation_ms": case.construction_ms,
                "representation_and_transfer_ms": preparation_ms,
                "one_time_total_ms": case.construction_ms + preparation_ms,
                "included_in_forward_timing": False,
                "included_in_train_step_timing": False,
                "neighbor_discovery_performed": False,
                "candidate_edge_generation_measured_separately": True,
            },
        }
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        oom = _is_oom_error(error)
        return {
            "status": "skipped_oom" if oom else "failed",
            "arm": name,
            "nodes": case.metadata["nodes"],
            "degree": case.metadata["requested_degree"],
            "edges": case.metadata["supplied_edges"],
            "variant": case.metadata["variant"],
            "graph_identity_sha256": case.metadata["edge_index_sha256"],
            "edge_consumption": definition.edge_consumption,
            "error_type": type(error).__name__,
            "reason": _bounded_error_message(error),
            "graph": case.metadata,
            "graph_construction": {
                "synthetic_generation_shared_across_arms": True,
                "synthetic_generation_ms": case.construction_ms,
                "included_in_forward_timing": False,
                "included_in_train_step_timing": False,
                "neighbor_discovery_performed": False,
            },
        }
    finally:
        del optimizer, model, prepared
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _measure_forward(
    model: torch.nn.Module,
    batch: GraphBatch,
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    model.eval()

    def operation() -> torch.Tensor:
        with torch.inference_mode():
            return predict_graph_scalar(model, batch)

    for _ in range(warmup):
        output = operation()
        _require_finite_tensor("forward output", output)
    _synchronize(device)
    memory = _begin_memory_measurement(device)
    samples, last = _timed_repeats(operation, device=device, repeats=repeats)
    _require_finite_tensor("forward output", last)
    memory.update(_end_memory_measurement(device, memory))
    return {
        **_timing_summary(samples),
        "mode": "eval_inference_mode",
        "model_forward_only": True,
        "input_transfer_included": False,
        "graph_construction_included": False,
        **memory,
    }


def _measure_train_step(
    model: torch.nn.Module,
    batch: GraphBatch,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    model.train()

    def operation() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        prediction = predict_graph_scalar(model, batch)
        target = batch.target.reshape_as(prediction)
        loss = F.mse_loss(prediction.float(), target.float())
        _require_finite_tensor("train loss", loss)
        loss.backward()
        optimizer.step()
        return loss.detach()

    for _ in range(warmup):
        operation()
    _synchronize(device)
    memory = _begin_memory_measurement(device)
    samples, last_loss = _timed_repeats(
        operation,
        device=device,
        repeats=repeats,
    )
    memory.update(_end_memory_measurement(device, memory))
    return {
        **_timing_summary(samples),
        "mode": "train",
        "optimizer": type(optimizer).__name__,
        "optimizer_inclusive": True,
        "zero_grad_included": True,
        "forward_included": True,
        "loss_included": True,
        "backward_included": True,
        "optimizer_step_included": True,
        "input_transfer_included": False,
        "graph_construction_included": False,
        "last_loss": float(last_loss.float().cpu().item()),
        **memory,
    }


def _timed_repeats(
    operation: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    repeats: int,
) -> tuple[list[float], torch.Tensor]:
    samples = []
    last = torch.empty(())
    for _ in range(repeats):
        _synchronize(device)
        started = time.perf_counter()
        last = operation()
        _synchronize(device)
        samples.append((time.perf_counter() - started) * 1000.0)
    return samples, last


def _prepare_batch(
    batch: GraphBatch,
    *,
    edge_consumption: str,
    local_backend: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[GraphBatch, str]:
    if batch.edge_index is None:
        raise RuntimeError("synthetic base batch has no edge_index")
    if edge_consumption == "none":
        prepared = replace(
            batch,
            edge_index=None,
            edge_index_is_validated=False,
            packed_neighbors=None,
        )
        return prepared.to(device=device, dtype=dtype), "not_consumed"
    packed_backends = {
        "auto",
        "segment_csr",
        "streamed_csr",
        "ell",
        "custom",
    }
    if edge_consumption == "sparse_residual" and local_backend in packed_backends:
        receiver_degree = torch.bincount(
            batch.edge_index[0],
            minlength=batch.node_feats.shape[0],
        )
        max_degree = int(receiver_degree.max().item())
        ell_padding_ratio = (
            batch.node_feats.shape[0] * max_degree / batch.edge_index.shape[1]
        )
        build_ell = local_backend == "ell" or (
            local_backend == "auto" and ell_padding_ratio <= 64.0
        )
        packed = build_receiver_csr(
            batch.edge_index,
            num_nodes=batch.node_feats.shape[0],
            build_ell=build_ell,
            ell_max_degree=max(64, max_degree),
            ell_max_padding_ratio=64.0,
        )
        prepared = replace(
            batch,
            edge_index=None,
            edge_index_is_validated=False,
            packed_neighbors=packed,
        )
        return prepared.to(device=device, dtype=dtype), "packed_receiver_csr"
    prepared = replace(batch, packed_neighbors=None)
    return prepared.to(device=device, dtype=dtype), "validated_coo"


def _execution_receipt(
    *,
    config: ArchitectureConfig,
    batch: GraphBatch,
    edge_count: int,
    device: torch.device,
    dtype: torch.dtype,
    sparse_local_executed: bool,
    transient_workspace_executed: bool,
) -> dict[str, object]:
    packed = batch.packed_neighbors
    feature_width, value_width = _global_transport_widths(config)
    receipt = resolve_execution_metadata(
        requested_global_lane=config.global_transport.reduction_backend,
        requested_local_backend=config.local.requested_backend,
        requested_cache_mode=config.neighbor.geometry_cache_mode,
        graph_layout=batch.graph_layout,
        node_count=batch.node_feats.shape[0],
        edge_count=edge_count,
        neighbor_policy="precomputed_fixed_candidates",
        provider_capabilities={
            "provider": "synthetic_precomputed_exact_e_equals_k_n",
            "complexity": "O(N+E)_deterministic_candidate_construction",
            "production_ready": False,
            "deterministic_selection": True,
            "supports_skin": False,
            "supports_pbc": False,
            "supports_cell_list": False,
            "supports_hard_learned_topk": False,
        },
        symmetry_group=config.symmetry_group,
        architecture_profile=config.profile,
        dtype=dtype,
        device=device,
        num_heads=config.num_heads,
        feature_width=feature_width,
        value_width=value_width,
        has_receiver_csr=packed is not None,
        has_ell=packed is not None and packed.ell_sender is not None,
        max_degree=(
            0
            if packed is None or packed.max_degree is None
            else packed.max_degree
        ),
        local_operation=config.local.sparse_residual_normalization,
    )
    resolved = receipt.to_dict()
    resolved["schema"] = (
        "equivariant_attention.architecture_matrix_execution"
    )
    resolved["source_resolver_schema"] = receipt.SCHEMA
    resolved["source_resolver_schema_version"] = receipt.SCHEMA_VERSION
    resolved["global_feature_width"] = feature_width
    resolved["global_value_width"] = value_width
    if sparse_local_executed:
        resolved["local_backend_status"] = "executed"
    else:
        resolved["configured_local_backend"] = resolved[
            "requested_local_backend"
        ]
        resolved["requested_local_backend"] = "not_executed"
        resolved["effective_local_backend"] = "not_executed"
        resolved["local_operation"] = "not_executed"
        resolved["local_backend_status"] = "not_executed"
        resolved["local_execution_reason"] = (
            "architecture arm has no sparse/local backend operator"
        )
        resolved["fallbacks"] = [
            fallback
            for fallback in resolved["fallbacks"]
            if fallback["subsystem"] != "local"
        ]
    resolved["transient_workspace_status"] = (
        "executed" if transient_workspace_executed else "not_configured"
    )
    resolved["transient_workspace_backend"] = (
        "filtered_coo_reference"
        if transient_workspace_executed
        else "not_executed"
    )
    return resolved


def _arm_receipts(
    configs: dict[str, ArchitectureConfig],
    *,
    widths: dict[str, int],
    legacy_reference_width: int,
) -> dict[str, object]:
    receipts: dict[str, object] = {}
    reference_count = None
    raw_counts: dict[str, int] = {}
    raw_bytes: dict[str, dict[str, int]] = {}
    for name, config in configs.items():
        model = build_regression_model(
            config.node_dim,
            architecture_config=config,
        )
        count = _count_parameters(model)
        raw_counts[name] = count
        raw_bytes[name] = _model_bytes(model)
        if name == "legacy_lgl":
            reference_count = count
    if reference_count is None:
        reference_config = build_arm_config(
            "legacy_lgl",
            node_dim=next(iter(configs.values())).node_dim,
            width=legacy_reference_width,
            num_heads=next(iter(configs.values())).num_heads,
        )
        reference_count = _count_parameters(
            build_regression_model(
                reference_config.node_dim,
                architecture_config=reference_config,
            )
        )
    for name, config in configs.items():
        ratio = raw_counts[name] / reference_count
        definition = ARM_DEFINITIONS[name]
        receipts[name] = {
            "description": definition.description,
            "width": widths[name],
            "depth": definition.depth,
            "local_rank": definition.local_rank,
            "edge_consumption": definition.edge_consumption,
            "parameter_count": raw_counts[name],
            "parameter_ratio_to_legacy_lgl": ratio,
            "parameter_matched_within_one_percent": abs(ratio - 1.0) <= 0.01,
            **raw_bytes[name],
            "architecture": config.to_dict(),
        }
    return receipts


def _relative_comparisons(
    rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, object, object], list[dict[str, object]]] = {}
    for row in rows:
        key = (row["nodes"], row["degree"], row["variant"])
        grouped.setdefault(key, []).append(row)
    comparisons = []
    for (nodes, degree, variant), group in grouped.items():
        legacy = next(
            (
                row
                for row in group
                if row["arm"] == "legacy_lgl" and row["status"] == "completed"
            ),
            None,
        )
        for row in group:
            comparison: dict[str, object] = {
                "nodes": nodes,
                "degree": degree,
                "variant": variant,
                "arm": row["arm"],
                "reference_arm": "legacy_lgl",
            }
            if legacy is None or row["status"] != "completed":
                comparison.update(
                    {
                        "status": "unavailable",
                        "reason": (
                            "completed legacy_lgl and candidate rows are required"
                        ),
                    }
                )
            else:
                train_ratio = _nested_median(row, "train_step") / _nested_median(
                    legacy,
                    "train_step",
                )
                forward_ratio = _nested_median(row, "forward") / _nested_median(
                    legacy,
                    "forward",
                )
                parameter_ratio = float(row["parameter_count"]) / float(
                    legacy["parameter_count"]
                )
                comparison.update(
                    {
                        "status": "measured",
                        "parameter_ratio": parameter_ratio,
                        "parameters_within_one_percent": (
                            abs(parameter_ratio - 1.0) <= 0.01
                        ),
                        "forward_median_ratio": forward_ratio,
                        "optimizer_step_median_ratio": train_ratio,
                        "optimizer_step_within_five_percent": (
                            abs(train_ratio - 1.0) <= 0.05
                        ),
                        "resource_match_is_observed_not_enforced": True,
                    }
                )
            comparisons.append(comparison)
    return comparisons


def _topology_plan(
    num_nodes: int,
    *,
    degree: int,
    variant: str,
) -> tuple[tuple[int, ...], tuple[int, ...], dict[str, object]]:
    expected_edges = num_nodes * degree
    if degree > num_nodes:
        raise TopologyInfeasibleError(
            "unique directed edges with one self edge per node require "
            f"k <= N, but received N={num_nodes}, k={degree}"
        )
    if variant in {"uniform", "skew"}:
        graph_sizes = (num_nodes,)
        graph_edge_counts = (expected_edges,)
        limitations = []
        if variant == "skew" and degree in {1, num_nodes}:
            limitations.append(
                "unique-degree capacity forces a uniform receiver degree "
                "at this boundary"
            )
        return graph_sizes, graph_edge_counts, {
            "requested_topology_variant": variant,
            "effective_topology_variant": (
                "uniform"
                if variant == "skew" and limitations
                else variant
            ),
            "topology_limitations": limitations,
            "ragged_partition_adjusted": False,
            "ragged_partition_collapsed": False,
        }

    preferred = _preferred_ragged_sizes(num_nodes)
    graph_sizes = preferred
    limitations: list[str] = []
    capacity = sum(size * size for size in graph_sizes)
    partition_adjusted = False
    partition_collapsed = False
    if capacity < expected_edges:
        most_ragged_two_graph_partition = (
            (1, num_nodes - 1) if num_nodes > 1 else (num_nodes,)
        )
        two_graph_capacity = sum(
            size * size for size in most_ragged_two_graph_partition
        )
        partition_adjusted = graph_sizes != most_ragged_two_graph_partition
        if two_graph_capacity >= expected_edges:
            graph_sizes = most_ragged_two_graph_partition
            limitations.append(
                "preferred ragged partition lacked unique-edge capacity; "
                "adjusted to a maximally ragged two-graph partition"
            )
        else:
            graph_sizes = (num_nodes,)
            partition_collapsed = True
            limitations.append(
                "no multi-graph partition had enough unique-edge capacity; "
                "collapsed to one graph"
            )
    graph_edge_counts = _allocate_graph_edges(
        graph_sizes,
        total_edges=expected_edges,
    )
    is_effectively_ragged = (
        len(graph_sizes) > 1 and len(set(graph_sizes)) > 1
    )
    if len(graph_sizes) > 1 and not is_effectively_ragged:
        limitations.append(
            "node count cannot form the requested unequal graph-size partition"
        )
    return graph_sizes, graph_edge_counts, {
        "requested_topology_variant": variant,
        "effective_topology_variant": (
            "ragged"
            if is_effectively_ragged
            else (
                "uniform_graph_size_fallback"
                if len(graph_sizes) > 1
                else "single_graph_capacity_fallback"
            )
        ),
        "topology_limitations": limitations,
        "preferred_ragged_graph_sizes": list(preferred),
        "ragged_partition_adjusted": partition_adjusted,
        "ragged_partition_collapsed": partition_collapsed,
        "per_graph_edge_counts": list(graph_edge_counts),
    }


def _preferred_ragged_sizes(num_nodes: int) -> tuple[int, ...]:
    if num_nodes == 1:
        return (1,)
    if num_nodes < 6:
        return (1, num_nodes - 1)
    small = max(1, num_nodes // 16)
    medium = max(2, num_nodes // 4)
    large = num_nodes - small - medium
    if large <= 0:
        return (1, num_nodes - 1)
    return (small, medium, large)


def _allocate_graph_edges(
    graph_sizes: Sequence[int],
    *,
    total_edges: int,
) -> tuple[int, ...]:
    minimum = sum(graph_sizes)
    maximum = sum(size * size for size in graph_sizes)
    if not minimum <= total_edges <= maximum:
        raise TopologyInfeasibleError(
            "ragged partition cannot realize the requested exact unique-edge count"
        )
    edge_counts = list(graph_sizes)
    remaining = total_edges - minimum
    for index in sorted(
        range(len(graph_sizes)),
        key=lambda item: (-graph_sizes[item], item),
    ):
        capacity = graph_sizes[index] * (graph_sizes[index] - 1)
        addition = min(remaining, capacity)
        edge_counts[index] += addition
        remaining -= addition
    if remaining:
        raise RuntimeError("unique-edge capacity allocation is incomplete")
    return tuple(edge_counts)


def _balanced_receiver_degrees(
    num_nodes: int,
    *,
    edge_count: int,
) -> tuple[int, ...]:
    if not num_nodes <= edge_count <= num_nodes * num_nodes:
        raise TopologyInfeasibleError(
            "graph edge count must lie between one self per node and full capacity"
        )
    extras, remainder = divmod(edge_count - num_nodes, num_nodes)
    degrees = tuple(
        1 + extras + (index < remainder)
        for index in range(num_nodes)
    )
    if any(degree > num_nodes for degree in degrees):
        raise RuntimeError("balanced receiver allocation exceeded unique capacity")
    return degrees


def _skew_receiver_degrees(
    num_nodes: int,
    *,
    edge_count: int,
) -> tuple[int, ...]:
    if not num_nodes <= edge_count <= num_nodes * num_nodes:
        raise TopologyInfeasibleError(
            "graph edge count must lie between one self per node and full capacity"
        )
    degrees = [1] * num_nodes
    remaining = edge_count - num_nodes
    for receiver in range(num_nodes):
        addition = min(remaining, num_nodes - 1)
        degrees[receiver] += addition
        remaining -= addition
    if remaining:
        raise RuntimeError("skew receiver allocation exceeded unique capacity")
    return tuple(degrees)


def _graph_edges(
    num_nodes: int,
    *,
    receiver_degrees: Sequence[int],
    node_offset: int,
) -> torch.Tensor:
    if len(receiver_degrees) != num_nodes:
        raise ValueError("receiver_degrees must contain one degree per node")
    degrees = torch.tensor(receiver_degrees, dtype=torch.long)
    if bool(((degrees < 1) | (degrees > num_nodes)).any().item()):
        raise TopologyInfeasibleError(
            "receiver degree exceeds simple directed graph capacity"
        )
    receiver = torch.repeat_interleave(
        torch.arange(num_nodes, dtype=torch.long),
        degrees,
        output_size=int(degrees.sum().item()),
    )
    row_start = torch.cumsum(degrees, dim=0) - degrees
    slot = torch.arange(receiver.numel(), dtype=torch.long) - (
        torch.repeat_interleave(
            row_start,
            degrees,
            output_size=receiver.numel(),
        )
    )
    # slot=0 is self. Slots 1..degree-1 are distinct cyclic nonself senders.
    sender = (receiver + slot) % num_nodes
    return torch.stack([receiver + node_offset, sender + node_offset])


def _assert_topology_postconditions(
    edge_index: torch.Tensor,
    *,
    batch_index: torch.Tensor,
    num_nodes: int,
    expected_edges: int,
) -> None:
    if edge_index.shape != (2, expected_edges):
        raise RuntimeError("synthetic graph violated the exact E=kN contract")
    receiver, sender = edge_index
    if not torch.equal(batch_index[receiver], batch_index[sender]):
        raise RuntimeError("synthetic graph contains cross-graph edges")
    pair_id = receiver * num_nodes + sender
    if torch.unique(pair_id).numel() != expected_edges:
        raise RuntimeError("synthetic graph contains duplicate directed edges")
    self_counts = torch.bincount(
        receiver[receiver == sender],
        minlength=num_nodes,
    )
    if not torch.equal(self_counts, torch.ones_like(self_counts)):
        raise RuntimeError(
            "synthetic graph must contain exactly one self edge for every node"
        )


def _begin_memory_measurement(device: torch.device) -> dict[str, object]:
    if device.type != "cuda":
        return {
            "cuda_peak_allocated_bytes": None,
            "cuda_peak_reserved_bytes": None,
            "cuda_peak_allocated_delta_bytes": None,
            "cuda_peak_reserved_delta_bytes": None,
            "cuda_memory_available": False,
        }
    _synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    return {
        "_cuda_baseline_allocated": torch.cuda.memory_allocated(device),
        "_cuda_baseline_reserved": torch.cuda.memory_reserved(device),
    }


def _end_memory_measurement(
    device: torch.device,
    memory: dict[str, object],
) -> dict[str, object]:
    if device.type != "cuda":
        return {}
    _synchronize(device)
    baseline_allocated = int(memory.pop("_cuda_baseline_allocated"))
    baseline_reserved = int(memory.pop("_cuda_baseline_reserved"))
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    return {
        "cuda_peak_allocated_bytes": peak_allocated,
        "cuda_peak_reserved_bytes": peak_reserved,
        "cuda_peak_allocated_delta_bytes": max(
            0,
            peak_allocated - baseline_allocated,
        ),
        "cuda_peak_reserved_delta_bytes": max(
            0,
            peak_reserved - baseline_reserved,
        ),
        "cuda_memory_available": True,
    }


def _timing_summary(samples_ms: Sequence[float]) -> dict[str, object]:
    return {
        "samples_ms": list(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "minimum_ms": min(samples_ms),
        "maximum_ms": max(samples_ms),
        "repeat_count": len(samples_ms),
    }


def _model_bytes(model: torch.nn.Module) -> dict[str, int]:
    parameter_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in model.parameters()
    )
    buffer_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in model.buffers()
    )
    state_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in model.state_dict().values()
        if isinstance(tensor, torch.Tensor)
    )
    return {
        "parameter_bytes": parameter_bytes,
        "buffer_bytes": buffer_bytes,
        "model_state_bytes": state_bytes,
    }


def _optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    return sum(
        value.numel() * value.element_size()
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )


def _batch_tensor_bytes(batch: GraphBatch) -> int:
    tensors: list[torch.Tensor] = [
        batch.node_feats,
        batch.pos,
        batch.batch,
        batch.target,
    ]
    for value in (
        batch.edge_index,
        batch.readout_mask,
        batch.edge_relation_id,
        batch.node_role_id,
        batch.hierarchy_id,
    ):
        if value is not None:
            tensors.append(value)
    if batch.node_masks is not None:
        tensors.extend(batch.node_masks.values())
    if batch.packed_neighbors is not None:
        packed = batch.packed_neighbors
        for value in (
            packed.row_ptr,
            packed.sender,
            packed.edge_order,
            packed.relation_id,
            packed.reverse_row_ptr,
            packed.reverse_edge_order,
            packed.degree,
            packed.degree_histogram,
            packed.degree_bucket,
            packed.ell_sender,
            packed.ell_mask,
        ):
            if value is not None:
                tensors.append(value)
    if batch.graph_layout is not None:
        layout = batch.graph_layout
        for value in (
            layout.batch,
            layout.graph_counts,
            layout.graph_ptr,
            layout.order,
            layout.inverse_order,
            layout.dense_index,
            layout.dense_mask,
        ):
            if value is not None:
                tensors.append(value)
        for bucket in layout.buckets:
            tensors.extend(
                (bucket.graph_index, bucket.node_index, bucket.mask)
            )
    seen: set[tuple[int, int]] = set()
    total = 0
    for tensor in tensors:
        identity = (tensor.untyped_storage().data_ptr(), tensor.storage_offset())
        if identity in seen:
            continue
        seen.add(identity)
        total += tensor.numel() * tensor.element_size()
    return total


def _environment_receipt(device: torch.device) -> dict[str, object]:
    return {
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "device_type": device.type,
        "gpu_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "cuda_available": torch.cuda.is_available(),
    }


def _skipped_row(
    *,
    name: str,
    case: SyntheticCase,
    status: str,
    reason: str,
) -> dict[str, object]:
    return {
        "status": status,
        "reason": reason,
        "arm": name,
        "nodes": case.metadata["nodes"],
        "degree": case.metadata["requested_degree"],
        "edges": case.metadata["supplied_edges"],
        "variant": case.metadata["variant"],
        "graph_identity_sha256": case.metadata["edge_index_sha256"],
        "edge_consumption": ARM_DEFINITIONS[name].edge_consumption,
        "graph": case.metadata,
        "graph_construction": {
            "synthetic_generation_shared_across_arms": True,
            "synthetic_generation_ms": case.construction_ms,
            "included_in_forward_timing": False,
            "included_in_train_step_timing": False,
            "neighbor_discovery_performed": False,
        },
    }


def _infeasible_topology_row(
    *,
    name: str,
    num_nodes: int,
    degree: int,
    variant: str,
    reason: str,
) -> dict[str, object]:
    return {
        "status": "skipped_infeasible_topology",
        "reason": reason,
        "arm": name,
        "nodes": num_nodes,
        "degree": degree,
        "edges": num_nodes * degree,
        "variant": variant,
        "graph_identity_sha256": None,
        "edge_consumption": ARM_DEFINITIONS[name].edge_consumption,
        "graph": {
            "variant": variant,
            "nodes": num_nodes,
            "requested_degree": degree,
            "expected_edges": num_nodes * degree,
            "supplied_edges": None,
            "exact_e_equals_k_n": False,
            "topology_feasible": False,
            "topology_limitations": [reason],
        },
        "graph_construction": {
            "included_in_forward_timing": False,
            "included_in_train_step_timing": False,
            "neighbor_discovery_performed": False,
        },
    }


def _nested_median(row: dict[str, object], field: str) -> float:
    nested = row[field]
    if not isinstance(nested, dict):
        raise TypeError(f"{field} must be a timing mapping")
    return float(nested["median_ms"])


def _hidden_scalar_width(config: ArchitectureConfig) -> int:
    legacy = config.to_legacy()
    return CartesianIrreps.parse(legacy.hidden_irreps).scalars


def _global_transport_widths(
    config: ArchitectureConfig,
) -> tuple[int, int]:
    """Mirror the live factorized feature/value payload dimensions."""

    head_dim = _hidden_scalar_width(config) // config.num_heads
    query_scalar_width = head_dim
    if config.representation.use_tensor_product_kernel:
        # Constant plus flattened 3x3 symmetric-traceless matrix.
        query_scalar_width += 10
    if config.representation.use_quartic_kernel:
        # Symmetric degree-four map of one three-vector.
        query_scalar_width += math.comb(3 + 4 - 1, 4)
    angular_width = 3 * config.representation.angular_bandwidth
    feature_width = (
        query_scalar_width
        + 1
        + angular_width
        + angular_width * (angular_width + 1) // 2
    )
    # `_quadratic_gaussian_spatial_features` has ten polynomial coordinates.
    # Static multiscale uses one scale per head, so each head still sees ten
    # features. Adaptive multiscale concatenates four scales per head.
    spatial_base_width = 10
    if config.global_transport.use_multiscale_spatial_kernel:
        feature_width += spatial_base_width
    if config.global_transport.use_adaptive_multiscale_spatial_kernel:
        feature_width += spatial_base_width * 4

    value_width = head_dim + 16
    if config.global_transport.use_radial_trace:
        value_width += 5
    if config.global_transport.use_global_tensor_value_transport:
        value_width += 5
    return feature_width, value_width


def _count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _tensor_sha256(value: torch.Tensor) -> str:
    materialized = (
        value.detach().to(device="cpu").contiguous().numpy().tobytes()
    )
    return hashlib.sha256(materialized).hexdigest()


def _require_finite_tensor(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"{name} is non-finite")


def _is_oom_error(error: BaseException) -> bool:
    if isinstance(error, (MemoryError, torch.cuda.OutOfMemoryError)):
        return True
    message = str(error).lower()
    return "out of memory" in message and (
        "cuda" in message or "memory" in message
    )


def _bounded_error_message(error: BaseException, limit: int = 500) -> str:
    message = " ".join(str(error).split())
    return message[:limit]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _resolve_dtype(value: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    try:
        return mapping[value]
    except KeyError as error:
        raise ValueError(f"unsupported dtype: {value}") from error


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_arm_names(names: Sequence[str]) -> None:
    if not names:
        raise ValueError("at least one arm is required")
    if len(set(names)) != len(names):
        raise ValueError("arm names must be unique")
    unknown = [name for name in names if name not in ARM_DEFINITIONS]
    if unknown:
        raise ValueError(f"unknown arms: {', '.join(unknown)}")


def _validate_run_request(
    *,
    nodes: Sequence[int],
    degrees: Sequence[int],
    variants: Sequence[str],
    arms: Sequence[str],
    node_dim: int,
    width: int,
    num_heads: int,
    workspace_channels: int,
    warmup: int,
    repeats: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    max_wall_seconds: float,
    threads: int | None,
) -> None:
    if not nodes or not degrees or not variants:
        raise ValueError("nodes, degrees, and variants must not be empty")
    for value in nodes:
        _positive_integer("nodes", value)
    for value in degrees:
        _positive_integer("degrees", value)
    if any(variant not in GRAPH_VARIANTS for variant in variants):
        raise ValueError(f"variants must be chosen from: {', '.join(GRAPH_VARIANTS)}")
    if len(set(variants)) != len(variants):
        raise ValueError("variants must be unique")
    _validate_arm_names(arms)
    for name, value in (
        ("node_dim", node_dim),
        ("width", width),
        ("num_heads", num_heads),
        ("workspace_channels", workspace_channels),
        ("repeats", repeats),
    ):
        _positive_integer(name, value)
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not math.isfinite(max_wall_seconds) or max_wall_seconds < 0.0:
        raise ValueError("max_wall_seconds must be finite and nonnegative")
    if threads is not None:
        _positive_integer("threads", threads)


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item) for item in value.split(",") if item)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return parsed


def _parse_csv_strings(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("expected a nonempty comma-separated list")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark generic 3D architecture mechanics without evaluating accuracy."
        )
    )
    parser.add_argument("--nodes", type=_parse_csv_ints, default=DEFAULT_NODES)
    parser.add_argument("--degrees", type=_parse_csv_ints, default=DEFAULT_DEGREES)
    parser.add_argument(
        "--variants",
        type=_parse_csv_strings,
        default=DEFAULT_VARIANTS,
    )
    parser.add_argument(
        "--arms",
        type=_parse_csv_strings,
        default=REQUIRED_ARMS,
    )
    parser.add_argument("--include-standard", action="store_true")
    parser.add_argument("--node-dim", type=int, default=8)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--workspace-channels", type=int, default=1)
    parser.add_argument(
        "--parameter-match",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--parameter-search-max-width", type=int)
    parser.add_argument(
        "--global-backend",
        choices=("outer_scatter", "feature_gemm", "auto"),
        default="auto",
    )
    parser.add_argument(
        "--local-backend",
        choices=(
            "materialized",
            "segment_csr",
            "streamed_csr",
            "ell",
            "custom",
            "auto",
        ),
        default="auto",
    )
    parser.add_argument(
        "--geometry-cache-mode",
        choices=("full", "compact", "recompute", "auto"),
        default="auto",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32", "float64"),
        default="float32",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--max-wall-seconds", type=float, default=120.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    arms = tuple(args.arms)
    if args.include_standard and "standard" not in arms:
        arms = (*arms, "standard")
    result = run_architecture_matrix(
        nodes=args.nodes,
        degrees=args.degrees,
        variants=args.variants,
        arms=arms,
        node_dim=args.node_dim,
        width=args.width,
        num_heads=args.num_heads,
        workspace_channels=args.workspace_channels,
        parameter_match=args.parameter_match,
        parameter_search_max_width=args.parameter_search_max_width,
        global_backend=args.global_backend,
        local_backend=args.local_backend,
        geometry_cache_mode=args.geometry_cache_mode,
        device=args.device,
        dtype=args.dtype,
        warmup=args.warmup,
        repeats=args.repeats,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        max_wall_seconds=args.max_wall_seconds,
        threads=args.threads,
    )
    encoded = json.dumps(
        result,
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    if args.output is None:
        print(encoded)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
