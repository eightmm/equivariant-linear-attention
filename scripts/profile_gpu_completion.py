#!/usr/bin/env python3
"""Bounded CUDA evidence for the ELA completion performance lanes.

The profiler deliberately keeps each scope explicit.  Public ingestion lanes
use ``ELAGraph -> ELA -> ELAGraph``; kernel-only lanes import private helpers and
label that fact in the JSON report.  It never interprets an internal argsort as
having disappeared merely because the explicit COO-to-CSR sort was avoided.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any
from unittest.mock import patch
import warnings

import torch
import torch.nn.functional as F

from equivariant_linear_attention import ELA, ELAGraph
from equivariant_linear_attention.batch import ELABatch
from equivariant_linear_attention.geometry.layout import pack_graph_layout
from equivariant_linear_attention.geometry.radius import radius_graph
from equivariant_linear_attention.inference import (
    _CompiledCoreInferenceModule,
    prepare_for_inference,
)
from equivariant_linear_attention.kernels import kernel_backend, triton_available
from equivariant_linear_attention.nn.parity import (
    _can_use_grouped_mm,
    _exact_balanced_attention,
    _grouped_mm_feature_gemm,
    _layout_feature_gemm,
    _segmented_feature_gemm,
)


SCHEMA_VERSION = 1
LANE_NAMES = (
    "trusted_prepared_cache",
    "ragged_global",
    "radius_ingestion",
    "triton_training_local",
    "compiled_numerical_core",
)
RAGGED_RELATIVE_L2_MAX = 0.05
RAGGED_LATENCY_RATIO_MAX = 1.0
RADIUS_ERROR_MAX = 1.0e-5
TRITON_RELATIVE_L2_MAX = 1.0e-3
TRITON_ABSOLUTE_ERROR_MAX = 5.0e-3
COMPILED_ABSOLUTE_ERROR_MAX = 1.0e-3
COMPILED_LATENCY_RATIO_MAX = 1.0
PEAK_ALLOCATED_LIMIT_BYTES = 16 * 1024**3


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or value <= 0:
        raise argparse.ArgumentTypeError(f"{name} must be a positive integer")
    return value


def _counts(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "ragged counts must be comma-separated integers"
        ) from error
    if len(parsed) < 2 or any(count <= 0 for count in parsed):
        raise argparse.ArgumentTypeError(
            "ragged counts must contain at least two positive integers"
        )
    return parsed


def _percentile(samples: Sequence[float], quantile: float) -> float:
    if not samples:
        raise ValueError("at least one sample is required")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in samples)
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _timing_summary(samples: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in samples]
    return {
        "median_ms": statistics.median(values),
        "p95_ms": _percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
        "samples_ms": values,
    }


def _sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _measure_cuda_lanes(
    functions: Mapping[str, Callable[[], object]],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
    setups: Mapping[str, Callable[[], None]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Measure alternating CUDA lanes with raw wall/event samples and peaks."""

    if device.type != "cuda":
        raise ValueError("CUDA timing requires a CUDA device")
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be nonnegative and repeats must be positive")
    lane_setups = {} if setups is None else dict(setups)
    unknown_setups = set(lane_setups) - set(functions)
    if unknown_setups:
        raise ValueError(f"setup supplied for unknown lanes: {sorted(unknown_setups)}")
    for name, function in functions.items():
        for _ in range(warmup):
            setup = lane_setups.get(name)
            if setup is not None:
                setup()
            result = function()
            del result
    _sync(device)

    wall = {name: [] for name in functions}
    event = {name: [] for name in functions}
    peak = {name: 0 for name in functions}
    incremental = {name: 0 for name in functions}
    names = tuple(functions)
    for repeat in range(repeats):
        order = names if repeat % 2 == 0 else tuple(reversed(names))
        for name in order:
            setup = lane_setups.get(name)
            if setup is not None:
                setup()
            _sync(device)
            torch.cuda.reset_peak_memory_stats(device)
            baseline = torch.cuda.memory_allocated(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            wall_start = time.perf_counter()
            start_event.record()
            result = functions[name]()
            end_event.record()
            _sync(device)
            wall[name].append((time.perf_counter() - wall_start) * 1000.0)
            event[name].append(float(start_event.elapsed_time(end_event)))
            observed = int(torch.cuda.max_memory_allocated(device))
            peak[name] = max(peak[name], observed)
            incremental[name] = max(incremental[name], max(0, observed - baseline))
            del result

    return {
        name: {
            "wall": _timing_summary(wall[name]),
            "cuda_event": _timing_summary(event[name]),
            "peak_allocated_bytes": peak[name],
            "peak_incremental_allocated_bytes": incremental[name],
        }
        for name in names
    }


def _measure_one_cuda(
    function: Callable[[], object],
    *,
    device: torch.device,
) -> dict[str, Any]:
    return _measure_cuda_lanes(
        {"sample": function},
        device=device,
        warmup=0,
        repeats=1,
    )["sample"]


def _maximum_recorded_peak_allocated_bytes(value: object) -> int | None:
    """Return the largest timed absolute CUDA allocation in a nested receipt."""

    peaks: list[int] = []

    def visit(item: object) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if key == "peak_allocated_bytes":
                    if isinstance(child, bool) or not isinstance(child, int):
                        raise TypeError("peak_allocated_bytes must be an integer")
                    peaks.append(child)
                else:
                    visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return max(peaks) if peaks else None


def _attach_resource_gate(lane: dict[str, Any]) -> bool:
    """Attach the preregistered absolute-memory gate to one completed lane."""

    peak = _maximum_recorded_peak_allocated_bytes(lane.get("measurements"))
    if peak is None:
        raise RuntimeError("completed CUDA lane did not record peak allocation")
    passed = peak < PEAK_ALLOCATED_LIMIT_BYTES
    lane["resource_gate"] = {
        "passed": passed,
        "thresholds": {
            "peak_allocated_bytes_lt": PEAK_ALLOCATED_LIMIT_BYTES,
            "peak_allocated_gib_lt": 16,
        },
        "observed": {
            "maximum_peak_allocated_bytes": peak,
            "maximum_peak_allocated_gib": peak / 1024**3,
        },
    }
    return passed


def _fixed_degree_edges(
    nodes: int,
    degree: int,
    device: torch.device,
    *,
    shift: int = 0,
) -> torch.Tensor:
    if not 0 < degree < nodes:
        raise ValueError("degree must satisfy 0 < degree < nodes")
    receiver = torch.arange(nodes, device=device).repeat_interleave(degree)
    offset = torch.arange(1, degree + 1, device=device).repeat(nodes)
    sender = (receiver + offset + shift) % nodes
    return torch.stack((receiver, sender))


def _grid_positions(
    nodes: int,
    *,
    spacing: float,
    device: torch.device,
) -> torch.Tensor:
    side = max(1, math.ceil(nodes ** (1.0 / 3.0)))
    index = torch.arange(nodes, device=device, dtype=torch.long)
    return (
        torch.stack(
            (
                index % side,
                (index // side) % side,
                index // (side * side),
            ),
            dim=-1,
        ).to(dtype=torch.float32)
        * spacing
    )


def _batched_grid(
    graphs: int,
    nodes_per_graph: int,
    *,
    cutoff: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    local = _grid_positions(
        nodes_per_graph,
        spacing=0.72 * cutoff,
        device=device,
    )
    positions = local.repeat(graphs, 1)
    batch = torch.arange(graphs, device=device).repeat_interleave(nodes_per_graph)
    return positions, batch


def _activate_local_outputs(model: ELA) -> None:
    with torch.no_grad():
        for layer in model.layers:
            for name in (
                "local_scalar_out",
                "local_odd_out",
                "local_polar_out",
                "local_axial_out",
                "local_even_tensor_out",
                "local_odd_tensor_out",
                "local_mass_out",
            ):
                module = getattr(layer, name)
                weight = getattr(module, "weight", None)
                if weight is not None:
                    weight.normal_(mean=0.0, std=0.02)


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0:
        return 0.0
    return float((left.float() - right.float()).abs().max().item())


def _relative_l2(left: torch.Tensor, right: torch.Tensor) -> float:
    difference = (left.float() - right.float()).double()
    denominator = right.float().double().square().sum().sqrt().clamp_min(1e-30)
    return float((difference.square().sum().sqrt() / denominator).item())


def _canonical_topology_codes(edge_index: torch.Tensor, nodes: int) -> torch.Tensor:
    """Canonicalize private receiver/sender COO for topology-only comparison."""

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2,E)")
    receiver, sender = edge_index.to(dtype=torch.long)
    return torch.sort(receiver * nodes + sender).values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_source_manifest(path: Path, repository: Path) -> dict[str, Any]:
    """Bind a dirty worktree run to a manifest and recheck every source byte."""

    repository = repository.resolve()
    manifest_path = path.resolve()
    try:
        relative_manifest = manifest_path.relative_to(repository).as_posix()
    except ValueError as error:
        raise ValueError("source manifest must stay inside the repository") from error
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload.get("files")
    expected_combined = payload.get("combined_sha256")
    if not isinstance(records, list) or not isinstance(expected_combined, str):
        raise ValueError("source manifest is missing files or combined_sha256")
    combined = hashlib.sha256()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("source manifest file records must be objects")
        relative = record.get("path")
        size = record.get("size_bytes")
        expected_sha = record.get("sha256")
        if (
            not isinstance(relative, str)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or not isinstance(expected_sha, str)
        ):
            raise ValueError("source manifest file record is malformed")
        source = (repository / relative).resolve()
        try:
            source.relative_to(repository)
        except ValueError as error:
            raise ValueError("source manifest path escapes the repository") from error
        actual_sha = _sha256(source)
        if source.stat().st_size != size or actual_sha != expected_sha:
            raise ValueError(f"source manifest mismatch: {relative}")
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(str(size).encode("ascii"))
        combined.update(b"\0")
        combined.update(actual_sha.encode("ascii"))
        combined.update(b"\0")
    actual_combined = combined.hexdigest()
    if actual_combined != expected_combined:
        raise ValueError("source manifest combined hash mismatch")
    return {
        "path": relative_manifest,
        "manifest_sha256": _sha256(manifest_path),
        "combined_sha256": actual_combined,
        "file_count": len(records),
        "verified_against_current_bytes": True,
    }


def _profile_trusted_cache(
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    nodes = args.cache_graphs * args.cache_nodes_per_graph
    positions, batch = _batched_grid(
        args.cache_graphs,
        args.cache_nodes_per_graph,
        cutoff=args.radius_cutoff,
        device=device,
    )
    model = (
        ELA(
            "8x0e",
            width=32,
            depth=2,
            cutoff=args.radius_cutoff,
            max_neighbors=args.radius_max_neighbors,
        )
        .to(device)
        .eval()
    )
    _activate_local_outputs(model)
    features = torch.randn(nodes, 8, device=device)
    local_edges = _fixed_degree_edges(
        args.cache_nodes_per_graph,
        min(8, args.cache_nodes_per_graph - 1),
        device,
    )
    internal_edges = torch.cat(
        [
            local_edges + graph_index * args.cache_nodes_per_graph
            for graph_index in range(args.cache_graphs)
        ],
        dim=1,
    )
    graph = ELAGraph(
        features,
        positions,
        batch=batch,
        edge_index=internal_edges.flip(0).contiguous(),
    ).assume_immutable()
    with torch.inference_mode():
        model(graph)
    first = graph._prepared_graph
    provenance = graph._prepared_provenance
    if first is None:
        raise RuntimeError("public radius execution did not attach a prepared cache")
    packed = graph._to_packed()
    immutable_trusted_prepared_admitted = packed._trusted_prepared
    packed_template_identity_reused = packed is graph._packed_template
    validated = model._prepare_packed(packed)
    trusted_cache_reused = validated._prepared_graph is first
    with torch.inference_mode():
        model(graph)
    identity_reused = graph._prepared_graph is first
    immutable_provenance_active = (
        provenance is not None and graph._prepared_provenance is provenance
    )

    probe = ELAGraph(
        torch.randn(3, 8, device=device),
        torch.randn(3, 3, device=device),
        edge_index=torch.tensor([[0, 1], [1, 0]], device=device),
    )
    probe = model._prepare_graph(probe)
    probe_first = probe._prepared_graph
    if probe_first is None or probe.edge_index is None:
        raise RuntimeError("DLPack invalidation probe was not prepared")
    alias = torch.from_dlpack(probe.edge_index)
    alias.copy_(torch.tensor([[0, 2], [2, 0]], device=device))
    probe_reprepared = model._prepare_graph(probe)
    probe_second = probe_reprepared._prepared_graph
    unsealed_dlpack_alias_invalidates_cache = (
        probe_second is not None
        and probe_second is not probe_first
        and torch.equal(
            probe_second.neighbors.original_edge_index().long(),
            probe.edge_index[[1, 0]].long(),
        )
    )

    def reuse() -> torch.Tensor:
        with torch.inference_mode():
            return model(graph).x

    measurements = _measure_cuda_lanes(
        {"prepared_cache_reuse": reuse},
        device=device,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    identity_after_measurement = graph._prepared_graph is first
    if not (
        immutable_trusted_prepared_admitted
        and packed_template_identity_reused
        and trusted_cache_reused
        and identity_reused
        and immutable_provenance_active
        and identity_after_measurement
        and unsealed_dlpack_alias_invalidates_cache
    ):
        raise RuntimeError(
            "safe/trusted CUDA prepared-cache contract was not preserved"
        )
    gate = {
        "passed": True,
        "thresholds": {
            "prepared_object_identity_reused": True,
            "immutable_trusted_prepared_admitted": True,
            "packed_template_identity_reused": True,
            "unsealed_dlpack_alias_invalidates_cache": True,
        },
        "observed": {
            "prepared_object_identity_reused": identity_reused,
            "immutable_trusted_prepared_admitted": (
                immutable_trusted_prepared_admitted and trusted_cache_reused
            ),
            "packed_template_identity_reused": packed_template_identity_reused,
            "unsealed_dlpack_alias_invalidates_cache": (
                unsealed_dlpack_alias_invalidates_cache
            ),
        },
    }
    return {
        "status": "completed",
        "scope": (
            "explicit immutable ELAGraph O(1) packed-template reuse plus safe unsealed "
            "DLPack invalidation"
        ),
        "nodes": nodes,
        "graphs": args.cache_graphs,
        "cache_source": first.spec.source,
        "immutable_trusted_prepared_admitted": immutable_trusted_prepared_admitted,
        "packed_template_identity_reused": packed_template_identity_reused,
        "prepared_object_identity_reused": identity_reused,
        "immutable_provenance_active": immutable_provenance_active,
        "unsealed_dlpack_alias_invalidates_cache": (
            unsealed_dlpack_alias_invalidates_cache
        ),
        "identity_preserved_through_measurement": identity_after_measurement,
        "excluded_setup": "one untimed public radius preparation",
        "gate": gate,
        "measurements": measurements,
    }


def _profile_ragged_global(
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is unavailable")
    if not callable(getattr(F, "grouped_mm", None)):
        raise RuntimeError("torch.nn.functional.grouped_mm is unavailable")
    counts = torch.tensor(args.ragged_counts, device=device, dtype=torch.long)
    batch = torch.repeat_interleave(
        torch.arange(counts.numel(), device=device),
        counts,
        output_size=sum(args.ragged_counts),
    )
    layout = pack_graph_layout(
        batch,
        graph_counts=counts,
        assume_grouped=True,
        maximum_buckets=1,
        minimum_extreme_graphs=1_000_000,
    )
    if layout.structure != "ragged":
        raise RuntimeError(f"expected ragged layout, got {layout.structure}")
    shape = (layout.num_nodes, args.ragged_heads, args.ragged_feature_width)
    query = torch.rand(shape, device=device, dtype=torch.bfloat16).add_(0.25)
    key = torch.rand_like(query).add_(0.25)
    value = torch.randn(
        layout.num_nodes,
        args.ragged_heads,
        args.ragged_value_width,
        device=device,
        dtype=torch.bfloat16,
    )
    reference_query = query.detach().requires_grad_(True)
    reference_key = key.detach().requires_grad_(True)
    reference_value = value.detach().requires_grad_(True)
    with torch.inference_mode():
        native_path_selected = _can_use_grouped_mm(query, key, value)
        if not native_path_selected:
            raise RuntimeError("native grouped-MM dispatch contract is not active")
        native_probe = _grouped_mm_feature_gemm(query, key, value, layout)
        dispatched_probe = _layout_feature_gemm(query, key, value, layout)
        reference_probe = _segmented_feature_gemm(query, key, value, layout)
    error = _max_abs(native_probe, reference_probe)
    relative_error = _relative_l2(native_probe, reference_probe)
    dispatch_error = _max_abs(dispatched_probe, native_probe)
    del native_probe, dispatched_probe, reference_probe

    import equivariant_linear_attention.nn.parity as parity_module

    with patch.object(parity_module, "_can_use_grouped_mm", return_value=False):
        with torch.inference_mode():
            balanced_reference = _exact_balanced_attention(
                query,
                key,
                value,
                batch,
                layout,
                eps=1.0e-6,
            )
    integrated_calls = 0
    original_grouped_mm = parity_module._grouped_mm_feature_gemm

    def observed_grouped_mm(*values: object, **options: object) -> torch.Tensor:
        nonlocal integrated_calls
        integrated_calls += 1
        return original_grouped_mm(*values, **options)

    with patch.object(
        parity_module,
        "_grouped_mm_feature_gemm",
        side_effect=observed_grouped_mm,
    ):
        with torch.inference_mode():
            balanced_native = _exact_balanced_attention(
                query,
                key,
                value,
                batch,
                layout,
                eps=1.0e-6,
            )
    balanced_error = _max_abs(balanced_native, balanced_reference)
    balanced_relative_error = _relative_l2(
        balanced_native,
        balanced_reference,
    )
    del balanced_native, balanced_reference

    def native() -> torch.Tensor:
        with torch.inference_mode():
            return _grouped_mm_feature_gemm(query, key, value, layout)

    def segmented_inference() -> torch.Tensor:
        with torch.inference_mode():
            return _segmented_feature_gemm(query, key, value, layout)

    def differentiable_segmented_fwd_bwd() -> tuple[torch.Tensor, ...]:
        with torch.enable_grad():
            output = _segmented_feature_gemm(
                reference_query,
                reference_key,
                reference_value,
                layout,
            )
            return torch.autograd.grad(
                output.float().square().mean(),
                (reference_query, reference_key, reference_value),
            )

    inference_measurements = _measure_cuda_lanes(
        {
            "native_grouped_mm_inference": native,
            "tiled_segmented_inference": segmented_inference,
        },
        device=device,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    training_measurement = _measure_cuda_lanes(
        {
            "tiled_segmented_training_fwd_bwd": (differentiable_segmented_fwd_bwd),
        },
        device=device,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    measurements = {**inference_measurements, **training_measurement}
    native_median = measurements["native_grouped_mm_inference"]["cuda_event"][
        "median_ms"
    ]
    reference_median = measurements["tiled_segmented_inference"]["cuda_event"][
        "median_ms"
    ]
    gate = {
        "passed": (
            integrated_calls == 1
            and dispatch_error == 0.0
            and relative_error <= RAGGED_RELATIVE_L2_MAX
            and balanced_relative_error <= RAGGED_RELATIVE_L2_MAX
            and native_median <= reference_median * RAGGED_LATENCY_RATIO_MAX
        ),
        "thresholds": {
            "transport_relative_l2_max": RAGGED_RELATIVE_L2_MAX,
            "balanced_relative_l2_max": RAGGED_RELATIVE_L2_MAX,
            "native_over_segmented_inference_median_max": (RAGGED_LATENCY_RATIO_MAX),
            "integrated_native_calls_exact": 1,
            "dispatch_vs_direct_max_abs": 0.0,
        },
        "observed": {
            "transport_relative_l2": relative_error,
            "balanced_relative_l2": balanced_relative_error,
            "native_over_segmented_inference_median": (
                native_median / reference_median
            ),
            "integrated_native_calls": integrated_calls,
            "dispatch_vs_direct_max_abs": dispatch_error,
        },
    }
    return {
        "status": "completed",
        "scope": "private global feature-transport kernels",
        "layout": layout.structure,
        "counts": list(args.ragged_counts),
        "nodes": layout.num_nodes,
        "dtype": "bfloat16",
        "heads": args.ragged_heads,
        "feature_width": args.ragged_feature_width,
        "value_width": args.ragged_value_width,
        "native_regime": "CUDA BF16 inference with torch.nn.functional.grouped_mm",
        "inference_reference_regime": (
            "same BF16 inference-mode inputs and synchronization as native"
        ),
        "training_reference_regime": (
            "grad-enabled tiled segmented forward plus first backward"
        ),
        "dispatch_native_path_selected": native_path_selected,
        "dispatch_vs_direct_native_max_abs_error": dispatch_error,
        "integrated_balanced_attention_native_calls": integrated_calls,
        "integrated_balanced_attention_max_abs_error": balanced_error,
        "integrated_balanced_attention_relative_l2_error": (balanced_relative_error),
        "fallback_reference": "tiled_segmented_inference",
        "numerical_max_abs_error": error,
        "numerical_relative_l2_error": relative_error,
        "gate": gate,
        "measurements": measurements,
    }


def _profile_radius_ingestion(
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    nodes = args.radius_graphs * args.radius_nodes_per_graph
    positions, batch = _batched_grid(
        args.radius_graphs,
        args.radius_nodes_per_graph,
        cutoff=args.radius_cutoff,
        device=device,
    )
    features = torch.randn(nodes, 8, device=device)
    model = (
        ELA(
            "8x0e",
            width=32,
            depth=2,
            cutoff=args.radius_cutoff,
            max_neighbors=args.radius_max_neighbors,
        )
        .to(device)
        .eval()
    )
    _activate_local_outputs(model)

    automatic_probe = model._prepare_graph(ELAGraph(features, positions, batch=batch))
    internal_coo = radius_graph(
        positions,
        cutoff=args.radius_cutoff,
        batch=batch,
        max_neighbors=args.radius_max_neighbors,
    )
    public_coo = internal_coo.flip(0).contiguous()
    explicit_probe = model._prepare_graph(
        ELAGraph(features, positions, edge_index=public_coo, batch=batch)
    )
    automatic_prepared = automatic_probe._prepared_graph
    explicit_prepared = explicit_probe._prepared_graph
    if automatic_prepared is None or explicit_prepared is None:
        raise RuntimeError("topology probes were not prepared")
    automatic_edges = automatic_prepared.neighbors.original_edge_index().long()
    explicit_edges = explicit_prepared.neighbors.original_edge_index().long()
    automatic_codes = _canonical_topology_codes(automatic_edges, nodes)
    explicit_codes = _canonical_topology_codes(explicit_edges, nodes)
    topology_equal = torch.equal(automatic_codes, explicit_codes)
    if not topology_equal:
        raise RuntimeError("automatic direct CSR and explicit COO topology differ")
    with torch.inference_mode():
        automatic_output = model(automatic_probe).x
        explicit_output = model(explicit_probe).x
    output_max_abs_error = _max_abs(automatic_output, explicit_output)
    output_relative_l2_error = _relative_l2(automatic_output, explicit_output)
    gate = {
        "passed": (
            topology_equal
            and output_max_abs_error <= RADIUS_ERROR_MAX
            and output_relative_l2_error <= RADIUS_ERROR_MAX
        ),
        "thresholds": _functional_thresholds("radius_ingestion"),
        "observed": {
            "topology_equal": topology_equal,
            "output_max_abs_error": output_max_abs_error,
            "output_relative_l2_error": output_relative_l2_error,
        },
    }
    del automatic_output, explicit_output
    del automatic_probe, explicit_probe, automatic_prepared, explicit_prepared
    del automatic_codes, explicit_codes

    def automatic_public_ingestion() -> torch.Tensor:
        graph = ELAGraph(features, positions, batch=batch)
        with torch.inference_mode():
            return model(graph).x

    def explicit_public_ingestion() -> torch.Tensor:
        graph = ELAGraph(features, positions, edge_index=public_coo, batch=batch)
        with torch.inference_mode():
            return model(graph).x

    measurements = _measure_cuda_lanes(
        {
            "automatic_radius_direct_csr": automatic_public_ingestion,
            "explicit_coo_to_csr_reference": explicit_public_ingestion,
        },
        device=device,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    return {
        "status": "completed",
        "scope": "public cold ELAGraph -> ELA -> ELAGraph graph ingestion and execution",
        "nodes": nodes,
        "graphs": args.radius_graphs,
        "edges": int(automatic_edges.shape[1]),
        "cutoff": args.radius_cutoff,
        "max_neighbors": args.radius_max_neighbors,
        "topology_parity": {
            "equal_after_receiver_sender_canonicalization": topology_equal,
            "automatic_edges": int(automatic_edges.shape[1]),
            "explicit_edges": int(explicit_edges.shape[1]),
        },
        "output_parity": {
            "max_abs_error": output_max_abs_error,
            "relative_l2_error": output_relative_l2_error,
        },
        "automatic_argsort_scope": {
            "explicit_coo_to_csr_sort_included": False,
            "radius_internal_argsorts_may_execute": True,
            "all_argsorts_removed": False,
            "note": (
                "direct receiver-major CSR avoids a separate explicit COO-to-CSR "
                "conversion; radius cell ordering, shell limiting, or receiver "
                "grouping may still sort"
            ),
        },
        "explicit_argsort_scope": {
            "explicit_coo_to_csr_sort_included": True,
            "radius_discovery_in_timed_region": False,
            "reference_coo_built_untimed": True,
        },
        "gate": gate,
        "measurements": measurements,
    }


def _training_batch(
    template: ELABatch,
    features: torch.Tensor,
    positions: torch.Tensor,
) -> ELABatch:
    if template._prepared_graph is None:
        raise RuntimeError("training template must be prepared")
    return ELABatch(
        node_irreps=features,
        positions=positions,
        ptr=template.ptr,
        edge_index=template.edge_index,
        edge_relation_id=template.edge_relation_id,
        _prepared_graph=template._prepared_graph,
        _trusted_prepared=True,
    )


def _training_step(
    model: ELA,
    batch: ELABatch,
    *,
    backend: str,
) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    batch.node_irreps.grad = None
    batch.positions.grad = None
    with kernel_backend(backend):
        output = model._forward_prepared(batch)["node_irreps"]
        output.float().square().mean().backward()
    return output


def _training_probe(
    model: ELA,
    batch: ELABatch,
    *,
    backend: str,
) -> dict[str, Any]:
    output = _training_step(model, batch, backend=backend)
    if batch.node_irreps.grad is None or batch.positions.grad is None:
        raise RuntimeError("training probe input gradients are missing")
    parameter_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return {
        "output": output.detach().clone(),
        "feature_gradient": batch.node_irreps.grad.detach().clone(),
        "position_gradient": batch.positions.grad.detach().clone(),
        "parameter_gradients": parameter_gradients,
    }


def _parameter_gradient_error(
    candidate: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    candidate_names = set(candidate)
    reference_names = set(reference)
    common = sorted(candidate_names & reference_names)
    if not common:
        raise RuntimeError("no common parameter gradients were produced")
    maximum = max(_max_abs(candidate[name], reference[name]) for name in common)
    difference_square = sum(
        float(
            (candidate[name].float() - reference[name].float())
            .double()
            .square()
            .sum()
            .item()
        )
        for name in common
    )
    reference_square = sum(
        float(reference[name].float().double().square().sum().item()) for name in common
    )
    relative = math.sqrt(difference_square) / max(math.sqrt(reference_square), 1e-30)
    return {
        "max_abs": maximum,
        "relative_l2": relative,
        "compared_parameter_tensors": len(common),
        "candidate_only": sorted(candidate_names - reference_names),
        "reference_only": sorted(reference_names - candidate_names),
    }


def _expected_triton_dispatch(depth: int) -> dict[str, list[list[int]]]:
    return {
        "weighted_pair_gate_lanes": [[0, 1], [2, 3]] * depth,
        "tensor_pair_gate_lanes": [[4, 5]] * depth,
        "direction_triple_gate_lanes": [[6, 7, 8]] * depth,
    }


def _profile_triton_training(
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    if not triton_available():
        raise RuntimeError("Triton runtime is unavailable")
    model = (
        ELA(
            "16x0e + 2x1o",
            "1x0e + 1x1o",
            width=args.training_width,
            depth=args.training_depth,
            cutoff=8.0,
        )
        .to(device)
        .train()
    )
    _activate_local_outputs(model)
    input_dim = model.config.input_layout.dim
    features = torch.randn(
        args.training_nodes,
        input_dim,
        device=device,
        requires_grad=True,
    )
    positions = torch.randn(
        args.training_nodes,
        3,
        device=device,
        requires_grad=True,
    )
    edges = _fixed_degree_edges(
        args.training_nodes,
        args.training_degree,
        device,
    )
    template = model._prepare_packed(
        ELABatch(features.detach(), positions.detach(), edge_index=edges)
    )
    batch = _training_batch(template, features, positions)

    reference = _training_probe(model, batch, backend="torch")

    import equivariant_linear_attention.kernels.local as local_module

    original_pair = local_module._trusted_weighted_gather_reduce_pair
    original_tensor = local_module._trusted_local_tensor_reduce_pair
    original_direction = local_module._trusted_direction_reduce_triple
    observed_dispatch: dict[str, list[list[int]]] = {
        "weighted_pair_gate_lanes": [],
        "tensor_pair_gate_lanes": [],
        "direction_triple_gate_lanes": [],
    }

    def observe_pair(
        *values: object,
        gate_lanes: tuple[int, int],
        **options: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observed_dispatch["weighted_pair_gate_lanes"].append(list(gate_lanes))
        return original_pair(*values, gate_lanes=gate_lanes, **options)

    def observe_tensor(
        *values: object,
        gate_lanes: tuple[int, int],
        **options: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observed_dispatch["tensor_pair_gate_lanes"].append(list(gate_lanes))
        return original_tensor(*values, gate_lanes=gate_lanes, **options)

    def observe_direction(
        *values: object,
        gate_lanes: tuple[int, int, int],
        **options: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observed_dispatch["direction_triple_gate_lanes"].append(list(gate_lanes))
        return original_direction(*values, gate_lanes=gate_lanes, **options)

    with (
        patch.object(
            local_module,
            "_trusted_weighted_gather_reduce_pair",
            side_effect=observe_pair,
        ),
        patch.object(
            local_module,
            "_trusted_local_tensor_reduce_pair",
            side_effect=observe_tensor,
        ),
        patch.object(
            local_module,
            "_trusted_direction_reduce_triple",
            side_effect=observe_direction,
        ),
    ):
        candidate = _training_probe(model, batch, backend="triton")
    expected_dispatch = _expected_triton_dispatch(args.training_depth)
    dispatch_proven = observed_dispatch == expected_dispatch
    equivalence = {
        "output": {
            "max_abs": _max_abs(candidate["output"], reference["output"]),
            "relative_l2": _relative_l2(candidate["output"], reference["output"]),
        },
        "feature_gradient": {
            "max_abs": _max_abs(
                candidate["feature_gradient"], reference["feature_gradient"]
            ),
            "relative_l2": _relative_l2(
                candidate["feature_gradient"], reference["feature_gradient"]
            ),
        },
        "position_gradient": {
            "max_abs": _max_abs(
                candidate["position_gradient"], reference["position_gradient"]
            ),
            "relative_l2": _relative_l2(
                candidate["position_gradient"], reference["position_gradient"]
            ),
        },
        "parameter_gradient": _parameter_gradient_error(
            candidate["parameter_gradients"],
            reference["parameter_gradients"],
        ),
    }
    del reference, candidate
    model.zero_grad(set_to_none=True)
    features.grad = None
    positions.grad = None

    def clear_training_state() -> None:
        model.zero_grad(set_to_none=True)
        features.grad = None
        positions.grad = None

    measurements = _measure_cuda_lanes(
        {
            "torch_complete_forward_backward": lambda: _training_step(
                model, batch, backend="torch"
            ),
            "forced_triton_complete_forward_backward": lambda: _training_step(
                model, batch, backend="triton"
            ),
        },
        device=device,
        warmup=args.warmup,
        repeats=args.repeats,
        setups={
            "torch_complete_forward_backward": clear_training_state,
            "forced_triton_complete_forward_backward": clear_training_state,
        },
    )
    maximum_relative_error = max(
        equivalence[name]["relative_l2"]
        for name in (
            "output",
            "feature_gradient",
            "position_gradient",
            "parameter_gradient",
        )
    )
    maximum_absolute_error = max(
        equivalence[name]["max_abs"]
        for name in (
            "output",
            "feature_gradient",
            "position_gradient",
            "parameter_gradient",
        )
    )
    gate = {
        "passed": (
            maximum_relative_error <= TRITON_RELATIVE_L2_MAX
            and maximum_absolute_error <= TRITON_ABSOLUTE_ERROR_MAX
            and dispatch_proven
            and not equivalence["parameter_gradient"]["candidate_only"]
            and not equivalence["parameter_gradient"]["reference_only"]
        ),
        "thresholds": {
            "maximum_relative_l2": TRITON_RELATIVE_L2_MAX,
            "maximum_absolute_error": TRITON_ABSOLUTE_ERROR_MAX,
            "complete_fused_dispatch_proven": True,
            "parameter_gradient_name_sets_equal": True,
        },
        "observed": {
            "maximum_relative_l2": maximum_relative_error,
            "maximum_absolute_error": maximum_absolute_error,
            "complete_fused_dispatch_proven": dispatch_proven,
            "parameter_gradient_name_sets_equal": (
                not equivalence["parameter_gradient"]["candidate_only"]
                and not equivalence["parameter_gradient"]["reference_only"]
            ),
        },
    }
    return {
        "status": "completed",
        "scope": (
            "private prepared complete model forward+loss+backward; backend switch "
            "changes local reductions, while global and remaining model work is shared"
        ),
        "nodes": args.training_nodes,
        "edges": int(edges.shape[1]),
        "degree": args.training_degree,
        "width": args.training_width,
        "depth": args.training_depth,
        "dtype": "float32",
        "loss": "mean(square(float32(node_irreps)))",
        "triton_training_primitive": (
            "private custom-autograd fusion for scalar, vector, l=2 tensor, and "
            "three directional local transports; Triton forward with exact "
            "differentiable PyTorch recomputation backward"
        ),
        "dispatch_evidence": {
            "observed": observed_dispatch,
            "expected": expected_dispatch,
            "complete_fused_dispatch_proven": dispatch_proven,
        },
        "equivalence": equivalence,
        "gate": gate,
        "measurements": measurements,
    }


def _compiler_counter_snapshot() -> dict[str, dict[str, int]] | None:
    try:
        from torch._dynamo.utils import counters
    except ImportError:
        return None
    return {
        category: {
            str(name): int(value)
            for name, value in values.items()
            if isinstance(value, int)
        }
        for category, values in counters.items()
        if values
    }


def _counter_value(
    snapshot: dict[str, dict[str, int]] | None,
    category: str,
    name: str,
) -> int | None:
    if snapshot is None:
        return None
    return snapshot.get(category, {}).get(name)


def _counter_delta(
    before: dict[str, dict[str, int]] | None,
    after: dict[str, dict[str, int]] | None,
    category: str,
    name: str,
) -> int | None:
    after_value = _counter_value(after, category, name)
    if after_value is None:
        return None
    before_value = _counter_value(before, category, name)
    return after_value - (0 if before_value is None else before_value)


def _compile_graph(
    model: ELA,
    *,
    nodes: int,
    degree: int,
    device: torch.device,
    topology_shift: int,
) -> ELAGraph:
    features = torch.randn(nodes, model.config.input_layout.dim, device=device)
    positions = torch.randn(nodes, 3, device=device)
    internal = _fixed_degree_edges(
        nodes,
        degree,
        device,
        shift=topology_shift,
    )
    public = internal.flip(0).contiguous()
    return model._prepare_graph(ELAGraph(features, positions, edge_index=public))


def _profile_compiled_core(
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    model = (
        ELA(
            "8x0e",
            "1x0e",
            width=args.compile_width,
            depth=args.compile_depth,
            cutoff=8.0,
        )
        .to(device)
        .eval()
    )
    _activate_local_outputs(model)
    graph = _compile_graph(
        model,
        nodes=args.compile_nodes,
        degree=args.compile_degree,
        device=device,
        topology_shift=0,
    )
    with torch.inference_mode():
        eager_reference = model(graph).x.detach().cpu()
    compiled = prepare_for_inference(
        model,
        device=device,
        dtype="fp32",
        compile_model=True,
        compile_mode=args.compile_mode,
    )
    if not isinstance(compiled, _CompiledCoreInferenceModule):
        raise RuntimeError("ELA did not select the private compiled-core wrapper")
    initial_execute = compiled._execute

    try:
        from torch._dynamo.utils import counters
    except ImportError:
        counters = None
    if counters is not None:
        counters.clear()
    torch._dynamo.reset()
    before = _compiler_counter_snapshot()
    cold_warnings: list[str] = []
    cold_outputs: list[torch.Tensor] = []

    def cold() -> torch.Tensor:
        with warnings.catch_warnings(record=True) as records, torch.inference_mode():
            warnings.simplefilter("always")
            output = compiled(graph).x
        cold_warnings.extend(str(record.message) for record in records)
        cold_outputs.append(output.detach())
        return output

    cold_measurement = _measure_one_cuda(cold, device=device)
    after_cold = _compiler_counter_snapshot()
    if len(cold_outputs) != 1:
        raise RuntimeError("cold compiled execution did not produce one output")
    output_error = _max_abs(cold_outputs[0].cpu(), eager_reference)
    fallback_after_cold = compiled._execute is not initial_execute
    cold_outputs.clear()
    del eager_reference
    warm_warnings: list[str] = []

    def warm() -> torch.Tensor:
        with warnings.catch_warnings(record=True) as records, torch.inference_mode():
            warnings.simplefilter("always")
            output = compiled(graph).x
        warm_warnings.extend(str(record.message) for record in records)
        return output

    def eager() -> torch.Tensor:
        with torch.inference_mode():
            return model(graph).x

    steady_measurements = _measure_cuda_lanes(
        {
            "compiled_same_shape_and_topology": warm,
            "eager_same_shape_and_topology": eager,
        },
        device=device,
        warmup=1,
        repeats=args.repeats,
    )
    after_warm = _compiler_counter_snapshot()

    topology_graph = _compile_graph(
        model,
        nodes=args.compile_nodes,
        degree=args.compile_degree,
        device=device,
        topology_shift=3,
    )
    with torch.inference_mode():
        topology_eager_reference = model(topology_graph).x.detach().cpu()
    topology_warnings: list[str] = []
    topology_outputs: list[torch.Tensor] = []

    def changed_topology() -> torch.Tensor:
        with warnings.catch_warnings(record=True) as records, torch.inference_mode():
            warnings.simplefilter("always")
            output = compiled(topology_graph).x
        topology_warnings.extend(str(record.message) for record in records)
        topology_outputs.append(output.detach())
        return output

    topology_measurement = _measure_one_cuda(changed_topology, device=device)
    if len(topology_outputs) != 1:
        raise RuntimeError("changed-topology execution did not produce one output")
    topology_output_error = _max_abs(
        topology_outputs[0].cpu(),
        topology_eager_reference,
    )
    topology_outputs.clear()
    del topology_eager_reference
    after_topology = _compiler_counter_snapshot()

    shape_nodes = args.compile_nodes + args.compile_shape_delta
    shape_degree = min(args.compile_degree, shape_nodes - 1)
    shape_graph = _compile_graph(
        model,
        nodes=shape_nodes,
        degree=shape_degree,
        device=device,
        topology_shift=1,
    )
    with torch.inference_mode():
        shape_eager_reference = model(shape_graph).x.detach().cpu()
    shape_warnings: list[str] = []
    shape_outputs: list[torch.Tensor] = []

    def changed_shape() -> torch.Tensor:
        with warnings.catch_warnings(record=True) as records, torch.inference_mode():
            warnings.simplefilter("always")
            output = compiled(shape_graph).x
        shape_warnings.extend(str(record.message) for record in records)
        shape_outputs.append(output.detach())
        return output

    shape_measurement = _measure_one_cuda(changed_shape, device=device)
    if len(shape_outputs) != 1:
        raise RuntimeError("changed-shape execution did not produce one output")
    shape_output_error = _max_abs(shape_outputs[0].cpu(), shape_eager_reference)
    shape_outputs.clear()
    del shape_eager_reference
    after_shape = _compiler_counter_snapshot()
    fallback_final = compiled._execute is not initial_execute
    cold_graph_delta = _counter_delta(before, after_cold, "stats", "unique_graphs")
    warm_graph_delta = _counter_delta(
        after_cold,
        after_warm,
        "stats",
        "unique_graphs",
    )
    topology_graph_delta = _counter_delta(
        after_warm,
        after_topology,
        "stats",
        "unique_graphs",
    )
    shape_graph_delta = _counter_delta(
        after_topology,
        after_shape,
        "stats",
        "unique_graphs",
    )
    compiled_median = steady_measurements["compiled_same_shape_and_topology"][
        "cuda_event"
    ]["median_ms"]
    eager_median = steady_measurements["eager_same_shape_and_topology"]["cuda_event"][
        "median_ms"
    ]
    compiled_over_eager = compiled_median / eager_median
    fallback_warning_observed = any(
        "falling back" in message.lower()
        for message in (
            *cold_warnings,
            *warm_warnings,
            *topology_warnings,
            *shape_warnings,
        )
    )
    compile_gate = {
        "passed": (
            not fallback_final
            and not fallback_warning_observed
            and cold_graph_delta is not None
            and cold_graph_delta >= 1
            and warm_graph_delta == 0
            and topology_graph_delta == 0
            and output_error <= COMPILED_ABSOLUTE_ERROR_MAX
            and topology_output_error <= COMPILED_ABSOLUTE_ERROR_MAX
            and shape_output_error <= COMPILED_ABSOLUTE_ERROR_MAX
            and compiled_over_eager <= COMPILED_LATENCY_RATIO_MAX
        ),
        "thresholds": {
            "fallback_allowed": False,
            "fallback_warning_allowed": False,
            "cold_unique_graph_delta_min": 1,
            "warm_unique_graph_delta": 0,
            "same_shape_new_topology_unique_graph_delta": 0,
            "maximum_absolute_error": COMPILED_ABSOLUTE_ERROR_MAX,
            "compiled_over_eager_inference_median_max": (COMPILED_LATENCY_RATIO_MAX),
        },
        "observed": {
            "fallback_occurred": fallback_final,
            "fallback_warning_observed": fallback_warning_observed,
            "cold_unique_graph_delta": cold_graph_delta,
            "warm_unique_graph_delta": warm_graph_delta,
            "same_shape_new_topology_unique_graph_delta": topology_graph_delta,
            "initial_max_abs_error": output_error,
            "same_shape_new_topology_max_abs_error": topology_output_error,
            "changed_shape_max_abs_error": shape_output_error,
            "compiled_over_eager_inference_median": compiled_over_eager,
        },
    }

    return {
        "status": "completed",
        "scope": (
            "public prepared wrapper timing with only ELA._execute_numerical "
            "compiled; graph packing, cache lookup, pooling, and output wrapping eager"
        ),
        "compile_mode": args.compile_mode,
        "nodes": args.compile_nodes,
        "edges": args.compile_nodes * args.compile_degree,
        "fallback_status_after_cold": (
            "eager_fallback" if fallback_after_cold else "compiled_callable_retained"
        ),
        "fallback_status_final": (
            "eager_fallback" if fallback_final else "compiled_callable_retained"
        ),
        "fallback_warning_observed": fallback_warning_observed,
        "warnings": {
            "cold": cold_warnings,
            "warm": warm_warnings,
            "same_shape_new_topology": topology_warnings,
            "changed_shape": shape_warnings,
        },
        "eager_max_abs_error": output_error,
        "eager_parity": {
            "initial_max_abs_error": output_error,
            "same_shape_new_topology_max_abs_error": topology_output_error,
            "changed_shape_max_abs_error": shape_output_error,
        },
        "gate": compile_gate,
        "measurements": {
            "cold_first_execution_compile_included": cold_measurement,
            "warm_steady_execution": steady_measurements,
            "same_shape_new_topology_single_execution": topology_measurement,
            "changed_shape_single_execution": shape_measurement,
        },
        "recompile_scenarios": {
            "same_shape_new_topology": {
                "nodes": args.compile_nodes,
                "edges": args.compile_nodes * args.compile_degree,
                "unique_graph_counter_delta": topology_graph_delta,
            },
            "changed_shape": {
                "nodes": shape_nodes,
                "edges": shape_nodes * shape_degree,
                "unique_graph_counter_delta_from_topology": shape_graph_delta,
            },
            "counter_note": (
                "torch._dynamo counters are private diagnostic telemetry, not a "
                "stable public API; this bounded run nevertheless requires one "
                "cold graph and no steady/topology-only recompilation"
            ),
        },
        "compiler_counter_snapshots": {
            "before": before,
            "after_cold": after_cold,
            "after_warm": after_warm,
            "after_topology": after_topology,
            "after_shape": after_shape,
        },
    }


def _git_state(repository: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "sha": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "seed": args.seed,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "cache_graphs": args.cache_graphs,
        "cache_nodes_per_graph": args.cache_nodes_per_graph,
        "ragged_counts": list(args.ragged_counts),
        "ragged_heads": args.ragged_heads,
        "ragged_feature_width": args.ragged_feature_width,
        "ragged_value_width": args.ragged_value_width,
        "radius_graphs": args.radius_graphs,
        "radius_nodes_per_graph": args.radius_nodes_per_graph,
        "radius_cutoff": args.radius_cutoff,
        "radius_max_neighbors": args.radius_max_neighbors,
        "training_nodes": args.training_nodes,
        "training_degree": args.training_degree,
        "training_width": args.training_width,
        "training_depth": args.training_depth,
        "compile_nodes": args.compile_nodes,
        "compile_degree": args.compile_degree,
        "compile_width": args.compile_width,
        "compile_depth": args.compile_depth,
        "compile_shape_delta": args.compile_shape_delta,
        "compile_mode": args.compile_mode,
    }


def _planned_lanes() -> dict[str, dict[str, Any]]:
    return {
        "trusted_prepared_cache": {
            "status": "planned",
            "scope": (
                "public explicit-immutable trusted reuse and safe unsealed "
                "cache invalidation"
            ),
        },
        "ragged_global": {
            "status": "planned",
            "scope": "private BF16 grouped-MM and segmented global kernels",
        },
        "radius_ingestion": {
            "status": "planned",
            "scope": "public automatic radius and explicit COO ingestion",
        },
        "triton_training_local": {
            "status": "planned",
            "scope": "private prepared complete forward+backward backend comparison",
        },
        "compiled_numerical_core": {
            "status": "planned",
            "scope": "public wrapper around private compiled numerical core",
        },
    }


def _acceptance_contract() -> dict[str, dict[str, object]]:
    """Return the immutable gates written before CUDA results are observed."""

    return {
        "trusted_prepared_cache": {
            "prepared_object_identity_reused": True,
            "immutable_trusted_prepared_admitted": True,
            "packed_template_identity_reused": True,
            "unsealed_dlpack_alias_invalidates_cache": True,
            "peak_allocated_bytes_lt": PEAK_ALLOCATED_LIMIT_BYTES,
        },
        "ragged_global": {
            "transport_relative_l2_max": RAGGED_RELATIVE_L2_MAX,
            "balanced_relative_l2_max": RAGGED_RELATIVE_L2_MAX,
            "native_over_segmented_inference_median_max": (RAGGED_LATENCY_RATIO_MAX),
            "integrated_native_calls_exact": 1,
            "dispatch_vs_direct_max_abs": 0.0,
            "peak_allocated_bytes_lt": PEAK_ALLOCATED_LIMIT_BYTES,
        },
        "radius_ingestion": {
            "topology_equal": True,
            "output_max_abs_error": RADIUS_ERROR_MAX,
            "output_relative_l2_error": RADIUS_ERROR_MAX,
            "latency_promotion_gate": None,
            "peak_allocated_bytes_lt": PEAK_ALLOCATED_LIMIT_BYTES,
        },
        "triton_training_local": {
            "maximum_relative_l2": TRITON_RELATIVE_L2_MAX,
            "maximum_absolute_error": TRITON_ABSOLUTE_ERROR_MAX,
            "complete_fused_dispatch_proven": True,
            "parameter_gradient_name_sets_equal": True,
            "peak_allocated_bytes_lt": PEAK_ALLOCATED_LIMIT_BYTES,
        },
        "compiled_numerical_core": {
            "fallback_allowed": False,
            "fallback_warning_allowed": False,
            "cold_unique_graph_delta_min": 1,
            "warm_unique_graph_delta": 0,
            "same_shape_new_topology_unique_graph_delta": 0,
            "maximum_absolute_error": COMPILED_ABSOLUTE_ERROR_MAX,
            "compiled_over_eager_inference_median_max": (COMPILED_LATENCY_RATIO_MAX),
            "peak_allocated_bytes_lt": PEAK_ALLOCATED_LIMIT_BYTES,
        },
    }


def _functional_thresholds(lane: str) -> dict[str, object]:
    """Return a lane's frozen non-resource thresholds for its receipt."""

    return {
        key: value
        for key, value in _acceptance_contract()[lane].items()
        if key != "peak_allocated_bytes_lt"
    }


def _base_report(
    args: argparse.Namespace,
    *,
    repository: Path,
    schema_only: bool,
    source_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "ela_cuda_completion_profile",
        "status": "schema_only" if schema_only else "running",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git": _git_state(repository),
        "source_manifest": None if source_manifest is None else dict(source_manifest),
        "device": {
            "requested": "cuda",
            "actual": None if schema_only else "cuda",
            "name": None,
            "capability": None,
        },
        "budget_contract": {
            "target_wall_time_minutes_lt": 10,
            "target_peak_allocated_gib_lt": 16,
            "defaults_are_bounded": True,
            "claim": "hard per-lane timed peak-allocation acceptance gate",
        },
        "timing_contract": {
            "raw_samples": True,
            "statistics": ["median_ms", "p95_ms", "min_ms", "max_ms"],
            "cuda_events": True,
            "synchronized_wall_clock": True,
            "alternating_lane_order": True,
            "peak_metric": "torch.cuda.max_memory_allocated",
            "peak_baseline": ("after optional lane setup and device synchronization"),
        },
        "acceptance_contract": _acceptance_contract(),
        "config": _config(args),
        "lanes": _planned_lanes(),
        "failures": [],
    }


def _validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected report schema version")
    if report.get("experiment") != "ela_cuda_completion_profile":
        raise ValueError("unexpected experiment name")
    if report.get("acceptance_contract") != _acceptance_contract():
        raise ValueError("report acceptance contract differs from frozen gates")
    status = report.get("status")
    if status not in {"schema_only", "completed", "partial_failure"}:
        raise ValueError("report has an invalid top-level status")
    failures = report.get("failures")
    if not isinstance(failures, list):
        raise ValueError("report failures must be a list")
    if status == "completed" and failures:
        raise ValueError("completed report cannot retain failures")
    if status == "partial_failure" and not failures:
        raise ValueError("partial_failure report must retain failures")
    if status != "schema_only":
        source_manifest = report.get("source_manifest")
        if (
            not isinstance(source_manifest, Mapping)
            or source_manifest.get("verified_against_current_bytes") is not True
        ):
            raise ValueError("CUDA report must bind to verified source bytes")
    lanes = report.get("lanes")
    if not isinstance(lanes, Mapping) or tuple(lanes) != LANE_NAMES:
        raise ValueError("report must contain the five ordered completion lanes")
    for name, lane in lanes.items():
        if not isinstance(lane, Mapping):
            raise ValueError(f"lane {name} must be an object")
        if lane.get("status") not in {"planned", "completed", "error"}:
            raise ValueError(f"lane {name} has an invalid status")
        if not isinstance(lane.get("scope"), str):
            raise ValueError(f"lane {name} must declare its scope")
        if lane.get("status") == "completed":
            gate = lane.get("gate")
            resource_gate = lane.get("resource_gate")
            if not isinstance(gate, Mapping) or gate.get("passed") is not True:
                raise ValueError(f"completed lane {name} must pass its functional gate")
            if (
                not isinstance(resource_gate, Mapping)
                or resource_gate.get("passed") is not True
            ):
                raise ValueError(f"completed lane {name} must pass its resource gate")
            if not isinstance(lane.get("measurements"), Mapping):
                raise ValueError(f"completed lane {name} must retain measurements")
        if status == "schema_only" and lane.get("status") != "planned":
            raise ValueError("schema-only report must contain only planned lanes")
    # This is also the canonical finite-number check used before writing.
    json.dumps(report, allow_nan=False)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile all CUDA completion lanes with bounded truthful scopes"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help="pre-execution source manifest; required for CUDA execution",
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="write a CPU-safe planned report without executing CUDA kernels",
    )
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--cache-graphs", type=int, default=8)
    parser.add_argument("--cache-nodes-per-graph", type=int, default=128)
    parser.add_argument(
        "--ragged-counts",
        type=_counts,
        default=_counts("257,509,1021,2047"),
    )
    parser.add_argument("--ragged-heads", type=int, default=4)
    parser.add_argument("--ragged-feature-width", type=int, default=32)
    parser.add_argument("--ragged-value-width", type=int, default=33)
    parser.add_argument("--radius-graphs", type=int, default=8)
    parser.add_argument("--radius-nodes-per-graph", type=int, default=192)
    parser.add_argument("--radius-cutoff", type=float, default=1.25)
    parser.add_argument("--radius-max-neighbors", type=int, default=32)
    parser.add_argument("--training-nodes", type=int, default=1024)
    parser.add_argument("--training-degree", type=int, default=32)
    parser.add_argument("--training-width", type=int, default=64)
    parser.add_argument("--training-depth", type=int, default=2)
    parser.add_argument("--compile-nodes", type=int, default=256)
    parser.add_argument("--compile-degree", type=int, default=16)
    parser.add_argument("--compile-width", type=int, default=32)
    parser.add_argument("--compile-depth", type=int, default=2)
    parser.add_argument("--compile-shape-delta", type=int, default=32)
    parser.add_argument("--compile-mode", default="reduce-overhead")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not args.schema_only and args.source_manifest is None:
        raise ValueError("CUDA execution requires --source-manifest")
    if args.warmup < 0:
        raise ValueError("warmup must be nonnegative")
    for name in (
        "repeats",
        "cache_graphs",
        "cache_nodes_per_graph",
        "ragged_heads",
        "ragged_feature_width",
        "ragged_value_width",
        "radius_graphs",
        "radius_nodes_per_graph",
        "radius_max_neighbors",
        "training_nodes",
        "training_degree",
        "training_width",
        "training_depth",
        "compile_nodes",
        "compile_degree",
        "compile_width",
        "compile_depth",
        "compile_shape_delta",
    ):
        _positive_int(name, getattr(args, name))
    if not math.isfinite(args.radius_cutoff) or args.radius_cutoff <= 0.0:
        raise ValueError("radius_cutoff must be finite and positive")
    if args.training_degree >= args.training_nodes:
        raise ValueError("training_degree must be smaller than training_nodes")
    if args.compile_degree >= args.compile_nodes:
        raise ValueError("compile_degree must be smaller than compile_nodes")


def _write_report(report: dict[str, Any], output: Path) -> None:
    _validate_report(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, allow_nan=False)
    output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


def main() -> None:
    args = _parser().parse_args()
    _validate_args(args)
    repository = Path(__file__).resolve().parents[1]
    source_manifest = (
        None
        if args.source_manifest is None
        else _verify_source_manifest(args.source_manifest, repository)
    )
    report = _base_report(
        args,
        repository=repository,
        schema_only=args.schema_only,
        source_manifest=source_manifest,
    )
    if args.schema_only:
        _write_report(report, args.output)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; use --schema-only for a CPU-safe check")
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    report["device"] = {
        "requested": "cuda",
        "actual": str(device),
        "name": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "bfloat16_supported": torch.cuda.is_bf16_supported(),
        "triton_available": triton_available(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
    }

    functions = {
        "trusted_prepared_cache": _profile_trusted_cache,
        "ragged_global": _profile_ragged_global,
        "radius_ingestion": _profile_radius_ingestion,
        "triton_training_local": _profile_triton_training,
        "compiled_numerical_core": _profile_compiled_core,
    }
    failures: list[dict[str, str]] = []
    for name, function in functions.items():
        try:
            lane_result = function(args, device)
            gate = lane_result.get("gate")
            resource_passed = _attach_resource_gate(lane_result)
            if (
                not isinstance(gate, Mapping)
                or gate.get("passed") is not True
                or not resource_passed
            ):
                failure = {
                    "lane": name,
                    "type": "AcceptanceGateFailure",
                    "message": (
                        "one or more preregistered numerical, performance, "
                        "or resource thresholds failed"
                    ),
                }
                failures.append(failure)
                lane_result["status"] = "error"
                lane_result["error"] = failure
            report["lanes"][name] = lane_result
        except Exception as error:  # keep the other independent evidence lanes
            failure = {
                "lane": name,
                "type": type(error).__name__,
                "message": str(error),
            }
            failures.append(failure)
            report["lanes"][name] = {
                "status": "error",
                "scope": report["lanes"][name]["scope"],
                "error": failure,
            }
    report["failures"] = failures
    report["status"] = "completed" if not failures else "partial_failure"
    _write_report(report, args.output)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
