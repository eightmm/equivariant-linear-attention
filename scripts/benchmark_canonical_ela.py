#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import statistics
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

import equivariant_attention as ela_package
from equivariant_attention import ELA, ELAConfig, SparseGeometry
from equivariant_attention.equivariant_linear_attention import (
    EquivariantLinearAttention,
)
from equivariant_attention.migration import load_advanced_ela_state
from equivariant_attention.reproducibility import configure_reproducibility
from equivariant_attention.unified import prepare_3d_graph


def _dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
    }[name]


def _autocast(
    device: torch.device,
    dtype: torch.dtype,
) -> contextlib.AbstractContextManager[Any]:
    if dtype != torch.bfloat16:
        return contextlib.nullcontext()
    return torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=device.type in {"cpu", "cuda"},
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _release_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def _device_fingerprint(device: torch.device) -> dict[str, object] | None:
    if device.type != "cuda":
        return None
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "type": "cuda",
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "compute_capability": [
            properties.major,
            properties.minor,
        ],
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
    }


def _fixed_degree_edges(
    nodes: int,
    degree: int,
    device: torch.device,
) -> torch.Tensor:
    if degree <= 0 or degree >= nodes:
        raise ValueError("degree must satisfy 0 < degree < nodes")
    receiver = torch.arange(nodes, device=device).repeat_interleave(degree)
    offsets = torch.arange(1, degree + 1, device=device).repeat(nodes)
    sender = (receiver + offsets) % nodes
    return torch.stack([receiver, sender])


def _clear_gradients(values: Sequence[torch.Tensor]) -> None:
    for value in values:
        value.grad = None


def _tensor_sha256(values: Sequence[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, value in values:
        contiguous = value.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _strict_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"{name} contains a nonfinite value")


def _assert_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> float:
    _strict_finite(f"{name}.actual", actual)
    _strict_finite(f"{name}.expected", expected)
    tolerance = {
        torch.float64: (1e-8, 1e-8),
        torch.float32: (2e-5, 2e-5),
        torch.bfloat16: (1e-2, 1e-2),
    }[dtype]
    torch.testing.assert_close(
        actual,
        expected,
        atol=tolerance[0],
        rtol=tolerance[1],
        msg=lambda message: f"{name} mismatch: {message}",
    )
    if actual.numel() == 0:
        return 0.0
    return float((actual - expected).abs().max().item())


def _measure(
    function: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
    backward: bool,
    gradients: Sequence[torch.Tensor],
) -> tuple[list[float], int | None]:
    for _ in range(warmup):
        output = function()
        if backward:
            output.float().square().mean().backward()
            _clear_gradients(gradients)
    _synchronize(device)

    times: list[float] = []
    peak = 0 if device.type == "cuda" else None
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        output = function()
        if backward:
            output.float().square().mean().backward()
        _synchronize(device)
        times.append((time.perf_counter() - start) * 1000.0)
        if device.type == "cuda":
            if peak is None:
                raise RuntimeError("CUDA memory accounting is unavailable")
            peak = max(peak, torch.cuda.max_memory_allocated(device))
        if backward:
            _clear_gradients(gradients)
    return times, peak


def _timing_summary(samples: Sequence[float]) -> dict[str, float | list[float]]:
    ordered = sorted(samples)
    if not ordered:
        raise ValueError("timing samples must not be empty")
    if len(ordered) == 1:
        lower = upper = ordered[0]
    else:
        lower, _, upper = statistics.quantiles(
            ordered,
            n=4,
            method="inclusive",
        )
    return {
        "median_ms": statistics.median(ordered),
        "iqr_ms": upper - lower,
        "samples_ms": list(samples),
    }


def _profile_model(
    model: torch.nn.Module,
    features: torch.Tensor,
    positions: torch.Tensor,
    graph: Any,
    *,
    device: torch.device,
    compute_dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    parameters = tuple(model.parameters())
    model.eval()

    def forward() -> torch.Tensor:
        with torch.inference_mode(), _autocast(device, compute_dtype):
            return model(features, positions, graph)["node_irreps"]

    inference_samples, inference_peak = _measure(
        forward,
        device=device,
        warmup=warmup,
        repeats=repeats,
        backward=False,
        gradients=(),
    )

    training_features = features.detach().clone().requires_grad_(True)
    training_positions = positions.detach().clone().requires_grad_(True)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def train_step() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        training_features.grad = None
        training_positions.grad = None
        with _autocast(device, compute_dtype):
            output = model(
                training_features,
                training_positions,
                graph,
            )["node_irreps"]
            loss = output.float().square().mean()
        loss.backward()
        optimizer.step()
        return output

    training_samples, training_peak = _measure(
        train_step,
        device=device,
        warmup=warmup,
        repeats=repeats,
        backward=False,
        gradients=(*parameters, training_features, training_positions),
    )
    model.eval()
    return {
        "inference": _timing_summary(inference_samples),
        "inference_peak_allocated_bytes": inference_peak,
        "optimizer_train_step": _timing_summary(training_samples),
        "optimizer_train_step_peak_allocated_bytes": training_peak,
    }


def _functional_probe(
    model: torch.nn.Module,
    features: torch.Tensor,
    positions: torch.Tensor,
    graph: Any,
    *,
    common_parameter_names: Sequence[str],
    device: torch.device,
    model_dtype: torch.dtype,
    compute_dtype: torch.dtype,
) -> dict[str, Any]:
    model.to(device=device, dtype=model_dtype).eval()
    probe_features = features.detach().clone().requires_grad_(True)
    probe_positions = positions.detach().clone().requires_grad_(True)
    named_parameters = dict(model.named_parameters())
    common_parameters = tuple(
        named_parameters[name] for name in common_parameter_names
    )
    branch_parameters = tuple(
        parameter
        for name, parameter in named_parameters.items()
        if ".branch_fusion." in name
    )
    with _autocast(device, compute_dtype):
        output = model(probe_features, probe_positions, graph)
        node_output = output["node_irreps"]
        graph_output = output["graph_irreps"]
        loss = (
            node_output.float().square().mean()
            + graph_output.float().square().mean()
        )
    gradients = torch.autograd.grad(
        loss,
        (probe_features, probe_positions, *common_parameters),
        allow_unused=True,
        retain_graph=bool(branch_parameters),
    )
    feature_gradient, position_gradient, *parameter_gradients = gradients
    if feature_gradient is None or position_gradient is None:
        raise RuntimeError("input gradients must be present")
    parameter_result = {
        name: None if gradient is None else gradient.float().cpu()
        for name, gradient in zip(
            common_parameter_names,
            parameter_gradients,
            strict=True,
        )
    }
    for name, gradient in parameter_result.items():
        if gradient is not None:
            _strict_finite(f"common_gradient.{name}", gradient)
    branch_gradients = torch.autograd.grad(
        loss,
        branch_parameters,
        allow_unused=True,
        retain_graph=False,
    ) if branch_parameters else ()
    branch_finite = all(
        gradient is None or bool(torch.isfinite(gradient).all().item())
        for gradient in branch_gradients
    )
    branch_nonzero = any(
        gradient is not None and bool(torch.count_nonzero(gradient).item())
        for gradient in branch_gradients
    )
    result = {
        "node_output": node_output.float().cpu(),
        "graph_output": graph_output.float().cpu(),
        "feature_gradient": feature_gradient.float().cpu(),
        "position_gradient": position_gradient.float().cpu(),
        "parameter_gradients": parameter_result,
        "branch_gradients_finite": branch_finite,
        "branch_gradients_nonzero": branch_nonzero,
    }
    model.to(device="cpu")
    _release_cuda(device)
    return result


def _profile_isolated(
    model: torch.nn.Module,
    features: torch.Tensor,
    positions: torch.Tensor,
    graph: Any,
    *,
    device: torch.device,
    model_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    model.to(device=device, dtype=model_dtype)
    result = _profile_model(
        model,
        features,
        positions,
        graph,
        device=device,
        compute_dtype=compute_dtype,
        warmup=warmup,
        repeats=repeats,
    )
    model.to(device="cpu")
    _release_cuda(device)
    return result


def _compare_functional_probes(
    control: dict[str, Any],
    candidate: dict[str, Any],
    *,
    compute_dtype: torch.dtype,
) -> dict[str, Any]:
    receipt = {
        "node_output_max_abs": _assert_close(
            "node_output",
            candidate["node_output"],
            control["node_output"],
            dtype=compute_dtype,
        ),
        "graph_output_max_abs": _assert_close(
            "graph_output",
            candidate["graph_output"],
            control["graph_output"],
            dtype=compute_dtype,
        ),
        "feature_gradient_max_abs": _assert_close(
            "feature_gradient",
            candidate["feature_gradient"],
            control["feature_gradient"],
            dtype=compute_dtype,
        ),
        "position_gradient_max_abs": _assert_close(
            "position_gradient",
            candidate["position_gradient"],
            control["position_gradient"],
            dtype=compute_dtype,
        ),
    }
    parameter_max_abs = 0.0
    control_gradients = control["parameter_gradients"]
    candidate_gradients = candidate["parameter_gradients"]
    if control_gradients.keys() != candidate_gradients.keys():
        raise RuntimeError("common gradient keys differ")
    for name, expected in control_gradients.items():
        actual = candidate_gradients[name]
        if (expected is None) != (actual is None):
            raise RuntimeError(f"common gradient presence differs for {name}")
        if expected is not None:
            if actual is None:
                raise RuntimeError(f"candidate gradient is missing for {name}")
            parameter_max_abs = max(
                parameter_max_abs,
                _assert_close(
                    f"common_parameter_gradient.{name}",
                    actual,
                    expected,
                    dtype=compute_dtype,
                ),
            )
    receipt["common_parameter_gradient_max_abs"] = parameter_max_abs
    receipt["candidate_branch_gradients_finite"] = bool(
        candidate["branch_gradients_finite"]
    )
    receipt["candidate_branch_gradients_nonzero"] = bool(
        candidate["branch_gradients_nonzero"]
    )
    if not receipt["candidate_branch_gradients_finite"]:
        raise RuntimeError("candidate branch gradients are nonfinite")
    if not receipt["candidate_branch_gradients_nonzero"]:
        raise RuntimeError("candidate branch gradients are all zero")
    return receipt


def _memory_ratio(
    candidate: int | None,
    control: int | None,
) -> float | None:
    if candidate is None or control is None:
        return None
    return candidate / max(1, control)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure the branch-router overhead of canonical ELA"
    )
    parser.add_argument("--nodes", type=int, default=1024)
    parser.add_argument("--degree", type=int, default=32)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--order",
        choices=["control-first", "candidate-first"],
        default="control-first",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64", "bfloat16"],
        default="float32",
    )
    parser.add_argument("--expected-source-root", type=Path)
    parser.add_argument("--enforce-resource-gate", action="store_true")
    parser.add_argument("--max-parameter-ratio", type=float, default=1.05)
    parser.add_argument("--max-inference-ratio", type=float, default=1.20)
    parser.add_argument("--max-train-step-ratio", type=float, default=1.20)
    parser.add_argument("--max-memory-ratio", type=float, default=1.20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.nodes <= 1 or args.width < 16 or args.depth <= 0:
        raise ValueError("invalid nodes, width, or depth")
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be nonnegative and repeats positive")
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in (
            args.max_parameter_ratio,
            args.max_inference_ratio,
            args.max_train_step_ratio,
            args.max_memory_ratio,
        )
    ):
        raise ValueError("resource ceilings must be finite and positive")
    source_file = Path(ela_package.__file__).resolve()
    source_root = source_file.parents[1]
    repository_root = Path(__file__).resolve().parents[1]
    canonical_source_root = (repository_root / "src").resolve()
    if args.expected_source_root is not None:
        expected_source_root = args.expected_source_root.resolve()
        if expected_source_root != canonical_source_root:
            raise RuntimeError(
                "--expected-source-root must name this script checkout; "
                f"{expected_source_root} != {canonical_source_root}"
            )
    if source_root != canonical_source_root:
        raise RuntimeError(
            "benchmark imported a different source checkout; "
            f"{source_root} != {canonical_source_root}"
        )
    reproducibility = configure_reproducibility(
        seed=args.seed,
        mode="strict",
    )
    device = torch.device(args.device)
    compute_dtype = _dtype(args.dtype)
    model_dtype = (
        torch.float32 if compute_dtype == torch.bfloat16 else compute_dtype
    )
    minimal = ELAConfig(
        input_irreps="16x0e",
        output_irreps="1x0e + 1x1o",
        width=args.width,
        depth=args.depth,
        geometry=SparseGeometry(cutoff=6.0, num_rbf=16),
    )

    torch.manual_seed(args.seed)
    control = EquivariantLinearAttention(
        minimal.to_advanced_config()
    ).to(dtype=model_dtype)
    candidate = ELA(minimal).to(dtype=model_dtype)
    migration = load_advanced_ela_state(candidate, control.state_dict())

    control_parameters = sum(parameter.numel() for parameter in control.parameters())
    candidate_parameters = sum(
        parameter.numel() for parameter in candidate.parameters()
    )
    router_parameters = sum(
        parameter.numel()
        for name, parameter in candidate.named_parameters()
        if ".branch_fusion." in name
    )
    control_state = control.state_dict()
    candidate_state = candidate.state_dict()
    if not all(
        key in candidate_state and torch.equal(value, candidate_state[key])
        for key, value in control_state.items()
    ):
        raise RuntimeError("common state tensors are not byte-identical")
    common_state_sha256 = _tensor_sha256(tuple(control_state.items()))
    common_parameter_names = tuple(
        name for name, _ in control.named_parameters()
    )

    features = torch.randn(
        args.nodes,
        16,
        device=device,
        dtype=model_dtype,
    )
    positions = torch.randn(
        args.nodes,
        3,
        device=device,
        dtype=model_dtype,
    )
    batch = torch.zeros(args.nodes, device=device, dtype=torch.long)
    edge_index = _fixed_degree_edges(args.nodes, args.degree, device)
    graph = prepare_3d_graph(batch, edge_index)

    control_probe = _functional_probe(
        control,
        features,
        positions,
        graph,
        common_parameter_names=common_parameter_names,
        device=device,
        model_dtype=model_dtype,
        compute_dtype=compute_dtype,
    )
    candidate_probe = _functional_probe(
        candidate,
        features,
        positions,
        graph,
        common_parameter_names=common_parameter_names,
        device=device,
        model_dtype=model_dtype,
        compute_dtype=compute_dtype,
    )
    functional_receipt = _compare_functional_probes(
        control_probe,
        candidate_probe,
        compute_dtype=compute_dtype,
    )

    models = (
        [("control", control), ("candidate", candidate)]
        if args.order == "control-first"
        else [("candidate", candidate), ("control", control)]
    )
    profiles: dict[str, dict[str, Any]] = {}
    for name, model in models:
        profiles[name] = _profile_isolated(
            model,
            features,
            positions,
            graph,
            device=device,
            model_dtype=model_dtype,
            compute_dtype=compute_dtype,
            warmup=args.warmup,
            repeats=args.repeats,
        )
    control_profile = profiles["control"]
    candidate_profile = profiles["candidate"]
    ratios = {
        "parameters": candidate_parameters / control_parameters,
        "inference_median": (
            candidate_profile["inference"]["median_ms"]
            / control_profile["inference"]["median_ms"]
        ),
        "optimizer_train_step_median": (
            candidate_profile["optimizer_train_step"]["median_ms"]
            / control_profile["optimizer_train_step"]["median_ms"]
        ),
        "inference_peak_allocated": _memory_ratio(
            candidate_profile["inference_peak_allocated_bytes"],
            control_profile["inference_peak_allocated_bytes"],
        ),
        "optimizer_train_step_peak_allocated": _memory_ratio(
            candidate_profile["optimizer_train_step_peak_allocated_bytes"],
            control_profile["optimizer_train_step_peak_allocated_bytes"],
        ),
    }
    gate_checks = {
        "parameters": ratios["parameters"] <= args.max_parameter_ratio,
        "inference": (
            ratios["inference_median"] <= args.max_inference_ratio
        ),
        "optimizer_train_step": (
            ratios["optimizer_train_step_median"]
            <= args.max_train_step_ratio
        ),
        "inference_memory": (
            ratios["inference_peak_allocated"] is None
            or ratios["inference_peak_allocated"] <= args.max_memory_ratio
        ),
        "optimizer_train_step_memory": (
            ratios["optimizer_train_step_peak_allocated"] is None
            or ratios["optimizer_train_step_peak_allocated"]
            <= args.max_memory_ratio
        ),
    }
    resource_gate_passed = all(gate_checks.values())
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        text=True,
    ).strip()
    git_dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repository_root,
            text=True,
        ).strip()
    )
    if args.enforce_resource_gate and git_dirty:
        raise RuntimeError(
            "an enforced canonical resource gate requires a clean worktree"
        )

    payload = {
        "schema_version": 3,
        "experiment": "canonical_ela_overhead",
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "source_file": str(source_file),
        "source_verified": True,
        "device": str(device),
        "device_fingerprint": _device_fingerprint(device),
        "dtype": args.dtype,
        "seed": args.seed,
        "reproducibility": reproducibility,
        "profile_order": args.order,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "nodes": args.nodes,
        "supplied_candidate_edges": args.nodes * args.degree,
        "supplied_candidate_degree": args.degree,
        "width": args.width,
        "depth": args.depth,
        "neighbor_discovery_included": False,
        "graph_packing_included": False,
        "host_device_preparation_included": False,
        "models_profiled_one_at_a_time": True,
        "same_common_weights": True,
        "common_state_sha256": common_state_sha256,
        "input_sha256": _tensor_sha256(
            (
                ("features", features),
                ("positions", positions),
                ("batch", batch),
                ("edge_index", edge_index),
            )
        ),
        "migration": {
            "loaded_keys": migration.loaded_keys,
            "missing_keys": list(migration.missing_keys),
            "unexpected_keys": list(migration.unexpected_keys),
            "router_initialized": migration.router_initialized,
        },
        "functional_equivalence": functional_receipt,
        "control": {
            "model": "EquivariantLinearAttention",
            "parameters": control_parameters,
            **control_profile,
        },
        "candidate": {
            "model": "ELA",
            "parameters": candidate_parameters,
            "router_parameters": router_parameters,
            **candidate_profile,
        },
        "ratios": ratios,
        "resource_gate": {
            "enforced": args.enforce_resource_gate,
            "limits": {
                "parameters": args.max_parameter_ratio,
                "inference": args.max_inference_ratio,
                "optimizer_train_step": args.max_train_step_ratio,
                "memory": args.max_memory_ratio,
            },
            "checks": gate_checks,
            "passed": resource_gate_passed,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, allow_nan=False)
    args.output.write_text(serialized, encoding="utf-8")
    print(serialized)
    if args.enforce_resource_gate and not resource_gate_passed:
        raise SystemExit("canonical ELA resource gate failed")


if __name__ == "__main__":
    main()
