#!/usr/bin/env python3
"""Run the bounded Stage-0 activation gate for interacting global memory.

This is a mechanism probe, not a model-quality benchmark.  It compares M=1
against larger memory counts with an identical parameter state on one fixed
16-node graph.  A larger-memory arm is admitted only when every global head
has non-collapsed assignments, nontrivial coupling, a nonconstant effective
pair gate, and a measurably different full model output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
import equivariant_attention.moment as moment

from equivariant_attention import EquivariantAttention, EquivariantAttentionConfig
from equivariant_attention.diagnostics import (
    memory_center_summary,
    memory_pair_gate_summary,
)


SCHEMA_VERSION = 2
NONCONSTANT_RELATIVE_TOLERANCE = 1e-3
ENTROPY_MIN = 0.05
ENTROPY_MAX = 0.995
MARGINAL_ENTROPY_MIN = 0.05
MUTUAL_INFORMATION_MIN = 1e-3
OCCUPANCY_FRACTION_MIN = 1e-4
COUPLING_Q00_MAX = 0.99
PAIR_GATE_CENTERED_FROBENIUS_MIN = 1e-2
PAIR_GATE_NONCONSTANT_FRACTION_MIN = 0.10
RELATIVE_OUTPUT_RMS_MIN = 1e-5
MECHANISM_RMS_MIN = 1e-5
CV_IDENTITY_ABS_TOLERANCE = 1e-10
IDENTITY_MIX_CANDIDATES = (0.10, 0.25, 0.50)


def probe_graph(
    *,
    dtype: torch.dtype,
    device: torch.device,
    scenario: str = "aligned",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return one deterministic graph from the registered Stage-0 suite."""

    if dtype not in {torch.float32, torch.float64}:
        raise ValueError("probe dtype must be float32 or float64")
    if scenario not in {"aligned", "crossed", "spatial_only", "semantic_only"}:
        raise ValueError(f"unknown probe scenario: {scenario!r}")
    cluster_ids = torch.arange(4, device=device).repeat_interleave(4)
    feature_cluster_ids = cluster_ids
    if scenario == "crossed":
        feature_cluster_ids = torch.tensor(
            [0, 2, 3, 1], dtype=torch.long, device=device
        )[cluster_ids]
    node_feats = F.one_hot(feature_cluster_ids, num_classes=4).to(dtype=dtype)
    phase = (
        torch.arange(16, dtype=dtype, device=device) + node_feats.new_tensor(0.5)
    ) / 16.0
    continuous = torch.stack(
        (
            torch.sin(2.0 * math.pi * phase),
            torch.cos(2.0 * math.pi * phase),
            2.0 * phase - 1.0,
            (2.0 * phase - 1.0).square(),
        ),
        dim=-1,
    )
    node_feats = torch.cat((node_feats, continuous), dim=-1)
    if scenario == "spatial_only":
        node_feats = node_feats.new_zeros(node_feats.shape)
        node_feats[:, 0] = 1.0

    centers = node_feats.new_tensor(
        [
            [-2.0, -2.0, -0.75],
            [2.0, -2.0, 0.75],
            [-2.0, 2.0, 0.75],
            [2.0, 2.0, -0.75],
        ]
    )
    offsets = node_feats.new_tensor(
        [
            [-0.18, -0.06, 0.03],
            [0.15, -0.09, -0.04],
            [-0.07, 0.17, 0.05],
            [0.10, 0.12, -0.02],
        ]
    )
    pos = centers[cluster_ids] + offsets[torch.arange(16, device=device) % 4]
    if scenario == "semantic_only":
        pos = torch.stack(
            (
                0.35 * torch.sin(2.0 * math.pi * phase),
                0.35 * torch.cos(2.0 * math.pi * phase),
                0.25 * (2.0 * phase - 1.0),
            ),
            dim=-1,
        )
    batch = torch.zeros(16, dtype=torch.long, device=device)
    return node_feats, pos, batch


def symmetric_relative_rms(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return a scale-free symmetric RMS difference, with zero for two zeros."""

    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        raise TypeError("symmetric_relative_rms inputs must be tensors")
    if left.shape != right.shape or left.numel() == 0:
        raise ValueError("symmetric_relative_rms inputs must have equal nonempty shapes")
    left64 = left.detach().to(dtype=torch.float64)
    right64 = right.detach().to(dtype=torch.float64, device=left64.device)
    if not bool(torch.isfinite(left64).all().item()) or not bool(
        torch.isfinite(right64).all().item()
    ):
        raise ValueError("symmetric_relative_rms inputs must be finite")
    denominator_square = left64.square().sum() + right64.square().sum()
    if float(denominator_square.item()) == 0.0:
        return 0.0
    numerator_square = 2.0 * (left64 - right64).square().sum()
    return float((numerator_square / denominator_square).sqrt().item())


def relative_output_rms(
    baseline: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
) -> float:
    """Return RMS(candidate-baseline) / RMS(baseline) over the full output."""

    if baseline.keys() != candidate.keys() or not baseline:
        raise ValueError("baseline and candidate outputs must have identical keys")
    reference_square = 0.0
    difference_square = 0.0
    count = 0
    for name in baseline:
        reference = baseline[name].detach().to(dtype=torch.float64)
        comparison = (
            candidate[name]
            .detach()
            .to(
                dtype=torch.float64,
                device=reference.device,
            )
        )
        if reference.shape != comparison.shape:
            raise ValueError(f"output shape mismatch for {name!r}")
        if not bool(torch.isfinite(reference).all().item()) or not bool(
            torch.isfinite(comparison).all().item()
        ):
            raise ValueError("model outputs must contain only finite values")
        reference_square += float(reference.square().sum().item())
        difference_square += float((comparison - reference).square().sum().item())
        count += reference.numel()
    baseline_rms = math.sqrt(reference_square / count)
    if baseline_rms == 0.0:
        raise ValueError("baseline RMS must be positive")
    return math.sqrt(difference_square / count) / baseline_rms


def stage0_decision(
    activation: Mapping[str, object],
    *,
    mechanism: Mapping[str, object],
    relative_output_rms: float,
) -> dict[str, object]:
    """Apply the frozen all-head Stage-0 thresholds to one memory arm."""

    heads = activation.get("heads")
    head_count = activation.get("head_count")
    if not isinstance(heads, list) or not heads:
        raise ValueError("activation must contain at least one head summary")
    if head_count != len(heads):
        raise ValueError("activation head_count must match the head summaries")
    output_change = float(relative_output_rms)
    if not math.isfinite(output_change) or output_change < 0.0:
        raise ValueError("relative_output_rms must be finite and nonnegative")

    checks: list[dict[str, object]] = []

    def add_check(
        name: str,
        value: float,
        operator: str,
        threshold: float,
        passed: bool,
    ) -> None:
        if not math.isfinite(value):
            passed = False
        checks.append(
            {
                "name": name,
                "value": value,
                "operator": operator,
                "threshold": threshold,
                "passed": bool(passed),
            }
        )

    for head_offset, raw_head in enumerate(heads):
        if not isinstance(raw_head, Mapping):
            raise ValueError("each activation head must be a mapping")
        assignment = raw_head.get("assignment")
        coupling = raw_head.get("coupling")
        pair_gate = raw_head.get("pair_gate")
        if not all(
            isinstance(summary, Mapping)
            for summary in (assignment, coupling, pair_gate)
        ):
            raise ValueError(
                "each head must contain assignment, coupling, and pair_gate"
            )
        head_index = int(raw_head.get("head_index", head_offset))
        prefix = f"head[{head_index}]"

        entropy = float(assignment["assignment_entropy_over_log_m"])
        marginal_entropy = float(assignment["marginal_entropy_over_log_m"])
        mutual_information = float(assignment["mutual_information_over_log_m"])
        occupancy = float(assignment["occupancy_fraction.min"])
        coupling_min = float(coupling["coupling.q00"])
        cv = float(pair_gate["cv"])
        centered = float(pair_gate["centered_frobenius_ratio"])
        nonconstant = float(pair_gate["nonconstant_fraction"])
        expected_centered = cv / math.sqrt(1.0 + cv * cv)
        identity_error = abs(centered - expected_centered)

        add_check(
            f"{prefix}.entropy_min", entropy, ">=", ENTROPY_MIN, entropy >= ENTROPY_MIN
        )
        add_check(
            f"{prefix}.entropy_max", entropy, "<=", ENTROPY_MAX, entropy <= ENTROPY_MAX
        )
        add_check(
            f"{prefix}.marginal_entropy_min",
            marginal_entropy,
            ">=",
            MARGINAL_ENTROPY_MIN,
            marginal_entropy >= MARGINAL_ENTROPY_MIN,
        )
        add_check(
            f"{prefix}.mutual_information_min",
            mutual_information,
            ">=",
            MUTUAL_INFORMATION_MIN,
            mutual_information >= MUTUAL_INFORMATION_MIN,
        )
        add_check(
            f"{prefix}.occupancy_fraction_min",
            occupancy,
            ">=",
            OCCUPANCY_FRACTION_MIN,
            occupancy >= OCCUPANCY_FRACTION_MIN,
        )
        add_check(
            f"{prefix}.coupling_q00",
            coupling_min,
            "<=",
            COUPLING_Q00_MAX,
            coupling_min <= COUPLING_Q00_MAX,
        )
        add_check(
            f"{prefix}.pair_gate_centered_frobenius_ratio",
            centered,
            ">=",
            PAIR_GATE_CENTERED_FROBENIUS_MIN,
            centered >= PAIR_GATE_CENTERED_FROBENIUS_MIN,
        )
        add_check(
            f"{prefix}.pair_gate_nonconstant_fraction",
            nonconstant,
            ">=",
            PAIR_GATE_NONCONSTANT_FRACTION_MIN,
            nonconstant >= PAIR_GATE_NONCONSTANT_FRACTION_MIN,
        )
        add_check(
            f"{prefix}.cv_centered_identity_error",
            identity_error,
            "<=",
            CV_IDENTITY_ABS_TOLERANCE,
            identity_error <= CV_IDENTITY_ABS_TOLERANCE,
        )

    messages = mechanism.get("messages")
    post_middle = mechanism.get("post_middle")
    gradients = mechanism.get("gradients")
    if not all(isinstance(value, Mapping) for value in (messages, post_middle, gradients)):
        raise ValueError("mechanism must contain messages, post_middle, and gradients")
    mechanism_checks = (
        ("middle_messages.aggregate", float(messages["aggregate"])),
        ("post_middle.aggregate", float(post_middle["aggregate"])),
        ("gradients.scalars", float(gradients["scalars"])),
        ("gradients.vectors", float(gradients["vectors"])),
        ("gradients.positions", float(gradients["positions"])),
    )
    for name, value in mechanism_checks:
        add_check(name, value, ">=", MECHANISM_RMS_MIN, value >= MECHANISM_RMS_MIN)

    add_check(
        "full_output.relative_rms",
        output_change,
        ">=",
        RELATIVE_OUTPUT_RMS_MIN,
        output_change >= RELATIVE_OUTPUT_RMS_MIN,
    )
    return {
        "passed": all(bool(check["passed"]) for check in checks),
        "checks": checks,
    }


def _model_config(
    *,
    memory_count: int,
    hidden_dim: int,
    num_heads: int,
) -> EquivariantAttentionConfig:
    return EquivariantAttentionConfig(
        node_dim=8,
        hidden_irreps=f"{hidden_dim}x0e + {max(1, hidden_dim // 16)}x1o",
        output_irreps="1x0e + 1x1o + 1x2e",
        num_layers=3,
        num_heads=num_heads,
        local_head_counts=(num_heads, 0, num_heads),
        global_memory_count=memory_count,
        use_memory_interaction=True,
    )


def _state_hashes(model: torch.nn.Module) -> dict[str, str]:
    state_digest = hashlib.sha256()
    schema_digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        metadata = json.dumps(
            [name, str(tensor.dtype), list(tensor.shape)],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
        for digest in (state_digest, schema_digest):
            digest.update(len(metadata).to_bytes(8, "big"))
            digest.update(metadata)
        raw = (
            tensor.detach()
            .cpu()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
        state_digest.update(len(raw).to_bytes(8, "big"))
        state_digest.update(raw)
    return {
        "state_sha256": state_digest.hexdigest(),
        "state_schema_sha256": schema_digest.hexdigest(),
    }


def _source_hash() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "src").rglob("*.py"))
    paths.extend(
        path
        for path in (
            root / "scripts" / "probe_memory_activation.py",
            root / "pyproject.toml",
            root / "uv.lock",
        )
        if path.is_file()
    )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _fixed_projection_loss(messages: Sequence[torch.Tensor]) -> torch.Tensor:
    loss = messages[0].new_zeros(())
    for index, value in enumerate(messages):
        phase = torch.arange(
            1,
            value.numel() + 1,
            dtype=value.dtype,
            device=value.device,
        ).reshape(value.shape)
        probe = torch.sin((0.731 + 0.137 * index) * phase)
        loss = loss + (value * probe).sum() / math.sqrt(value.numel())
    return loss


def _aggregate_symmetric_relative_rms(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> float:
    if left.keys() != right.keys() or not left:
        raise ValueError("mechanism groups must have identical nonempty keys")
    left_flat = torch.cat([left[name].detach().reshape(-1) for name in left])
    right_flat = torch.cat([right[name].detach().reshape(-1) for name in left])
    return symmetric_relative_rms(left_flat, right_flat)


def _select_counterfactual_coupling(
    radial: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if name == "radial":
        return radial
    if name == "ones":
        return torch.ones_like(radial)
    memories = radial.shape[-1]
    identity = torch.eye(
        memories,
        dtype=radial.dtype,
        device=radial.device,
    ).expand_as(radial)
    if name == "identity":
        return identity
    if name.startswith("lambda_"):
        identity_mix = float(name.removeprefix("lambda_"))
        return moment._mix_memory_coupling(radial, identity_mix)
    raise ValueError(f"unknown counterfactual coupling: {name!r}")


def _trace_model(
    model: EquivariantAttention,
    node_feats: torch.Tensor,
    pos: torch.Tensor,
    batch: torch.Tensor,
    *,
    counterfactual: str,
) -> dict[str, object]:
    """Capture actual middle transport, state, gradients, and memory activation."""

    layer = model.layers[1]
    captured: dict[str, object] = {}
    original_assignment = moment._memory_assignments_and_coupling
    original_messages = moment._global_moment_messages

    def capture_layer_input(
        _module: torch.nn.Module,
        inputs: tuple[object, ...],
    ) -> None:
        captured["scalars"] = inputs[0]
        captured["vectors"] = inputs[1]

    def capture_layer_output(
        _module: torch.nn.Module,
        _inputs: tuple[object, ...],
        output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        captured["post_middle"] = output

    def wrapped_assignment(*args: object, **kwargs: object) -> tuple[torch.Tensor, ...]:
        kwargs["identity_mix"] = 0.0
        assignment, radial, centers = original_assignment(*args, **kwargs)
        coupling = _select_counterfactual_coupling(radial, counterfactual)
        captured["assignment"] = assignment
        captured["radial_coupling"] = radial
        captured["coupling"] = coupling
        captured["centers"] = centers
        return assignment, coupling, centers

    def wrapped_messages(*args: object, **kwargs: object) -> tuple[torch.Tensor, ...]:
        messages = original_messages(*args, **kwargs)
        captured["messages"] = messages
        return messages

    training_states = [(module, module.training) for module in model.modules()]
    pre_handle = layer.register_forward_pre_hook(capture_layer_input)
    post_handle = layer.register_forward_hook(capture_layer_output)
    moment._memory_assignments_and_coupling = wrapped_assignment
    moment._global_moment_messages = wrapped_messages
    run_pos = pos.detach().clone().requires_grad_(True)
    try:
        model.eval()
        output = model(node_feats, run_pos, batch=batch)
        messages = captured.get("messages")
        if not isinstance(messages, tuple) or len(messages) != 5:
            raise RuntimeError("middle global message capture did not execute")
        scalars = captured.get("scalars")
        vectors = captured.get("vectors")
        if not isinstance(scalars, torch.Tensor) or not isinstance(vectors, torch.Tensor):
            raise RuntimeError("middle layer input capture did not execute")
        raw_gradients = torch.autograd.grad(
            _fixed_projection_loss(messages),
            (scalars, vectors, run_pos),
            allow_unused=True,
        )
        gradients = tuple(
            torch.zeros_like(target) if gradient is None else gradient
            for gradient, target in zip(
                raw_gradients,
                (scalars, vectors, run_pos),
                strict=True,
            )
        )
    finally:
        moment._memory_assignments_and_coupling = original_assignment
        moment._global_moment_messages = original_messages
        pre_handle.remove()
        post_handle.remove()
        for module, training in training_states:
            module.training = training

    message_names = (
        "scalar_message",
        "vector_base",
        "relative",
        "tensor",
        "radial_trace",
    )
    post_middle = captured.get("post_middle")
    if not isinstance(post_middle, tuple) or len(post_middle) != 3:
        raise RuntimeError("middle layer output capture did not execute")
    trace: dict[str, object] = {
        "output": {name: value.detach() for name, value in output.items()},
        "messages": {
            name: value.detach()
            for name, value in zip(message_names, messages, strict=True)
        },
        "post_middle": {
            name: value.detach()
            for name, value in zip(
                ("scalars", "vectors", "transient_tensor"),
                post_middle,
                strict=True,
            )
        },
        "gradients": {
            name: value.detach()
            for name, value in zip(
                ("scalars", "vectors", "positions"),
                gradients,
                strict=True,
            )
        },
    }
    assignment = captured.get("assignment")
    coupling = captured.get("coupling")
    centers = captured.get("centers")
    if isinstance(assignment, torch.Tensor):
        if not isinstance(coupling, torch.Tensor) or not isinstance(centers, torch.Tensor):
            raise RuntimeError("memory coupling/center capture is incomplete")
        if assignment.shape[0] != node_feats.shape[0] or coupling.shape[0] != 1:
            raise RuntimeError("Stage-0 activation probe requires exactly one graph")
        trace["activation"] = memory_pair_gate_summary(
            assignment,
            coupling[0],
            nonconstant_relative_tolerance=NONCONSTANT_RELATIVE_TOLERANCE,
        )
        trace["centers"] = memory_center_summary(
            centers[0],
            interaction_cutoff=layer.memory_interaction_cutoff,
        )
    return trace


def _mechanism_difference(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for group in ("messages", "post_middle", "gradients"):
        left = baseline[group]
        right = candidate[group]
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            raise ValueError(f"trace group {group!r} must be a mapping")
        per_item = {
            name: symmetric_relative_rms(left[name], right[name]) for name in left
        }
        per_item["aggregate"] = _aggregate_symmetric_relative_rms(left, right)
        result[group] = per_item
    return result


def _activation_materially_nonconstant(activation: Mapping[str, object]) -> bool:
    heads = activation.get("heads")
    if not isinstance(heads, list) or not heads:
        return False
    for head in heads:
        if not isinstance(head, Mapping):
            return False
        assignment = head.get("assignment")
        pair_gate = head.get("pair_gate")
        if not isinstance(assignment, Mapping) or not isinstance(pair_gate, Mapping):
            return False
        if (
            float(assignment["mutual_information_over_log_m"])
            < MUTUAL_INFORMATION_MIN
            or float(pair_gate["centered_frobenius_ratio"])
            < PAIR_GATE_CENTERED_FROBENIUS_MIN
            or float(pair_gate["nonconstant_fraction"])
            < PAIR_GATE_NONCONSTANT_FRACTION_MIN
        ):
            return False
    return True


def run_probe(
    *,
    memory_counts: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    hidden_dim: int,
    num_heads: int,
    scenario: str = "aligned",
) -> dict[str, object]:
    """Run matched-state M=1 versus coupling counterfactuals for one lane."""

    counts = tuple(int(count) for count in memory_counts)
    if (
        not counts
        or any(count <= 1 for count in counts)
        or len(set(counts)) != len(counts)
    ):
        raise ValueError("memory_counts must contain unique integers greater than one")
    if hidden_dim <= 0 or num_heads <= 0 or hidden_dim % num_heads != 0:
        raise ValueError("hidden_dim must be positive and divisible by num_heads")
    if dtype not in {torch.float32, torch.float64}:
        raise ValueError("probe dtype must be float32 or float64")

    fork_devices: list[int] = []
    if device.type == "cuda":
        fork_devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        node_feats, pos, batch = probe_graph(
            dtype=dtype,
            device=device,
            scenario=scenario,
        )
        baseline_model = EquivariantAttention(
            _model_config(
                memory_count=1,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
            )
        ).to(device=device, dtype=dtype)
        baseline_state = baseline_model.state_dict()
        baseline_hashes = _state_hashes(baseline_model)
        baseline_trace = _trace_model(
            baseline_model,
            node_feats,
            pos,
            batch,
            counterfactual="ones",
        )

        arms: list[dict[str, object]] = []
        for memory_count in counts:
            candidate = EquivariantAttention(
                _model_config(
                    memory_count=memory_count,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                )
            ).to(device=device, dtype=dtype)
            candidate.load_state_dict(baseline_state, strict=True)
            candidate_hashes = _state_hashes(candidate)
            counterfactuals: dict[str, object] = {}
            counterfactual_names = (
                "ones",
                "radial",
                "identity",
                *(f"lambda_{value:.2f}" for value in IDENTITY_MIX_CANDIDATES),
            )
            centers: dict[str, object] | None = None
            for name in counterfactual_names:
                candidate_trace = _trace_model(
                    candidate,
                    node_feats,
                    pos,
                    batch,
                    counterfactual=name,
                )
                activation = candidate_trace.get("activation")
                if not isinstance(activation, Mapping):
                    raise RuntimeError("interacting memory did not emit activation")
                mechanism = _mechanism_difference(baseline_trace, candidate_trace)
                baseline_output = baseline_trace["output"]
                candidate_output = candidate_trace["output"]
                if not isinstance(baseline_output, Mapping) or not isinstance(
                    candidate_output, Mapping
                ):
                    raise RuntimeError("model trace output must be a mapping")
                output_change = relative_output_rms(
                    baseline_output,
                    candidate_output,
                )
                decision = stage0_decision(
                    activation,
                    mechanism=mechanism,
                    relative_output_rms=output_change,
                )
                counterfactuals[name] = {
                    "activation": activation,
                    "mechanism": mechanism,
                    "relative_output_rms": output_change,
                    "decision": decision,
                }
                if centers is None:
                    raw_centers = candidate_trace.get("centers")
                    if not isinstance(raw_centers, dict):
                        raise RuntimeError("interacting memory did not emit centers")
                    centers = raw_centers

            identity_activation = counterfactuals["identity"]["activation"]
            radial_activation = counterfactuals["radial"]["activation"]
            if not _activation_materially_nonconstant(identity_activation):
                diagnosis = "router_functionally_inactive"
            elif not _activation_materially_nonconstant(radial_activation):
                diagnosis = "radial_coupling_collapsed"
            else:
                diagnosis = "identity_activates"
            arms.append(
                {
                    "memory_count": memory_count,
                    **candidate_hashes,
                    "router": "shared_invariant_mlp_v1",
                    "centers": centers,
                    "counterfactuals": counterfactuals,
                    "diagnosis": diagnosis,
                }
            )

    selected_identity_mix: float | None = None
    for value in IDENTITY_MIX_CANDIDATES:
        name = f"lambda_{value:.2f}"
        if all(
            bool(arm["counterfactuals"][name]["decision"]["passed"])
            for arm in arms
        ):
            selected_identity_mix = value
            break
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "probe": "hemm_stage0_pair_gate_activation",
        "test_evaluated": False,
        "seed": int(seed),
        "scenario": scenario,
        "hidden_dim": int(hidden_dim),
        "num_heads": int(num_heads),
        "source_sha256": _source_hash(),
        "device": str(device),
        "dtype": str(dtype).removeprefix("torch."),
        "node_count": 16,
        "thresholds": {
            "assignment_entropy_over_log_m": [ENTROPY_MIN, ENTROPY_MAX],
            "marginal_entropy_over_log_m_min": MARGINAL_ENTROPY_MIN,
            "mutual_information_over_log_m_min": MUTUAL_INFORMATION_MIN,
            "occupancy_fraction_min": OCCUPANCY_FRACTION_MIN,
            "coupling_q00_max": COUPLING_Q00_MAX,
            "pair_gate_centered_frobenius_ratio_min": PAIR_GATE_CENTERED_FROBENIUS_MIN,
            "pair_gate_nonconstant_fraction_min": PAIR_GATE_NONCONSTANT_FRACTION_MIN,
            "pair_gate_nonconstant_relative_tolerance": NONCONSTANT_RELATIVE_TOLERANCE,
            "relative_output_rms_min": RELATIVE_OUTPUT_RMS_MIN,
            "mechanism_symmetric_relative_rms_min": MECHANISM_RMS_MIN,
        },
        "baseline": {"memory_count": 1, **baseline_hashes},
        "arms": arms,
        "selected_identity_mix": selected_identity_mix,
        "decision": (
            "admit_interacting_memory_arms"
            if selected_identity_mix is not None
            else "block_interacting_memory_arms"
        ),
    }
    json.dumps(result, allow_nan=False)
    return result


def run_suite(
    *,
    memory_counts: Sequence[int],
    hidden_dims: Sequence[int],
    seeds: Sequence[int],
    scenarios: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
    num_heads: int,
) -> dict[str, object]:
    """Run and conservatively aggregate the registered Stage-0 lane matrix."""

    widths = tuple(int(value) for value in hidden_dims)
    fixed_seeds = tuple(int(value) for value in seeds)
    graph_scenarios = tuple(str(value) for value in scenarios)
    if not widths or len(widths) != len(set(widths)):
        raise ValueError("hidden_dims must contain unique values")
    if not fixed_seeds or len(fixed_seeds) != len(set(fixed_seeds)):
        raise ValueError("seeds must contain unique values")
    if "aligned" not in graph_scenarios or len(graph_scenarios) != len(
        set(graph_scenarios)
    ):
        raise ValueError("scenarios must uniquely include aligned")

    lanes = [
        run_probe(
            memory_counts=memory_counts,
            device=device,
            dtype=dtype,
            seed=seed,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            scenario=scenario,
        )
        for scenario in graph_scenarios
        for hidden_dim in widths
        for seed in fixed_seeds
    ]
    aligned_lanes = [lane for lane in lanes if lane["scenario"] == "aligned"]
    selected_identity_mix: float | None = None
    for value in IDENTITY_MIX_CANDIDATES:
        name = f"lambda_{value:.2f}"
        if all(
            bool(arm["counterfactuals"][name]["decision"]["passed"])
            for lane in aligned_lanes
            for arm in lane["arms"]
        ):
            selected_identity_mix = value
            break

    semantic_lanes = [lane for lane in lanes if lane["scenario"] == "semantic_only"]
    semantic_identity_active = bool(semantic_lanes) and all(
        _activation_materially_nonconstant(
            arm["counterfactuals"]["identity"]["activation"]
        )
        for lane in semantic_lanes
        for arm in lane["arms"]
    )
    result = {
        "schema_version": 1,
        "probe": "hemm_stage0_registered_suite",
        "test_evaluated": False,
        "source_sha256": _source_hash(),
        "device": str(device),
        "dtype": str(dtype).removeprefix("torch."),
        "hidden_dims": list(widths),
        "seeds": list(fixed_seeds),
        "scenarios": list(graph_scenarios),
        "memory_counts": [int(value) for value in memory_counts],
        "identity_mix_candidates": list(IDENTITY_MIX_CANDIDATES),
        "thresholds": lanes[0]["thresholds"],
        "lanes": lanes,
        "semantic_identity_active": semantic_identity_active,
        "selected_identity_mix": selected_identity_mix,
        "scientific_decision": (
            "admit_interacting_memory_arms"
            if selected_identity_mix is not None
            else "block_interacting_memory_arms"
        ),
    }
    json.dumps(result, allow_nan=False)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-counts", type=int, nargs="+", default=(4, 8))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument(
        "--scenario",
        choices=("aligned", "crossed", "spatial_only", "semantic_only"),
        default="aligned",
    )
    parser.add_argument("--metrics-out", type=Path)
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=(16, 64))
    parser.add_argument("--seeds", type=int, nargs="+", default=(401, 402, 403))
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=("aligned", "crossed", "spatial_only", "semantic_only"),
        default=("aligned", "crossed", "spatial_only", "semantic_only"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = {"float32": torch.float32, "float64": torch.float64}[args.dtype]
    if args.suite:
        result = run_suite(
            memory_counts=args.memory_counts,
            hidden_dims=args.hidden_dims,
            seeds=args.seeds,
            scenarios=args.scenarios,
            device=torch.device(args.device),
            dtype=dtype,
            num_heads=args.num_heads,
        )
    else:
        result = run_probe(
            memory_counts=args.memory_counts,
            device=torch.device(args.device),
            dtype=dtype,
            seed=args.seed,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            scenario=args.scenario,
        )
    serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.metrics_out is not None:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
