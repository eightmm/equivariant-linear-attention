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

from equivariant_attention import EquivariantAttention, EquivariantAttentionConfig
from equivariant_attention.diagnostics import memory_pair_gate_summary
from equivariant_attention.moment import (
    _memory_assignments_and_coupling,
    _normalize_positive_features,
)


SCHEMA_VERSION = 1
NONCONSTANT_RELATIVE_TOLERANCE = 1e-3
ENTROPY_MIN = 0.05
ENTROPY_MAX = 0.995
OCCUPANCY_FRACTION_MIN = 1e-4
COUPLING_Q00_MAX = 0.99
PAIR_GATE_CENTERED_FROBENIUS_MIN = 1e-2
PAIR_GATE_NONCONSTANT_FRACTION_MIN = 0.10
RELATIVE_OUTPUT_RMS_MIN = 1e-5
CV_IDENTITY_ABS_TOLERANCE = 1e-10


def probe_graph(
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the fixed heterogeneous graph registered for the Stage-0 gate."""

    if dtype not in {torch.float32, torch.float64}:
        raise ValueError("probe dtype must be float32 or float64")
    cluster_ids = torch.arange(4, device=device).repeat_interleave(4)
    node_feats = F.one_hot(cluster_ids, num_classes=4).to(dtype=dtype)
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
    batch = torch.zeros(16, dtype=torch.long, device=device)
    return node_feats, pos, batch


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


def _forward_and_activation(
    model: EquivariantAttention,
    node_feats: torch.Tensor,
    pos: torch.Tensor,
    batch: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    layer = model.layers[1]
    captured: dict[str, torch.Tensor | int] = {}

    def capture_layer_input(
        _module: torch.nn.Module,
        inputs: tuple[object, ...],
    ) -> None:
        captured["scalars"] = inputs[0].detach()
        captured["global_pos"] = inputs[2].detach()
        captured["batch"] = inputs[4].detach()
        captured["num_graphs"] = int(inputs[5])

    training_states = [(module, module.training) for module in model.modules()]
    handle = layer.register_forward_pre_hook(capture_layer_input)
    try:
        model.eval()
        with torch.no_grad():
            output = model(node_feats, pos, batch=batch)
    finally:
        handle.remove()
        for module, training in training_states:
            module.training = training
    if not captured:
        raise RuntimeError("middle global layer hook did not execute")

    with torch.no_grad():
        scalars = captured["scalars"]
        global_pos = captured["global_pos"]
        captured_batch = captured["batch"]
        num_graphs = int(captured["num_graphs"])
        if num_graphs != 1:
            raise RuntimeError("Stage-0 activation probe requires exactly one graph")
        normalized_scalars = layer.norm(scalars)
        key_scalar = _normalize_positive_features(
            F.elu(
                layer.key_scalar(normalized_scalars).reshape(
                    scalars.shape[0],
                    layer.num_heads,
                    layer.head_dim,
                )
            )
            + 1.0,
            layer.eps,
        )
        global_heads = slice(layer.local_head_count, layer.num_heads)
        assignment, coupling, _ = _memory_assignments_and_coupling(
            key_scalar[:, global_heads],
            global_pos,
            captured_batch,
            num_graphs=num_graphs,
            memory_count=layer.global_memory_count,
            temperature=layer.memory_assignment_temperature,
            assignment_scale=layer.memory_assignment_scale,
            interaction_cutoff=layer.memory_interaction_cutoff,
            interact=True,
        )
        activation = memory_pair_gate_summary(
            assignment,
            coupling[0],
            nonconstant_relative_tolerance=NONCONSTANT_RELATIVE_TOLERANCE,
        )
    return output, activation


def run_probe(
    *,
    memory_counts: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    hidden_dim: int,
    num_heads: int,
) -> dict[str, object]:
    """Run matched-state M=1 versus M>1 activation arms."""

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
        node_feats, pos, batch = probe_graph(dtype=dtype, device=device)
        baseline_model = EquivariantAttention(
            _model_config(
                memory_count=1,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
            )
        ).to(device=device, dtype=dtype)
        baseline_state = baseline_model.state_dict()
        baseline_hashes = _state_hashes(baseline_model)
        with torch.no_grad():
            baseline_model.eval()
            baseline_output = baseline_model(node_feats, pos, batch=batch)

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
            candidate_output, activation = _forward_and_activation(
                candidate,
                node_feats,
                pos,
                batch,
            )
            output_change = relative_output_rms(baseline_output, candidate_output)
            decision = stage0_decision(
                activation,
                relative_output_rms=output_change,
            )
            arms.append(
                {
                    "memory_count": memory_count,
                    **candidate_hashes,
                    "relative_output_rms": output_change,
                    "activation": activation,
                    "decision": decision,
                }
            )

    passed = all(bool(arm["decision"]["passed"]) for arm in arms)
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "probe": "hemm_stage0_pair_gate_activation",
        "test_evaluated": False,
        "seed": int(seed),
        "source_sha256": _source_hash(),
        "device": str(device),
        "dtype": str(dtype).removeprefix("torch."),
        "node_count": 16,
        "thresholds": {
            "assignment_entropy_over_log_m": [ENTROPY_MIN, ENTROPY_MAX],
            "occupancy_fraction_min": OCCUPANCY_FRACTION_MIN,
            "coupling_q00_max": COUPLING_Q00_MAX,
            "pair_gate_centered_frobenius_ratio_min": PAIR_GATE_CENTERED_FROBENIUS_MIN,
            "pair_gate_nonconstant_fraction_min": PAIR_GATE_NONCONSTANT_FRACTION_MIN,
            "pair_gate_nonconstant_relative_tolerance": NONCONSTANT_RELATIVE_TOLERANCE,
            "relative_output_rms_min": RELATIVE_OUTPUT_RMS_MIN,
        },
        "baseline": {"memory_count": 1, **baseline_hashes},
        "arms": arms,
        "decision": (
            "admit_interacting_memory_arms"
            if passed
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
    parser.add_argument("--metrics-out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = {"float32": torch.float32, "float64": torch.float64}[args.dtype]
    result = run_probe(
        memory_counts=args.memory_counts,
        device=torch.device(args.device),
        dtype=dtype,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
    )
    serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.metrics_out is not None:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
