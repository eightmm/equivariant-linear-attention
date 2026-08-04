#!/usr/bin/env python3
"""Deterministic paired mechanics ablations for private ELA lanes.

The candidate and control in each lane have identical parameter schemas,
bit-identical initial state, and identical task tensors. Shared parameters are
frozen in both arms. Only the selected zero-initialized lane is trainable in
the candidate; the control keeps it frozen at identity and privately disabled.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from ablate_relation_transport import (
    _schema_sha256,
    _state_sha256,
    _update_hash_with_tensor,
)
from equivariant_linear_attention import ELA, ELAGraph
from equivariant_linear_attention.nn.multipoles import (
    _set_l1_l2_closure_enabled,
)


_CG12_TOKEN = "tensor_closure.l1_l2_"
_MULTISCALE_SUFFIXES = (
    "local_scale_score_mix",
    "local_scale_value_mix",
)
_CG12_INPUT_IRREPS = "2x0e + 1x1o + 1x1e + 1x2e + 1x2o"
_CG12_OUTPUT_IRREPS = "1x1o + 1x1e + 1x2e + 1x2o"


def _complete_directed_edges(nodes: int) -> torch.Tensor:
    pairs = [
        (sender, receiver)
        for sender in range(nodes)
        for receiver in range(nodes)
        if sender != receiver
    ]
    return torch.tensor(pairs, dtype=torch.long).T.contiguous()


def _task_sha256(graph: ELAGraph, target: torch.Tensor) -> str:
    if graph.edge_index is None or graph.batch is None:
        raise RuntimeError("architecture ablation requires explicit batched edges")
    digest = hashlib.sha256()
    for name, tensor in (
        ("x", graph.x),
        ("pos", graph.pos),
        ("edge_index", graph.edge_index),
        ("batch", graph.batch),
        ("target", target),
    ):
        _update_hash_with_tensor(digest, name, tensor)
    return digest.hexdigest()


def _target_parameter_names(model: ELA, lane: str) -> tuple[str, ...]:
    if lane == "cg12":
        names = tuple(
            name for name, _ in model.named_parameters() if _CG12_TOKEN in name
        )
        expected = 4 * model.config.depth
    elif lane == "multiscale":
        names = tuple(
            name
            for name, _ in model.named_parameters()
            if name.endswith(_MULTISCALE_SUFFIXES)
        )
        expected = 2 * model.config.depth
    else:
        raise ValueError(f"unsupported architecture lane: {lane}")
    if len(names) != expected:
        raise RuntimeError(f"expected {expected} {lane} parameters, found {len(names)}")
    return names


def _set_target_identity(model: ELA, names: tuple[str, ...]) -> None:
    selected = set(names)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in selected:
                parameter.zero_()


def _freeze_except(model: ELA, names: tuple[str, ...]) -> tuple[str, ...]:
    selected = set(names)
    trainable: list[str] = []
    for name, parameter in model.named_parameters():
        enabled = name in selected
        parameter.requires_grad_(enabled)
        if enabled:
            trainable.append(name)
    if tuple(trainable) != names:
        raise RuntimeError("trainable target-lane schema changed unexpectedly")
    return tuple(trainable)


def _freeze_all(model: ELA) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _parameter_max_abs(
    model: ELA,
    names: tuple[str, ...],
) -> dict[str, float]:
    selected = set(names)
    return {
        name: float(parameter.detach().abs().max().item())
        for name, parameter in model.named_parameters()
        if name in selected
    }


def _cg12_graph(model: ELA, *, seed: int) -> tuple[ELAGraph, dict[str, float]]:
    graphs = 8
    nodes = 5
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(
        graphs * nodes,
        model.config.input_layout.dim,
        generator=generator,
    )
    # Remove scalar input information: all task energy resides in persistent
    # l=1 and l=2 sectors before the target 1 x 2 closure is applied.
    features[:, :2].zero_()
    positions = torch.cat(
        tuple(torch.randn(nodes, 3, generator=generator) * 0.4 for _ in range(graphs))
    )
    local_edges = _complete_directed_edges(nodes)
    edges = torch.cat(
        tuple(local_edges + graph_index * nodes for graph_index in range(graphs)),
        dim=1,
    )
    graph = ELAGraph(
        features,
        positions,
        edge_index=edges,
        batch=torch.arange(graphs).repeat_interleave(nodes),
    )
    sector_rms = {
        "scalar_l0": float(features[:, :2].square().mean().sqrt().item()),
        "vector_l1": float(features[:, 2:8].square().mean().sqrt().item()),
        "tensor_l2": float(features[:, 8:].square().mean().sqrt().item()),
    }
    return graph, sector_rms


def _multiscale_graph() -> tuple[ELAGraph, torch.Tensor]:
    graphs = 8
    nodes = 4
    features = torch.tensor(
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
        )
    ).repeat(graphs, 1)
    template = torch.tensor(
        (
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    radial_scales = torch.linspace(0.25, 0.95, graphs)
    positions = torch.cat(tuple(template * scale for scale in radial_scales))
    local_edges = _complete_directed_edges(nodes)
    edges = torch.cat(
        tuple(local_edges + graph_index * nodes for graph_index in range(graphs)),
        dim=1,
    )
    graph = ELAGraph(
        features,
        positions,
        edge_index=edges,
        batch=torch.arange(graphs).repeat_interleave(nodes),
    )
    return graph, radial_scales


def _set_teacher_parameters(
    teacher: ELA,
    names: tuple[str, ...],
    *,
    lane: str,
) -> None:
    selected = set(names)
    index = 0
    with torch.no_grad():
        for name, parameter in teacher.named_parameters():
            if name not in selected:
                continue
            if lane == "cg12":
                values = torch.linspace(
                    -0.7,
                    0.7,
                    parameter.numel(),
                    dtype=parameter.dtype,
                    device=parameter.device,
                ).reshape_as(parameter)
                parameter.copy_(values + 0.1 * index)
            elif name.endswith("local_scale_score_mix"):
                parameter.copy_(
                    torch.tensor(
                        ((2.0,), (-2.0,)),
                        dtype=parameter.dtype,
                        device=parameter.device,
                    ).expand_as(parameter)
                )
            else:
                parameter.copy_(
                    torch.linspace(
                        -2.0,
                        2.0,
                        parameter.numel(),
                        dtype=parameter.dtype,
                        device=parameter.device,
                    ).reshape_as(parameter)
                )
            index += 1


def _set_lane_enabled(model: ELA, lane: str, enabled: bool) -> int:
    if lane == "cg12":
        return _set_l1_l2_closure_enabled(model, enabled)
    if lane == "multiscale":
        for layer in model.layers:
            layer._set_multiscale_local_enabled(enabled)
        return len(model.layers)
    raise ValueError(f"unsupported architecture lane: {lane}")


def _train_candidate(
    model: ELA,
    graph: ELAGraph,
    target: torch.Tensor,
    *,
    steps: int,
    learning_rate: float,
) -> dict[str, Any]:
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(trainable, lr=learning_rate)
    with torch.no_grad():
        initial_loss = float((model(graph).x - target).square().mean().item())
    losses: list[float] = []
    start = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(graph).x - target).square().mean()
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("non-finite architecture-lane loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
    elapsed = time.perf_counter() - start
    with torch.no_grad():
        final_loss = float((model(graph).x - target).square().mean().item())
    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "best_pre_update_loss": min(losses, default=initial_loss),
        "steps": steps,
        "elapsed_seconds": elapsed,
        "mean_step_ms": 0.0 if steps == 0 else elapsed * 1000.0 / steps,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
    }


def _run_lane(
    lane: str,
    *,
    seed: int,
    steps: int,
    learning_rate: float,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if lane == "cg12":
        base = ELA(
            _CG12_INPUT_IRREPS,
            _CG12_OUTPUT_IRREPS,
            width=16,
            depth=1,
            cutoff=4.0,
        )
        graph, sector_rms = _cg12_graph(base, seed=seed + 1)
        task_details: dict[str, Any] = {
            "kind": "rich_irreps_vector_tensor_teacher_reconstruction",
            "graphs": 8,
            "nodes_per_graph": 5,
            "edges_per_graph": 20,
            "input_sector_rms": sector_rms,
            "requires_nonzero_l1_and_l2_inputs": True,
        }
        shared_bootstrap: dict[str, Any] = {}
    else:
        base = ELA("3x0e", "1x0e", width=16, depth=1, cutoff=4.0)
        opened: list[str] = []
        with torch.no_grad():
            for name, parameter in base.named_parameters():
                if name.endswith("local_scalar_out.weight"):
                    parameter.fill_(0.5)
                    opened.append(name)
        graph, radial_scales = _multiscale_graph()
        task_details = {
            "kind": "same_csr_radial_scale_teacher_reconstruction",
            "graphs": 8,
            "nodes_per_graph": 4,
            "edges_per_graph": 12,
            "radial_scales": [float(value) for value in radial_scales],
            "same_receiver_sender_csr_pattern": True,
        }
        shared_bootstrap = {
            "local_scalar_out": 0.5,
            "opened_parameters": opened,
        }

    names = _target_parameter_names(base, lane)
    _set_target_identity(base, names)
    candidate = copy.deepcopy(base)
    control = copy.deepcopy(base)
    teacher = copy.deepcopy(base)
    _set_lane_enabled(candidate, lane, True)
    _set_lane_enabled(control, lane, False)
    _set_lane_enabled(teacher, lane, True)
    trainable_names = _freeze_except(candidate, names)
    _freeze_all(control)
    _freeze_all(teacher)
    _set_teacher_parameters(teacher, names, lane=lane)

    candidate_state = _state_sha256(candidate)
    control_state = _state_sha256(control)
    candidate_schema = _schema_sha256(candidate)
    control_schema = _schema_sha256(control)
    if candidate_state != control_state or candidate_schema != control_schema:
        raise RuntimeError(
            "paired architecture arms must have identical state and schema"
        )
    with torch.no_grad():
        candidate_initial = candidate(graph).x
        control_initial = control(graph).x
        target = teacher(graph).x
    initial_output_max_abs = float(
        (candidate_initial - control_initial).abs().max().item()
    )
    if initial_output_max_abs != 0.0:
        raise RuntimeError("paired architecture arms must have equal initial outputs")
    teacher_signal = target - control_initial
    teacher_signal_rms = float(teacher_signal.square().mean().sqrt().item())
    if teacher_signal_rms == 0.0:
        raise RuntimeError("architecture teacher did not activate its target lane")
    task_hash = _task_sha256(graph, target)

    control_loss = float((control_initial - target).square().mean().item())
    candidate_result = _train_candidate(
        candidate,
        graph,
        target,
        steps=steps,
        learning_rate=learning_rate,
    )
    candidate_final = candidate_result["final_loss"]
    assert isinstance(candidate_final, float)

    if lane == "multiscale":
        graphs = int(task_details["graphs"])
        response = teacher_signal.reshape(graphs, -1).mean(dim=-1)
        task_details["teacher_radial_response_span"] = float(
            (response.max() - response.min()).abs().item()
        )

    return {
        "lane": lane,
        "seed": seed,
        "learning_rate": learning_rate,
        "task": {
            **task_details,
            "sha256": task_hash,
            "teacher_signal_rms": teacher_signal_rms,
        },
        "model": {
            "input_irreps": str(base.config.input_layout),
            "output_irreps": str(base.config.output_layout),
            "width": base.config.width,
            "depth": base.config.depth,
            "total_parameters": sum(
                parameter.numel() for parameter in base.parameters()
            ),
            "target_parameter_names": list(names),
            "shared_bootstrap": shared_bootstrap,
        },
        "initial_identity": {
            "candidate_state_sha256": candidate_state,
            "control_state_sha256": control_state,
            "full_state_equal": candidate_state == control_state,
            "candidate_schema_sha256": candidate_schema,
            "control_schema_sha256": control_schema,
            "parameter_schema_equal": candidate_schema == control_schema,
            "task_tensors_shared": True,
            "initial_output_max_abs": initial_output_max_abs,
        },
        "ablation": {
            "candidate_lane_enabled": True,
            "control_lane_enabled": False,
            "candidate_only_target_lane_trainable": list(trainable_names)
            == list(names),
            "candidate_trainable_parameters": list(trainable_names),
            "control_trainable_parameters": [],
            "shared_parameters_frozen": True,
        },
        "arms": {
            "candidate": candidate_result,
            "control": {
                "initial_loss": control_loss,
                "final_loss": control_loss,
                "steps": 0,
                "trainable_parameters": 0,
            },
        },
        "target_parameter_max_abs_after_training": {
            "candidate": _parameter_max_abs(candidate, names),
            "control": _parameter_max_abs(control, names),
            "teacher": _parameter_max_abs(teacher, names),
        },
        "paired_outcome": {
            "candidate_minus_control_final_loss": candidate_final - control_loss,
            "candidate_lower_final_loss": candidate_final < control_loss,
            "candidate_to_control_final_loss_ratio": (
                candidate_final / control_loss if control_loss > 0.0 else None
            ),
        },
        "scope": (
            "synthetic frozen-backbone mechanics ablation; not downstream "
            "accuracy, generalization, or architecture-superiority evidence"
        ),
    }


def run_ablations(
    *,
    lane: str,
    seed: int,
    steps: int,
    learning_rate: float,
    threads: int,
) -> dict[str, Any]:
    if lane not in {"all", "cg12", "multiscale"}:
        raise ValueError("lane must be all, cg12, or multiscale")
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive and finite")
    if threads <= 0:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    requested = ("cg12", "multiscale") if lane == "all" else (lane,)
    receipts = {
        name: _run_lane(
            name,
            seed=seed + index * 1009,
            steps=steps,
            learning_rate=learning_rate,
        )
        for index, name in enumerate(requested)
    }
    return {
        "schema_version": 1,
        "experiment": "private_architecture_lane_paired_ablations",
        "device": "cpu",
        "dtype": "float32",
        "seed": seed,
        "threads": threads,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "lanes": receipts,
        "scope": (
            "synthetic mechanics only; run preregistered real-data paired "
            "experiments before any utility or superiority claim"
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=("all", "cg12", "multiscale"), default="all")
    parser.add_argument("--seed", type=int, default=991)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=5.0e-2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    payload = run_ablations(
        lane=args.lane,
        seed=args.seed,
        steps=args.steps,
        learning_rate=args.learning_rate,
        threads=args.threads,
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
