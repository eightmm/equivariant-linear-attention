#!/usr/bin/env python3
"""Paired mechanics ablation for relation-conditioned local transport.

The synthetic target is carried only by invariant edge relation IDs.  Both
arms receive exactly the same graph, start from an identical state, and retain
the same relation schema and cutoffs.  The control freezes only the private
relation-specific modulation parameters at their identity initialization.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from equivariant_linear_attention import ELA, ELAGraph


_RELATION_PARAMETER_SUFFIXES = (
    "relation_score_bias",
    "relation_radial_scale",
    "relation_value_gate",
)


def _relation_parameter_names(model: ELA) -> tuple[str, ...]:
    return tuple(
        name
        for name, _ in model.named_parameters()
        if name.endswith(_RELATION_PARAMETER_SUFFIXES)
    )


def _update_hash_with_tensor(
    digest: Any,
    name: str,
    tensor: torch.Tensor,
) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    if value.numel():
        digest.update(bytes(value.reshape(-1).view(torch.uint8).tolist()))


def _state_sha256(model: ELA) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        _update_hash_with_tensor(digest, name, tensor)
    return digest.hexdigest()


def _schema_sha256(model: ELA) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(str(parameter.dtype).encode("ascii"))
        digest.update(str(tuple(parameter.shape)).encode("ascii"))
    return digest.hexdigest()


def _task_sha256(graph: ELAGraph, target: torch.Tensor) -> str:
    if graph.edge_index is None or graph.edge_type is None or graph.batch is None:
        raise RuntimeError("relation ablation task must use an explicit packed graph")
    digest = hashlib.sha256()
    tensors = {
        "x": graph.x,
        "pos": graph.pos,
        "edge_index": graph.edge_index,
        "batch": graph.batch,
        "edge_type": graph.edge_type,
        "target": target,
    }
    for name, tensor in tensors.items():
        _update_hash_with_tensor(digest, name, tensor)
    return digest.hexdigest()


def _complete_directed_edges(nodes: int) -> torch.Tensor:
    pairs = [(sender, receiver) for sender in range(nodes) for receiver in range(nodes)]
    pairs = [pair for pair in pairs if pair[0] != pair[1]]
    return torch.tensor(pairs, dtype=torch.long).T.contiguous()


def _relation_task(graphs: int) -> tuple[ELAGraph, torch.Tensor]:
    if graphs < 2 or graphs % 2:
        raise ValueError("graphs must be an even integer of at least two")
    nodes = 4
    node_features = torch.tensor(
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
        )
    )
    positions = torch.tensor(
        (
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    local_edges = _complete_directed_edges(nodes)
    edge_count = local_edges.shape[1]
    edge_index = torch.cat(
        tuple(local_edges + graph_index * nodes for graph_index in range(graphs)),
        dim=1,
    )
    relation = torch.cat(
        tuple(
            torch.full(
                (edge_count,),
                graph_index % 2,
                dtype=torch.long,
            )
            for graph_index in range(graphs)
        )
    )
    target = torch.where(
        torch.arange(graphs) % 2 == 0,
        -torch.ones(graphs),
        torch.ones(graphs),
    )
    graph = ELAGraph(
        x=node_features.repeat(graphs, 1),
        pos=positions.repeat(graphs, 1),
        edge_index=edge_index,
        batch=torch.arange(graphs).repeat_interleave(nodes),
        edge_type=relation,
        y=target[:, None],
    )
    return graph, target


def _open_zero_initialized_local_readout(model: ELA, value: float) -> tuple[str, ...]:
    names: list[str] = []
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("local_scalar_out.weight"):
                parameter.fill_(value)
                names.append(name)
    if len(names) != model.config.depth:
        raise RuntimeError("expected one local scalar output projection per layer")
    return tuple(names)


def _freeze_relation_modulation(model: ELA) -> tuple[str, ...]:
    names = _relation_parameter_names(model)
    expected = model.config.depth * len(_RELATION_PARAMETER_SUFFIXES)
    if len(names) != expected:
        raise RuntimeError(
            f"expected {expected} relation parameters, found {len(names)}"
        )
    for name, parameter in model.named_parameters():
        if name in names:
            with torch.no_grad():
                parameter.zero_()
            parameter.requires_grad_(False)
    return names


def _set_relation_identity(model: ELA) -> tuple[str, ...]:
    names = _relation_parameter_names(model)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in names:
                parameter.zero_()
    return names


def _parameter_max_abs(model: ELA, names: Iterable[str]) -> dict[str, float]:
    selected = set(names)
    return {
        name: float(parameter.detach().abs().max().item())
        for name, parameter in model.named_parameters()
        if name in selected
    }


def _predict(model: ELA, graph: ELAGraph) -> torch.Tensor:
    output = model(graph).graph_x
    if output is None or output.shape[-1] != 1:
        raise RuntimeError("relation ablation expects one invariant graph output")
    return output[:, 0]


def _train_arm(
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
        initial_prediction = _predict(model, graph)
        initial_loss = (initial_prediction - target).square().mean()

    losses: list[float] = []
    start = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = _predict(model, graph)
        loss = (prediction - target).square().mean()
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("non-finite relation ablation loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
    elapsed = time.perf_counter() - start

    with torch.no_grad():
        final_prediction = _predict(model, graph)
        final_loss = float((final_prediction - target).square().mean().item())
        by_relation = {
            str(relation): float(final_prediction[relation::2].mean().item())
            for relation in (0, 1)
        }
    return {
        "initial_loss": float(initial_loss.item()),
        "final_loss": final_loss,
        "best_pre_update_loss": min(losses, default=float(initial_loss.item())),
        "final_prediction_mean_by_relation": by_relation,
        "elapsed_seconds": elapsed,
        "mean_step_ms": 0.0 if steps == 0 else elapsed * 1000.0 / steps,
        "steps": steps,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def run_ablation(
    *,
    seed: int,
    graphs: int,
    steps: int,
    width: int,
    depth: int,
    learning_rate: float,
    threads: int,
    local_readout_bootstrap: float,
) -> dict[str, Any]:
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    if width <= 0 or depth <= 0:
        raise ValueError("width and depth must be positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive and finite")
    if threads <= 0:
        raise ValueError("threads must be positive")
    if not math.isfinite(local_readout_bootstrap):
        raise ValueError("local_readout_bootstrap must be finite")

    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)

    base = ELA(
        "3x0e",
        "1x0e",
        width=width,
        depth=depth,
        cutoff=4.0,
        edge_types=2,
    )
    opened_parameters = _open_zero_initialized_local_readout(
        base,
        local_readout_bootstrap,
    )
    _set_relation_identity(base)
    candidate = copy.deepcopy(base)
    control = copy.deepcopy(base)
    frozen_relation_parameters = _freeze_relation_modulation(control)

    candidate_state = _state_sha256(candidate)
    control_state = _state_sha256(control)
    candidate_schema = _schema_sha256(candidate)
    control_schema = _schema_sha256(control)
    graph_candidate, target_candidate = _relation_task(graphs)
    graph_control, target_control = _relation_task(graphs)
    candidate_task = _task_sha256(graph_candidate, target_candidate)
    control_task = _task_sha256(graph_control, target_control)
    if candidate_state != control_state or candidate_schema != control_schema:
        raise RuntimeError("paired arms must start from identical state and schema")
    if candidate_task != control_task:
        raise RuntimeError("paired arms must receive identical task tensors")
    with torch.no_grad():
        candidate_initial = _predict(candidate, graph_candidate)
        control_initial = _predict(control, graph_control)
    initial_output_max_abs = float(
        (candidate_initial - control_initial).abs().max().item()
    )
    if initial_output_max_abs != 0.0:
        raise RuntimeError("paired arms must have exactly equal initial outputs")

    candidate_result = _train_arm(
        candidate,
        graph_candidate,
        target_candidate,
        steps=steps,
        learning_rate=learning_rate,
    )
    control_result = _train_arm(
        control,
        graph_control,
        target_control,
        steps=steps,
        learning_rate=learning_rate,
    )
    candidate_final = candidate_result["final_loss"]
    control_final = control_result["final_loss"]
    assert isinstance(candidate_final, float)
    assert isinstance(control_final, float)

    return {
        "schema_version": 1,
        "experiment": "relation_conditioned_transport_paired_ablation",
        "device": "cpu",
        "dtype": "float32",
        "seed": seed,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "threads": threads,
        "task": {
            "graphs": graphs,
            "nodes_per_graph": 4,
            "edges_per_graph": 12,
            "relation_types": 2,
            "shared_cutoff": 4.0,
            "target": "-1 for relation 0; +1 for relation 1",
            "only_varying_input": "edge_type",
            "sha256": candidate_task,
        },
        "model": {
            "input_irreps": "3x0e",
            "output_irreps": "1x0e",
            "width": width,
            "depth": depth,
            "local_readout_bootstrap": local_readout_bootstrap,
            "opened_parameters": list(opened_parameters),
            "relation_parameter_names": list(frozen_relation_parameters),
        },
        "initial_identity": {
            "candidate_state_sha256": candidate_state,
            "control_state_sha256": control_state,
            "full_state_equal": candidate_state == control_state,
            "candidate_schema_sha256": candidate_schema,
            "control_schema_sha256": control_schema,
            "parameter_schema_equal": candidate_schema == control_schema,
            "task_tensors_equal": candidate_task == control_task,
            "initial_output_max_abs": initial_output_max_abs,
        },
        "ablation": {
            "candidate": "all ELA parameters trainable",
            "control": "only relation modulation frozen at identity",
            "control_frozen_parameters": list(frozen_relation_parameters),
            "shared_relation_cutoffs": True,
            "shared_topology": True,
            "shared_inputs": True,
        },
        "arms": {
            "candidate": candidate_result,
            "control": control_result,
        },
        "relation_parameter_max_abs_after_training": {
            "candidate": _parameter_max_abs(candidate, frozen_relation_parameters),
            "control": _parameter_max_abs(control, frozen_relation_parameters),
        },
        "paired_outcome": {
            "candidate_minus_control_final_loss": candidate_final - control_final,
            "candidate_lower_final_loss": candidate_final < control_final,
            "candidate_to_control_final_loss_ratio": (
                candidate_final / control_final if control_final > 0.0 else None
            ),
        },
        "scope": (
            "synthetic mechanics ablation; not downstream accuracy, "
            "generalization, or model-superiority evidence"
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--graphs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5.0e-3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--local-readout-bootstrap", type=float, default=0.1)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    payload = run_ablation(
        seed=args.seed,
        graphs=args.graphs,
        steps=args.steps,
        width=args.width,
        depth=args.depth,
        learning_rate=args.learning_rate,
        threads=args.threads,
        local_readout_bootstrap=args.local_readout_bootstrap,
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
