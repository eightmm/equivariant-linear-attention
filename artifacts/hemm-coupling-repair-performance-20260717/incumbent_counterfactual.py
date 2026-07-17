#!/usr/bin/env python3
"""Reproduce same-assignment coupling counterfactuals on the clean incumbent."""

from __future__ import annotations

import argparse
import json
import math
import runpy
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F

from equivariant_attention import EquivariantAttention
from equivariant_attention.diagnostics import pair_gate_summary
from equivariant_attention.moment import (
    _memory_assignments_and_coupling,
    _normalize_positive_features,
)


ROOT = Path(__file__).resolve().parents[2]
PROBE = runpy.run_path(ROOT / "scripts" / "probe_memory_activation.py")


def _entropy(probabilities: torch.Tensor) -> dict[str, float]:
    log_m = math.log(probabilities.shape[-1])
    safe_log = torch.where(
        probabilities > 0.0,
        probabilities.log(),
        torch.zeros_like(probabilities),
    )
    conditional = float((-(probabilities * safe_log).sum(-1).mean() / log_m).item())
    marginal = probabilities.mean(0)
    marginal_log = torch.where(
        marginal > 0.0,
        marginal.log(),
        torch.zeros_like(marginal),
    )
    marginal_entropy = float((-(marginal * marginal_log).sum() / log_m).item())
    return {
        "conditional_entropy_over_log_m": conditional,
        "marginal_entropy_over_log_m": marginal_entropy,
        "mutual_information_over_log_m": marginal_entropy - conditional,
    }


def _capture_assignment(
    *, hidden_dim: int, seed: int, memory_count: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str]:
    torch.manual_seed(seed)
    node_feats, pos, batch = PROBE["probe_graph"](
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    baseline = EquivariantAttention(
        PROBE["_model_config"](
            memory_count=1,
            hidden_dim=hidden_dim,
            num_heads=4,
        )
    ).double()
    state = baseline.state_dict()
    candidate = EquivariantAttention(
        PROBE["_model_config"](
            memory_count=memory_count,
            hidden_dim=hidden_dim,
            num_heads=4,
        )
    ).double()
    candidate.load_state_dict(state, strict=True)
    captured: dict[str, object] = {}

    def hook(_module: torch.nn.Module, inputs: tuple[object, ...]) -> None:
        captured["scalars"] = inputs[0]
        captured["global_pos"] = inputs[2]
        captured["batch"] = inputs[4]
        captured["num_graphs"] = inputs[5]

    handle = candidate.layers[1].register_forward_pre_hook(hook)
    try:
        candidate.eval()
        with torch.no_grad():
            candidate(node_feats, pos, batch=batch)
    finally:
        handle.remove()

    layer = candidate.layers[1]
    scalars = captured["scalars"]
    assert isinstance(scalars, torch.Tensor)
    normalized = layer.norm(scalars)
    key_scalar = _normalize_positive_features(
        F.elu(
            layer.key_scalar(normalized).reshape(
                scalars.shape[0], layer.num_heads, layer.head_dim
            )
        )
        + 1.0,
        layer.eps,
    )
    global_pos = captured["global_pos"]
    captured_batch = captured["batch"]
    assert isinstance(global_pos, torch.Tensor)
    assert isinstance(captured_batch, torch.Tensor)
    assignment, radial, centers = _memory_assignments_and_coupling(
        key_scalar,
        global_pos,
        captured_batch,
        num_graphs=int(captured["num_graphs"]),
        memory_count=memory_count,
        temperature=layer.memory_assignment_temperature,
        assignment_scale=layer.memory_assignment_scale,
        interaction_cutoff=layer.memory_interaction_cutoff,
        interact=True,
    )
    hashes = PROBE["_state_hashes"](candidate)
    return (
        assignment,
        radial[0],
        centers[0],
        hashes["state_sha256"],
        hashes["state_schema_sha256"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    lambdas = (0.10, 0.25, 0.50)
    rows: list[dict[str, object]] = []
    for hidden_dim in (16, 64):
        for seed in (401, 402, 403):
            for memory_count in (4, 8):
                assignment, radial, centers, state_hash, schema_hash = (
                    _capture_assignment(
                        hidden_dim=hidden_dim,
                        seed=seed,
                        memory_count=memory_count,
                    )
                )
                identity = torch.eye(memory_count, dtype=torch.float64).expand(
                    4, -1, -1
                )
                ones = torch.ones_like(radial)
                couplings = {
                    "ones": ones,
                    "radial": radial,
                    "identity": identity,
                    **{
                        f"lambda_{value:.2f}": (1.0 - value) * radial
                        + value * identity
                        for value in lambdas
                    },
                }
                head_metrics: list[dict[str, object]] = []
                for head in range(4):
                    per_coupling: dict[str, object] = {}
                    for name, coupling in couplings.items():
                        gate = torch.einsum(
                            "im,mn,jn->ij",
                            assignment[:, head],
                            coupling[head],
                            assignment[:, head],
                        )
                        per_coupling[name] = pair_gate_summary(gate)
                    head_metrics.append(
                        {
                            "head": head,
                            "assignment": _entropy(assignment[:, head]),
                            "pair_gates": per_coupling,
                        }
                    )
                center_delta = centers.unsqueeze(-2) - centers.unsqueeze(-3)
                center_distance = torch.linalg.vector_norm(center_delta, dim=-1)
                offdiag = center_distance[
                    ~torch.eye(memory_count, dtype=torch.bool).expand(4, -1, -1)
                ]
                rows.append(
                    {
                        "hidden_dim": hidden_dim,
                        "seed": seed,
                        "memory_count": memory_count,
                        "state_sha256": state_hash,
                        "state_schema_sha256": schema_hash,
                        "center_offdiagonal_distance": {
                            "min": float(offdiag.min().item()),
                            "median": float(offdiag.median().item()),
                            "max": float(offdiag.max().item()),
                        },
                        "heads": head_metrics,
                    }
                )

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    )
    result = {
        "schema_version": 1,
        "probe": "incumbent_same_assignment_coupling_counterfactual",
        "base_commit": commit,
        "source_sha256": PROBE["_source_hash"](),
        "working_tree_dirty": bool(status),
        "source_code_modified_from_base": False,
        "test_evaluated": False,
        "dtype": "float64",
        "device": "cpu",
        "hidden_dims": [16, 64],
        "seeds": [401, 402, 403],
        "memory_counts": [4, 8],
        "lambdas": list(lambdas),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
