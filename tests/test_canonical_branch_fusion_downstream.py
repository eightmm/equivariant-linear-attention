from __future__ import annotations

from pathlib import Path
import json
import runpy

import pytest
import torch

from equivariant_attention.benchmarking import GraphSample, collate_graphs


def _symbols() -> dict[str, object]:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_canonical_branch_fusion_downstream.py"
    )
    return runpy.run_path(script)


def _sample(index: int) -> GraphSample:
    generator = torch.Generator().manual_seed(100 + index)
    nodes = 4
    receiver = torch.arange(nodes).repeat_interleave(nodes)
    sender = torch.arange(nodes).repeat(nodes)
    return GraphSample(
        node_feats=torch.randn(nodes, 4, generator=generator),
        pos=torch.randn(nodes, 3, generator=generator),
        target=torch.tensor([float(index + 1)]),
        sample_id=f"sample-{index}",
        edge_index=torch.stack([receiver, sender]),
    )


def test_plan_keeps_qm9_test_and_all_lba_evaluation_splits_closed() -> None:
    symbols = _symbols()
    args = symbols["parse_args"](
        [
            "artifacts/canonical-fusion",
            "--task",
            "both",
            "--qm9-num-samples",
            "20",
            "--qm9-train-size",
            "12",
            "--qm9-val-size",
            "4",
            "--qm9-steps",
            "3",
            "--lba-steps",
            "4",
            "--dry-run",
        ]
    )
    plan = symbols["build_plan"](args)

    assert plan["determinism"] == "strict"
    assert plan["arms"] == ["identity_locked", "trainable_fusion"]
    assert plan["qm9"]["updates_per_arm"] == 3
    assert plan["qm9"]["validation_evaluated"] is True
    assert plan["qm9"]["test_evaluated"] is False
    assert plan["lba"]["updates_per_arm"] == 4
    assert plan["lba"]["split"] == "train"
    assert plan["lba"]["validation_evaluated"] is False
    assert plan["lba"]["test_evaluated"] is False
    assert plan["lba"]["claim_boundary"] == "train-only capacity/overfit"
    assert plan["registered_protocol"]["matched"] is False
    assert not hasattr(args, "evaluate_test")


def test_default_downstream_plan_matches_registered_protocol() -> None:
    symbols = _symbols()
    args = symbols["parse_args"](
        ["artifacts/canonical-fusion", "--dry-run"]
    )

    plan = symbols["build_plan"](args)

    assert plan["registered_protocol"]["matched"] is True
    assert all(plan["registered_protocol"]["checks"].values())


def test_initial_prediction_identity_is_a_hard_prerequisite() -> None:
    symbols = _symbols()

    symbols["_require_paired_initial_output"](
        {"byte_identical": True, "max_abs_difference": 0.0}
    )
    with pytest.raises(RuntimeError, match="identical initial predictions"):
        symbols["_require_paired_initial_output"](
            {"byte_identical": False, "max_abs_difference": 0.0}
        )


def test_sample_content_hash_accepts_length_one_strided_target() -> None:
    symbols = _symbols()
    sample = _sample(0)
    backing = torch.arange(19, dtype=torch.float32)
    strided = GraphSample(
        node_feats=sample.node_feats,
        pos=sample.pos,
        target=backing[::19],
        sample_id=sample.sample_id,
        edge_index=sample.edge_index,
    )

    digest = symbols["_sample_content_sha256"]([strided])

    assert isinstance(digest, str)
    assert len(digest) == 64


def test_paired_models_have_byte_identical_state_and_locked_control() -> None:
    symbols = _symbols()
    control, candidate, receipt = symbols["build_paired_models"](
        node_dim=4,
        width=16,
        depth=1,
        cutoff=5.0,
        num_rbf=4,
        seed=17,
    )

    assert receipt["byte_identical"] is True
    assert receipt["tensor_mismatch_count"] == 0
    assert (
        receipt["full_initial_state_sha256"]
        == (receipt["candidate_full_initial_state_sha256"])
    )
    assert receipt["state_schema_sha256"] == (receipt["candidate_state_schema_sha256"])
    assert receipt["branch_parameter_count"] > 0
    assert all(
        parameter.requires_grad
        for name, parameter in control.named_parameters()
        if ".branch_fusion." in name
    )
    assert all(
        parameter.requires_grad
        for name, parameter in candidate.named_parameters()
        if ".branch_fusion." in name
    )


def test_router_probe_and_gradient_diagnostics_separate_locked_control() -> None:
    symbols = _symbols()
    samples = [_sample(0), _sample(1)]
    batch = collate_graphs(samples)
    control, candidate, _receipt = symbols["build_paired_models"](
        node_dim=4,
        width=16,
        depth=1,
        cutoff=5.0,
        num_rbf=4,
        seed=23,
    )

    control_probe = symbols["fusion_probe"](control, batch)
    candidate_probe = symbols["fusion_probe"](candidate, batch)
    assert control_probe["mean_abs_weight_deviation_from_identity"] == 0.0
    assert candidate_probe["mean_abs_weight_deviation_from_identity"] == 0.0
    assert control_probe["max_abs_balance_strength"] == 0.0

    optimizer = torch.optim.AdamW(
        candidate.parameters(),
        lr=1e-3,
        weight_decay=0.0,
    )
    normalizer = symbols["fit_target_normalizer"](samples)
    step = symbols["train_step"](
        candidate,
        batch,
        optimizer,
        target_normalizer=normalizer,
        grad_clip=1.0,
    )

    assert step["router_gradient"]["trainable_parameter_count"] > 0
    assert step["router_gradient"]["with_gradient_count"] > 0
    assert step["router_gradient"]["nonzero_gradient_count"] > 0
    assert step["router_gradient"]["nonfinite_gradient_count"] == 0
    assert step["router_gradient"]["l2_norm"] > 0.0
    assert step["router_gradient"]["squared_norm_share"] > 0.0


def test_identity_locked_control_keeps_branch_state_with_optimizer_schema() -> None:
    symbols = _symbols()
    samples = [_sample(0), _sample(1)]
    batch = collate_graphs(samples)
    control, _candidate, _receipt = symbols["build_paired_models"](
        node_dim=4,
        width=16,
        depth=1,
        cutoff=5.0,
        num_rbf=4,
        seed=29,
    )
    initial = symbols["_branch_snapshot"](control)
    optimizer = symbols["_paired_optimizer"](
        control,
        learning_rate=1e-3,
        weight_decay=0.01,
    )
    normalizer = symbols["fit_target_normalizer"](samples)

    step = symbols["train_step"](
        control,
        batch,
        optimizer,
        target_normalizer=normalizer,
        grad_clip=1.0,
    )

    assert step["router_gradient"]["trainable_parameter_count"] > 0
    assert step["router_gradient"]["with_gradient_count"] > 0
    assert step["router_gradient"]["nonzero_gradient_count"] == 0
    assert step["router_gradient"]["squared_norm_share"] == 0.0
    assert symbols["_branch_delta"](initial, control)["changed_tensor_count"] == 0


def _resource_summary(*, include_lba_shape: bool = True) -> dict[str, object]:
    shapes = [
        {
            "nodes": 128,
            "degree": 8,
            "width": 64,
            "depth": 3,
            "pair_count": 5,
            "passed": True,
        }
    ]
    if include_lba_shape:
        shapes.append(
            {
                "nodes": 512,
                "degree": 32,
                "width": 64,
                "depth": 3,
                "pair_count": 5,
                "passed": True,
            }
        )
    return {
        "schema_version": 1,
        "experiment": "canonical_ela_resource_ab_ba",
        "git_sha": "a" * 40,
        "source_file": "/repo/src/equivariant_attention/__init__.py",
        "same_common_weights": True,
        "environment_contract": {
            "device": "cuda",
            "dtype": "float32",
            "minimum_warmup": 10,
            "minimum_repeats": 30,
            "device_fingerprint": {"type": "cuda"},
        },
        "shape_results": shapes,
        "resource_gate": {"passed": True},
    }


def test_downstream_resource_receipt_requires_registered_two_shape_aggregate(
    tmp_path: Path,
) -> None:
    symbols = _symbols()
    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps(_resource_summary()), encoding="utf-8")

    receipts = symbols["_load_resource_receipts"]([valid])

    assert len(receipts) == 1
    assert len(receipts[0]["shape_results"]) == 2


def test_downstream_resource_receipt_rejects_partial_or_multiple_aggregates(
    tmp_path: Path,
) -> None:
    symbols = _symbols()
    partial = tmp_path / "partial.json"
    partial.write_text(
        json.dumps(_resource_summary(include_lba_shape=False)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="registered shapes"):
        symbols["_load_resource_receipts"]([partial])
    with pytest.raises(ValueError, match="exactly one"):
        symbols["_load_resource_receipts"]([partial, partial])
