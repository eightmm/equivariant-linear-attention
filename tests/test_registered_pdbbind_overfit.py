from __future__ import annotations

import runpy
from pathlib import Path

import pytest
import torch

from equivariant_attention._egnn_baseline import _StaticEGNNBaseline
from equivariant_attention.benchmarking import GraphSample
from equivariant_attention.training import build_regression_model


def _symbols() -> dict[str, object]:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_registered_pdbbind_overfit.py"
    )
    return runpy.run_path(script)


def test_packet_constants_freeze_train_only_subset_and_compute() -> None:
    symbols = _symbols()

    assert symbols["DATASET_REVISION"] == (
        "f93dd2d150a47c270f624620f84e07451a158705"
    )
    assert symbols["SUBSET_INDICES"] == tuple(range(16))
    assert symbols["MAX_STEPS"] == 3_000
    assert symbols["TRAIN_MAE_THRESHOLD_PK"] == 0.10
    assert symbols["MAX_GPU_SECONDS"] == 1_800
    assert symbols["MAX_ARM_GPU_SECONDS"] == 900
    assert symbols["EGNN_CUTOFF_ANGSTROM"] == 6.0
    assert symbols["registered_arms"]() == ("attention", "egnn")


def test_parameter_matching_selects_closest_static_egnn_width() -> None:
    symbols = _symbols()
    attention = build_regression_model(
        node_dim=140,
        hidden_dim=64,
        num_layers=3,
        num_heads=4,
        use_key_balancing=False,
        use_multiscale_spatial_kernel=True,
        hidden_tensor_dim=4,
    )
    target = sum(parameter.numel() for parameter in attention.parameters())

    width = symbols["matched_egnn_width"](
        target_parameter_count=target,
        node_dim=140,
        num_layers=3,
    )
    selected = _StaticEGNNBaseline(
        node_dim=140,
        hidden_dim=width,
        num_layers=3,
    )
    selected_count = sum(parameter.numel() for parameter in selected.parameters())
    alternatives = []
    for candidate_width in range(8, 257):
        candidate = _StaticEGNNBaseline(
            node_dim=140,
            hidden_dim=candidate_width,
            num_layers=3,
        )
        alternatives.append(
            abs(
                sum(parameter.numel() for parameter in candidate.parameters())
                - target
            )
        )

    assert abs(selected_count - target) == min(alternatives)


def test_cyclic_batch_order_is_shared_and_deterministic() -> None:
    symbols = _symbols()
    first = [
        symbols["cyclic_batch_indices"](
            step=step,
            batch_size=3,
            sample_count=8,
            seed=20260723,
        )
        for step in range(6)
    ]
    second = [
        symbols["cyclic_batch_indices"](
            step=step,
            batch_size=3,
            sample_count=8,
            seed=20260723,
        )
        for step in range(6)
    ]

    assert first == second
    assert all(len(batch) == 3 for batch in first)
    assert all(0 <= index < 8 for batch in first for index in batch)


def test_cli_defaults_cannot_enable_validation_test_or_coordinate_updates() -> None:
    symbols = _symbols()
    args = symbols["parse_args"](["artifacts/run/result.json", "--dry-run"])
    plan = symbols["_run_plan"](args)

    assert args.device == "cuda"
    assert plan["device"] == "cuda"
    assert args.max_steps == 3_000
    assert args.threshold == 0.10
    assert args.budget_seconds == 1_800
    assert args.dry_run is True
    assert not hasattr(args, "evaluate_test")
    assert not hasattr(args, "coordinate_updates")


def test_spatial_lba_plan_records_matched_operator_contract() -> None:
    symbols = _symbols()
    args = symbols["parse_args"](
        [
            "artifacts/run/spatial.json",
            "--arms",
            "ela_explicit",
            "ela_implicit",
            "ela_hybrid",
            "--dry-run",
        ]
    )
    plan = symbols["_run_plan"](args)

    assert plan["packet_id"] == symbols["SPATIAL_PACKET_ID"]
    assert plan["determinism"] == "strict"
    assert plan["arms_registered"] == [
        "ela_explicit",
        "ela_implicit",
        "ela_hybrid",
    ]
    assert plan["ela_spatial"]["common_node_multipoles"] == (
        "edge_free_zero_neighbor_context"
    )
    assert plan["ela_spatial"]["readout"] == "ligand_mask_mean_0e"


@pytest.mark.parametrize(
    ("arm", "expects_edges"),
    [
        ("explicit", True),
        ("implicit", False),
        ("hybrid", True),
    ],
)
def test_spatial_lba_builders_share_parameter_schema(
    arm: str,
    expects_edges: bool,
) -> None:
    symbols = _symbols()
    model = symbols["_build_spatial"](arm)

    assert model.consumes_external_neighbors is expects_edges
    assert sum(parameter.numel() for parameter in model.parameters()) == 258_072


def test_radius_edges_include_self_and_respect_cutoff() -> None:
    symbols = _symbols()
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [7.0, 0.0, 0.0]]
    )

    edge_index = symbols["radius_edge_index"](
        pos,
        cutoff=6.0,
    )
    pairs = set(map(tuple, edge_index.T.tolist()))

    assert {(0, 0), (1, 1), (2, 2)} <= pairs
    assert (0, 1) in pairs and (1, 0) in pairs
    assert (0, 2) not in pairs and (2, 0) not in pairs


def test_frozen_sample_validator_rejects_identity_or_count_drift() -> None:
    symbols = _symbols()
    sample_ids = symbols["FROZEN_SAMPLE_IDS"]
    node_counts = symbols["FROZEN_NODE_COUNTS"]
    ligand_counts = symbols["FROZEN_LIGAND_COUNTS"]
    samples = [
        GraphSample(
            node_feats=torch.zeros(node_count, 140),
            pos=torch.zeros(node_count, 3),
            target=torch.tensor([1.0]),
            sample_id=sample_id,
            readout_mask=(
                torch.arange(node_count) < ligand_count
            ),
        )
        for sample_id, node_count, ligand_count in zip(
            sample_ids,
            node_counts,
            ligand_counts,
            strict=True,
        )
    ]

    symbols["_validate_frozen_samples"](samples)
    changed_id = [*samples]
    changed_id[0] = GraphSample(
        node_feats=samples[0].node_feats,
        pos=samples[0].pos,
        target=samples[0].target,
        sample_id="drifted",
        readout_mask=samples[0].readout_mask,
    )
    with pytest.raises(ValueError, match="identity"):
        symbols["_validate_frozen_samples"](changed_id)
    changed_mask = [*samples]
    assert samples[0].readout_mask is not None
    drifted_mask = samples[0].readout_mask.clone()
    drifted_mask[27] = True
    changed_mask[0] = GraphSample(
        node_feats=samples[0].node_feats,
        pos=samples[0].pos,
        target=samples[0].target,
        sample_id=samples[0].sample_id,
        readout_mask=drifted_mask,
    )
    with pytest.raises(ValueError, match="ligand count"):
        symbols["_validate_frozen_samples"](changed_mask)


def test_cpu_peak_cuda_memory_is_unavailable_not_zero() -> None:
    symbols = _symbols()

    assert symbols["_peak_cuda_memory_bytes"](torch.device("cpu")) is None
