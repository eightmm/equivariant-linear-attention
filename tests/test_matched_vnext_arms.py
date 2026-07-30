from __future__ import annotations

import torch

from equivariant_attention.benchmarking import GraphBatch
from equivariant_attention._matched_vnext_arms import (
    MATCHED_VNEXT_ARMS,
    build_matched_vnext_config,
)
from equivariant_attention.training import (
    build_regression_model,
    predict_graph_scalar,
)


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def test_structured_lgl_baseline_matches_flat_candidate_exactly() -> None:
    torch.manual_seed(17)
    flat = build_regression_model(
        node_dim=11,
        hidden_dim=64,
        num_layers=3,
        num_heads=4,
        local_head_counts=(4, 0, 4),
        local_cutoff=2.5,
        use_key_balancing=False,
        use_gated_local_transport=True,
        use_grouped_invariant_normalization=True,
    )
    torch.manual_seed(17)
    structured = build_regression_model(
        node_dim=11,
        architecture_config=build_matched_vnext_config(
            "lgl",
            node_dim=11,
            hidden_dim=64,
            num_layers=3,
            num_heads=4,
            local_cutoff=2.5,
            global_backend="outer_scatter",
            geometry_cache_mode="full",
        ),
    )

    assert tuple(structured.state_dict()) == tuple(flat.state_dict())
    assert all(
        torch.equal(structured.state_dict()[name], flat.state_dict()[name])
        for name in flat.state_dict()
    )


def test_l3_arm_has_a_parameter_matched_persistent_2e_control() -> None:
    models = {
        arm: build_regression_model(
            node_dim=11,
            architecture_config=build_matched_vnext_config(
                arm,
                node_dim=11,
                hidden_dim=64,
                num_layers=3,
                num_heads=4,
                local_cutoff=2.5,
            ),
        )
        for arm in ("lgl", "lgl_2e", "lgl_2e_l3")
    }
    baseline_parameters = _parameter_count(models["lgl"])

    assert models["lgl"].hidden_irreps.tensors == 0
    assert models["lgl_2e"].hidden_irreps.tensors == 1
    assert models["lgl_2e_l3"].hidden_irreps.tensors == 1
    assert models["lgl_2e"].config.use_transient_l3_workspace is False
    assert models["lgl_2e_l3"].config.use_transient_l3_workspace is True
    assert models["lgl_2e_l3"].config.transient_l3_layers == (0, 2)
    for model in models.values():
        assert _parameter_count(model) / baseline_parameters <= 1.01


def test_full_high_order_sparse_arm_is_bounded_and_parameter_matched() -> None:
    baseline = build_regression_model(
        node_dim=11,
        architecture_config=build_matched_vnext_config(
            "lgl",
            node_dim=11,
            hidden_dim=64,
            num_layers=3,
            num_heads=4,
            local_cutoff=2.5,
        ),
    )
    candidate = build_regression_model(
        node_dim=11,
        architecture_config=build_matched_vnext_config(
            "high_order_sparse",
            node_dim=11,
            hidden_dim=64,
            num_layers=3,
            num_heads=4,
            local_cutoff=2.5,
        ),
    )

    assert MATCHED_VNEXT_ARMS == (
        "lgl",
        "lgl_2e",
        "lgl_2e_l3",
        "global_only",
        "global_local",
        "high_order_sparse",
    )
    assert candidate.config.use_sparse_low_rank_local_residual is True
    assert candidate.config.local_residual_rank == 2
    assert candidate.config.local_residual_layers == (0,)
    assert candidate.config.use_transient_l3_workspace is True
    assert candidate.config.transient_l3_layers == (0,)
    assert _parameter_count(candidate) / _parameter_count(baseline) <= 1.01


def test_homogeneous_global_local_keeps_every_global_head() -> None:
    control = build_regression_model(
        node_dim=11,
        architecture_config=build_matched_vnext_config(
            "global_only",
            node_dim=11,
            hidden_dim=64,
            num_layers=3,
            num_heads=4,
            local_cutoff=2.5,
            global_backend="feature_gemm",
        ),
    )
    candidate = build_regression_model(
        node_dim=11,
        architecture_config=build_matched_vnext_config(
            "global_local",
            node_dim=11,
            hidden_dim=64,
            num_layers=3,
            num_heads=4,
            local_cutoff=2.5,
            global_backend="feature_gemm",
        ),
    )
    lgl = build_regression_model(
        node_dim=11,
        architecture_config=build_matched_vnext_config(
            "lgl",
            node_dim=11,
            hidden_dim=64,
            num_layers=3,
            num_heads=4,
            local_cutoff=2.5,
        ),
    )

    assert control.config.local_head_counts == (0, 0, 0)
    assert candidate.config.local_head_counts == (0, 0, 0)
    assert control.config.use_sparse_low_rank_local_residual is False
    assert candidate.config.use_sparse_low_rank_local_residual is True
    assert candidate.config.local_residual_rank == 4
    assert candidate.config.local_residual_layers == (0, 2)
    assert candidate.config.global_reduction_backend == "feature_gemm"
    assert _parameter_count(candidate) / _parameter_count(lgl) <= 1.05


def test_global_only_control_ignores_shared_topology_at_training_boundary() -> None:
    model = build_regression_model(
        node_dim=3,
        architecture_config=build_matched_vnext_config(
            "global_only",
            node_dim=3,
            hidden_dim=8,
            num_layers=2,
            num_heads=2,
            local_cutoff=2.5,
            global_backend="feature_gemm",
        ),
    )
    batch = GraphBatch(
        node_feats=torch.randn(4, 3),
        pos=torch.randn(4, 3),
        batch=torch.tensor([0, 0, 1, 1]),
        target=torch.randn(2, 1),
        sample_ids=("a", "b"),
        edge_index=torch.tensor(
            [[0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 0, 1, 2, 3, 2, 3]]
        ),
        edge_index_is_validated=True,
    )

    prediction = predict_graph_scalar(model, batch)

    assert prediction.shape == (2, 1)
