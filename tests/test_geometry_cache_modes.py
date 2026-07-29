from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from equivariant_attention import EquivariantAttention, EquivariantAttentionConfig
from equivariant_attention import moment
from equivariant_attention.training import build_regression_model


def _config(**overrides: object) -> EquivariantAttentionConfig:
    base = EquivariantAttentionConfig(
        node_dim=4,
        hidden_irreps="8x0e + 2x1o",
        output_irreps="1x0e + 1x1o",
        num_layers=2,
        num_heads=2,
        local_head_counts=(2, 2),
        local_cutoff=3.0,
        use_gated_local_transport=True,
        coordinate_updates=True,
        coordinate_neighbor_policy="fixed",
    )
    return replace(base, **overrides)


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(912)
    feats = torch.randn(6, 4, generator=generator, dtype=torch.float64)
    pos = torch.randn(6, 3, generator=generator, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    receiver: list[int] = []
    sender: list[int] = []
    for start in (0, 3):
        for target in range(start, start + 3):
            for source in range(start, start + 3):
                receiver.append(target)
                sender.append(source)
    edge_index = torch.tensor([receiver, sender], dtype=torch.long)
    return feats, pos, batch, edge_index


@pytest.mark.parametrize("mode", ["full", "compact", "recompute", "auto"])
def test_geometry_cache_modes_are_valid_config_values(mode: str) -> None:
    EquivariantAttention(_config(geometry_cache_mode=mode))

    with pytest.raises(ValueError, match="geometry_cache_mode"):
        EquivariantAttention(_config(geometry_cache_mode="unknown"))


def test_regression_builder_forwards_geometry_cache_mode() -> None:
    model = build_regression_model(
        node_dim=4,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        geometry_cache_mode="recompute",
    )

    assert isinstance(model, EquivariantAttention)
    assert model.config.geometry_cache_mode == "recompute"


def test_geometry_cache_storage_policy_and_tuple_values_are_exact() -> None:
    _feats, pos, batch, edge_index = _inputs()
    geometries = {
        mode: moment._local_geometry(
            pos,
            batch,
            num_graphs=2,
            cutoff=3.0,
            num_rbf=8,
            edge_index=edge_index,
            cache_mode=mode,
        )
        for mode in ("full", "compact", "recompute")
    }

    assert geometries["full"].cache_mode == "full"
    assert geometries["full"]._rbf is not None
    assert geometries["full"]._nonself_tensor_features is not None
    assert geometries["compact"]._displacement is not None
    assert geometries["compact"]._rbf is None
    assert geometries["compact"]._nonself_cutoff is None
    assert geometries["recompute"]._displacement is None
    assert geometries["recompute"]._squared_distance is None
    assert geometries["recompute"]._rbf is None

    reference = tuple(geometries["full"])
    for mode in ("compact", "recompute"):
        for actual, expected in zip(
            tuple(geometries[mode]),
            reference,
            strict=True,
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        nonself = moment._nonself_local_geometry(geometries[mode])
        reference_nonself = moment._nonself_local_geometry(geometries["full"])
        for actual, expected in zip(nonself, reference_nonself, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["compact", "recompute", "auto"])
def test_cache_modes_match_full_model_forward_and_input_gradients(
    mode: str,
) -> None:
    feats, pos, batch, edge_index = _inputs()
    torch.manual_seed(913)
    reference = EquivariantAttention(_config(geometry_cache_mode="full")).double()
    torch.manual_seed(913)
    candidate = EquivariantAttention(_config(geometry_cache_mode=mode)).double()
    assert reference.state_dict().keys() == candidate.state_dict().keys()
    reference_feats = feats.clone().requires_grad_(True)
    candidate_feats = feats.clone().requires_grad_(True)
    reference_pos = pos.clone().requires_grad_(True)
    candidate_pos = pos.clone().requires_grad_(True)

    expected = reference(
        reference_feats,
        reference_pos,
        batch=batch,
        edge_index=edge_index,
    )
    actual = candidate(
        candidate_feats,
        candidate_pos,
        batch=batch,
        edge_index=edge_index,
    )

    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    expected_loss = sum(value.square().sum() for value in expected.values())
    actual_loss = sum(value.square().sum() for value in actual.values())
    expected_gradients = torch.autograd.grad(
        expected_loss,
        (reference_feats, reference_pos),
        create_graph=True,
    )
    actual_gradients = torch.autograd.grad(
        actual_loss,
        (candidate_feats, candidate_pos),
        create_graph=True,
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=2e-12,
            atol=2e-12,
        )
    reference_parameters = tuple(
        parameter for parameter in reference.parameters() if parameter.requires_grad
    )
    candidate_parameters = tuple(
        parameter for parameter in candidate.parameters() if parameter.requires_grad
    )
    expected_parameter_gradients = torch.autograd.grad(
        expected_loss,
        reference_parameters,
        allow_unused=True,
        retain_graph=True,
    )
    actual_parameter_gradients = torch.autograd.grad(
        actual_loss,
        candidate_parameters,
        allow_unused=True,
        retain_graph=True,
    )
    for actual_gradient, expected_gradient in zip(
        actual_parameter_gradients,
        expected_parameter_gradients,
        strict=True,
    ):
        if actual_gradient is None or expected_gradient is None:
            assert actual_gradient is expected_gradient
        else:
            torch.testing.assert_close(
                actual_gradient,
                expected_gradient,
                rtol=2e-12,
                atol=2e-12,
            )
    second = torch.autograd.grad(
        sum(gradient.square().sum() for gradient in actual_gradients),
        (candidate_feats, candidate_pos),
    )
    assert all(torch.isfinite(value).all() for value in second)


@pytest.mark.parametrize(
    ("edge_count", "expected"),
    [(64, "full"), (8192, "compact"), (131072, "recompute")],
)
def test_auto_cache_policy_is_deterministic(
    edge_count: int,
    expected: str,
) -> None:
    assert moment._resolve_geometry_cache_mode("auto", edge_count) == expected


def test_bfloat16_projections_keep_geometry_and_coordinate_gradients_float32() -> None:
    feats64, pos64, batch, edge_index = _inputs()
    feats = feats64.float()
    pos = pos64.float()
    torch.manual_seed(914)
    reference = EquivariantAttention(
        _config(geometry_cache_mode="recompute")
    ).float()
    torch.manual_seed(914)
    candidate = EquivariantAttention(
        _config(geometry_cache_mode="recompute")
    ).to(dtype=torch.bfloat16)
    candidate.load_state_dict(reference.state_dict(), strict=True)
    reference_feats = feats.clone().requires_grad_(True)
    candidate_feats = feats.to(dtype=torch.bfloat16).requires_grad_(True)
    reference_pos = pos.clone().requires_grad_(True)
    candidate_pos = pos.clone().requires_grad_(True)

    expected = reference(
        reference_feats,
        reference_pos,
        batch=batch,
        edge_index=edge_index,
    )
    actual = candidate(
        candidate_feats,
        candidate_pos,
        batch=batch,
        edge_index=edge_index,
    )

    assert actual["node_positions"].dtype == torch.float32
    torch.testing.assert_close(
        actual["node_positions"],
        expected["node_positions"],
        rtol=2e-2,
        atol=2e-2,
    )
    torch.testing.assert_close(
        actual["graph_scalars"].float(),
        expected["graph_scalars"],
        rtol=5e-2,
        atol=5e-2,
    )
    expected_loss = sum(value.float().square().sum() for value in expected.values())
    actual_loss = sum(value.float().square().sum() for value in actual.values())
    reference_parameters = tuple(
        parameter for parameter in reference.parameters() if parameter.requires_grad
    )
    candidate_parameters = tuple(
        parameter for parameter in candidate.parameters() if parameter.requires_grad
    )
    expected_gradients = torch.autograd.grad(
        expected_loss,
        (reference_feats, reference_pos, *reference_parameters),
        allow_unused=True,
    )
    actual_gradients = torch.autograd.grad(
        actual_loss,
        (candidate_feats, candidate_pos, *candidate_parameters),
        allow_unused=True,
    )
    actual_feature_gradient, actual_position_gradient = actual_gradients[:2]
    expected_feature_gradient, expected_position_gradient = expected_gradients[:2]
    assert actual_position_gradient.dtype == torch.float32
    assert torch.isfinite(actual_feature_gradient).all()
    assert torch.isfinite(actual_position_gradient).all()
    torch.testing.assert_close(
        actual_position_gradient,
        expected_position_gradient,
        rtol=8e-2,
        atol=8e-2,
    )
    compared_parameters = 0
    for actual_gradient, expected_gradient in zip(
        actual_gradients[2:],
        expected_gradients[2:],
        strict=True,
    ):
        if actual_gradient is None or expected_gradient is None:
            assert actual_gradient is expected_gradient
            continue
        assert torch.isfinite(actual_gradient).all()
        torch.testing.assert_close(
            actual_gradient.float(),
            expected_gradient,
            rtol=1.5e-1,
            atol=1.5e-1,
        )
        compared_parameters += 1
    assert compared_parameters > 20
