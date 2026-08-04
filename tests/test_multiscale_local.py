from __future__ import annotations

import torch

from equivariant_linear_attention import ELA, ELAGraph
from equivariant_linear_attention.advanced import ELAConfig, SparseGeometry
from equivariant_linear_attention.migration import load_advanced_ela_state


def _complete_edges(nodes: int) -> torch.Tensor:
    receiver = torch.arange(nodes).repeat_interleave(nodes)
    sender = torch.arange(nodes).repeat(nodes)
    return torch.stack((receiver, sender))


def _open_local_lane(model: ELA) -> None:
    with torch.no_grad():
        for layer in model.layers:
            for name in (
                "local_scalar_out",
                "local_odd_out",
                "local_polar_out",
                "local_axial_out",
                "local_even_tensor_out",
                "local_odd_tensor_out",
                "local_chiral_scalar_out",
                "local_chiral_axial_out",
                "local_chiral_tensor_out",
            ):
                getattr(layer, name).weight.normal_(mean=0.0, std=0.08)
            layer.local_mass_out.weight.normal_(mean=0.0, std=0.05)


def test_multiscale_local_is_identity_initialized_and_privately_ablatable() -> None:
    torch.manual_seed(701)
    model = ELA("3x0e", "2x0e", width=16, depth=1, cutoff=3.0).double()
    _open_local_lane(model)
    layer = model.layers[0]
    torch.testing.assert_close(
        layer.local_scale_score_mix,
        torch.zeros_like(layer.local_scale_score_mix),
    )
    torch.testing.assert_close(
        layer.local_scale_value_mix,
        torch.zeros_like(layer.local_scale_value_mix),
    )
    graph = ELAGraph(
        torch.randn(4, 3, dtype=torch.float64),
        torch.randn(4, 3, dtype=torch.float64),
        edge_index=_complete_edges(4),
    )

    enabled = model(graph).x
    layer._set_multiscale_local_enabled(False)
    ablated = model(graph).x

    torch.testing.assert_close(enabled, ablated, atol=0.0, rtol=0.0)


def test_historical_checkpoint_migration_keeps_identity_scale_initialization() -> None:
    source = ELA("3x0e", "2x0e", width=16, depth=1, cutoff=3.0)
    target = ELA("3x0e", "2x0e", width=16, depth=1, cutoff=3.0)
    state = {
        key: value
        for key, value in source.state_dict().items()
        if not key.endswith(
            (".local_scale_score_mix", ".local_scale_value_mix")
        )
    }

    receipt = load_advanced_ela_state(target, state)

    assert receipt.canonical_initialized is True
    assert len(receipt.missing_keys) == 2
    for layer in target.layers:
        torch.testing.assert_close(
            layer.local_scale_score_mix,
            torch.zeros_like(layer.local_scale_score_mix),
        )
        torch.testing.assert_close(
            layer.local_scale_value_mix,
            torch.zeros_like(layer.local_scale_value_mix),
        )


def test_multiscale_local_changes_the_function_and_has_finite_gradients() -> None:
    torch.manual_seed(703)
    model = ELA("3x0e", "2x0e", width=16, depth=1, cutoff=4.0).double()
    _open_local_lane(model)
    layer = model.layers[0]
    with torch.no_grad():
        layer.local_scale_score_mix.copy_(
            torch.tensor([[0.35], [-0.2]], dtype=torch.float64).expand_as(
                layer.local_scale_score_mix
            )
        )
        layer.local_scale_value_mix.normal_(mean=0.12, std=0.03)

    features = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    positions = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    graph = ELAGraph(features, positions, edge_index=_complete_edges(5))
    active = model(graph).x
    layer._set_multiscale_local_enabled(False)
    ablated = model(graph).x
    layer._set_multiscale_local_enabled(True)

    assert not torch.allclose(active, ablated, atol=1e-10, rtol=1e-10)
    active.square().sum().backward()
    for gradient in (
        layer.local_scale_score_mix.grad,
        layer.local_scale_value_mix.grad,
        features.grad,
        positions.grad,
    ):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0


def test_multiscale_local_cannot_revive_relation_masked_edges() -> None:
    torch.manual_seed(705)
    model = ELA.from_config(
        ELAConfig(
            input_irreps="2x0e",
            output_irreps="1x0e",
            width=16,
            depth=1,
            geometry=SparseGeometry(
                cutoff=3.0,
                num_rbf=8,
                relation_cutoffs=(1.0,),
            ),
        )
    ).double()
    _open_local_lane(model)
    layer = model.layers[0]
    with torch.no_grad():
        layer.local_scale_score_mix.fill_(2.0)
        layer.local_scale_value_mix.fill_(2.0)

    # Both explicit edges are inside the prepared max cutoff but outside the
    # relation-specific support. Their relation-aware C2 envelope is exactly
    # zero, so the extra scales must also be identity there.
    graph = ELAGraph(
        torch.randn(2, 2, dtype=torch.float64),
        torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        edge_type=torch.zeros(2, dtype=torch.long),
    )
    active = model(graph).x
    layer._set_multiscale_local_enabled(False)
    ablated = model(graph).x

    torch.testing.assert_close(active, ablated, atol=0.0, rtol=0.0)


def test_active_multiscale_local_preserves_o3_and_node_permutation_equivariance() -> None:
    torch.manual_seed(707)
    model = ELA("3x0e", "1x1o", width=16, depth=1, cutoff=5.0).double().eval()
    _open_local_lane(model)
    layer = model.layers[0]
    with torch.no_grad():
        layer.local_scale_score_mix.normal_(mean=0.1, std=0.2)
        layer.local_scale_value_mix.normal_(mean=0.05, std=0.1)

    nodes = 5
    features = torch.randn(nodes, 3, dtype=torch.float64)
    positions = torch.randn(nodes, 3, dtype=torch.float64)
    edges = _complete_edges(nodes)
    reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))
    translation = torch.tensor([0.7, -0.2, 1.1], dtype=torch.float64)
    permutation = torch.tensor([3, 0, 4, 1, 2])

    with torch.inference_mode():
        reference = model(ELAGraph(features, positions, edge_index=edges)).x
        transformed = model(
            ELAGraph(
                features,
                positions @ reflection.T + translation,
                edge_index=edges,
            )
        ).x
        permuted = model(
            ELAGraph(
                features[permutation],
                positions[permutation],
                edge_index=edges,
            )
        ).x

    torch.testing.assert_close(
        transformed,
        reference @ reflection.T,
        atol=2e-8,
        rtol=2e-8,
    )
    torch.testing.assert_close(
        permuted,
        reference[permutation],
        atol=2e-8,
        rtol=2e-8,
    )
