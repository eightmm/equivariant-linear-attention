from __future__ import annotations

import torch

from equivariant_linear_attention.geometry import prepare_3d_graph
from equivariant_linear_attention.model.runtime import (
    _BaseStackConfig,
    _ELARuntime,
)
from equivariant_linear_attention.nn.layers import _BaseELALayer


def _complete_edge_index(num_nodes: int) -> torch.Tensor:
    receiver = torch.arange(num_nodes).repeat_interleave(num_nodes)
    sender = torch.arange(num_nodes).repeat(num_nodes)
    return torch.stack([receiver, sender])


def _fixture(*, num_nodes: int = 6) -> tuple[torch.Tensor, torch.Tensor, object]:
    features = torch.randn(num_nodes, 4, dtype=torch.float64)
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.2, -0.1],
            [-0.3, 1.2, 0.4],
            [0.4, -0.6, 1.4],
            [-1.1, -0.2, 0.7],
            [0.8, 1.1, -0.9],
        ],
        dtype=torch.float64,
    )[:num_nodes]
    batch = torch.zeros(num_nodes, dtype=torch.long)
    graph = prepare_3d_graph(batch, _complete_edge_index(num_nodes))
    return features, positions, graph


def _assert_state_close(left: object, right: object) -> None:
    for name in (
        "even_scalar",
        "odd_scalar",
        "polar_vector",
        "axial_vector",
        "even_tensor",
        "odd_tensor",
    ):
        torch.testing.assert_close(
            getattr(left, name),
            getattr(right, name),
            atol=2e-9,
            rtol=2e-9,
        )


def test_stack_is_exact_composition_of_public_layers() -> None:
    torch.manual_seed(101)
    config = _BaseStackConfig(
        input_irreps="4x0e",
        output_irreps="1x0e + 1x1o",
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        local_rank=3,
        local_cutoff=10.0,
        num_rbf=8,
    )
    model = _ELARuntime(config).double().eval()
    features, positions, graph = _fixture()

    state, context = model.embed_input(features, positions, graph)
    manual = state
    for layer in model.layers:
        assert isinstance(layer, _BaseELALayer)
        manual = layer(manual, context).state

    reference = model(features, positions, graph)
    torch.testing.assert_close(
        model.project_state(manual),
        reference["node_irreps"],
        atol=2e-9,
        rtol=2e-9,
    )
    torch.testing.assert_close(reference["positions"], positions)
    torch.testing.assert_close(
        reference["coordinate_delta"],
        torch.zeros_like(positions),
    )


def test_attention_and_ffn_residuals_compose_to_layer_forward() -> None:
    torch.manual_seed(103)
    config = _BaseStackConfig(
        input_irreps="4x0e",
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        local_rank=3,
        local_cutoff=10.0,
        num_rbf=8,
    )
    model = _ELARuntime(config).double().eval()
    features, positions, graph = _fixture()
    state, context = model.embed_input(features, positions, graph)
    layer = model.layers[0]

    after_attention = layer.attention_residual(state, context)
    after_ffn = layer.ffn_residual(after_attention, context)
    combined = layer(state, context).state
    _assert_state_close(after_ffn, combined)


def test_dit_condition_is_neutral_at_initialization_and_trainable() -> None:
    torch.manual_seed(107)
    config = _BaseStackConfig(
        input_irreps="4x0e",
        output_irreps="1x0e",
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        local_rank=3,
        local_cutoff=10.0,
        num_rbf=8,
        condition_dim=5,
    )
    model = _ELARuntime(config).double()
    features, positions, graph = _fixture()
    condition_a = torch.randn(1, 5, dtype=torch.float64)
    condition_b = torch.randn(1, 5, dtype=torch.float64)

    output_a = model(features, positions, graph, condition=condition_a)[
        "node_irreps"
    ]
    output_b = model(features, positions, graph, condition=condition_b)[
        "node_irreps"
    ]
    torch.testing.assert_close(output_a, output_b, atol=0.0, rtol=0.0)

    output_a.square().mean().backward()
    projection = model.layers[0].conditioner.projection[-1]
    assert projection.weight.grad is not None
    assert torch.isfinite(projection.weight.grad).all()
    assert torch.count_nonzero(projection.weight.grad) > 0


def test_graph_and_node_conditions_have_identical_broadcast_semantics() -> None:
    torch.manual_seed(109)
    config = _BaseStackConfig(
        input_irreps="4x0e",
        output_irreps="1x0e",
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        local_rank=3,
        local_cutoff=10.0,
        num_rbf=8,
        condition_dim=3,
    )
    model = _ELARuntime(config).double().eval()
    with torch.no_grad():
        output = model.layers[0].conditioner.projection[-1]
        output.weight.normal_(mean=0.0, std=0.03)
        output.bias.normal_(mean=0.0, std=0.03)
    features, positions, graph = _fixture()
    graph_condition = torch.randn(1, 3, dtype=torch.float64)
    node_condition = graph_condition.expand(features.shape[0], -1).clone()

    graph_output = model(
        features,
        positions,
        graph,
        condition=graph_condition,
    )["node_irreps"]
    node_output = model(
        features,
        positions,
        graph,
        condition=node_condition,
    )["node_irreps"]
    torch.testing.assert_close(graph_output, node_output, atol=2e-9, rtol=2e-9)


def test_coordinate_refinement_is_se3_equivariant_and_bounded() -> None:
    torch.manual_seed(113)
    config = _BaseStackConfig(
        input_irreps="4x0e",
        output_irreps="1x0e",
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        local_rank=3,
        local_cutoff=10.0,
        num_rbf=8,
        coordinate_updates=True,
        max_coordinate_step=0.2,
    )
    model = _ELARuntime(config).double().eval()
    with torch.no_grad():
        for layer in model.layers:
            layer.coordinate_vector.weight.normal_(mean=0.0, std=0.2)
    features, positions, graph = _fixture()

    orthogonal, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.linalg.det(orthogonal) < 0:
        orthogonal[:, 0] = -orthogonal[:, 0]
    translation = torch.tensor([1.3, -0.7, 0.5], dtype=torch.float64)

    reference = model(features, positions, graph)
    transformed = model(
        features,
        positions @ orthogonal.T + translation,
        graph,
    )
    expected_positions = reference["positions"] @ orthogonal.T + translation
    expected_delta = reference["coordinate_delta"] @ orthogonal.T
    torch.testing.assert_close(
        transformed["positions"],
        expected_positions,
        atol=3e-8,
        rtol=3e-8,
    )
    torch.testing.assert_close(
        transformed["coordinate_delta"],
        expected_delta,
        atol=3e-8,
        rtol=3e-8,
    )
    assert torch.linalg.vector_norm(
        reference["coordinate_delta"],
        dim=-1,
    ).max() <= config.num_layers * config.max_coordinate_step + 1e-10


def test_condition_and_coordinate_contract_is_explicit() -> None:
    config = _BaseStackConfig(
        input_irreps="4x0e",
        condition_dim=8,
        coordinate_updates=True,
    )
    contract = config.canonical_contract()
    assert contract["layer_api"] == (
        "attention_residual_tensor_closure_ffn_residual"
    )
    assert contract["condition_irrep"] == "0e"
    assert contract["coordinate_output"] == "bounded_polar_residual"
    assert contract["coordinate_topology"] == (
        "fixed_candidate_recompute_geometry_each_layer"
    )
    assert contract["node_geometry"] == "dynamic_l0_l1_l2_radial_multipoles"
