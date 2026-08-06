from __future__ import annotations

import torch

from equivariant_linear_attention import ELA, ELAGraph
from equivariant_linear_attention.model.edge_free import (
    _EdgeFreeRelativeMomentBank,
)
from equivariant_linear_attention.nn.parity import _st_from_vector


def _normalized_positions(
    positions: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty_like(positions)
    for graph in batch.unique(sorted=True):
        selected = batch == graph
        centered = positions[selected] - positions[selected].mean(dim=0)
        radius = centered.square().sum(dim=-1).mean().sqrt().clamp_min(1e-12)
        output[selected] = centered / radius
    return output


def test_edge_free_moments_match_explicit_pair_oracle() -> None:
    torch.manual_seed(7)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    positions = _normalized_positions(
        torch.randn(7, 3, dtype=torch.float64),
        batch,
    )
    bank = _EdgeFreeRelativeMomentBank(rank=4, eps=1e-12).double()
    moments = bank(positions, batch, num_graphs=2)

    radius_squared = positions.square().sum(dim=-1)
    radial_coordinate = radius_squared / (1.0 + radius_squared)
    radial = torch.exp(
        (radial_coordinate.unsqueeze(-1) - 0.5)
        * bank._radial_scales.to(dtype=positions.dtype)
    )
    expected_polar = torch.zeros_like(moments.polar)
    expected_tensor = torch.zeros_like(moments.even_tensor)
    expected_mass = torch.zeros_like(moments.mass)
    for receiver in range(positions.shape[0]):
        for sender in range(positions.shape[0]):
            if receiver == sender or batch[receiver] != batch[sender]:
                continue
            displacement = positions[sender] - positions[receiver]
            weight = radial[sender]
            expected_mass[receiver] += weight
            expected_polar[receiver] += weight[:, None] * displacement
            expected_tensor[receiver] += (
                weight[:, None] * _st_from_vector(displacement)[None, :]
            )
    denominator = 1.0 + expected_mass
    expected_polar = expected_polar / denominator.unsqueeze(-1)
    expected_tensor = expected_tensor / denominator.unsqueeze(-1)

    torch.testing.assert_close(moments.mass, expected_mass)
    torch.testing.assert_close(moments.polar, expected_polar)
    torch.testing.assert_close(moments.even_tensor, expected_tensor)


def test_default_forward_never_discovers_radius_edges(monkeypatch) -> None:
    import equivariant_linear_attention.api as sparse_api

    def fail_radius_graph(*args: object, **kwargs: object) -> None:
        raise AssertionError("the edge-free default must not build a radius graph")

    monkeypatch.setattr(sparse_api, "_prepare_radius_3d_graph", fail_radius_graph)
    torch.manual_seed(11)
    model = ELA("4x0e", "2x0e", width=32, depth=2)
    graph = ELAGraph(
        x=torch.randn(8, 4),
        pos=torch.randn(8, 3),
    )
    packed = model._prepare_packed(graph._to_packed())
    assert packed._prepared_graph is not None
    assert packed._prepared_graph.spec.source == "explicit"
    assert packed._prepared_graph.num_edges == 0

    output = model(graph)
    assert output.x.shape == (8, 2)
    assert torch.isfinite(output.x).all()


def test_edge_free_default_is_cutoff_independent() -> None:
    torch.manual_seed(17)
    narrow = ELA("3x0e", "1x0e", width=32, depth=2, cutoff=2.0)
    wide = ELA("3x0e", "1x0e", width=32, depth=2, cutoff=50.0)
    wide.load_state_dict(narrow.state_dict())
    graph = ELAGraph(
        x=torch.randn(9, 3),
        pos=torch.randn(9, 3),
    )
    with torch.no_grad():
        narrow_output = narrow(graph)
        wide_output = wide(graph)
    torch.testing.assert_close(narrow_output.x, wide_output.x)


def test_krylov_orders_are_identity_safe_but_trainable() -> None:
    torch.manual_seed(23)
    model = ELA("4x0e", "1x0e", width=32, depth=1)
    graph = ELAGraph(
        x=torch.randn(10, 4),
        pos=torch.randn(10, 3),
    )
    with torch.no_grad():
        first_order_only = model(graph).x
        layer = model.layers[0]
        layer.global_krylov_gate.bias.fill_(0.75)
        higher_order = model(graph).x
    assert not torch.allclose(first_order_only, higher_order)


def test_explicit_edges_activate_only_the_optional_sparse_residual() -> None:
    torch.manual_seed(29)
    model = ELA("3x0e", "1x0e", width=32, depth=1)
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 0],
            [1, 0, 2, 1, 3, 2, 0, 3],
        ],
        dtype=torch.long,
    )
    graph = ELAGraph(
        x=torch.randn(4, 3),
        pos=torch.randn(4, 3),
        edge_index=edge_index,
    )
    packed = model._prepare_packed(graph._to_packed())
    assert packed._prepared_graph is not None
    assert packed._prepared_graph.num_edges == edge_index.shape[1]
    output = model(graph)
    assert torch.isfinite(output.x).all()
