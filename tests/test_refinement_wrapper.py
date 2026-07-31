from __future__ import annotations

import pytest
import torch

from equivariant_attention.canonical import ELA, ELAConfig, SparseGeometry
from equivariant_attention.refinement import (
    CoordinateRefinementConfig,
    ELACoordinateRefiner,
)
from equivariant_attention.unified import prepare_3d_graph


def _complete_edges(nodes: int) -> torch.Tensor:
    receiver = torch.arange(nodes).repeat_interleave(nodes)
    sender = torch.arange(nodes).repeat(nodes)
    return torch.stack([receiver, sender])


def _model() -> ELA:
    return ELA(
        ELAConfig(
            input_irreps="4x0e",
            output_irreps="1x0e",
            width=32,
            depth=1,
            geometry=SparseGeometry(cutoff=10.0, num_rbf=8),
        )
    ).double()


def test_refiner_is_identity_at_initialization() -> None:
    torch.manual_seed(17)
    model = _model()
    refiner = ELACoordinateRefiner(
        model,
        CoordinateRefinementConfig(steps=2, max_step=0.1),
    ).double()
    nodes = 6
    features = torch.randn(nodes, 4, dtype=torch.float64)
    positions = torch.randn(nodes, 3, dtype=torch.float64)
    graph = prepare_3d_graph(
        torch.zeros(nodes, dtype=torch.long),
        _complete_edges(nodes),
    )
    output = refiner(features, positions, graph)
    torch.testing.assert_close(output["positions"], positions, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        output["coordinate_delta"],
        torch.zeros_like(positions),
        atol=0.0,
        rtol=0.0,
    )


def test_refinement_displacement_is_bounded_and_centered() -> None:
    torch.manual_seed(19)
    model = _model()
    refiner = ELACoordinateRefiner(
        model,
        CoordinateRefinementConfig(
            steps=1,
            max_step=0.05,
            centering="selected",
        ),
    ).double()
    with torch.no_grad():
        refiner.vector_head.base_weight.normal_(mean=0.0, std=0.2)
    nodes = 6
    features = torch.randn(nodes, 4, dtype=torch.float64)
    positions = torch.randn(nodes, 3, dtype=torch.float64)
    batch = torch.zeros(nodes, dtype=torch.long)
    graph = prepare_3d_graph(batch, _complete_edges(nodes))
    state, _, _ = model.forward_features(features, positions, graph)
    delta = refiner.displacement(state, batch)

    assert torch.linalg.vector_norm(delta, dim=-1).max() <= 0.05 + 1e-12
    torch.testing.assert_close(
        delta.mean(dim=0),
        torch.zeros(3, dtype=torch.float64),
        atol=2e-12,
        rtol=2e-12,
    )


def test_refiner_rejects_nonboolean_update_mask() -> None:
    model = _model()
    refiner = ELACoordinateRefiner(model).double()
    nodes = 4
    features = torch.randn(nodes, 4, dtype=torch.float64)
    positions = torch.randn(nodes, 3, dtype=torch.float64)
    batch = torch.zeros(nodes, dtype=torch.long)
    graph = prepare_3d_graph(batch, _complete_edges(nodes))
    state, _, _ = model.forward_features(features, positions, graph)

    with pytest.raises(TypeError, match="boolean dtype"):
        refiner.displacement(
            state,
            batch,
            update_mask=torch.ones(nodes, dtype=torch.float64),
        )


def test_activated_refiner_is_o3_and_translation_equivariant() -> None:
    torch.manual_seed(23)
    refiner = ELACoordinateRefiner(
        _model(),
        CoordinateRefinementConfig(
            steps=1,
            max_step=0.05,
            centering="selected",
        ),
    ).double().eval()
    with torch.no_grad():
        refiner.vector_head.base_weight.normal_(mean=0.0, std=0.2)
    nodes = 6
    features = torch.randn(nodes, 4, dtype=torch.float64)
    positions = torch.randn(nodes, 3, dtype=torch.float64)
    graph = prepare_3d_graph(
        torch.zeros(nodes, dtype=torch.long),
        _complete_edges(nodes),
    )
    orthogonal, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.linalg.det(orthogonal) > 0:
        orthogonal[:, 0] = -orthogonal[:, 0]
    translation = torch.tensor([0.4, -0.8, 1.3], dtype=torch.float64)

    with torch.inference_mode():
        reference = refiner(features, positions, graph)["positions"]
        actual = refiner(
            features,
            positions @ orthogonal.T + translation,
            graph,
        )["positions"]
    torch.testing.assert_close(
        actual,
        reference @ orthogonal.T + translation,
        atol=8e-8,
        rtol=8e-8,
    )
