from __future__ import annotations

import torch
import pytest

from equivariant_attention import prepare_3d_graph
from equivariant_attention.equivariant_linear_attention import (
    EquivariantLinearAttentionConfig,
)
from equivariant_attention.implicit_spatial import ImplicitSpatialKernelConfig
from equivariant_attention.spatial_ablation import (
    SpatialOperatorAblationConfig,
    SpatialOperatorAblationModel,
    empty_prepared_graph_like,
    state_dict_sha256,
)
from equivariant_attention.spatial_benchmarks import make_synthetic_spatial_batch


def _config(*, scale_init: float = 0.0) -> SpatialOperatorAblationConfig:
    return SpatialOperatorAblationConfig(
        model=EquivariantLinearAttentionConfig(
            input_irreps="4x0e + 1x1o",
            output_irreps="1x0e",
            hidden_dim=16,
            num_layers=1,
            num_heads=4,
            local_rank=3,
            local_cutoff=2.5,
            num_rbf=8,
        ),
        implicit=ImplicitSpatialKernelConfig(
            scales=(2.0, 4.0),
            chunk_size=8,
            learnable_scale_weights=True,
        ),
        implicit_residual_scale_init=scale_init,
    )


def _fixture() -> tuple[object, object, object]:
    batch = make_synthetic_spatial_batch(
        task="mixed",
        num_graphs=2,
        nodes_per_graph=6,
        seed=3,
        cutoff=2.0,
        candidate_skin=0.5,
        dtype=torch.float64,
    )
    graph = prepare_3d_graph(batch.batch, batch.edge_index)
    return batch, graph, empty_prepared_graph_like(graph)


def test_all_arms_have_one_parameter_schema_and_hash() -> None:
    torch.manual_seed(5)
    config = _config()
    template = SpatialOperatorAblationModel(config, arm="explicit").double()
    state = template.state_dict()
    reference_hash = state_dict_sha256(template)
    counts = set()
    hashes = set()
    for arm in ("explicit", "implicit", "hybrid"):
        model = SpatialOperatorAblationModel(config, arm=arm).double()
        model.load_state_dict(state, strict=True)
        counts.add(sum(parameter.numel() for parameter in model.parameters()))
        hashes.add(state_dict_sha256(model))
    assert len(counts) == 1
    assert hashes == {reference_hash}


def test_zero_initialized_hybrid_is_exactly_explicit() -> None:
    torch.manual_seed(7)
    config = _config(scale_init=0.0)
    template = SpatialOperatorAblationModel(config, arm="explicit").double().eval()
    state = template.state_dict()
    explicit = SpatialOperatorAblationModel(config, arm="explicit").double().eval()
    hybrid = SpatialOperatorAblationModel(config, arm="hybrid").double().eval()
    explicit.load_state_dict(state, strict=True)
    hybrid.load_state_dict(state, strict=True)
    batch, graph, no_edge_graph = _fixture()

    with torch.inference_mode():
        explicit_output = explicit(
            batch.node_irreps,
            batch.positions,
            graph,
            no_edge_graph=no_edge_graph,
        )["node_irreps"]
        hybrid_output = hybrid(
            batch.node_irreps,
            batch.positions,
            graph,
            no_edge_graph=no_edge_graph,
        )["node_irreps"]
    torch.testing.assert_close(
        hybrid_output,
        explicit_output,
        atol=0.0,
        rtol=0.0,
    )


def test_implicit_arm_is_independent_of_explicit_edge_metadata() -> None:
    torch.manual_seed(11)
    config = _config(scale_init=0.1)
    model = SpatialOperatorAblationModel(config, arm="implicit").double().eval()
    batch, graph, no_edge_graph = _fixture()
    reversed_graph = prepare_3d_graph(
        batch.batch,
        batch.edge_index.flip(1),
    )

    with torch.inference_mode():
        first = model(
            batch.node_irreps,
            batch.positions,
            graph,
            no_edge_graph=no_edge_graph,
        )["node_irreps"]
        second = model(
            batch.node_irreps,
            batch.positions,
            reversed_graph,
            no_edge_graph=no_edge_graph,
        )["node_irreps"]
    torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)


def test_no_edge_graph_requires_identical_batch_membership() -> None:
    config = _config(scale_init=0.1)
    model = SpatialOperatorAblationModel(config, arm="implicit").double()
    batch, graph, _ = _fixture()
    mismatched_batch = graph.batch.roll(1)
    mismatched = prepare_3d_graph(
        mismatched_batch,
        torch.empty((2, 0), dtype=torch.long),
    )

    with pytest.raises(ValueError, match="reuse graph.batch"):
        model(
            batch.node_irreps,
            batch.positions,
            graph,
            no_edge_graph=mismatched,
        )


def test_zero_implicit_layerscale_receives_gradient() -> None:
    torch.manual_seed(13)
    config = _config(scale_init=0.0)
    model = SpatialOperatorAblationModel(config, arm="hybrid").double()
    batch, graph, no_edge_graph = _fixture()
    output = model(
        batch.node_irreps,
        batch.positions,
        graph,
        no_edge_graph=no_edge_graph,
    )["graph_irreps"]
    output.square().mean().backward()

    scale = model.implicit_residuals[0].even_scale
    assert scale.grad is not None
    assert torch.isfinite(scale.grad).all()
    assert torch.count_nonzero(scale.grad) > 0
