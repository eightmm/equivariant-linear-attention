from __future__ import annotations

import inspect

import pytest
import torch

from equivariant_linear_attention import ELA, ELAGraph
from equivariant_linear_attention.advanced import ELAConfig, OrderContext


def _edges(nodes: int) -> torch.Tensor:
    sender = torch.arange(nodes).repeat(nodes)
    receiver = torch.arange(nodes).repeat_interleave(nodes)
    return torch.stack([sender, receiver])


def _graph(*, edge_type: torch.Tensor | None = None) -> ELAGraph:
    nodes = 3
    return ELAGraph(
        x=torch.randn(nodes, 3),
        pos=torch.randn(nodes, 3),
        edge_index=_edges(nodes),
        edge_type=edge_type,
    )


def test_constructor_rejects_invalid_surface_values() -> None:
    with pytest.raises(TypeError, match="input_irreps"):
        ELA(3)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="update_positions"):
        ELA("3x0e", update_positions=1)  # type: ignore[arg-type]
    for value in (True, -1, 1.5):
        error = TypeError if value is True or isinstance(value, float) else ValueError
        with pytest.raises(error, match="edge_types"):
            ELA("3x0e", edge_types=value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        ELA("3x0e", max_coordinate_step=0.0)
    with pytest.raises(TypeError, match="ELAConfig"):
        ELA.from_config("bad")  # type: ignore[arg-type]
    assert isinstance(ELA.from_config(ELAConfig(input_irreps="3x0e")), ELA)


def test_description_matches_the_single_contract() -> None:
    model = ELA(
        "3x0e",
        "2x0e",
        width=16,
        depth=1,
        cutoff=4.0,
        edge_types=2,
        update_positions=True,
    )
    description = model.describe()
    assert description["model"] == "ELA"
    assert description["graph"] == "ELAGraph"
    assert description["public_contract"] == "ELAGraph -> ELA -> ELAGraph"
    assert description["internal_graph_ir"] == "packed receiver-major CSR"
    assert description["edge_types"] == 2
    assert description["update_positions"] is True
    assert description["coordinate_updates"] == 1
    assert description["coordinate_update_layers"] == (1,)
    assert isinstance(description["num_parameters"], int)
    assert "input_irreps='3x0e'" in repr(model)
    assert "ELAGraph" in inspect.getdoc(ELA)


def test_model_accepts_exactly_one_public_input_type() -> None:
    model = ELA("3x0e", width=16, depth=1)
    graph = _graph()
    assert isinstance(model(graph), ELAGraph)
    for invalid in (
        graph.x,
        {"x": graph.x, "pos": graph.pos},
        (graph.x, graph.pos),
    ):
        with pytest.raises(TypeError, match="exactly one public input type"):
            model(invalid)  # type: ignore[arg-type]


def test_public_surface_hides_prepared_graph_execution_and_cache_trust() -> None:
    graph_parameters = inspect.signature(ELAGraph).parameters
    for name in (
        "_prepared_graph",
        "_prepared_provenance",
        "_packed_template",
        "_assume_immutable_storage",
    ):
        assert name not in graph_parameters

    model = ELA("3x0e", width=16, depth=1)
    for name in (
        "prepare_context",
        "embed_input",
        "project_state",
        "encode_context",
        "forward_features",
    ):
        assert not hasattr(model, name)

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        ELAGraph(
            torch.randn(3, 3),
            torch.randn(3, 3),
            _assume_immutable_storage=True,
        )  # type: ignore[call-arg]


def test_typed_edges_fail_closed() -> None:
    untyped = ELA("3x0e", width=16, depth=1)
    with pytest.raises(ValueError, match="edge_types=0"):
        untyped(_graph(edge_type=torch.zeros(9, dtype=torch.long)))

    typed = ELA("3x0e", width=16, depth=1, edge_types=2)
    with pytest.raises(ValueError, match="edge_type is required"):
        typed(_graph())
    with pytest.raises(ValueError, match=r"\[0, 2\)"):
        typed(_graph(edge_type=torch.full((9,), 2)))
    output = typed(_graph(edge_type=torch.arange(9) % 2))
    assert isinstance(output, ELAGraph)


def test_private_prepared_boundaries_fail_closed() -> None:
    model = ELA("3x0e", width=16, depth=1)
    graph = _graph()
    unpacked = graph._to_packed()

    with pytest.raises(TypeError, match="internal preparation expects ELABatch"):
        model._prepare_packed(graph)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="_prepare_graph expects an ELAGraph"):
        model._prepare_graph(unpacked)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="preparation invariant"):
        model._execute_numerical(unpacked)
    with pytest.raises(RuntimeError, match="preparation invariant"):
        model._execute_packed(unpacked)
    with pytest.raises(TypeError, match="_forward_prepared expects"):
        model._forward_prepared(graph)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not prepared"):
        model._forward_prepared(unpacked)

    prepared = model._prepare_packed(unpacked)
    incompatible = ELA("3x0e", width=16, depth=1, edge_types=1)
    with pytest.raises(ValueError, match="stale or incompatible"):
        incompatible._forward_prepared(prepared)


def test_update_mask_is_only_valid_for_coordinate_updating_models() -> None:
    mask = torch.tensor([True, False, True])
    graph = ELAGraph(torch.randn(3, 3), torch.randn(3, 3), update_mask=mask)
    with pytest.raises(ValueError, match="update_positions=False"):
        ELA("3x0e", width=16, depth=1)(graph)
    output = ELA(
        "3x0e",
        width=16,
        depth=1,
        update_positions=True,
    )(graph)
    assert output.delta is not None
    torch.testing.assert_close(
        output.delta[~mask], torch.zeros_like(output.delta[~mask])
    )


def test_graph_rejects_invalid_optional_runtime_inputs_early() -> None:
    x = torch.randn(3, 3)
    pos = torch.randn(3, 3)
    edges = torch.tensor([[0, 1], [1, 0]])

    with pytest.raises(TypeError, match="condition must be a tensor"):
        ELAGraph(x, pos, condition=[1.0])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="condition must be floating point"):
        ELAGraph(x, pos, condition=torch.ones(1, dtype=torch.long))
    with pytest.raises(ValueError, match="condition leading dimension"):
        ELAGraph(x, pos, condition=torch.randn(2, 4))
    with pytest.raises(ValueError, match="finite"):
        ELAGraph(x, pos, condition=torch.tensor([float("nan")]))

    with pytest.raises(TypeError, match="OrderContext"):
        ELAGraph(x, pos, order=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"shape \(N,K\)"):
        ELAGraph(x, pos, order=OrderContext(torch.randn(2, 1)))
    with pytest.raises(TypeError, match="periods must be floating point"):
        ELAGraph(
            x,
            pos,
            order=OrderContext(
                coordinates=torch.randn(3, 1),
                periods=torch.ones(1, dtype=torch.long),
            ),
        )

    with pytest.raises(ValueError, match="edge_type values must be nonnegative"):
        ELAGraph(x, pos, edge_index=edges, edge_type=torch.tensor([0, -1]))
    with pytest.raises(ValueError, match="interaction-group boundaries"):
        ELAGraph(
            x,
            pos,
            edge_index=edges,
            group=torch.tensor([0, 1, 1]),
        )
    with pytest.raises(TypeError, match="ids must be a tuple"):
        ELAGraph(x, pos, ids=["sample"])  # type: ignore[arg-type]


def test_graph_rejects_nonfinite_primary_and_output_tensors() -> None:
    x = torch.randn(3, 3)
    pos = torch.randn(3, 3)
    with pytest.raises(ValueError, match="x must contain only finite"):
        bad_x = x.clone()
        bad_x[0, 0] = float("inf")
        ELAGraph(bad_x, pos)
    with pytest.raises(ValueError, match="pos must contain only finite"):
        bad_pos = pos.clone()
        bad_pos[0, 0] = float("nan")
        ELAGraph(x, bad_pos)
    with pytest.raises(TypeError, match="graph_x must be floating point"):
        ELAGraph(x, pos, graph_x=torch.ones(1, 1, dtype=torch.long))
    with pytest.raises(ValueError, match="delta must contain only finite"):
        ELAGraph(x, pos, delta=torch.full_like(pos, float("nan")))
