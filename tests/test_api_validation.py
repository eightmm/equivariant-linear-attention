from __future__ import annotations

import pytest
import torch

from equivariant_linear_attention import ELA, ELABatch, ELAConfig


def _edges(nodes: int) -> torch.Tensor:
    receiver = torch.arange(nodes).repeat_interleave(nodes)
    sender = torch.arange(nodes).repeat(nodes)
    return torch.stack([receiver, sender])


def _batch(*, relation: torch.Tensor | None = None) -> ELABatch:
    nodes = 3
    return ELABatch(
        torch.randn(nodes, 3),
        torch.randn(nodes, 3),
        edge_index=_edges(nodes),
        edge_relation_id=relation,
    )


def test_constructor_rejects_ambiguous_or_invalid_surfaces() -> None:
    config = ELAConfig(input_irreps="3x0e", width=16, depth=1)
    with pytest.raises(TypeError, match="ELAConfig"):
        ELA("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mutually exclusive"):
        ELA(config, input_irreps="3x0e")
    with pytest.raises(ValueError, match="input_irreps is required"):
        ELA()
    for value in (True, -1, 1.5):
        with pytest.raises(ValueError, match="nonnegative integer"):
            ELA(input_irreps="3x0e", num_edge_types=value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must match"):
        ELA(
            input_irreps="3x0e",
            num_edge_types=2,
            relation_cutoffs=(4.0,),
        )


def test_description_and_convenience_constructors_share_one_contract() -> None:
    model = ELA(
        input_irreps="3x0e",
        output_irreps="2x0e",
        width=16,
        depth=1,
        cutoff=4.0,
        num_edge_types=2,
    )
    description = model.describe()
    assert description["model"] == "ELA"
    assert description["graph_input"] == "ELABatch"
    assert description["num_edge_types"] == 2
    assert "input_irreps='3x0e'" in repr(model)

    flat = ELA.batch(
        torch.randn(3, 3),
        torch.randn(3, 3),
        edge_index=_edges(3),
    )
    assert isinstance(flat, ELABatch)

    padded = ELA.padded(
        torch.randn(1, 3, 3),
        torch.randn(1, 3, 3),
        edge_index=[_edges(3)],
        edge_type=[torch.zeros(9, dtype=torch.long)],
        y=torch.tensor([[1.0]]),
    )
    assert padded.target is not None
    assert padded.edge_relation_id is not None
    with pytest.raises(ValueError, match="shapes"):
        ELA.padded(torch.randn(3, 3), torch.randn(3, 3))
    with pytest.raises(ValueError, match="mutually exclusive"):
        ELA.padded(
            torch.randn(1, 2, 3),
            torch.randn(1, 2, 3),
            edge_type=[torch.zeros(0, dtype=torch.long)],
            edge_relation_id=[torch.zeros(0, dtype=torch.long)],
        )


def test_relation_preparation_fails_closed_and_caches_valid_graphs() -> None:
    untyped = ELA(input_irreps="3x0e", width=16, depth=1)
    with pytest.raises(ValueError, match="no relation capacity"):
        untyped.prepare(_batch(relation=torch.zeros(9, dtype=torch.long)))

    typed = ELA(
        input_irreps="3x0e",
        width=16,
        depth=1,
        num_edge_types=2,
    )
    with pytest.raises(ValueError, match="multiple edge types"):
        typed.prepare(_batch())
    with pytest.raises(ValueError, match=r"\[0, 2\)"):
        typed.prepare(_batch(relation=torch.full((9,), 2)))

    one_type = ELA(
        input_irreps="3x0e",
        width=16,
        depth=1,
        num_edge_types=1,
    )
    prepared = one_type.prepare(_batch())
    assert prepared.is_prepared
    assert one_type.prepare(prepared) is prepared
    forced = one_type.prepare(prepared, force=True)
    assert forced.is_prepared
    assert forced is not prepared


def test_forward_entry_points_require_elabatch_and_preparation() -> None:
    model = ELA(input_irreps="3x0e", width=16, depth=1)
    batch = _batch()
    with pytest.raises(TypeError, match="ELABatch"):
        model.prepare(torch.randn(3, 3))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ELABatch"):
        model.forward_prepared(torch.randn(3, 3))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not prepared"):
        model.forward_prepared(batch)
    with pytest.raises(TypeError, match="accepts one ELABatch"):
        model(torch.randn(3, 3))  # type: ignore[arg-type]
