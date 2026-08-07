from __future__ import annotations

import pytest
import torch
from conftest import orthogonal, transform_irreps

from equivariant_linear_attention import ELA, ELAGraph


@pytest.mark.parametrize("reflection", [False, True])
def test_full_model_obeys_o3_translation_and_mixed_irreps(reflection: bool) -> None:
    generator = torch.Generator().manual_seed(301)
    input_layout = "3x0e + 1x0o + 1x1o + 1x1e + 1x2e + 1x2o"
    output_layout = "2x0e + 1x0o + 1x1o + 1x1e + 1x2e + 1x2o"
    model = ELA(input_layout, output_layout, width=32, depth=2).double().eval()
    x = torch.randn(8, model.input_irreps.dim, generator=generator, dtype=torch.float64)
    pos = torch.randn(8, 3, generator=generator, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    transform = orthogonal(reflection=reflection, seed=303)
    moved_x = transform_irreps(x, model.input_irreps, transform)
    moved_pos = pos @ transform.T + torch.tensor([3.0, -2.0, 1.0])
    with torch.no_grad():
        reference = model(ELAGraph(x, pos, batch=batch))
        moved = model(ELAGraph(moved_x, moved_pos, batch=batch))
    expected = transform_irreps(reference.x, model.output_irreps, transform)
    torch.testing.assert_close(moved.x, expected, atol=5e-10, rtol=5e-10)
    assert reference.graph_x is not None and moved.graph_x is not None
    expected_graph = transform_irreps(reference.graph_x, model.output_irreps, transform)
    torch.testing.assert_close(moved.graph_x, expected_graph, atol=5e-10, rtol=5e-10)


def test_node_permutation_equivariance() -> None:
    generator = torch.Generator().manual_seed(307)
    model = ELA("4x0e", "2x0e + 1x1o", width=32, depth=2).double().eval()
    x = torch.randn(9, 4, generator=generator, dtype=torch.float64)
    pos = torch.randn(9, 3, generator=generator, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1])
    group = torch.tensor([0, 0, 1, 1, 0, 0, 0, 1, 1])
    permutation = torch.tensor([2, 0, 3, 1, 8, 5, 7, 4, 6])
    with torch.no_grad():
        reference = model(ELAGraph(x, pos, batch=batch, group=group))
        moved = model(
            ELAGraph(
                x[permutation],
                pos[permutation],
                batch=batch[permutation],
                group=group[permutation],
            )
        )
    torch.testing.assert_close(moved.x, reference.x[permutation], atol=4e-10, rtol=4e-10)


def test_interaction_components_are_isolated() -> None:
    generator = torch.Generator().manual_seed(311)
    model = ELA("3x0e", "1x0e", width=32, depth=2).double().eval()
    x = torch.randn(6, 3, generator=generator, dtype=torch.float64)
    pos = torch.randn(6, 3, generator=generator, dtype=torch.float64)
    group = torch.tensor([0, 0, 0, 1, 1, 1])
    with torch.no_grad():
        reference = model(ELAGraph(x, pos, group=group)).x
        changed_x = x.clone()
        changed_pos = pos.clone()
        changed_x[3:] += 100.0
        changed_pos[3:] += 1000.0
        changed = model(ELAGraph(changed_x, changed_pos, group=group)).x
    torch.testing.assert_close(changed[:3], reference[:3], atol=3e-10, rtol=3e-10)


def test_model_has_no_quadratic_or_edge_state_parameters() -> None:
    model = ELA("2x0e", width=32, depth=1)
    forbidden = ("edge", "neighbor", "pair", "radius", "cutoff")
    names = tuple(name.lower() for name, _ in model.named_parameters())
    assert not any(any(token in name for token in forbidden) for name in names)
