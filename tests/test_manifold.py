from __future__ import annotations

import pytest
import torch
from conftest import orthogonal

from equivariant_linear_attention import ELA, ELAGraph
from equivariant_linear_attention.nn.manifold import quotient_rigid_shape_step


@pytest.mark.parametrize("reflection", [False, True])
def test_quotient_step_removes_full_rigid_gauge_and_is_o3_equivariant(
    reflection: bool,
) -> None:
    generator = torch.Generator().manual_seed(401)
    index = torch.tensor([0, 0, 0, 0, 1, 1, 1])
    counts = torch.tensor([4, 3])
    positions = torch.randn(7, 3, generator=generator, dtype=torch.float64)
    centers = torch.stack([positions[index == graph].mean(0) for graph in range(2)])
    translation = torch.tensor(
        [[0.3, -0.1, 0.2], [-0.2, 0.4, 0.1]], dtype=torch.float64
    )
    angular = torch.tensor([[0.1, -0.2, 0.3], [-0.3, 0.2, 0.1]], dtype=torch.float64)
    relative = positions - centers[index]
    raw = translation[index] + torch.cross(angular[index], relative, dim=-1)
    selected = torch.ones(7, dtype=torch.bool)
    gates = torch.ones(2, 3, dtype=torch.float64)
    reference = quotient_rigid_shape_step(
        raw,
        positions,
        index,
        num_segments=2,
        counts=counts,
        selected=selected,
        component_gates=gates,
        max_step=100.0,
        eps=1e-12,
    )
    torch.testing.assert_close(
        reference, torch.zeros_like(reference), atol=3e-10, rtol=0.0
    )

    transform = orthogonal(reflection=reflection, seed=403)
    moved = quotient_rigid_shape_step(
        raw @ transform.T,
        positions @ transform.T,
        index,
        num_segments=2,
        counts=counts,
        selected=selected,
        component_gates=gates,
        max_step=100.0,
        eps=1e-12,
    )
    torch.testing.assert_close(moved, reference @ transform.T, atol=3e-10, rtol=3e-10)


def test_partial_selection_retains_rigid_pose_and_fixes_unselected_nodes() -> None:
    index = torch.tensor([0, 0, 0, 0])
    counts = torch.tensor([4])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    raw = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    selected = torch.tensor([True, True, False, False])
    gates = torch.ones(1, 3, dtype=torch.float64)
    step = quotient_rigid_shape_step(
        raw,
        positions,
        index,
        num_segments=1,
        counts=counts,
        selected=selected,
        component_gates=gates,
        max_step=100.0,
        eps=1e-12,
    )
    torch.testing.assert_close(step[selected], raw[selected], atol=2e-10, rtol=2e-10)
    assert torch.count_nonzero(step[~selected]) == 0


def test_coordinate_update_mask_bound_and_double_backward() -> None:
    generator = torch.Generator().manual_seed(407)
    model = ELA(
        "4x0e",
        "1x0e",
        width=32,
        depth=2,
        update_positions=True,
        max_coordinate_step=0.2,
    ).double()
    # Open the vector lane so the manifold path is exercised.
    with torch.no_grad():
        assert model.coordinate_update is not None
        model.coordinate_update.vector.base_weight.fill_(0.25)
    pos = torch.randn(
        6, 3, generator=generator, dtype=torch.float64, requires_grad=True
    )
    mask = torch.tensor([True, True, True, False, False, False])
    output = model(
        ELAGraph(
            torch.randn(6, 4, generator=generator, dtype=torch.float64),
            pos,
            update_mask=mask,
        )
    )
    assert output.delta is not None
    assert torch.count_nonzero(output.delta[~mask]) == 0
    assert (
        float(torch.linalg.vector_norm(output.delta.detach(), dim=-1).max())
        <= 0.2000001
    )
    loss = output.x.square().sum() + output.pos.square().sum()
    gradient = torch.autograd.grad(loss, pos, create_graph=True)[0]
    second = torch.autograd.grad(gradient.square().sum(), pos)[0]
    assert torch.isfinite(gradient).all()
    assert torch.isfinite(second).all()
