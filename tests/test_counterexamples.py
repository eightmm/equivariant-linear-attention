import pytest
import torch

from equivariant_attention import EquivariantAttention, EquivariantAttentionConfig


def _model() -> EquivariantAttention:
    torch.manual_seed(211)
    return EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=3,
            hidden_irreps="8x0e + 2x1o",
            output_irreps="1x0e",
            num_layers=1,
            num_heads=2,
        )
    ).double()


def test_low_order_moment_collision_is_explicit() -> None:
    first = torch.tensor([-1.0, -1.0, 1.0, 1.0], dtype=torch.float64)
    second = torch.tensor([-2.0**0.5, 0.0, 0.0, 2.0**0.5], dtype=torch.float64)

    for order in range(3):
        assert torch.allclose(first.pow(order).mean(), second.pow(order).mean(), atol=1e-12, rtol=0.0)
    assert not torch.allclose(first.pow(4).mean(), second.pow(4).mean())


def test_global_normalized_geometry_has_no_cluster_decay_mechanism() -> None:
    def normalize(distance: float) -> torch.Tensor:
        pos = torch.tensor([[-distance / 2.0, 0.0, 0.0], [distance / 2.0, 0.0, 0.0]])
        centered = pos - pos.mean(dim=0, keepdim=True)
        rms = centered.square().sum(dim=-1).mean().sqrt()
        return centered / rms

    assert torch.equal(normalize(10.0), normalize(1_000_000.0))


def test_scalar_output_is_reflection_invariant_for_isolated_mirror() -> None:
    model = _model().eval()
    node_feats = torch.randn(5, 3, dtype=torch.float64)
    pos = torch.randn(5, 3, dtype=torch.float64)
    reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))

    reference = model(node_feats, pos)["graph_scalars"]
    mirrored = model(node_feats, pos @ reflection.T)["graph_scalars"]

    assert torch.allclose(reference, mirrored, atol=1e-6, rtol=0.0)


@pytest.mark.parametrize("reflection", [False, True])
def test_invariant_scalar_coordinate_gradient_is_o3_equivariant(reflection: bool) -> None:
    model = _model().eval()
    node_feats = torch.randn(6, 3, dtype=torch.float64)
    pos = torch.randn(6, 3, dtype=torch.float64, requires_grad=True)
    transform, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if reflection != (torch.linalg.det(transform) < 0):
        transform[:, 0] *= -1

    energy = model(node_feats, pos)["graph_scalars"].sum()
    gradient = torch.autograd.grad(energy, pos)[0]
    moved_pos = (pos.detach() @ transform.T).requires_grad_(True)
    moved_energy = model(node_feats, moved_pos)["graph_scalars"].sum()
    moved_gradient = torch.autograd.grad(moved_energy, moved_pos)[0]

    assert torch.allclose(moved_gradient, gradient @ transform.T, atol=1e-6, rtol=0.0)
