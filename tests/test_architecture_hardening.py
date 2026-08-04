from __future__ import annotations

import torch

from equivariant_linear_attention import ELA, ELAGraph
from equivariant_linear_attention.advanced import ELAConfig
from equivariant_linear_attention.model.stack import _EquivariantRMSNorm
from equivariant_linear_attention.nn.parity import _ParityState


def _state() -> _ParityState:
    return _ParityState(
        even_scalar=torch.randn(3, 8, dtype=torch.float64),
        odd_scalar=torch.randn(3, 2, dtype=torch.float64),
        polar_vector=torch.randn(3, 2, 3, dtype=torch.float64)
        * torch.tensor([1.0, 100.0], dtype=torch.float64)[None, :, None],
        axial_vector=torch.randn(3, 2, 3, dtype=torch.float64),
        even_tensor=torch.randn(3, 2, 5, dtype=torch.float64)
        * torch.tensor([1.0, 50.0], dtype=torch.float64)[None, :, None],
        odd_tensor=torch.randn(3, 2, 5, dtype=torch.float64),
    )


def test_geometric_capacity_scales_and_canonical_fusion_is_fixed() -> None:
    small = ELAConfig(input_irreps="4x0e", width=64)
    medium = ELAConfig(input_irreps="4x0e", width=128)
    large = ELAConfig(input_irreps="4x0e", width=256)

    assert small.num_heads < medium.num_heads < large.num_heads
    assert small.local_rank <= medium.local_rank <= large.local_rank
    assert large.canonical_contract()["message_fusion"] == (
        "fixed_exact_global_plus_local_sum"
    )


def test_active_equivariant_norm_is_copy_wise_for_vectors_and_tensors() -> None:
    norm = _EquivariantRMSNorm(scalar_width=8, num_heads=2, eps=1e-12).double()
    output = norm(_state())

    vector_rms = output.polar_vector.square().mean(dim=-1).sqrt()
    tensor_matrix = torch.stack(
        (
            output.even_tensor[..., 0],
            output.even_tensor[..., 1],
            output.even_tensor[..., 2],
            output.even_tensor[..., 3],
            output.even_tensor[..., 4],
        ),
        dim=-1,
    )
    assert torch.allclose(vector_rms, torch.ones_like(vector_rms), atol=2e-10)
    # The two copies started at very different scales and remain independently
    # finite after normalization; no copy borrows the other's RMS.
    assert torch.isfinite(tensor_matrix).all()
    ratio = output.even_tensor[:, 0].norm() / output.even_tensor[:, 1].norm()
    assert 0.5 < float(ratio.detach()) < 2.0


def test_second_moment_chirality_is_zero_initialized() -> None:
    model = ELA("4x0e", width=32, depth=2)
    for layer in model.layers:
        torch.testing.assert_close(
            layer.second_moment_chiral_mix,
            torch.zeros_like(layer.second_moment_chiral_mix),
        )


def test_relation_conditioning_is_active_and_differentiable() -> None:
    torch.manual_seed(17)
    model = ELA(
        "3x0e",
        "1x0e",
        width=16,
        depth=1,
        cutoff=3.0,
        edge_types=2,
    ).double()
    layer = model.layers[0]
    assert layer.relation_radial_scale is not None
    assert layer.relation_value_gate is not None
    with torch.no_grad():
        # The local output projection is deliberately zero-initialized. Open
        # one scalar route before checking the identity-initialized relation
        # conditioner itself.
        layer.local_scalar_out.weight.fill_(0.2)
        layer.relation_radial_scale[1].fill_(0.4)
        layer.relation_value_gate[1].fill_(0.25)

    x = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
    pos = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
    edges = torch.tensor([[0, 1, 2], [1, 2, 0]])
    first = model(
        ELAGraph(x, pos, edge_index=edges, edge_type=torch.zeros(3, dtype=torch.long))
    ).x
    second = model(
        ELAGraph(x, pos, edge_index=edges, edge_type=torch.ones(3, dtype=torch.long))
    ).x

    assert not torch.allclose(first, second)
    second.square().sum().backward()
    assert layer.relation_radial_scale.grad is not None
    assert layer.relation_value_gate.grad is not None
    assert torch.isfinite(layer.relation_radial_scale.grad).all()
    assert torch.isfinite(layer.relation_value_gate.grad).all()


def test_global_radial_shell_has_finite_double_backward_at_graph_center() -> None:
    model = ELA("2x0e", "1x0e", width=16, depth=1, cutoff=3.0).double()
    x = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    pos = torch.tensor(
        [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    edges = torch.arange(3).repeat(3).reshape(3, 3)
    edges = torch.stack((edges.flatten(), edges.T.flatten()))
    output = model(ELAGraph(x, pos, edge_index=edges)).x.sum()
    first = torch.autograd.grad(output, pos, create_graph=True)[0]
    second = torch.autograd.grad(first.square().sum(), pos)[0]

    assert torch.isfinite(first).all()
    assert torch.isfinite(second).all()
