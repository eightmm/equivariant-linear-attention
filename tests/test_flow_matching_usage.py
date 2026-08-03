from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

from equivariant_attention import ELA, ELABatch


def _complete_edges(ptr: torch.Tensor) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for start, stop in zip(ptr[:-1], ptr[1:], strict=True):
        first = int(start.item())
        nodes = int((stop - start).item())
        receiver = torch.arange(nodes).repeat_interleave(nodes) + first
        sender = torch.arange(nodes).repeat(nodes) + first
        parts.append(torch.stack([receiver, sender]))
    return torch.cat(parts, dim=1)


def _center_per_graph(
    coordinates: torch.Tensor,
    batch_index: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    total = coordinates.new_zeros((num_graphs, 3))
    total.index_add_(0, batch_index, coordinates)
    count = torch.bincount(batch_index, minlength=num_graphs).clamp_min(1)
    return coordinates - total[batch_index] / count[batch_index, None].to(
        coordinates.dtype
    )


def _fixture() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    ptr = torch.tensor([0, 4, 7])
    batch_index = torch.repeat_interleave(
        torch.arange(2),
        ptr[1:] - ptr[:-1],
    )
    features = torch.randn(7, 4, dtype=torch.float64)
    x0 = _center_per_graph(
        torch.randn(7, 3, dtype=torch.float64),
        batch_index,
        2,
    )
    x1 = _center_per_graph(
        torch.randn(7, 3, dtype=torch.float64),
        batch_index,
        2,
    )
    time = torch.tensor([[0.2], [0.7]], dtype=torch.float64)
    x_t = (1.0 - time[batch_index]) * x0 + time[batch_index] * x1
    target = x1 - x0
    return features, x_t, target, ptr, batch_index, time


def _model() -> ELA:
    return ELA(
        input_irreps="4x0e",
        output_irreps="1x1o",
        condition_dim=1,
        width=16,
        depth=1,
        cutoff=10.0,
        num_rbf=8,
    ).double()


def test_flow_matching_velocity_shape_and_backward() -> None:
    torch.manual_seed(101)
    features, x_t, target, ptr, _, time = _fixture()
    features.requires_grad_(True)
    x_t.requires_grad_(True)
    model = _model()
    batch = ELABatch(
        node_irreps=features,
        positions=x_t,
        ptr=ptr,
        edge_index=_complete_edges(ptr),
        condition=time,
    )

    output = model(batch)
    split_velocity = model.split_output(output["node"])["1o"]
    velocity = split_velocity.squeeze(-2)
    loss = F.mse_loss(velocity, target)
    loss.backward()

    assert output["node"].shape == (7, 3)
    assert split_velocity.shape == (7, 1, 3)
    assert velocity.shape == target.shape
    torch.testing.assert_close(output["delta"], torch.zeros_like(x_t))
    assert torch.isfinite(loss)
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert x_t.grad is not None and torch.isfinite(x_t.grad).all()
    assert float(features.grad.abs().sum()) > 0.0
    assert float(x_t.grad.abs().sum()) > 0.0
    parameter_grads = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert parameter_grads
    assert all(torch.isfinite(gradient).all() for gradient in parameter_grads)
    assert sum(float(gradient.abs().sum()) for gradient in parameter_grads) > 0.0


@pytest.mark.parametrize("determinant_sign", [1, -1], ids=["proper", "improper"])
def test_flow_matching_velocity_obeys_o3_translation_and_permutation(
    determinant_sign: int,
) -> None:
    torch.manual_seed(103)
    features, x_t, target, ptr, _, time = _fixture()
    edges = _complete_edges(ptr)
    model = _model().eval()
    orthogonal, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if int(torch.linalg.det(orthogonal).sign().item()) != determinant_sign:
        orthogonal[:, 0].neg_()
    translation = torch.tensor([0.8, -1.3, 0.4], dtype=torch.float64)

    with torch.inference_mode():
        reference_output = model(
            ELABatch(features, x_t, ptr=ptr, edge_index=edges, condition=time)
        )
        reference = model.split_output(reference_output["node"])["1o"].squeeze(-2)
        transformed_output = model(
            ELABatch(
                features,
                x_t @ orthogonal.T + translation,
                ptr=ptr,
                edge_index=edges,
                condition=time,
            )
        )
        transformed = model.split_output(transformed_output["node"])[
            "1o"
        ].squeeze(-2)

    expected_velocity = reference @ orthogonal.T
    transformed_target = target @ orthogonal.T
    torch.testing.assert_close(
        transformed,
        expected_velocity,
        atol=2e-8,
        rtol=2e-8,
    )
    torch.testing.assert_close(
        F.mse_loss(transformed, transformed_target),
        F.mse_loss(reference, target),
        atol=2e-10,
        rtol=2e-10,
    )

    permutation = torch.tensor([2, 0, 3, 1, 6, 4, 5])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel())
    with torch.inference_mode():
        permuted_output = model(
            ELABatch(
                features[permutation],
                x_t[permutation],
                ptr=ptr,
                edge_index=inverse[edges],
                condition=time,
            )
        )
        permuted = model.split_output(permuted_output["node"])["1o"].squeeze(-2)
    torch.testing.assert_close(
        permuted,
        reference[permutation],
        atol=2e-10,
        rtol=2e-10,
    )
    torch.testing.assert_close(
        F.mse_loss(permuted, target[permutation]),
        F.mse_loss(reference, target),
        atol=2e-10,
        rtol=2e-10,
    )
