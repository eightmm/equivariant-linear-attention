from __future__ import annotations

import pytest
import torch

from equivariant_attention import EquivariantAttention, EquivariantAttentionConfig


def _model(
    *,
    gated: bool = False,
    grouped: bool = False,
) -> EquivariantAttention:
    return EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=5,
            hidden_irreps="8x0e + 2x1o",
            output_irreps="2x0e + 1x1o + 1x2e",
            num_layers=2,
            num_heads=2,
            local_head_counts=(2, 0),
            use_key_balancing=False,
            use_gated_local_transport=gated,
            use_grouped_invariant_normalization=grouped,
        )
    ).double()


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    node_feats = torch.randn(7, 5, dtype=torch.float64)
    pos = 0.35 * torch.randn(7, 3, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6],
            [0, 1, 2, 3, 4, 5, 6, 1, 2, 0, 4, 5, 6, 3],
        ]
    )
    return node_feats, pos, batch, edge_index


def test_gated_local_defaults_are_exactly_backward_compatible() -> None:
    torch.manual_seed(2401)
    default = _model()
    torch.manual_seed(2401)
    explicit = _model(gated=False, grouped=False)
    node_feats, pos, batch, edge_index = _inputs()

    assert list(default.state_dict()) == list(explicit.state_dict())
    for name, value in default.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[name]), name
    default_output = default(
        node_feats, pos, batch=batch, edge_index=edge_index
    )
    explicit_output = explicit(
        node_feats, pos, batch=batch, edge_index=edge_index
    )
    for name in default_output:
        assert torch.equal(default_output[name], explicit_output[name]), name


def test_gated_local_preserves_every_common_initialization() -> None:
    torch.manual_seed(2402)
    baseline = _model()
    torch.manual_seed(2402)
    candidate = _model(gated=True, grouped=True)

    baseline_state = baseline.state_dict()
    candidate_state = candidate.state_dict()
    common = baseline_state.keys() & candidate_state.keys()

    assert common
    assert any("gated_local" in name for name in candidate_state)
    for name in common:
        assert torch.equal(baseline_state[name], candidate_state[name]), name


def test_gated_local_public_path_is_o3_translation_and_permutation_consistent() -> None:
    torch.manual_seed(2403)
    model = _model(gated=True, grouped=True)
    node_feats, pos, batch, edge_index = _inputs()
    orthogonal, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    orthogonal[:, 0].neg_()
    translation = torch.randn(1, 3, dtype=torch.float64)
    permutation = torch.tensor([4, 0, 6, 2, 3, 1, 5])
    inverse = torch.argsort(permutation)

    reference = model(node_feats, pos, batch=batch, edge_index=edge_index)
    moved = model(
        node_feats,
        pos @ orthogonal.T + translation,
        batch=batch,
        edge_index=edge_index,
    )
    permuted = model(
        node_feats[permutation],
        pos[permutation],
        batch=batch[permutation],
        edge_index=inverse[edge_index],
    )

    assert torch.allclose(moved["node_scalars"], reference["node_scalars"], atol=1e-9)
    assert torch.allclose(
        moved["node_vectors"],
        torch.einsum("nca,ba->ncb", reference["node_vectors"], orthogonal),
        atol=1e-9,
    )
    expected_tensor = torch.einsum(
        "ab,nkbc,dc->nkad", orthogonal, reference["node_tensors"], orthogonal
    )
    assert torch.allclose(moved["node_tensors"], expected_tensor, atol=1e-9)
    for name in ("node_scalars", "node_vectors", "node_tensors"):
        assert torch.allclose(permuted[name][inverse], reference[name], atol=1e-9)


def test_gated_local_receives_finite_nonzero_state_and_coordinate_gradients() -> None:
    torch.manual_seed(2404)
    model = _model(gated=True, grouped=True)
    node_feats, pos, batch, edge_index = _inputs()
    node_feats.requires_grad_()
    pos.requires_grad_()

    output = model(
        node_feats,
        pos,
        batch=batch,
        edge_index=edge_index,
    )
    output["graph_scalars"].square().sum().backward()

    gated_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "gated_local" in name
    ]
    assert gated_gradients
    assert all(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in gated_gradients
    )
    assert any(torch.count_nonzero(gradient) for gradient in gated_gradients)
    for gradient in (node_feats.grad, pos.grad):
        assert gradient is not None and torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient)


@pytest.mark.parametrize(
    "overrides",
    [
        {"local_head_counts": (0, 0), "use_gated_local_transport": True},
        {"local_head_counts": (1, 0), "use_gated_local_transport": True},
        {
            "local_head_counts": (2, 0),
            "use_gated_local_transport": True,
            "use_edge_conditioned_local_transport": True,
        },
    ],
)
def test_gated_local_rejects_inactive_partial_or_conflicting_routes(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="gated local"):
        EquivariantAttention(
            EquivariantAttentionConfig(
                node_dim=5,
                hidden_irreps="8x0e + 2x1o",
                num_layers=2,
                num_heads=2,
                **overrides,
            )
        )
