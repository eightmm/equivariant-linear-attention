import pytest
import torch

import equivariant_attention as package
from equivariant_attention.moment import _factorized_attention


def _make_model(dtype: torch.dtype = torch.float64) -> torch.nn.Module:
    return package.EquivariantAttention(
        package.EquivariantAttentionConfig(
            node_dim=4,
            hidden_irreps="8x0e + 2x1o",
            output_irreps="1x0e + 1x1o + 1x2e",
            num_layers=2,
            num_heads=2,
        )
    ).to(dtype=dtype)


def test_package_exposes_one_attention_implementation() -> None:
    assert package.EquivariantAttention.__module__ == "equivariant_attention.moment"
    assert not hasattr(package, "EquivariantMomentAttention")
    assert not hasattr(package, "RichEquivariantAttention")


def test_forward_returns_only_tensor_outputs_and_model_owns_metadata() -> None:
    model = _make_model()
    node_feats = torch.randn(5, 4, dtype=torch.float64)
    pos = torch.randn(5, 3, dtype=torch.float64)

    outputs = model(node_feats, pos)

    assert set(outputs) == {
        "node_scalars",
        "node_vectors",
        "node_tensors",
        "graph_scalars",
        "graph_vectors",
        "graph_tensors",
    }
    assert all(isinstance(value, torch.Tensor) for value in outputs.values())
    assert model.attention_kind == "factorized_moment"
    assert model.symmetry == "O3"


@pytest.mark.parametrize(
    ("batch", "error", "message"),
    [
        (torch.tensor([0.0, 0.0, 1.0, 1.0]), TypeError, "integer dtype"),
        (torch.tensor([0, 0, 2, 2]), ValueError, "contiguous"),
        (torch.tensor([-1, 0, 0, 0]), ValueError, "nonnegative"),
    ],
)
def test_batch_contract_rejects_ambiguous_graph_ids(
    batch: torch.Tensor,
    error: type[Exception],
    message: str,
) -> None:
    model = _make_model()
    node_feats = torch.randn(4, 4, dtype=torch.float64)
    pos = torch.randn(4, 3, dtype=torch.float64)

    with pytest.raises(error, match=message):
        model(node_feats, pos, batch=batch)


def test_factorized_attention_accumulates_low_precision_inputs_in_float32() -> None:
    query = torch.full((2, 1, 2), 1e-4, dtype=torch.float16)
    key = query.clone()
    value = torch.tensor([[[1.0]], [[3.0]]], dtype=torch.float16)
    batch = torch.zeros(2, dtype=torch.long)

    output = _factorized_attention(query, key, value, batch, balanced=True, eps=1e-12)

    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()
    assert torch.allclose(output.float(), torch.full_like(output.float(), 2.0), atol=2e-3, rtol=0.0)


def test_singleton_and_coincident_graphs_have_finite_backward() -> None:
    model = _make_model(dtype=torch.float32)
    node_feats = torch.randn(4, 4)
    pos = torch.zeros(4, 3)
    batch = torch.tensor([0, 1, 1, 1])

    loss = sum(value.float().square().mean() for value in model(node_feats, pos, batch=batch).values())
    loss.backward()

    assert torch.isfinite(loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_fp16_singleton_and_coincident_graphs_remain_finite() -> None:
    model = _make_model(dtype=torch.float16)
    node_feats = torch.randn(4, 4, dtype=torch.float16)
    pos = torch.zeros(4, 3, dtype=torch.float16)
    batch = torch.tensor([0, 1, 1, 1])

    outputs = model(node_feats, pos, batch=batch)

    assert all(torch.isfinite(value).all() for value in outputs.values())


def test_fp16_large_graph_geometry_remains_finite() -> None:
    model = package.EquivariantAttention(
        package.EquivariantAttentionConfig(
            node_dim=1,
            hidden_irreps="2x0e + 1x1o",
            output_irreps="1x0e + 1x1o + 1x2e",
            num_layers=1,
            num_heads=1,
        )
    ).to(dtype=torch.float16).eval()
    node_count = 65_537
    node_feats = torch.zeros(node_count, 1, dtype=torch.float16)
    pos = torch.zeros(node_count, 3, dtype=torch.float16)
    pos[-1, 0] = torch.finfo(torch.float16).max

    with torch.no_grad():
        outputs = model(node_feats, pos)

    assert all(torch.isfinite(value).all() for value in outputs.values())
