import pytest
import torch

import equivariant_attention as package
import equivariant_attention.moment as moment_module


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


def test_structured_attention_accumulates_low_precision_inputs_in_float32() -> None:
    query_scalar = torch.full((2, 1, 2), 1e-4, dtype=torch.float16)
    key_scalar = query_scalar.clone()
    query_vector = torch.zeros(2, 1, 3, dtype=torch.float16)
    key_vector = torch.zeros_like(query_vector)
    value = torch.tensor([[[1.0]], [[3.0]]], dtype=torch.float16)
    batch = torch.zeros(2, dtype=torch.long)

    output = moment_module._factorized_moment_attention(
        query_scalar,
        key_scalar,
        query_vector,
        key_vector,
        torch.ones(1, dtype=torch.float16),
        value,
        batch,
        num_graphs=1,
        balanced=True,
    )

    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()
    assert torch.allclose(
        output.float(), torch.full_like(output.float(), 2.0), atol=2e-3, rtol=0.0
    )


def test_graph_metadata_is_derived_once_per_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = moment_module._graph_metadata

    def counted(batch: torch.Tensor) -> tuple[int, torch.Tensor]:
        nonlocal calls
        calls += 1
        return original(batch)

    monkeypatch.setattr(moment_module, "_graph_metadata", counted)
    model = _make_model()
    node_feats = torch.randn(5, 4, dtype=torch.float64)
    pos = torch.randn(5, 3, dtype=torch.float64)
    batch = torch.tensor([0, 0, 1, 1, 1])

    model(node_feats, pos, batch=batch)

    assert calls == 1


def test_singleton_and_coincident_graphs_have_finite_backward() -> None:
    model = _make_model(dtype=torch.float32)
    node_feats = torch.randn(4, 4)
    pos = torch.zeros(4, 3)
    batch = torch.tensor([0, 1, 1, 1])

    loss = sum(
        value.float().square().mean()
        for value in model(node_feats, pos, batch=batch).values()
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_fp16_features_with_fp32_singleton_and_coincident_geometry_remain_finite() -> (
    None
):
    model = _make_model(dtype=torch.float16)
    node_feats = torch.randn(4, 4, dtype=torch.float16)
    pos = torch.zeros(4, 3, dtype=torch.float32)
    batch = torch.tensor([0, 1, 1, 1])

    outputs = model(node_feats, pos, batch=batch)

    assert all(torch.isfinite(value).all() for value in outputs.values())


def test_fp16_features_with_large_fp32_graph_geometry_remain_finite() -> None:
    model = (
        package.EquivariantAttention(
            package.EquivariantAttentionConfig(
                node_dim=1,
                hidden_irreps="2x0e + 1x1o",
                output_irreps="1x0e + 1x1o + 1x2e",
                num_layers=1,
                num_heads=1,
            )
        )
        .to(dtype=torch.float16)
        .eval()
    )
    node_count = 65_537
    node_feats = torch.zeros(node_count, 1, dtype=torch.float16)
    pos = torch.zeros(node_count, 3, dtype=torch.float32)
    pos[-1, 0] = 1e34

    with torch.no_grad():
        outputs = model(node_feats, pos)

    assert all(torch.isfinite(value).all() for value in outputs.values())


def test_fp16_features_preserve_large_fp32_coordinates_and_gradients() -> None:
    model = package.EquivariantAttention(
        package.EquivariantAttentionConfig(
            node_dim=1,
            hidden_irreps="2x0e + 1x1o",
            output_irreps="1x0e + 1x1o + 1x2e",
            num_layers=1,
            num_heads=1,
        )
    ).to(dtype=torch.float16)
    node_feats = torch.zeros(3, 1, dtype=torch.float16)
    pos = torch.tensor(
        [[-100_000.0, 0.0, 0.0], [0.0, 0.0, 0.0], [100_000.0, 0.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )

    outputs = model(node_feats, pos)
    loss = sum(value.float().square().mean() for value in outputs.values())
    loss.backward()

    assert all(torch.isfinite(value).all() for value in outputs.values())
    assert pos.dtype == torch.float32
    assert pos.grad is not None and torch.isfinite(pos.grad).all()
