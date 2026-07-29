from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from equivariant_attention import EquivariantAttention, EquivariantAttentionConfig
from equivariant_attention import moment


def _attention_inputs(
    *,
    dtype: torch.dtype = torch.float64,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260729)
    counts = torch.tensor([3, 5, 2], dtype=torch.long)
    batch = torch.repeat_interleave(torch.arange(3), counts)
    nodes, heads, scalar_features, value_features = 10, 2, 4, 37
    values = {
        "query_scalar": torch.rand(
            nodes, heads, scalar_features, generator=generator, dtype=dtype
        ),
        "key_scalar": torch.rand(
            nodes, heads, scalar_features, generator=generator, dtype=dtype
        ),
        "query_vector": 0.3
        * torch.randn(nodes, heads, 3, generator=generator, dtype=dtype),
        "key_vector": 0.3
        * torch.randn(nodes, heads, 3, generator=generator, dtype=dtype),
        "kernel_scale": torch.tensor([0.2, 0.4], dtype=dtype),
        "value": torch.randn(
            nodes, heads, value_features, generator=generator, dtype=dtype
        ),
        "alignment_scale": torch.tensor([0.12, 0.18], dtype=dtype),
        "alignment_dot_scale": torch.tensor([0.07, 0.11], dtype=dtype),
        "spatial_features": 0.2
        * torch.rand(nodes, heads, 7, generator=generator, dtype=dtype),
        "spatial_key_features": 0.2
        * torch.rand(nodes, heads, 7, generator=generator, dtype=dtype),
    }
    return values, batch, counts


def _run_attention(
    backend: str,
    *,
    balanced: bool,
    floor_mode: str,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    values, batch, counts = _attention_inputs()
    differentiable = {}
    for name, value in values.items():
        differentiable[name] = value.clone().requires_grad_(True)
    output = moment._factorized_moment_attention(
        differentiable["query_scalar"],
        differentiable["key_scalar"],
        differentiable["query_vector"],
        differentiable["key_vector"],
        differentiable["kernel_scale"],
        differentiable["value"],
        batch,
        num_graphs=3,
        balanced=balanced,
        alignment_scale=differentiable["alignment_scale"],
        alignment_dot_scale=differentiable["alignment_dot_scale"],
        kernel_floor=0.9,
        kernel_floor_mode=floor_mode,
        graph_counts=counts,
        spatial_features=differentiable["spatial_features"],
        spatial_key_features=differentiable["spatial_key_features"],
        reduction_backend=backend,
    )
    inputs = tuple(differentiable.values())
    gradients = torch.autograd.grad(output.square().sum(), inputs)
    return output, gradients


@pytest.mark.parametrize(
    ("balanced", "floor_mode"),
    [(False, "fixed"), (True, "fixed"), (False, "inverse_graph_size")],
)
def test_feature_gemm_matches_outer_scatter_forward_and_gradients(
    balanced: bool,
    floor_mode: str,
) -> None:
    reference, reference_gradients = _run_attention(
        "outer_scatter",
        balanced=balanced,
        floor_mode=floor_mode,
    )
    candidate, candidate_gradients = _run_attention(
        "feature_gemm",
        balanced=balanced,
        floor_mode=floor_mode,
    )

    torch.testing.assert_close(candidate, reference, rtol=2e-12, atol=2e-12)
    for candidate_gradient, reference_gradient in zip(
        candidate_gradients, reference_gradients, strict=True
    ):
        torch.testing.assert_close(
            candidate_gradient,
            reference_gradient,
            rtol=2e-11,
            atol=2e-11,
        )


def _global_config(**overrides: object) -> EquivariantAttentionConfig:
    config = EquivariantAttentionConfig(
        node_dim=5,
        hidden_irreps="16x0e + 2x1o",
        output_irreps="1x0e + 1x1o",
        num_layers=2,
        num_heads=2,
        local_head_counts=(0, 0),
    )
    return replace(config, **overrides)


def test_feature_gemm_full_model_matches_outer_backend_and_keeps_state_schema() -> None:
    torch.manual_seed(71)
    reference = EquivariantAttention(_global_config())
    torch.manual_seed(71)
    candidate = EquivariantAttention(
        _global_config(global_reduction_backend="feature_gemm")
    )

    assert reference.state_dict().keys() == candidate.state_dict().keys()
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(candidate.state_dict()[name], value, rtol=0, atol=0)

    generator = torch.Generator().manual_seed(19)
    node_feats = torch.randn(9, 5, generator=generator, dtype=torch.float64)
    pos = torch.randn(9, 3, generator=generator, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1])
    reference = reference.double()
    candidate = candidate.double()
    reference_feats = node_feats.clone().requires_grad_(True)
    candidate_feats = node_feats.clone().requires_grad_(True)
    reference_pos = pos.clone().requires_grad_(True)
    candidate_pos = pos.clone().requires_grad_(True)

    reference_output = reference(reference_feats, reference_pos, batch=batch)
    candidate_output = candidate(candidate_feats, candidate_pos, batch=batch)
    for key in reference_output:
        torch.testing.assert_close(
            candidate_output[key],
            reference_output[key],
            rtol=3e-11,
            atol=3e-11,
        )
    reference_loss = sum(value.square().sum() for value in reference_output.values())
    candidate_loss = sum(value.square().sum() for value in candidate_output.values())
    reference_gradients = torch.autograd.grad(
        reference_loss, (reference_feats, reference_pos)
    )
    candidate_gradients = torch.autograd.grad(
        candidate_loss, (candidate_feats, candidate_pos)
    )
    for candidate_gradient, reference_gradient in zip(
        candidate_gradients, reference_gradients, strict=True
    ):
        torch.testing.assert_close(
            candidate_gradient,
            reference_gradient,
            rtol=3e-10,
            atol=3e-10,
        )


def test_global_reduction_backend_validation_is_explicit() -> None:
    with pytest.raises(ValueError, match="global_reduction_backend"):
        EquivariantAttention(_global_config(global_reduction_backend="auto"))


def test_feature_gemm_ragged_fallback_matches_outer_scatter() -> None:
    counts = torch.tensor([50, *([1] * 19)], dtype=torch.long)
    batch = torch.repeat_interleave(torch.arange(20), counts)
    nodes = int(counts.sum())
    generator = torch.Generator().manual_seed(20260730)
    query_scalar = torch.rand(nodes, 1, 3, generator=generator, dtype=torch.float64)
    key_scalar = torch.rand(nodes, 1, 3, generator=generator, dtype=torch.float64)
    query_vector = 0.2 * torch.randn(
        nodes, 1, 3, generator=generator, dtype=torch.float64
    )
    key_vector = 0.2 * torch.randn(
        nodes, 1, 3, generator=generator, dtype=torch.float64
    )
    value = torch.randn(
        nodes, 1, 11, generator=generator, dtype=torch.float64
    )
    common = dict(
        num_graphs=20,
        balanced=True,
        alignment_scale=torch.tensor([0.1], dtype=torch.float64),
        alignment_dot_scale=torch.tensor([0.08], dtype=torch.float64),
        kernel_floor=1.0,
        graph_counts=counts,
    )
    assert moment._graph_padded_layout(batch, counts, 20) is None

    reference = moment._factorized_moment_attention(
        query_scalar,
        key_scalar,
        query_vector,
        key_vector,
        torch.tensor([0.2], dtype=torch.float64),
        value,
        batch,
        reduction_backend="outer_scatter",
        **common,
    )
    candidate = moment._factorized_moment_attention(
        query_scalar,
        key_scalar,
        query_vector,
        key_vector,
        torch.tensor([0.2], dtype=torch.float64),
        value,
        batch,
        reduction_backend="feature_gemm",
        **common,
    )
    torch.testing.assert_close(candidate, reference, rtol=2e-12, atol=2e-12)


def test_ragged_feature_gemm_groups_once_and_matches_backward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = torch.tensor([80, *([1] * 31)], dtype=torch.long)
    contiguous_batch = torch.repeat_interleave(torch.arange(32), counts)
    generator = torch.Generator().manual_seed(20260731)
    permutation = torch.randperm(contiguous_batch.numel(), generator=generator)
    batch = contiguous_batch[permutation]
    nodes = batch.numel()
    query_scalar = torch.rand(
        nodes, 1, 4, generator=generator, dtype=torch.float64
    )
    key_scalar = torch.rand(
        nodes, 1, 4, generator=generator, dtype=torch.float64
    )
    query_vector = 0.2 * torch.randn(
        nodes, 1, 3, generator=generator, dtype=torch.float64
    )
    key_vector = 0.2 * torch.randn(
        nodes, 1, 3, generator=generator, dtype=torch.float64
    )
    value = torch.randn(
        nodes, 1, 13, generator=generator, dtype=torch.float64
    )
    scale = torch.tensor([0.2], dtype=torch.float64)
    common = dict(
        num_graphs=32,
        balanced=True,
        alignment_scale=torch.tensor([0.1], dtype=torch.float64),
        alignment_dot_scale=torch.tensor([0.08], dtype=torch.float64),
        kernel_floor=1.0,
        graph_counts=counts,
    )
    layout = moment._graph_feature_layout(batch, counts, 32)
    assert isinstance(layout, moment._GraphRaggedLayout)

    def run(backend: str) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        differentiable = tuple(
            tensor.clone().requires_grad_(True)
            for tensor in (
                query_scalar,
                key_scalar,
                query_vector,
                key_vector,
                value,
            )
        )
        output = moment._factorized_moment_attention(
            differentiable[0],
            differentiable[1],
            differentiable[2],
            differentiable[3],
            scale,
            differentiable[4],
            batch,
            reduction_backend=backend,
            **common,
        )
        return output, torch.autograd.grad(output.square().sum(), differentiable)

    reference, reference_gradients = run("outer_scatter")

    def forbidden_nonzero(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("ragged GEMM must not rescan batch per graph")

    monkeypatch.setattr(torch, "nonzero", forbidden_nonzero)
    candidate, candidate_gradients = run("feature_gemm")

    torch.testing.assert_close(candidate, reference, rtol=2e-12, atol=2e-12)
    for candidate_gradient, reference_gradient in zip(
        candidate_gradients,
        reference_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            candidate_gradient,
            reference_gradient,
            rtol=2e-11,
            atol=2e-11,
        )


@pytest.mark.parametrize("candidate_kind", ["adaptive_spatial", "tensor_value"])
def test_feature_gemm_composes_with_wide_registered_global_payloads(
    candidate_kind: str,
) -> None:
    if candidate_kind == "adaptive_spatial":
        common = dict(
            node_dim=5,
            hidden_irreps="16x0e + 2x1o",
            output_irreps="1x0e + 1x1o",
            num_layers=3,
            num_heads=2,
            local_head_counts=(2, 0, 2),
            use_adaptive_multiscale_spatial_kernel=True,
            use_key_balancing=False,
        )
    else:
        common = dict(
            node_dim=5,
            hidden_irreps="16x0e + 2x1o + 2x2e",
            output_irreps="1x0e + 1x1o + 1x2e",
            num_layers=2,
            num_heads=2,
            local_head_counts=(0, 0),
            use_static_tensor_carrier=True,
            use_global_tensor_value_transport=True,
            use_key_balancing=False,
        )
    torch.manual_seed(177)
    reference = EquivariantAttention(EquivariantAttentionConfig(**common)).double()
    torch.manual_seed(177)
    candidate = EquivariantAttention(
        EquivariantAttentionConfig(
            **common,
            global_reduction_backend="feature_gemm",
        )
    ).double()
    generator = torch.Generator().manual_seed(178)
    node_feats = torch.randn(7, 5, generator=generator, dtype=torch.float64)
    pos = torch.randn(7, 3, generator=generator, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    edge_index = torch.tensor(
        [
            [0, 1, 2, 0, 1, 2, 3, 4, 5, 6, 3, 4, 5, 6],
            [0, 1, 2, 1, 2, 0, 3, 4, 5, 6, 4, 5, 6, 3],
        ]
    )
    kwargs = (
        {"edge_index": edge_index}
        if candidate_kind == "adaptive_spatial"
        else {}
    )

    reference_output = reference(node_feats, pos, batch=batch, **kwargs)
    candidate_output = candidate(node_feats, pos, batch=batch, **kwargs)
    for name in reference_output:
        torch.testing.assert_close(
            candidate_output[name],
            reference_output[name],
            rtol=3e-11,
            atol=3e-11,
        )
