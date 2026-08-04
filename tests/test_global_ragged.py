from __future__ import annotations

import pytest
import torch

from equivariant_linear_attention.geometry.layout import pack_graph_layout
from equivariant_linear_attention.nn.parity import (
    _exact_balanced_attention,
    _grouped_mm_feature_gemm,
    _graph_loop_feature_gemm_reference,
    _layout_feature_gemm,
)


def _ragged_layout(
    counts_value: tuple[int, ...] = (1, 2, 7, 9, 31, 33),
) -> tuple[torch.Tensor, object]:
    counts = torch.tensor(counts_value, dtype=torch.long)
    grouped = torch.repeat_interleave(torch.arange(counts.numel()), counts)
    permutation = torch.randperm(grouped.numel())
    batch = grouped.index_select(0, permutation)
    return batch, pack_graph_layout(
        batch,
        maximum_padding_ratio=1.0,
        maximum_buckets=1,
    )


def test_ragged_segmented_global_matches_graph_oracle_and_double_backward() -> None:
    torch.manual_seed(103)
    _, layout = _ragged_layout()
    assert layout.structure == "ragged"
    shapes = (
        (layout.num_nodes, 2, 5),
        (layout.num_nodes, 2, 5),
        (layout.num_nodes, 2, 4),
    )
    actual_inputs = tuple(
        torch.randn(shape, dtype=torch.float64, requires_grad=True) for shape in shapes
    )
    expected_inputs = tuple(
        value.detach().clone().requires_grad_(True) for value in actual_inputs
    )

    actual = _layout_feature_gemm(*actual_inputs, layout)
    expected = _graph_loop_feature_gemm_reference(*expected_inputs, layout)
    torch.testing.assert_close(actual, expected)

    actual_first = torch.autograd.grad(
        actual.square().sum(), actual_inputs, create_graph=True
    )
    expected_first = torch.autograd.grad(
        expected.square().sum(), expected_inputs, create_graph=True
    )
    for candidate, reference in zip(actual_first, expected_first, strict=True):
        torch.testing.assert_close(candidate, reference)

    actual_second = torch.autograd.grad(
        sum(value.square().sum() for value in actual_first), actual_inputs
    )
    expected_second = torch.autograd.grad(
        sum(value.square().sum() for value in expected_first), expected_inputs
    )
    for candidate, reference in zip(actual_second, expected_second, strict=True):
        torch.testing.assert_close(candidate, reference)


def test_ragged_segmented_lane_never_uses_cached_padding_buckets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(107)
    _, layout = _ragged_layout()
    query = torch.randn(layout.num_nodes, 2, 6)
    key = torch.randn_like(query)
    value = torch.randn(layout.num_nodes, 2, 3)

    def forbidden(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("ragged segmented execution must not gather a bucket")

    monkeypatch.setattr(type(layout.buckets[0]), "gather", forbidden)
    output = _layout_feature_gemm(query, key, value, layout)
    assert output.shape == value.shape
    assert torch.isfinite(output).all()


def test_public_extreme_layout_uses_padding_free_segmented_gemm() -> None:
    torch.manual_seed(108)
    counts = torch.tensor([129, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    batch = torch.repeat_interleave(torch.arange(counts.numel()), counts)
    layout = pack_graph_layout(batch)
    assert layout.structure == "extreme"
    query = torch.randn(layout.num_nodes, 2, 5, dtype=torch.float64)
    key = torch.randn_like(query)
    value = torch.randn(layout.num_nodes, 2, 4, dtype=torch.float64)

    actual = _layout_feature_gemm(query, key, value, layout)
    expected = _graph_loop_feature_gemm_reference(query, key, value, layout)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "counts_value",
    ((1, 2, 7, 9, 31, 33), (1, 3)),
)
def test_grouped_mm_ragged_layout_matches_oracle_without_node_padding(
    monkeypatch: pytest.MonkeyPatch,
    counts_value: tuple[int, ...],
) -> None:
    import equivariant_linear_attention.nn.parity as parity_module

    torch.manual_seed(109)
    _, layout = _ragged_layout(counts_value)
    query = torch.randn(layout.num_nodes, 2, 5, dtype=torch.float64)
    key = torch.randn_like(query)
    value = torch.randn(layout.num_nodes, 2, 4, dtype=torch.float64)
    observed: list[tuple[tuple[int, ...], tuple[int, ...], int]] = []

    def grouped_mm(
        left: torch.Tensor,
        right: torch.Tensor,
        *,
        offs: torch.Tensor,
        **_kwargs: object,
    ) -> torch.Tensor:
        boundaries = [0, *offs.tolist()]
        observed.append((tuple(left.shape), tuple(right.shape), int(offs[-1])))
        if right.ndim == 2:
            return torch.stack(
                [
                    left[:, start:stop] @ right[start:stop]
                    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
                ]
            )
        output = left.new_zeros((left.shape[0], right.shape[-1]))
        for group, (start, stop) in enumerate(
            zip(boundaries[:-1], boundaries[1:], strict=True)
        ):
            output[start:stop] = left[start:stop] @ right[group]
        return output

    monkeypatch.setattr(parity_module.F, "grouped_mm", grouped_mm)
    actual = _grouped_mm_feature_gemm(query, key, value, layout)
    expected = _graph_loop_feature_gemm_reference(query, key, value, layout)

    torch.testing.assert_close(actual, expected)
    assert len(observed) == 2
    # The ragged token dimension remains concatenated rather than G x max(Ng).
    tokens = layout.num_nodes * query.shape[1]
    assert observed[0][0][0] == query.shape[-1]
    assert tokens < observed[0][0][1] <= tokens + 8
    assert observed[1][0][0] == observed[0][0][1]
    assert observed[1][0][1] == query.shape[-1]
    assert observed[0][2] == observed[1][2] == tokens


def test_balanced_attention_reaches_bf16_grouped_mm_after_fp32_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import equivariant_linear_attention.nn.parity as parity_module

    torch.manual_seed(113)
    _, layout = _ragged_layout()
    query = torch.randn(
        layout.num_nodes,
        2,
        5,
        dtype=torch.bfloat16,
    ).abs().add_(0.25)
    key = torch.randn_like(query).abs().add_(0.25)
    value = torch.randn(layout.num_nodes, 2, 4, dtype=torch.bfloat16)
    calls: list[tuple[torch.dtype, torch.dtype, torch.dtype]] = []

    expected = _exact_balanced_attention(
        query,
        key,
        value,
        layout.batch,
        layout,
        eps=1.0e-6,
    )

    monkeypatch.setattr(
        parity_module,
        "_can_use_grouped_mm",
        lambda *_args: True,
    )

    def grouped(
        grouped_query: torch.Tensor,
        grouped_key: torch.Tensor,
        grouped_value: torch.Tensor,
        grouped_layout: object,
    ) -> torch.Tensor:
        calls.append(
            (grouped_query.dtype, grouped_key.dtype, grouped_value.dtype)
        )
        return _graph_loop_feature_gemm_reference(
            grouped_query,
            grouped_key,
            grouped_value,
            grouped_layout,
        )

    monkeypatch.setattr(parity_module, "_grouped_mm_feature_gemm", grouped)
    with torch.inference_mode():
        actual = _exact_balanced_attention(
            query,
            key,
            value,
            layout.batch,
            layout,
            eps=1.0e-6,
        )

    assert calls == [(torch.bfloat16, torch.bfloat16, torch.bfloat16)]
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0.08, rtol=0.05)
