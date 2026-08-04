from __future__ import annotations

import pytest
import torch

from equivariant_linear_attention import ELA, ELAGraph
from equivariant_linear_attention.geometry.layout import pack_graph_layout
from equivariant_linear_attention.nn.parity import (
    _can_use_grouped_mm,
    _exact_balanced_attention,
    _graph_loop_feature_gemm_reference,
    _grouped_mm_feature_gemm,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)


@pytest.mark.skipif(
    not torch.cuda.is_bf16_supported(),
    reason="CUDA BF16 is unavailable",
)
@pytest.mark.parametrize(
    "count_values",
    ((1, 2, 7, 9, 31, 33), (1, 3)),
)
def test_cuda_ragged_native_grouped_mm_matches_graph_oracle(
    count_values: tuple[int, ...],
) -> None:
    torch.manual_seed(701)
    device = torch.device("cuda")
    counts = torch.tensor(count_values, device=device)
    batch = torch.repeat_interleave(
        torch.arange(counts.numel(), device=device),
        counts,
    )
    layout = pack_graph_layout(
        batch,
        maximum_padding_ratio=1.0,
        maximum_buckets=1,
    )
    assert layout.structure == "ragged"
    query = torch.randn(
        layout.num_nodes,
        2,
        23,
        device=device,
        dtype=torch.bfloat16,
    )
    key = torch.randn_like(query)
    value = torch.randn(
        layout.num_nodes,
        2,
        37,
        device=device,
        dtype=torch.bfloat16,
    )

    with torch.inference_mode():
        assert _can_use_grouped_mm(query, key, value)
        actual = _grouped_mm_feature_gemm(query, key, value, layout)
        expected = _graph_loop_feature_gemm_reference(query, key, value, layout)

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, atol=0.5, rtol=0.04)


@pytest.mark.skipif(
    not torch.cuda.is_bf16_supported(),
    reason="CUDA BF16 is unavailable",
)
def test_cuda_balanced_attention_reaches_native_grouped_mm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import equivariant_linear_attention.nn.parity as parity_module

    torch.manual_seed(702)
    device = torch.device("cuda")
    counts = torch.tensor([1, 2, 7, 9, 31, 33], device=device)
    batch = torch.repeat_interleave(
        torch.arange(counts.numel(), device=device),
        counts,
    )
    layout = pack_graph_layout(
        batch,
        maximum_padding_ratio=1.0,
        maximum_buckets=1,
    )
    query = torch.rand(
        layout.num_nodes,
        2,
        23,
        device=device,
        dtype=torch.bfloat16,
    ).add_(0.25)
    key = torch.rand_like(query).add_(0.25)
    value = torch.randn(
        layout.num_nodes,
        2,
        37,
        device=device,
        dtype=torch.bfloat16,
    )

    with torch.inference_mode():
        monkeypatch.setattr(parity_module, "_can_use_grouped_mm", lambda *_: False)
        expected = _exact_balanced_attention(
            query,
            key,
            value,
            batch,
            layout,
            eps=1.0e-6,
        )

    calls = 0
    native = _grouped_mm_feature_gemm

    def observed(*args: object, **kwargs: object) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return native(*args, **kwargs)

    monkeypatch.setattr(parity_module, "_can_use_grouped_mm", _can_use_grouped_mm)
    monkeypatch.setattr(parity_module, "_grouped_mm_feature_gemm", observed)
    with torch.inference_mode():
        actual = _exact_balanced_attention(
            query,
            key,
            value,
            batch,
            layout,
            eps=1.0e-6,
        )

    assert calls == 1
    torch.testing.assert_close(actual, expected, atol=0.5, rtol=0.05)


@pytest.mark.skipif(
    not torch.cuda.is_bf16_supported(),
    reason="CUDA BF16 is unavailable",
)
def test_public_ela_extreme_batch_reaches_native_grouped_mm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import equivariant_linear_attention.nn.parity as parity_module

    torch.manual_seed(704)
    device = torch.device("cuda")
    counts = torch.tensor([129, 1, 1, 1, 1, 1, 1, 1], device=device)
    batch = torch.repeat_interleave(
        torch.arange(counts.numel(), device=device),
        counts,
    )
    graph = ELAGraph(
        torch.randn(
            int(counts.sum().item()),
            2,
            device=device,
            dtype=torch.bfloat16,
        ),
        torch.randn(int(counts.sum().item()), 3, device=device),
        edge_index=torch.empty((2, 0), device=device, dtype=torch.long),
        batch=batch,
    )
    model = ELA("2x0e", width=16, depth=1, cutoff=2.0).to(
        device=device,
        dtype=torch.bfloat16,
    ).eval()

    with monkeypatch.context() as context:
        context.setattr(parity_module, "_can_use_grouped_mm", lambda *_: False)
        with torch.inference_mode():
            expected = model(graph).x

    calls = 0
    native = parity_module._grouped_mm_feature_gemm

    def observed(*args: object, **kwargs: object) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return native(*args, **kwargs)

    monkeypatch.setattr(parity_module, "_grouped_mm_feature_gemm", observed)
    with torch.inference_mode():
        actual = model(graph).x

    assert graph._prepared_graph is not None
    assert graph._prepared_graph.graph_layout.structure == "extreme"
    assert calls > 0
    torch.testing.assert_close(actual, expected, atol=0.1, rtol=0.05)


def test_cuda_native_tensors_reuse_exact_validated_prepared_cache() -> None:
    torch.manual_seed(703)
    device = torch.device("cuda")
    sender = torch.tensor([0, 1, 2, 3], device=device)
    receiver = torch.tensor([1, 2, 3, 0], device=device)
    graph = ELAGraph(
        torch.randn(4, 2, device=device),
        torch.randn(4, 3, device=device),
        edge_index=torch.stack((sender, receiver)),
    )
    model = ELA("2x0e", width=16, depth=1, cutoff=2.0).to(device)

    with torch.inference_mode():
        model(graph)
        first = graph._prepared_graph
        model(graph)

    assert first is not None
    assert graph._prepared_graph is first
    assert graph._prepared_provenance is None
