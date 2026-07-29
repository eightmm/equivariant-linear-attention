from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from equivariant_attention.local_streaming import (
    LocalBackendCapability,
    default_local_backend_capabilities,
    select_local_backend,
    streamed_positive_csr,
    streamed_positive_ell,
    streamed_softmax_csr,
    streamed_softmax_ell,
)


def _csr_inputs(
    *,
    seed: int = 731,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    row_ptr = torch.tensor([0, 3, 3, 7, 9], dtype=torch.int32)
    score = (
        torch.rand(9, 3, generator=generator, dtype=torch.float64) + 0.2
    ).requires_grad_()
    value = torch.randn(
        9,
        3,
        4,
        generator=generator,
        dtype=torch.float64,
        requires_grad=True,
    )
    cutoff = (
        torch.rand(9, generator=generator, dtype=torch.float64) + 0.1
    ).requires_grad_()
    return score, value, cutoff, row_ptr


def _receiver_index(row_ptr: torch.Tensor) -> torch.Tensor:
    degree = row_ptr[1:] - row_ptr[:-1]
    return torch.repeat_interleave(
        torch.arange(degree.numel()),
        degree.to(dtype=torch.long),
    )


def _materialized_positive(
    score: torch.Tensor,
    value: torch.Tensor,
    cutoff: torch.Tensor,
    row_ptr: torch.Tensor,
) -> torch.Tensor:
    receiver = _receiver_index(row_ptr)
    weight = score * cutoff[:, None]
    mass = score.new_zeros((row_ptr.numel() - 1, score.shape[1]))
    numerator = value.new_zeros(
        (row_ptr.numel() - 1, *value.shape[1:]),
    )
    mass = mass.index_add(0, receiver, weight)
    numerator = numerator.index_add(0, receiver, weight[..., None] * value)
    return numerator / (1.0 + mass[..., None])


def _materialized_softmax(
    score: torch.Tensor,
    value: torch.Tensor,
    cutoff: torch.Tensor,
    row_ptr: torch.Tensor,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    boundaries = row_ptr.tolist()
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        if start == stop:
            rows.append(value.new_zeros(value.shape[1:]))
            continue
        effective = score[start:stop] + cutoff[start:stop, None].log()
        weight = effective.softmax(dim=0)
        rows.append(torch.einsum("eh,ehd->hd", weight, value[start:stop]))
    return torch.stack(rows)


def _loss(output: torch.Tensor) -> torch.Tensor:
    coefficients = torch.linspace(
        0.3,
        1.1,
        output.numel(),
        dtype=output.dtype,
        device=output.device,
    ).reshape_as(output)
    return (output.square() + output * coefficients).sum()


@pytest.mark.parametrize(
    ("streamed", "materialized"),
    [
        (streamed_positive_csr, _materialized_positive),
        (streamed_softmax_csr, _materialized_softmax),
    ],
)
def test_streamed_csr_matches_materialized_forward_and_input_gradients(
    streamed,
    materialized,
) -> None:
    score, value, cutoff, row_ptr = _csr_inputs()

    actual = streamed(
        score,
        value,
        cutoff,
        row_ptr,
        chunk_size=2,
    )
    expected = materialized(score, value, cutoff, row_ptr)

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    actual_gradients = torch.autograd.grad(
        _loss(actual),
        (score, value, cutoff),
        retain_graph=True,
    )
    expected_gradients = torch.autograd.grad(
        _loss(expected),
        (score, value, cutoff),
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=1e-11,
            atol=1e-11,
        )


def test_streamed_positive_is_mass_damped_and_handles_zero_degree() -> None:
    row_ptr = torch.tensor([0, 0, 2, 2], dtype=torch.int32)
    score = torch.tensor([[2.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
    cutoff = torch.tensor([0.5, 1.0], dtype=torch.float64)
    value = torch.tensor(
        [
            [[2.0, 4.0], [1.0, 5.0]],
            [[4.0, 2.0], [3.0, 1.0]],
        ],
        dtype=torch.float64,
    )

    output = streamed_positive_csr(score, value, cutoff, row_ptr)

    assert torch.equal(output[0], torch.zeros_like(output[0]))
    assert torch.equal(output[2], torch.zeros_like(output[2]))
    assert bool((output[1] >= 0).all())
    assert bool((output[1] <= value.max(dim=0).values).all())


def test_streamed_softmax_respects_cutoff_and_zero_degree_rows() -> None:
    row_ptr = torch.tensor([0, 0, 3, 3], dtype=torch.int32)
    score = torch.tensor(
        [[100.0, -5.0], [1.0, 2.0], [-4.0, 3.0]],
        dtype=torch.float64,
    )
    cutoff = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    value = torch.ones(3, 2, 1, dtype=torch.float64)

    output = streamed_softmax_csr(score, value, cutoff, row_ptr, chunk_size=1)

    assert torch.equal(output[0], torch.zeros_like(output[0]))
    assert torch.equal(output[1], torch.ones_like(output[1]))
    assert torch.equal(output[2], torch.zeros_like(output[2]))
    assert bool(torch.isfinite(output).all())


@pytest.mark.parametrize("streamed", [streamed_positive_csr, streamed_softmax_csr])
def test_streamed_csr_is_consistent_under_within_receiver_edge_order(
    streamed,
) -> None:
    score, value, cutoff, row_ptr = _csr_inputs()
    permutation = torch.tensor([2, 0, 1, 5, 3, 6, 4, 8, 7])

    reference = streamed(score, value, cutoff, row_ptr, chunk_size=2)
    candidate = streamed(
        score[permutation],
        value[permutation],
        cutoff[permutation],
        row_ptr,
        chunk_size=3,
    )

    torch.testing.assert_close(candidate, reference, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("csr_function", "ell_function"),
    [
        (streamed_positive_csr, streamed_positive_ell),
        (streamed_softmax_csr, streamed_softmax_ell),
    ],
)
def test_streamed_ell_matches_receiver_csr_with_masked_padding(
    csr_function,
    ell_function,
) -> None:
    score, value, cutoff, row_ptr = _csr_inputs()
    num_nodes = row_ptr.numel() - 1
    max_degree = 4
    ell_score = score.new_zeros((num_nodes, max_degree, score.shape[1]))
    ell_value = value.new_zeros(
        (num_nodes, max_degree, *value.shape[1:]),
    )
    ell_cutoff = cutoff.new_zeros((num_nodes, max_degree))
    ell_mask = torch.zeros(num_nodes, max_degree, dtype=torch.bool)
    boundaries = row_ptr.tolist()
    for receiver, (start, stop) in enumerate(
        zip(boundaries[:-1], boundaries[1:], strict=True),
    ):
        degree = stop - start
        ell_score[receiver, :degree] = score[start:stop]
        ell_value[receiver, :degree] = value[start:stop]
        ell_cutoff[receiver, :degree] = cutoff[start:stop]
        ell_mask[receiver, :degree] = True

    csr_output = csr_function(score, value, cutoff, row_ptr, chunk_size=2)
    ell_output = ell_function(
        ell_score,
        ell_value,
        ell_cutoff,
        ell_mask,
        chunk_size=2,
    )

    torch.testing.assert_close(ell_output, csr_output, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "streamed",
    [streamed_positive_csr, streamed_softmax_csr],
)
def test_streamed_reference_keeps_saved_intermediates_row_chunk_local(
    streamed,
) -> None:
    generator = torch.Generator().manual_seed(739)
    edge_count, heads, width = 19, 3, 5
    row_ptr = torch.tensor([0, 10, 19], dtype=torch.int32)
    score = (
        torch.rand(
            edge_count,
            heads,
            generator=generator,
            dtype=torch.float64,
        )
        + 0.1
    ).requires_grad_()
    cutoff = (
        torch.rand(edge_count, generator=generator, dtype=torch.float64) + 0.1
    ).requires_grad_()
    value = torch.randn(
        edge_count,
        heads,
        width,
        generator=generator,
        dtype=torch.float64,
        requires_grad=True,
    )
    saved_tensors: list[tuple[torch.Size, int]] = []

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        saved_tensors.append((tensor.shape, tensor.data_ptr()))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = streamed(
            score,
            value,
            cutoff,
            row_ptr,
            chunk_size=3,
        )
        output.sum().backward()

    input_pointers = {score.data_ptr(), value.data_ptr(), cutoff.data_ptr()}
    assert all(
        pointer in input_pointers
        for shape, pointer in saved_tensors
        if shape
        in {
            torch.Size((edge_count, heads)),
            torch.Size((edge_count, heads, width)),
        }
    )
    assert (
        max(
            (
                shape[0]
                for shape, pointer in saved_tensors
                if (
                    pointer not in input_pointers
                    and len(shape) >= 2
                    and shape[1] == heads
                )
            ),
            default=0,
        )
        <= 3
    )


def test_low_precision_inputs_accumulate_in_float32_by_default() -> None:
    row_ptr = torch.tensor([0, 2], dtype=torch.int32)
    score = torch.ones(2, 2, dtype=torch.bfloat16)
    cutoff = torch.ones(2, dtype=torch.bfloat16)
    value = torch.ones(2, 2, 3, dtype=torch.bfloat16)

    output = streamed_positive_csr(score, value, cutoff, row_ptr)

    assert output.dtype == torch.float32
    torch.testing.assert_close(output, torch.full_like(output, 2.0 / 3.0))


def test_streamed_pytorch_reference_supports_double_backward() -> None:
    score = torch.tensor(
        [[0.8], [1.1]],
        dtype=torch.float64,
        requires_grad=True,
    )
    cutoff = torch.tensor(
        [0.7, 0.4],
        dtype=torch.float64,
        requires_grad=True,
    )
    value = torch.tensor(
        [[[0.3]], [[-0.2]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    row_ptr = torch.tensor([0, 2], dtype=torch.int32)

    assert torch.autograd.gradgradcheck(
        lambda s, v, c: streamed_positive_csr(
            s,
            v,
            c,
            row_ptr,
            chunk_size=1,
        ),
        (score, value, cutoff),
        atol=1e-6,
        rtol=1e-5,
    )


def test_auto_backend_selection_is_degree_and_layout_deterministic() -> None:
    capabilities = default_local_backend_capabilities()
    capabilities["custom"] = replace(
        capabilities["custom"],
        available=True,
        supported_devices=("cpu", "cuda"),
    )

    cases = [
        (4, True, False, "materialized"),
        (16, True, True, "ell"),
        (16, True, False, "streamed_csr"),
        (64, True, False, "custom"),
        (64, False, True, "custom"),
        (64, False, False, "materialized"),
    ]
    for max_degree, has_csr, has_ell, expected in cases:
        first = select_local_backend(
            "auto",
            operation="positive",
            max_degree=max_degree,
            has_csr=has_csr,
            has_ell=has_ell,
            capabilities=capabilities,
        )
        second = select_local_backend(
            "auto",
            operation="positive",
            max_degree=max_degree,
            has_csr=has_csr,
            has_ell=has_ell,
            capabilities=capabilities,
        )
        assert first == second
        assert first.effective_backend == expected


def test_auto_backend_without_custom_uses_streamed_csr_for_large_degree() -> None:
    selection = select_local_backend(
        "auto",
        operation="softmax",
        max_degree=128,
        has_csr=True,
        has_ell=False,
    )

    assert selection.effective_backend == "streamed_csr"
    assert selection.fallback_reason is None
    assert selection.supports_gradgrad


def test_unwired_segment_backend_falls_back_to_streamed_csr() -> None:
    selection = select_local_backend(
        "segment_csr",
        operation="positive",
        max_degree=16,
        has_csr=True,
        has_ell=False,
    )

    assert selection.effective_backend == "streamed_csr"
    assert selection.used_fallback
    assert selection.fallback_reason is not None
    assert "unavailable" in selection.fallback_reason


def test_custom_without_gradgrad_falls_back_to_pytorch_reference() -> None:
    capabilities = default_local_backend_capabilities()
    capabilities["custom"] = LocalBackendCapability(
        backend="custom",
        available=True,
        operations=("positive", "softmax"),
        supports_gradgrad=False,
        layout="csr_or_ell",
        supported_dtypes=(torch.float32,),
        supported_devices=("cuda",),
    )

    selection = select_local_backend(
        "custom",
        operation="positive",
        max_degree=64,
        has_csr=True,
        has_ell=False,
        require_gradgrad=True,
        dtype=torch.float32,
        device_type="cuda",
        capabilities=capabilities,
    )

    assert selection.effective_backend == "streamed_csr"
    assert selection.used_fallback
    assert selection.supports_gradgrad
    assert selection.fallback_reason is not None
    assert "double backward" in selection.fallback_reason


def test_explicit_unavailable_backend_can_disable_safe_fallback() -> None:
    with pytest.raises(RuntimeError, match="ell.*unavailable"):
        select_local_backend(
            "ell",
            operation="positive",
            max_degree=16,
            has_csr=True,
            has_ell=False,
            allow_fallback=False,
        )


@pytest.mark.parametrize(
    ("argument", "message"),
    [
        ({"requested_backend": "unknown"}, "requested_backend"),
        ({"operation": "global"}, "operation"),
        ({"max_degree": -1}, "max_degree"),
        ({"small_degree_threshold": True}, "small_degree_threshold"),
        ({"streamed_degree_threshold": 4}, "streamed_degree_threshold"),
    ],
)
def test_backend_selector_rejects_invalid_configuration(
    argument: dict[str, object],
    message: str,
) -> None:
    kwargs: dict[str, object] = {
        "requested_backend": "auto",
        "operation": "positive",
        "max_degree": 16,
        "has_csr": True,
        "has_ell": False,
        "small_degree_threshold": 8,
        "streamed_degree_threshold": 32,
    }
    kwargs.update(argument)

    with pytest.raises((TypeError, ValueError), match=message):
        select_local_backend(**kwargs)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda s, v, c, r: (s[:, :0], v, c, r),
            "head",
        ),
        (
            lambda s, v, c, r: (s, v[:-1], c, r),
            "leading",
        ),
        (
            lambda s, v, c, r: (s, v, -c, r),
            "nonnegative",
        ),
        (
            lambda s, v, c, r: (-s, v, c, r),
            "nonnegative",
        ),
        (
            lambda s, v, c, r: (
                s,
                v,
                c,
                torch.tensor([0, 5, 4, 9], dtype=torch.int32),
            ),
            "nondecreasing",
        ),
    ],
)
def test_streamed_positive_rejects_invalid_inputs(mutator, message: str) -> None:
    score, value, cutoff, row_ptr = _csr_inputs()
    args = mutator(score, value, cutoff, row_ptr)

    with pytest.raises((TypeError, ValueError), match=message):
        streamed_positive_csr(*args)


def test_streamed_rejects_invalid_accumulation_and_chunk_controls() -> None:
    score, value, cutoff, row_ptr = _csr_inputs()

    with pytest.raises(TypeError, match="chunk_size"):
        streamed_positive_csr(
            score,
            value,
            cutoff,
            row_ptr,
            chunk_size=True,
        )
    with pytest.raises(ValueError, match="chunk_size"):
        streamed_positive_csr(
            score,
            value,
            cutoff,
            row_ptr,
            chunk_size=0,
        )
    with pytest.raises(ValueError, match="accumulation_dtype"):
        streamed_positive_csr(
            score,
            value,
            cutoff,
            row_ptr,
            accumulation_dtype=torch.float16,
        )
