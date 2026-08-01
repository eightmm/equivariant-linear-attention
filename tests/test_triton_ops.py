from __future__ import annotations

import pytest
import torch

from equivariant_attention.triton_ops import (
    active_backend,
    csr_sum,
    csr_sum_many,
)


def _fixture(dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    row_ptr = torch.tensor([0, 2, 2, 5, 6], dtype=torch.int64)
    value = torch.arange(18, dtype=dtype).reshape(6, 3) / 7.0
    return value, row_ptr


def test_cpu_auto_backend_is_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ELA_KERNEL_BACKEND", "auto")
    value, row_ptr = _fixture()
    assert active_backend(value, row_ptr) == "torch"


def test_csr_sum_matches_segment_reduce() -> None:
    value, row_ptr = _fixture()
    expected = torch.segment_reduce(value, reduce="sum", offsets=row_ptr)
    actual = csr_sum(value, row_ptr)
    torch.testing.assert_close(actual, expected)


def test_csr_sum_supports_first_and_second_derivatives() -> None:
    value, row_ptr = _fixture()
    value.requires_grad_(True)
    output = csr_sum(value, row_ptr)
    first = torch.autograd.grad(output.square().sum(), value, create_graph=True)[0]
    second = torch.autograd.grad(first.square().sum(), value)[0]
    assert torch.isfinite(first).all()
    assert torch.isfinite(second).all()


def test_csr_sum_many_uses_one_packed_contract() -> None:
    value, row_ptr = _fixture()
    vector = value.reshape(6, 1, 3)
    scalar = value[:, :2]
    actual_vector, actual_scalar = csr_sum_many((vector, scalar), row_ptr)
    expected_vector = torch.segment_reduce(vector, reduce="sum", offsets=row_ptr)
    expected_scalar = torch.segment_reduce(scalar, reduce="sum", offsets=row_ptr)
    torch.testing.assert_close(actual_vector, expected_vector)
    torch.testing.assert_close(actual_scalar, expected_scalar)


def test_forced_triton_fails_closed_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ELA_KERNEL_BACKEND", "triton")
    value, row_ptr = _fixture(torch.float32)
    with pytest.raises(RuntimeError, match="forced"):
        csr_sum(value, row_ptr)


def test_invalid_backend_policy_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ELA_KERNEL_BACKEND", "unknown")
    value, row_ptr = _fixture(torch.float32)
    with pytest.raises(ValueError, match="ELA_KERNEL_BACKEND"):
        csr_sum(value, row_ptr)
