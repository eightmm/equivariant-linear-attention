from __future__ import annotations

import pytest
import torch

from equivariant_linear_attention.kernels.triton import (
    active_backend,
    backend_policy,
    csr_sum,
    csr_sum_many,
    kernel_backend,
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


def test_csr_sum_supports_inference_tensors_without_version_counters() -> None:
    with torch.inference_mode():
        value, row_ptr = _fixture()
        expected = torch.segment_reduce(value, reduce="sum", offsets=row_ptr)
        actual = csr_sum(value, row_ptr)
    torch.testing.assert_close(actual, expected)


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


def test_backend_context_is_nested_and_reversible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ELA_KERNEL_BACKEND", "auto")
    assert backend_policy() == "auto"
    with kernel_backend("torch"):
        assert backend_policy() == "torch"
        with kernel_backend("triton"):
            assert backend_policy() == "triton"
        assert backend_policy() == "torch"
    assert backend_policy() == "auto"
    with pytest.raises(TypeError, match="string"):
        with kernel_backend(1):  # type: ignore[arg-type]
            pass
    with pytest.raises(ValueError, match="backend policy"):
        with kernel_backend("invalid"):
            pass


def test_noncontiguous_row_ptr_is_copied_before_reduction() -> None:
    value = torch.arange(5, dtype=torch.float64).reshape(5, 1)
    storage = torch.tensor([0, 99, 2, 99, 5], dtype=torch.int64)
    row_ptr = storage[::2]
    assert not row_ptr.is_contiguous()
    expected = torch.segment_reduce(value, reduce="sum", offsets=row_ptr)
    torch.testing.assert_close(csr_sum(value, row_ptr), expected)


def test_low_precision_reference_accumulates_in_fp32() -> None:
    value = torch.tensor(
        [4096.0, 1.0, 1.0, 1.0, 1.0, -4096.0],
        dtype=torch.bfloat16,
    ).reshape(-1, 1)
    row_ptr = torch.tensor([0, value.shape[0]])
    expected = value.float().sum(dim=0, keepdim=True).to(value.dtype)
    torch.testing.assert_close(csr_sum(value, row_ptr), expected)


def test_csr_metadata_and_shapes_fail_closed() -> None:
    value, row_ptr = _fixture()
    with pytest.raises(ValueError, match="edge dimension"):
        csr_sum(torch.tensor(1.0), row_ptr)
    with pytest.raises(ValueError, match="row_ptr"):
        csr_sum(value, torch.tensor(0))
    with pytest.raises(ValueError, match="row_ptr"):
        csr_sum(value, torch.empty(0, dtype=torch.long))
    with pytest.raises(TypeError, match="int32 or int64"):
        csr_sum(value, row_ptr.float())
    with pytest.raises(ValueError, match="start at zero"):
        csr_sum(value, torch.tensor([1, 2, 2, 5, 6]))
    with pytest.raises(ValueError, match="edge count"):
        csr_sum(value, torch.tensor([0, 2, 2, 5, 5]))
    with pytest.raises(ValueError, match="nondecreasing"):
        csr_sum(value, torch.tensor([0, 3, 2, 5, 6]))

    empty_value = value[:0].detach().clone().requires_grad_(True)
    empty = csr_sum(empty_value, torch.tensor([0, 0, 0]))
    assert empty.shape == (2, 3)
    assert empty.requires_grad
    empty.sum().backward()
    assert empty_value.grad is not None
    assert empty_value.grad.shape == empty_value.shape


def test_package_import_does_not_patch_canonical_module_globals() -> None:
    from equivariant_linear_attention.nn import core, parity

    block = core._CanonicalMultipoleBlock
    assert not hasattr(block, "_ela_torch_local_message")
    assert block._local_message.__module__ == core.__name__
    assert block._torch_local_message.__module__ == core.__name__
    assert parity._csr_sum.__module__ == parity.__name__


def test_csr_sum_many_validates_payload_group() -> None:
    value, row_ptr = _fixture()
    assert csr_sum_many((), row_ptr) == ()
    with pytest.raises(ValueError, match="edge dimension"):
        csr_sum_many((value, value[:-1]), row_ptr)
    with pytest.raises(ValueError, match="device and dtype"):
        csr_sum_many((value, value.float()), row_ptr)
    with pytest.raises(ValueError, match="edge dimension"):
        csr_sum_many((torch.tensor(1.0),), row_ptr)
    with pytest.raises(ValueError, match="nonzero"):
        csr_sum_many((value.reshape(6, 3, 1)[:, :, :0],), row_ptr)
