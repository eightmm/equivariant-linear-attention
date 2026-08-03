import pytest
import torch
from torch import nn

from equivariant_attention import ELA, ELABatch
from equivariant_attention.inference import (
    _AutocastInferenceModule,
    _resolve_dtype,
    autocast_dtype,
    prepare_for_inference,
)


def _fixture() -> tuple[ELA, ELABatch]:
    model = ELA(
        input_irreps="3x0e",
        output_irreps="1x0e",
        width=16,
        depth=1,
        cutoff=10.0,
    )
    nodes = 3
    receiver = torch.arange(nodes).repeat_interleave(nodes)
    sender = torch.arange(nodes).repeat(nodes)
    batch = ELABatch(
        node_irreps=torch.randn(nodes, 3),
        positions=torch.randn(nodes, 3),
        edge_index=torch.stack([receiver, sender]),
    )
    return model, batch


def test_auto_inference_keeps_parameters_in_float32_on_cpu() -> None:
    model, batch = _fixture()

    prepared = prepare_for_inference(model, device="cpu", dtype="auto", compile_model=False)

    assert {parameter.dtype for parameter in prepared.parameters()} == {torch.float32}
    assert not any(parameter.requires_grad for parameter in prepared.parameters())
    with torch.inference_mode():
        output = prepared(batch)
    assert torch.isfinite(output["graph_irreps"]).all()


def test_explicit_low_precision_inference_remains_available() -> None:
    model, _ = _fixture()

    prepared = prepare_for_inference(model, device="cpu", dtype="bf16", compile_model=False)

    assert {parameter.dtype for parameter in prepared.parameters()} == {torch.bfloat16}


def test_inference_dtype_resolution_is_explicit_on_cpu() -> None:
    device = torch.device("cpu")
    assert autocast_dtype(device) == torch.float32
    assert _resolve_dtype(None, device) == torch.float32
    assert _resolve_dtype(torch.float64, device) == torch.float64
    assert _resolve_dtype("auto", device) == torch.float32
    assert _resolve_dtype("bf16", device) == torch.bfloat16
    assert _resolve_dtype("fp16", device) == torch.float16
    assert _resolve_dtype("fp32", device) == torch.float32
    assert _resolve_dtype("float32", device) == torch.float32
    with pytest.raises(ValueError, match="unknown dtype"):
        _resolve_dtype("unknown", device)


def test_autocast_wrapper_forwards_and_copies_model_metadata() -> None:
    linear = nn.Linear(3, 2)
    linear.attention_kind = "probe"  # type: ignore[attr-defined]
    wrapped = _AutocastInferenceModule(linear, torch.bfloat16)
    output = wrapped(torch.randn(4, 3))
    assert output.shape == (4, 2)
    assert output.dtype == torch.bfloat16
    assert wrapped.attention_kind == "probe"  # type: ignore[attr-defined]


def test_prepare_for_inference_uses_requested_compile_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _ = _fixture()
    observed: dict[str, object] = {}

    def fake_compile(module: nn.Module, *, mode: str) -> nn.Module:
        observed.update(module=module, mode=mode)
        return module

    monkeypatch.setattr(torch, "compile", fake_compile)
    prepared = prepare_for_inference(
        model,
        device=None,
        dtype=None,
        compile_model=True,
        compile_mode="default",
    )
    assert observed == {"module": prepared, "mode": "default"}


def test_cpu_preparation_never_mutates_cuda_tf32_policy() -> None:
    model, _ = _fixture()
    original_matmul = torch.backends.cuda.matmul.allow_tf32
    original_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        prepare_for_inference(
            model,
            device="cpu",
            compile_model=False,
            allow_tf32=True,
        )
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_matmul
        torch.backends.cudnn.allow_tf32 = original_cudnn


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_tf32_policy_is_preserved_by_default_and_explicit_when_requested() -> None:
    original_matmul = torch.backends.cuda.matmul.allow_tf32
    original_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        default_model, _ = _fixture()
        prepare_for_inference(
            default_model,
            device="cuda",
            compile_model=False,
        )
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32

        explicit_model, _ = _fixture()
        prepare_for_inference(
            explicit_model,
            device="cuda",
            compile_model=False,
            allow_tf32=True,
        )
        assert torch.backends.cuda.matmul.allow_tf32
        assert torch.backends.cudnn.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_matmul
        torch.backends.cudnn.allow_tf32 = original_cudnn
