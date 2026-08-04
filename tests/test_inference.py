import pytest
import torch
from torch import nn

import equivariant_linear_attention.inference as inference_module
from equivariant_linear_attention import ELA, ELAGraph
from equivariant_linear_attention.inference import (
    _AutocastInferenceModule,
    _CompiledCoreInferenceModule,
    _resolve_dtype,
    autocast_dtype,
    prepare_for_inference,
)


def _fixture() -> tuple[ELA, ELAGraph]:
    model = ELA(
        input_irreps="3x0e",
        output_irreps="1x0e",
        width=16,
        depth=1,
        cutoff=10.0,
    )
    nodes = 3
    sender = torch.arange(nodes).repeat(nodes)
    receiver = torch.arange(nodes).repeat_interleave(nodes)
    graph = ELAGraph(
        x=torch.randn(nodes, 3),
        pos=torch.randn(nodes, 3),
        edge_index=torch.stack([sender, receiver]),
    )
    return model, graph


def test_auto_inference_keeps_parameters_in_float32_on_cpu() -> None:
    model, batch = _fixture()

    prepared = prepare_for_inference(model, device="cpu", dtype="auto", compile_model=False)

    assert {parameter.dtype for parameter in prepared.parameters()} == {torch.float32}
    assert not any(parameter.requires_grad for parameter in prepared.parameters())
    with torch.inference_mode():
        output = prepared(batch)
    assert output.graph_x is not None
    assert torch.isfinite(output.graph_x).all()


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


def test_cuda_autocast_dtype_prefers_bf16_and_falls_back_to_fp16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert autocast_dtype("cuda") == torch.bfloat16

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert autocast_dtype("cuda") == torch.float16


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

    def fake_compile(target: object, *, mode: str) -> object:
        observed.update(target=target, mode=mode)
        return target

    monkeypatch.setattr(torch, "compile", fake_compile)
    prepared = prepare_for_inference(
        model,
        device=None,
        dtype=None,
        compile_model=True,
        compile_mode="default",
    )
    assert isinstance(prepared, _CompiledCoreInferenceModule)
    assert observed["mode"] == "default"
    target = observed["target"]
    assert callable(target)
    assert getattr(target, "__self__", None) is prepared.model


def test_compiled_core_keeps_public_graph_work_outside_compiled_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, graph = _fixture()
    events: list[str] = []

    original_pack = model._pack_and_prepare
    original_execute = model._execute_numerical
    original_finalize = model._finalize_packed
    original_wrap = model._wrap_output

    def observed_pack(value: ELAGraph):
        events.append("pack")
        return original_pack(value)

    def observed_execute(value: object):
        events.append("execute")
        return original_execute(value)  # type: ignore[arg-type]

    def observed_finalize(value: object, raw: object):
        events.append("finalize")
        return original_finalize(value, raw)  # type: ignore[arg-type]

    def observed_wrap(value: ELAGraph, packed: object, output: object):
        events.append("wrap")
        return original_wrap(value, packed, output)  # type: ignore[arg-type]

    monkeypatch.setattr(model, "_pack_and_prepare", observed_pack)
    monkeypatch.setattr(model, "_execute_numerical", observed_execute)
    monkeypatch.setattr(model, "_finalize_packed", observed_finalize)
    monkeypatch.setattr(model, "_wrap_output", observed_wrap)

    compiled_targets: list[object] = []

    def fake_compile(target: object, *, mode: str) -> object:
        assert mode == "reduce-overhead"
        compiled_targets.append(target)
        return target

    monkeypatch.setattr(torch, "compile", fake_compile)
    prepared = prepare_for_inference(model, device="cpu", compile_model=True)
    output = prepared(graph)

    assert isinstance(output, ELAGraph)
    assert events == ["pack", "execute", "finalize", "wrap"]
    assert compiled_targets == [observed_execute]


def test_compiled_core_validates_its_private_boundary() -> None:
    model, _ = _fixture()

    with pytest.raises(TypeError, match="requires ELA's prepared interface"):
        _CompiledCoreInferenceModule(nn.Linear(3, 2), lambda value: value)
    with pytest.raises(TypeError, match="executor must be callable"):
        _CompiledCoreInferenceModule(model, object())


def test_compiled_core_failure_warns_once_and_permanently_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SyntheticCompilerFailure(RuntimeError):
        pass

    model, graph = _fixture()
    attempts = 0

    def broken_compiled_execute(_packed: object) -> object:
        nonlocal attempts
        attempts += 1
        raise SyntheticCompilerFailure("backend failed")

    monkeypatch.setattr(
        inference_module,
        "_COMPILER_FAILURES",
        (SyntheticCompilerFailure,),
    )
    wrapped = _CompiledCoreInferenceModule(model, broken_compiled_execute)

    with pytest.warns(RuntimeWarning, match="falling back to the exact eager core"):
        first = wrapped(graph)
    second = wrapped(graph)

    assert isinstance(first, ELAGraph)
    assert isinstance(second, ELAGraph)
    assert attempts == 1
    assert getattr(wrapped._execute, "__self__", None) is model


def test_generic_module_compile_keeps_the_supported_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = nn.Linear(3, 2)
    observed: dict[str, object] = {}

    def fake_compile(target: object, *, mode: str) -> object:
        observed.update(target=target, mode=mode)
        return target

    monkeypatch.setattr(torch, "compile", fake_compile)
    prepared = prepare_for_inference(
        model,
        device="cpu",
        dtype="auto",
        compile_model=True,
        compile_mode="default",
    )

    assert prepared is model
    assert observed == {"target": model, "mode": "default"}
    assert not any(parameter.requires_grad for parameter in prepared.parameters())


def test_stagewise_coordinate_inference_skips_whole_model_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = ELA(
        "3x0e",
        "1x0e",
        width=16,
        depth=2,
        update_positions=True,
    )

    def forbidden_compile(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("moving topology must not enter whole-model compile")

    monkeypatch.setattr(torch, "compile", forbidden_compile)
    with pytest.warns(RuntimeWarning, match="topology rebuilding eager"):
        prepared = prepare_for_inference(model, device="cpu", compile_model=True)
    assert prepared is model


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
