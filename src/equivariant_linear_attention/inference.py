from __future__ import annotations

import warnings

import torch
from torch import nn

try:  # PyTorch keeps these exception types in internal compiler namespaces.
    from torch._dynamo.exc import BackendCompilerFailed
    from torch._inductor.exc import InductorError
except ImportError:  # pragma: no cover - future PyTorch compatibility guard
    _COMPILER_FAILURES: tuple[type[BaseException], ...] = ()
else:
    _COMPILER_FAILURES = (BackendCompilerFailed, InductorError)


_MODEL_METADATA = (
    "attention_kind",
    "symmetry",
    "config",
    "input_irreps",
    "hidden_irreps",
    "hidden_irrep_layout",
    "workspace_irreps",
    "output_irreps",
    "output_irrep_layout",
    "tensor_product_plan",
)


def prepare_for_inference(
    model: nn.Module,
    device: str | torch.device | None = None,
    dtype: torch.dtype | str | None = "auto",
    compile_model: bool = True,
    compile_mode: str = "reduce-overhead",
    allow_tf32: bool | None = None,
) -> nn.Module:
    """Prepare a model for inference with FP32 parameters by default.

    ``dtype="auto"`` preserves FP32 parameters and uses CUDA autocast when a
    lower-precision CUDA lane is supported. Explicit ``bf16``/``fp16`` keeps
    the previous whole-model conversion available for controlled experiments.
    TF32 policy is preserved by default. Passing ``allow_tf32`` explicitly
    changes PyTorch's process-global CUDA backend policy, and only when the
    requested target device is CUDA.
    """

    target_device = (
        torch.device(device) if device is not None else next(model.parameters()).device
    )
    if allow_tf32 is not None and target_device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

    automatic = dtype == "auto"
    target_dtype = torch.float32 if automatic else _resolve_dtype(dtype, target_device)
    model = model.to(device=target_device, dtype=target_dtype).eval()
    for param in model.parameters():
        param.requires_grad_(False)

    if compile_model:
        if _supports_prepared_core(model):
            if bool(getattr(model, "updates_positions", False)):
                warnings.warn(
                    "stagewise coordinate updates keep topology rebuilding eager; "
                    "compile_model was skipped for this model",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                # ELA deliberately keeps Python graph validation, radius
                # discovery, cache lookup, pooling, and public output
                # construction outside Dynamo. Only stable tensor math enters
                # the compiled callable.
                execute = torch.compile(
                    model._execute_numerical,  # type: ignore[attr-defined]
                    mode=compile_mode,
                )
                model = _CompiledCoreInferenceModule(model, execute)
        else:
            model = torch.compile(model, mode=compile_mode)
    if automatic and target_device.type == "cuda":
        model = _AutocastInferenceModule(model, autocast_dtype(target_device))
    return model


def _supports_prepared_core(model: nn.Module) -> bool:
    return all(
        callable(getattr(model, name, None))
        for name in (
            "_pack_and_prepare",
            "_execute_numerical",
            "_finalize_packed",
            "_wrap_output",
        )
    )


def _copy_model_metadata(target: nn.Module, source: nn.Module) -> None:
    for name in _MODEL_METADATA:
        if hasattr(source, name):
            setattr(target, name, getattr(source, name))


class _CompiledCoreInferenceModule(nn.Module):
    """Compile ELA's numerical core while leaving public graph work eager."""

    def __init__(self, model: nn.Module, execute: object) -> None:
        super().__init__()
        if not _supports_prepared_core(model):
            raise TypeError("compiled-core inference requires ELA's prepared interface")
        if not callable(execute):
            raise TypeError("compiled packed executor must be callable")
        self.model = model
        # ``torch.compile`` returns a callable that closes over ``model``.  It
        # must not be registered as another child module because doing so would
        # duplicate the model's state-dict path.
        object.__setattr__(self, "_execute", execute)
        _copy_model_metadata(self, model)
        self.eval()

    def forward(self, graph: object) -> object:
        packed = self.model._pack_and_prepare(graph)  # type: ignore[attr-defined]
        try:
            raw = self._execute(packed)
        except _COMPILER_FAILURES as error:
            warnings.warn(
                "compiled ELA core failed on this backend; falling back to "
                f"the exact eager core ({type(error).__name__})",
                RuntimeWarning,
                stacklevel=2,
            )
            execute = self.model._execute_numerical  # type: ignore[attr-defined]
            object.__setattr__(self, "_execute", execute)
            raw = execute(packed)
        output = self.model._finalize_packed(packed, raw)  # type: ignore[attr-defined]
        return self.model._wrap_output(graph, packed, output)  # type: ignore[attr-defined]


class _AutocastInferenceModule(nn.Module):
    def __init__(self, model: nn.Module, dtype: torch.dtype) -> None:
        super().__init__()
        self.model = model
        self.dtype = dtype
        self.device_type = next(model.parameters()).device.type
        _copy_model_metadata(self, model)
        self.eval()

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> object:
        with torch.autocast(device_type=self.device_type, dtype=self.dtype):
            return self.model(*args, **kwargs)


def autocast_dtype(device: str | torch.device = "cuda") -> torch.dtype:
    target_device = torch.device(device)
    if target_device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if target_device.type == "cuda":
        return torch.float16
    return torch.float32


def _resolve_dtype(
    dtype: torch.dtype | str | None, device: torch.device
) -> torch.dtype:
    if dtype is None:
        return torch.float32
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype == "auto":
        return autocast_dtype(device)
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype in {"fp32", "float32"}:
        return torch.float32
    msg = f"unknown dtype: {dtype}"
    raise ValueError(msg)
