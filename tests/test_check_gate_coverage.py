"""The GPU gate must actually select every CUDA-gated test file.

``scripts/check.sh gpu`` names its pytest targets explicitly, which keeps the
smoke short but lets a new CUDA-only test go unrun forever: under ``fast`` it
self-skips because ``CUDA_VISIBLE_DEVICES`` is empty, and under ``gpu`` it is
simply never selected. This pins the two lists together.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"
CHECK_SCRIPT = REPO_ROOT / "scripts" / "check.sh"

_CUDA_TOKENS = ("cuda", "bf16", "triton")


def _requires_accelerator(decorator: ast.expr) -> bool:
    """True when a decorator skips the test unless an accelerator is present."""

    if not isinstance(decorator, ast.Call):
        return False
    target = decorator.func
    name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
    if name != "skipif":
        return False
    rendered = ast.unparse(decorator).lower()
    return any(token in rendered for token in _CUDA_TOKENS)


def _module_mark_requires_accelerator(tree: ast.Module) -> bool:
    """Detect ``pytestmark = pytest.mark.skipif(...)`` at module scope."""

    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue
        if value is None or not any(
            isinstance(target, ast.Name) and target.id == "pytestmark"
            for target in targets
        ):
            continue
        rendered = ast.unparse(value).lower()
        if "skipif" in rendered and any(
            token in rendered for token in _CUDA_TOKENS
        ):
            return True
    return False


def _source_requires_accelerator(source: str) -> bool:
    tree = ast.parse(source)
    if _module_mark_requires_accelerator(tree):
        return True
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if any(_requires_accelerator(item) for item in node.decorator_list):
            return True
    return False


def _accelerator_gated_files() -> set[str]:
    gated: set[str] = set()
    for path in sorted(TESTS_DIR.glob("test_*.py")):
        if path.stem.endswith("_cuda") or _source_requires_accelerator(
            path.read_text()
        ):
            gated.add(f"tests/{path.name}")
    return gated


def _gate_selected_files() -> set[str]:
    return set(re.findall(r"tests/test_\w+\.py", CHECK_SCRIPT.read_text()))


def test_gpu_gate_selects_every_cuda_gated_test_file() -> None:
    gated = _accelerator_gated_files()
    assert gated, "expected the suite to contain CUDA-gated tests"

    missing = sorted(gated - _gate_selected_files())
    assert not missing, (
        "CUDA-gated test files never run in either check.sh mode; "
        f"add them to run_gpu in scripts/check.sh: {missing}"
    )


def test_gpu_gate_does_not_name_missing_test_files() -> None:
    stale = sorted(
        selected
        for selected in _gate_selected_files()
        if not (REPO_ROOT / selected).exists()
    )
    assert not stale, f"scripts/check.sh selects test files that no longer exist: {stale}"


def test_module_level_pytestmark_is_detected() -> None:
    source = """
import pytest
import torch
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)
"""
    assert _source_requires_accelerator(source)


def test_gpu_gate_activates_the_triton_extra() -> None:
    script = CHECK_SCRIPT.read_text()
    run_gpu = script.split("run_gpu() {", 1)[1].split("\n}\n", 1)[0]
    uv_invocations = [
        line.strip() for line in run_gpu.splitlines() if "uv run" in line
    ]
    assert uv_invocations
    assert all("uv run --extra triton" in line for line in uv_invocations)
