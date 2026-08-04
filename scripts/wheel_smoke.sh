#!/usr/bin/env bash
# Build and import the installed wheel from an isolated temporary environment.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "wheel smoke requires uv" >&2
  exit 1
fi

wheel_tmp="$(mktemp -d)"
trap 'rm -rf -- "$wheel_tmp"' EXIT

project_python="$(uv run python -c 'import sys; print(sys.executable)')"
project_site="$($project_python - <<'PY'
import site

paths = site.getsitepackages()
if len(paths) != 1:
    raise SystemExit(f"expected one project site-packages directory, got {paths!r}")
print(paths[0])
PY
)"

uv build --wheel --out-dir "$wheel_tmp/wheels" --no-create-gitignore >/dev/null
uv venv --python "$project_python" "$wheel_tmp/venv" >/dev/null

wheel_files=("$wheel_tmp"/wheels/*.whl)
if [ "${#wheel_files[@]}" -ne 1 ] || [ ! -f "${wheel_files[0]}" ]; then
  echo "wheel smoke expected exactly one wheel" >&2
  exit 1
fi

# Dependencies are already locked and imported by the project environment.
# Expose those dependency modules through PYTHONPATH without processing the
# project's editable .pth file; the package itself must come from this wheel.
uv pip install \
  --python "$wheel_tmp/venv/bin/python" \
  --no-deps \
  "${wheel_files[0]}" >/dev/null

(
  cd "$wheel_tmp"
  PYTHONPATH="$project_site" "$wheel_tmp/venv/bin/python" - <<PY
import inspect
from pathlib import Path

import torch

import equivariant_linear_attention as ela
from equivariant_linear_attention import ELA, ELAGraph

wheel_root = Path(${wheel_tmp@Q}) / "venv"
module_path = Path(ela.__file__).resolve()
assert module_path.is_relative_to(wheel_root.resolve()), (module_path, wheel_root)
assert set(ela.__all__) == {"ELA", "ELAGraph"}
assert ELA is ela.ELA
assert ELAGraph is ela.ELAGraph
assert tuple(inspect.signature(ELA.forward).parameters) == ("self", "graph")
assert not {
    "_prepared_graph",
    "_prepared_provenance",
    "_packed_template",
    "_assume_immutable_storage",
} & set(inspect.signature(ELAGraph).parameters)

model = ELA("2x0e", width=16, depth=1, cutoff=2.0)
graph = ELAGraph(
    torch.randn(3, 2),
    torch.randn(3, 3),
    edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
)
output = model(graph)
assert isinstance(output, ELAGraph)
assert output.x.shape == (3, 1)
for name in (
    "prepare_context",
    "embed_input",
    "project_state",
    "encode_context",
    "forward_features",
):
    assert not hasattr(model, name), name
PY
)

echo "wheel smoke: ok"
