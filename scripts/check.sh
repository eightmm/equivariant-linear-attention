#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -m compileall -q src scripts examples
ruff check .
pytest -q
python scripts/ml_smoke.py
