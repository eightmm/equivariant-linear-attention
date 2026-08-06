from __future__ import annotations

import pytest

from equivariant_linear_attention.api import ELA as SparseCompatibilityELA


_SPARSE_PREPARATION_MODULES = {
    "test_fast_preparation",
    "test_preparation_provenance",
}


@pytest.fixture(autouse=True)
def _route_sparse_preparation_contracts(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the retained sparse engine's private preparation tests explicit.

    The package-root ``ELA`` now has an edge-free default and is covered by
    ``test_edge_free_krylov.py``. These two modules specifically test radius
    discovery, receiver-CSR packing, skin/provenance, and neighbor caps; route
    only their module-global ``ELA`` symbol to the retained compatibility engine
    instead of weakening or deleting those independent subsystem checks.
    """

    module_name = request.module.__name__.rsplit(".", maxsplit=1)[-1]
    if module_name in _SPARSE_PREPARATION_MODULES:
        monkeypatch.setattr(
            request.module,
            "ELA",
            SparseCompatibilityELA,
            raising=True,
        )
