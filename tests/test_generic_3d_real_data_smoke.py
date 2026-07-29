from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


_SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "smoke_generic_3d_real_data.py"
)
_SPEC = importlib.util.spec_from_file_location("generic_3d_real_smoke", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_typed_relation_ids_encode_receiver_and_sender_roles() -> None:
    edges = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
    roles = torch.tensor([0, 1])

    relation = _MODULE.typed_relation_ids(edges, roles)

    assert torch.equal(relation, torch.tensor([0, 1, 2, 3]))


def test_o3_smoke_transform_is_improper_orthogonal() -> None:
    transform = _MODULE._fixed_orthogonal(
        torch.float64,
        torch.device("cpu"),
    )

    torch.testing.assert_close(
        transform.mT @ transform,
        torch.eye(3, dtype=torch.float64),
    )
    assert torch.linalg.det(transform) == pytest.approx(-1.0)


def test_o3_smoke_verifier_is_dtype_aware_and_fails_closed() -> None:
    assert _MODULE._verify_o3_invariance(
        9e-5,
        dtype=torch.float32,
    ) == pytest.approx(1e-4)

    with pytest.raises(RuntimeError, match="O\\(3\\) invariance smoke failed"):
        _MODULE._verify_o3_invariance(
            2e-4,
            dtype=torch.float32,
        )
    with pytest.raises(RuntimeError, match="max_abs=nan"):
        _MODULE._verify_o3_invariance(
            float("nan"),
            dtype=torch.float32,
        )


def test_real_data_smoke_rejects_validation_before_loading_data() -> None:
    with pytest.raises(ValueError, match="train-only"):
        _MODULE.run_smoke(SimpleNamespace(split="val"))


def test_real_data_high_order_architecture_is_executable_and_generic() -> None:
    config = _MODULE.build_real_data_architecture(
        local_cutoff=6.0,
        num_layers=2,
        num_heads=4,
        width=32,
    )
    legacy = config.to_legacy()

    assert config.profile == "high_order"
    assert legacy.use_transient_l3_workspace
    assert legacy.num_node_roles == 2
    assert legacy.num_edge_relations == 4
    assert legacy.use_sparse_low_rank_local_residual
    assert legacy.global_reduction_backend == "auto"


def test_real_data_high_order_architecture_rejects_nondivisible_width() -> None:
    with pytest.raises(ValueError, match="divisible"):
        _MODULE.build_real_data_architecture(
            local_cutoff=6.0,
            num_layers=2,
            num_heads=4,
            width=30,
        )
