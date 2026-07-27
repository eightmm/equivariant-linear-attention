from __future__ import annotations

from pathlib import Path
import runpy

import torch


PROFILE = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "profile_train_step.py")
)


def test_ctp_profile_models_use_matched_static_lgl_harness() -> None:
    build = PROFILE["SCALING"]["_build_train_step_model"]
    control = build(
        "persistent_2e_static",
        device=torch.device("cpu"),
        model_seed=20260727,
    )
    ctp = build(
        "ctp_static",
        device=torch.device("cpu"),
        model_seed=20260727,
    )

    assert control.hidden_irreps.tensors == ctp.hidden_irreps.tensors == 4
    assert control.config.local_head_counts == ctp.config.local_head_counts == (4, 0, 4)
    assert control.config.use_static_tensor_carrier is True
    assert ctp.config.use_static_tensor_carrier is True
    assert control.config.use_cartesian_tensor_product_local_transport is False
    assert ctp.config.use_cartesian_tensor_product_local_transport is True


def test_tiny_cpu_profile_records_named_training_stages() -> None:
    result = PROFILE["run_train_step_profile"](
        model_name="spatial_static",
        num_nodes=8,
        edge_multiplier=2,
        device="cpu",
        model_seed=20260723,
        graph_seed=20260723,
        warmup=1,
        repeats=1,
    )

    assert result["status"] == "completed"
    assert result["device"] == "cpu"
    assert result["model"] == "spatial_static"
    assert result["edge_index_supplied"] is False
    assert result["gradient_validation"]["all_finite"] is True
    assert result["gradient_validation"]["nonzero_elements"] > 0
    assert set(result["stages"]) == {
        "stage.zero_grad",
        "stage.forward",
        "stage.loss",
        "stage.backward",
        "stage.optimizer_step",
    }
    assert result["operators"]
