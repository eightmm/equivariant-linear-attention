from __future__ import annotations

from pathlib import Path
import runpy


PROFILE = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "profile_train_step.py")
)


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
