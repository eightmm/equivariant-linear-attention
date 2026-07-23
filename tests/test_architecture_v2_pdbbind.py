from __future__ import annotations

import json
import math
from pathlib import Path
import runpy

import torch

from equivariant_attention.benchmarking import GraphSample
from equivariant_attention.training import TargetNormalizer, fit_target_normalizer


def _symbols() -> dict[str, object]:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_architecture_v2_pdbbind.py"
    )
    return runpy.run_path(script)


def test_plan_freezes_three_train_only_strict_arms() -> None:
    symbols = _symbols()
    args = symbols["parse_args"](["artifacts/run/result.json", "--dry-run"])
    plan = symbols["_run_plan"](args)

    assert symbols["SUBSET_INDICES"] == tuple(range(16))
    assert symbols["registered_arms"]() == ("incumbent", "candidate", "egnn")
    assert args.max_steps == 3_000
    assert args.batch_size == 2
    assert args.threshold == 0.10
    assert args.budget_seconds == 1_800
    assert plan["arm_budget_seconds"] == 600
    assert plan["split"] == "train"
    assert plan["determinism"] == "strict"
    assert plan["validation_evaluated"] is False
    assert plan["test_evaluated"] is False
    assert not hasattr(args, "evaluate_test")
    assert not hasattr(args, "split")


def test_attention_builders_change_only_registered_v2_controls() -> None:
    symbols = _symbols()
    incumbent = symbols["_build_incumbent_attention"]()
    candidate = symbols["_build_candidate_attention"]()

    common = {
        "hidden_irreps": "64x0e + 4x1o + 4x2e",
        "use_key_balancing": False,
        "use_multiscale_spatial_kernel": True,
    }
    for name, expected in common.items():
        assert getattr(incumbent.config, name) == expected
        assert getattr(candidate.config, name) == expected
    assert incumbent.config.scalar_content_mode == "unit"
    assert incumbent.config.use_tensor_product_kernel is False
    assert candidate.config.scalar_content_mode == "bounded"
    assert candidate.config.use_tensor_product_kernel is True
    assert sum(parameter.numel() for parameter in candidate.parameters()) > sum(
        parameter.numel() for parameter in incumbent.parameters()
    )


def test_main_configures_strict_mode_before_train_only_load_and_writes_all_arms(
    tmp_path: Path,
) -> None:
    symbols = _symbols()
    events: list[object] = []
    samples = [
        GraphSample(
            node_feats=torch.zeros(2, 140),
            pos=torch.zeros(2, 3),
            target=torch.tensor([float(index)]),
            sample_id=f"sample-{index}",
            readout_mask=torch.tensor([False, True]),
        )
        for index in range(16)
    ]

    def configure(*, seed: int, mode: str) -> dict[str, object]:
        events.append(("configure", seed, mode))
        return {"mode": mode, "seed": seed}

    def load(
        root: Path, *, indices: tuple[int, ...], revision: str
    ) -> list[GraphSample]:
        events.append(("load", root, indices, revision))
        return samples

    def validate(value: list[GraphSample]) -> None:
        events.append(("validate", len(value)))

    def build_incumbent() -> torch.nn.Module:
        return torch.nn.Linear(1, 1)

    def build_candidate() -> torch.nn.Module:
        return torch.nn.Sequential(torch.nn.Linear(1, 2), torch.nn.Linear(2, 1))

    def matched_width(
        *,
        target_parameter_count: int,
        node_dim: int,
        num_layers: int,
    ) -> int:
        events.append(("match", target_parameter_count, node_dim, num_layers))
        return 17

    def run_arm(**kwargs: object) -> dict[str, object]:
        arm = str(kwargs["arm"])
        events.append(("run", arm, kwargs["samples"]))
        return {
            "arm": arm,
            "status": "completed",
            "overfit_passed": arm != "egnn",
            "parameter_count": 10,
            "initial_state_sha256": f"initial-{arm}",
            "final_state_sha256": f"final-{arm}",
            "time_to_threshold_seconds": 0.1,
            "validation_evaluated": False,
            "test_evaluated": False,
        }

    script_globals = symbols["main"].__globals__
    script_globals["configure_reproducibility"] = configure
    script_globals["load_atom3d_lba_samples"] = load
    script_globals["_validate_frozen_samples"] = validate
    script_globals["_build_incumbent_attention"] = build_incumbent
    script_globals["_build_candidate_attention"] = build_candidate
    script_globals["matched_egnn_width"] = matched_width
    script_globals["_build_egnn"] = lambda width: torch.nn.Linear(1, width)
    script_globals["_with_egnn_radius_edges"] = lambda value: [
        "egnn-edges",
        *value,
    ]
    script_globals["_run_arm"] = run_arm
    script_globals["fit_target_normalizer"] = lambda value: TargetNormalizer(
        torch.tensor([0.0]),
        torch.tensor([1.0]),
    )
    output = tmp_path / "result.json"

    assert (
        symbols["main"](
            [
                str(output),
                "--device",
                "cpu",
                "--max-steps",
                "1",
                "--budget-seconds",
                "3",
                "--eval-interval",
                "1",
            ]
        )
        == 0
    )

    assert events[0] == ("configure", symbols["MODEL_SEED"], "strict")
    assert events[1] == (
        "load",
        Path("data/atom3d_lba"),
        tuple(range(16)),
        symbols["DATASET_REVISION"],
    )
    run_events = [event for event in events if event[0] == "run"]
    assert [event[1] for event in run_events] == ["incumbent", "candidate", "egnn"]
    assert run_events[0][2] is samples
    assert run_events[1][2] is samples
    assert run_events[2][2][0] == "egnn-edges"

    result = json.loads(output.read_text())
    assert result["status"] == "completed"
    assert result["determinism"] == "strict"
    assert result["reproducibility"] == {
        "mode": "strict",
        "seed": symbols["MODEL_SEED"],
    }
    assert result["matched_egnn_width"] == 17
    assert result["validation_evaluated"] is False
    assert result["test_evaluated"] is False
    assert [arm["arm"] for arm in result["arms"]] == [
        "incumbent",
        "candidate",
        "egnn",
    ]


def test_candidate_cpu_one_step_smoke_is_finite() -> None:
    symbols = _symbols()
    samples = [
        GraphSample(
            node_feats=torch.eye(140, dtype=torch.float32)[torch.tensor([1, 6, 7, 8])],
            pos=torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            + float(index),
            target=torch.tensor([1.0 + index]),
            sample_id=f"cpu-smoke-{index}",
            readout_mask=torch.tensor([False, False, True, True]),
        )
        for index in range(2)
    ]
    normalizer = fit_target_normalizer(samples)

    result = symbols["_run_arm"](
        arm="candidate",
        model=symbols["_build_candidate_attention"](),
        samples=samples,
        normalizer=normalizer,
        device=torch.device("cpu"),
        max_steps=1,
        threshold=0.10,
        batch_size=2,
        eval_interval=1,
        budget_seconds=30.0,
    )

    assert result["status"] == "completed"
    assert result["steps_completed"] == 1
    assert math.isfinite(result["train_mae_pK"])
    assert math.isfinite(result["train_rmse_pK"])
    assert result["gradient_parameters"]["nonfinite_gradient_count"] == 0
    assert result["validation_evaluated"] is False
    assert result["test_evaluated"] is False
