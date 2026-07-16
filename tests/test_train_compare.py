import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import torch


def _script_symbols() -> dict[str, object]:
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_compare.py"
    return runpy.run_path(script)


def test_qm9_data_identity_accepts_expected_hashes(tmp_path: Path) -> None:
    symbols = _script_symbols()
    hash_file = symbols["_hash_file"]
    data_identity = symbols["_qm9_data_identity"]
    relative_paths = ["raw/gdb9.sdf", "raw/gdb9.sdf.csv", "processed/data_v3.pt"]
    expected = {}
    for index, relative in enumerate(relative_paths):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture-{index}".encode("ascii"))
        expected[relative] = hash_file(path)

    assert data_identity(tmp_path, expected=expected) == expected


def test_qm9_data_identity_rejects_changed_data(tmp_path: Path) -> None:
    symbols = _script_symbols()
    data_identity = symbols["_qm9_data_identity"]
    path = tmp_path / "processed/data_v3.pt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"changed")

    with pytest.raises(ValueError, match="data identity mismatch"):
        data_identity(tmp_path, expected={"processed/data_v3.pt": "0" * 64})


def test_run_config_records_single_architecture() -> None:
    symbols = _script_symbols()
    args = symbols["parse_args"]([])

    config = symbols["_run_config"](
        args,
        split_seed=42,
        model_seed=43,
    )

    assert config["model"] == "factorized_moment"
    assert config["attention"] == "factorized_moment"
    assert config["balance_cycles"] == 1
    assert config["key_balancing"] is True
    assert config["linear_kernel_init"] > 0.0
    assert config["ffn_hidden_ratio"] == 2.0


def test_test_evaluation_is_opt_in() -> None:
    symbols = _script_symbols()
    default_args = symbols["parse_args"]([])
    enabled_args = symbols["parse_args"](["--evaluate-test"])

    assert default_args.evaluate_test is False
    assert enabled_args.evaluate_test is True
    assert (
        symbols["_run_config"](default_args, split_seed=42, model_seed=43)[
            "test_evaluated"
        ]
        is False
    )
    assert (
        symbols["_run_config"](enabled_args, split_seed=42, model_seed=43)[
            "test_evaluated"
        ]
        is True
    )


@pytest.mark.parametrize(
    "routing, expected",
    [
        ("ggg", (0, 0, 0)),
        ("lgl", (4, 0, 4)),
        ("lll", (4, 4, 4)),
    ],
)
def test_routing_presets_are_explicit(routing: str, expected: tuple[int, ...]) -> None:
    symbols = _script_symbols()

    assert (
        symbols["_routing_head_counts"](routing, num_layers=3, num_heads=4) == expected
    )


def test_run_config_records_local_memory_and_trace_controls() -> None:
    symbols = _script_symbols()
    args = symbols["parse_args"](
        [
            "--routing",
            "lgl",
            "--memory-count",
            "4",
            "--memory-interaction",
            "--radial-trace",
        ]
    )

    config = symbols["_run_config"](args, split_seed=42, model_seed=43)

    assert config["routing"] == "lgl"
    assert config["local_head_counts"] == [4, 0, 4]
    assert config["memory_count"] == 4
    assert config["memory_interaction"] is True
    assert config["radial_trace"] is True


def test_gradient_diagnostics_are_finite_json_scalars() -> None:
    symbols = _script_symbols()
    model = symbols["build_regression_model"](
        node_dim=4,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
    )
    sum(parameter.sum() for parameter in model.parameters()).backward()

    summary = symbols["_gradient_norms"](model)

    assert summary["all"] > 0.0
    assert summary["beta_raw"] > 0.0
    assert summary["gamma_raw"] > 0.0
    assert all(type(value) is float for value in summary.values())
    json.dumps(summary, allow_nan=False)


def test_initial_state_hash_is_common_across_memory_count_and_value_sensitive() -> None:
    symbols = _script_symbols()
    build_model = symbols["build_regression_model"]

    torch.manual_seed(71)
    single_memory = build_model(
        node_dim=4,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        global_memory_count=1,
    )
    torch.manual_seed(71)
    four_memories = build_model(
        node_dim=4,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        global_memory_count=4,
    )

    single_hashes = symbols["_model_state_hashes"](single_memory)
    memory_hashes = symbols["_model_state_hashes"](four_memories)
    assert single_hashes == memory_hashes
    assert len(single_hashes["initial_state_sha256"]) == 64
    assert len(single_hashes["state_schema_sha256"]) == 64

    with torch.no_grad():
        next(four_memories.parameters()).add_(1.0)
    changed_hashes = symbols["_model_state_hashes"](four_memories)
    assert (
        changed_hashes["initial_state_sha256"] != single_hashes["initial_state_sha256"]
    )
    assert changed_hashes["state_schema_sha256"] == single_hashes["state_schema_sha256"]


def test_nonzero_gradient_parameter_count_is_scalar_element_count() -> None:
    symbols = _script_symbols()
    model = symbols["build_regression_model"](
        node_dim=4,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
    )
    sum(parameter.sum() for parameter in model.parameters()).backward()

    summary = symbols["_gradient_parameter_diagnostics"](model)
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    assert summary["nonzero_gradient_parameter_count"] == trainable_count
    assert summary["parameters_with_gradient_count"] == trainable_count
    assert summary["nonfinite_gradient_parameter_count"] == 0
    json.dumps(summary, allow_nan=False)


def test_node_count_strata_match_registered_stage_two_boundaries() -> None:
    symbols = _script_symbols()
    node_counts = [1, 16, 17, 32, 33, 64, 65, 128, 129, 512, 513, 2048, 2049]
    dataset = [
        SimpleNamespace(node_feats=torch.zeros(count, 1)) for count in node_counts
    ]

    strata = symbols["_stratify_indices_by_node_count"](
        dataset, list(range(len(dataset)))
    )

    assert {
        name: [node_counts[index] for index in indices]
        for name, indices in strata.items()
    } == {
        "1-16": [1, 16],
        "17-32": [17, 32],
        "33-64": [33, 64],
        "65-128": [65, 128],
        "129-512": [129, 512],
        "513-2048": [513, 2048],
        "2049+": [2049],
    }


def test_run_config_records_inverse_positive_baseline_and_bounded_diagnostics() -> None:
    symbols = _script_symbols()
    args = symbols["parse_args"](
        [
            "--kernel-floor-mode",
            "inverse_graph_size",
            "--no-key-balancing",
            "--bounded-diagnostics",
            "--diagnostic-max-nodes",
            "32",
            "--diagnostic-effective-rank",
        ]
    )

    config = symbols["_run_config"](args, split_seed=42, model_seed=43)

    assert config["graph_size_scaled_positive_baseline"] is True
    assert config["kernel_scaling_formula_version"] == "positive_baseline_v1"
    assert (
        config["kernel_formula"] == "a_dot_b + (c + beta*(1 + delta*t))/N_g + gamma*t^2"
    )
    assert config["bounded_diagnostics"] is True
    assert config["diagnostic_max_nodes"] == 32
    assert config["diagnostic_effective_rank"] is True


def test_runner_json_connects_bounded_metrics_without_evaluating_test(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    symbols = _script_symbols()
    args = symbols["parse_args"](
        [
            "--num-samples",
            "12",
            "--train-size",
            "7",
            "--val-size",
            "2",
            "--batch-size",
            "2",
            "--steps",
            "1",
            "--hidden-dim",
            "8",
            "--num-layers",
            "1",
            "--num-heads",
            "2",
            "--bounded-diagnostics",
            "--diagnostic-max-nodes",
            "16",
            "--diagnostic-effective-rank",
        ]
    )
    monkeypatch.setattr(
        symbols["argparse"].ArgumentParser,
        "parse_args",
        lambda self, argv=None: args,
    )

    symbols["main"]()
    metrics = json.loads(capsys.readouterr().out)

    assert len(metrics["initial_state_sha256"]) == 64
    assert len(metrics["state_schema_sha256"]) == 64
    assert metrics["nonzero_gradient_parameter_count"] > 0
    assert metrics["test_evaluated"] is False
    assert "test_mae" not in metrics
    assert metrics["node_count_strata"]["split"] == "validation"
    assert metrics["node_count_strata"]["metrics"]
    diagnostics = metrics["bounded_diagnostics"]
    assert diagnostics["enabled"] is True
    assert diagnostics["status"] == "ok"
    assert diagnostics["instrumentation"]["activation_source"] == (
        "exact_recompute_from_captured_layer_input"
    )
    assert diagnostics["kernel_attention"]["attention.effective_rank"] >= 1.0
    assert diagnostics["memory"]["status"] == "disabled"
    json.dumps(metrics, allow_nan=False)


def test_bounded_diagnostics_connect_active_memory_without_mutating_model() -> None:
    symbols = _script_symbols()
    dataset = symbols["SyntheticMoleculeDataset"](
        num_samples=4,
        node_dim=4,
        min_nodes=4,
        max_nodes=5,
        seed=83,
    )
    model = symbols["build_regression_model"](
        node_dim=4,
        hidden_dim=8,
        num_layers=3,
        num_heads=2,
        local_head_counts=(2, 0, 2),
        global_memory_count=4,
        use_memory_interaction=True,
    )
    model.train()
    before_hashes = symbols["_model_state_hashes"](model)
    before_training = [module.training for module in model.modules()]

    diagnostics = symbols["_bounded_model_diagnostics"](
        model,
        dataset,
        [0],
        max_nodes=8,
        include_effective_rank=False,
    )

    assert diagnostics["status"] == "ok"
    assert diagnostics["instrumentation"]["layer_index"] == 1
    assert diagnostics["memory"]["status"] == "active"
    assert diagnostics["memory"]["transport_connected"] is True
    assert diagnostics["memory"]["assignment"]["memory_count"] == 4
    assert "coupling.q50" in diagnostics["memory"]["coupling"]
    assert symbols["_model_state_hashes"](model) == before_hashes
    assert [module.training for module in model.modules()] == before_training
    json.dumps(diagnostics, allow_nan=False)
