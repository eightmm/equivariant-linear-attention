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
    assert config["global_transport_mode"] == "learned"
    assert config["global_attention_formula"] == "factorized_learned_kernel"


def test_determinism_mode_is_explicit_and_recorded_in_run_config() -> None:
    symbols = _script_symbols()
    default_args = symbols["parse_args"]([])
    strict_args = symbols["parse_args"](["--determinism", "strict"])

    assert default_args.determinism == "seeded"
    assert strict_args.determinism == "strict"
    assert (
        symbols["_run_config"](strict_args, split_seed=42, model_seed=43)[
            "determinism"
        ]
        == "strict"
    )


def test_synthetic_runner_records_effective_reproducibility_state(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    symbols = _script_symbols()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    args = symbols["parse_args"](
        [
            "--determinism",
            "strict",
            "--num-samples",
            "8",
            "--train-size",
            "4",
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
        ]
    )
    monkeypatch.setattr(
        symbols["argparse"].ArgumentParser,
        "parse_args",
        lambda self, argv=None: args,
    )

    try:
        symbols["main"]()
        metrics = json.loads(capsys.readouterr().out)
    finally:
        torch.use_deterministic_algorithms(
            previous_deterministic,
            warn_only=previous_warn_only,
        )
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        torch.backends.cudnn.benchmark = previous_cudnn_benchmark

    assert metrics["reproducibility"]["mode"] == "strict"
    assert metrics["reproducibility"]["deterministic_algorithms"] is True
    assert metrics["reproducibility"]["deterministic_warn_only"] is False
    assert metrics["run_config"]["determinism"] == "strict"


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
        ("lgg", (4, 0, 0)),
        ("ggl", (0, 0, 4)),
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
    assert config["hemm_admission_status"] == "stage0_blocked_experimental"


def test_run_config_records_pairwise_local_content_and_mass_contract() -> None:
    symbols = _script_symbols()
    args = symbols["parse_args"](
        ["--routing", "lgl", "--pairwise-local-content"]
    )

    config = symbols["_run_config"](args, split_seed=42, model_seed=43)
    model = symbols["_build_benchmark_model"](args, node_dim=11)

    assert config["pairwise_local_content"] is True
    assert config["pairwise_local_formula"] == (
        "shared_receiver_sender_rbf_mlp_plus_degree_and_cutoff_mass"
    )
    assert config["pairwise_local_aggregation"] == "cutoff_sum_over_sqrt_degree"
    assert model.config.use_pairwise_local_content is True
    assert config["pairwise_residual_scale_init"] == 0.1
    assert any("local_pairwise_content" in name for name, _ in model.named_parameters())

    zero_args = symbols["parse_args"](
        [
            "--routing",
            "lgl",
            "--pairwise-local-content",
            "--pairwise-residual-scale-init",
            "0.0",
        ]
    )
    zero_model = symbols["_build_benchmark_model"](zero_args, node_dim=11)
    assert zero_model.local_pairwise_content.residual_scale.item() == 0.0


@pytest.mark.parametrize(
    ("mode", "formula", "balancing"),
    [
        ("learned", "factorized_learned_kernel", "one_cycle"),
        ("uniform", "exact_graph_mean", "not_applicable"),
        ("none", "no_global_transport", "not_applicable"),
    ],
)
def test_run_config_records_actual_global_transport(
    mode: str,
    formula: str,
    balancing: str,
) -> None:
    symbols = _script_symbols()
    args = symbols["parse_args"](["--global-transport-mode", mode])

    config = symbols["_run_config"](args, split_seed=42, model_seed=43)

    assert config["global_transport_mode"] == mode
    assert config["global_attention_formula"] == formula
    assert config["global_key_balancing"] == balancing


@pytest.mark.parametrize("mode", ["learned", "uniform", "none"])
def test_local_only_run_config_marks_global_transport_not_executed(mode: str) -> None:
    symbols = _script_symbols()
    args = symbols["parse_args"](
        ["--routing", "lll", "--global-transport-mode", mode]
    )

    config = symbols["_run_config"](args, split_seed=42, model_seed=43)

    assert config["global_transport_mode"] == mode
    assert config["global_transport_executed"] is False
    assert config["global_attention_formula"] == "not_applicable_no_global_heads"
    assert config["global_key_balancing"] == "not_applicable"


def test_internal_static_egnn_run_config_is_explicitly_nonofficial() -> None:
    symbols = _script_symbols()
    args = symbols["parse_args"](
        [
            "--benchmark-model",
            "internal_static_egnn_baseline",
            "--hidden-dim",
            "91",
        ]
    )

    config = symbols["_run_config"](args, split_seed=42, model_seed=43)
    model = symbols["_build_benchmark_model"](args, node_dim=11)

    assert config["model"] == "internal_static_egnn_baseline"
    assert config["comparison_role"] == "internal_same_harness_baseline"
    assert config["official_reproduction"] is False
    assert config["coordinate_updates"] is False
    assert config["global_transport_mode"] == "not_applicable"
    assert config["readout"] == "layernorm_node_linear_graph_mean"
    assert sum(parameter.numel() for parameter in model.parameters()) == 152_065


def test_dynamic_coordinate_models_record_the_exact_private_and_public_controls() -> None:
    symbols = _script_symbols()
    attention_args = symbols["parse_args"](
        ["--routing", "lgl", "--coordinate-updates"]
    )
    egnn_args = symbols["parse_args"](
        ["--benchmark-model", "internal_dynamic_egnn_baseline"]
    )

    attention_config = symbols["_run_config"](
        attention_args, split_seed=42, model_seed=43
    )
    egnn_config = symbols["_run_config"](
        egnn_args, split_seed=42, model_seed=43
    )
    attention = symbols["_build_benchmark_model"](attention_args, node_dim=11)
    egnn = symbols["_build_benchmark_model"](egnn_args, node_dim=11)

    assert attention_config["coordinate_updates"] is True
    assert attention_config["coordinate_update_max_step_angstrom"] == 0.25
    assert attention_config["coordinate_update_count"] == 2
    assert attention.config.coordinate_updates is True
    assert egnn_config["model"] == "internal_dynamic_egnn_baseline"
    assert egnn_config["comparison_role"] == "internal_same_harness_baseline"
    assert egnn_config["official_reproduction"] is False
    assert egnn_config["coordinate_updates"] is True
    assert egnn_config["coordinate_update_formula"] == (
        "centered_bounded_relative_vectors_times_invariant_edge_scalars"
    )
    assert egnn_args.hidden_dim == 91
    assert abs(
        sum(parameter.numel() for parameter in attention.parameters())
        - sum(parameter.numel() for parameter in egnn.parameters())
    ) / sum(parameter.numel() for parameter in attention.parameters()) < 0.01


def test_model_specific_hidden_width_defaults_are_parameter_matched() -> None:
    symbols = _script_symbols()

    factorized = symbols["parse_args"]([])
    egnn = symbols["parse_args"](
        ["--benchmark-model", "internal_static_egnn_baseline"]
    )
    dynamic_egnn = symbols["parse_args"](
        ["--benchmark-model", "internal_dynamic_egnn_baseline"]
    )
    explicit = symbols["parse_args"](
        [
            "--benchmark-model",
            "internal_static_egnn_baseline",
            "--hidden-dim",
            "16",
        ]
    )

    assert factorized.hidden_dim == 64
    assert egnn.hidden_dim == 91
    assert dynamic_egnn.hidden_dim == 91
    assert explicit.hidden_dim == 16


@pytest.mark.parametrize(
    ("arguments", "control"),
    [
        (["--num-heads", "8"], "num_heads"),
        (["--linear-kernel-init", "0.9"], "linear_kernel_init"),
        (["--local-cutoff", "9.0"], "local_cutoff"),
        (["--num-rbf", "9"], "num_rbf"),
        (
            ["--memory-assignment-temperature", "0.5"],
            "memory_assignment_temperature",
        ),
        (["--memory-assignment-scale", "3.0"], "memory_assignment_scale"),
        (["--memory-interaction-cutoff", "3.0"], "memory_interaction_cutoff"),
        (["--diagnostic-max-nodes", "32"], "diagnostic_max_nodes"),
        (["--diagnostic-effective-rank"], "diagnostic_effective_rank"),
    ],
)
def test_internal_egnn_rejects_ignored_factorized_controls(
    arguments: list[str],
    control: str,
) -> None:
    symbols = _script_symbols()
    args = symbols["parse_args"](
        ["--benchmark-model", "internal_static_egnn_baseline", *arguments]
    )

    with pytest.raises(ValueError, match=control):
        symbols["_build_benchmark_model"](args, node_dim=11)


def test_internal_static_egnn_runner_uses_shared_training_and_hides_test(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    symbols = _script_symbols()
    args = symbols["parse_args"](
        [
            "--benchmark-model",
            "internal_static_egnn_baseline",
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
            "2",
            "--bounded-diagnostics",
        ]
    )
    monkeypatch.setattr(
        symbols["argparse"].ArgumentParser,
        "parse_args",
        lambda self, argv=None: args,
    )

    symbols["main"]()
    metrics = json.loads(capsys.readouterr().out)

    assert metrics["model"] == "internal_static_egnn_baseline"
    assert metrics["test_evaluated"] is False
    assert "test_mae" not in metrics
    assert metrics["run_config"]["official_reproduction"] is False
    assert metrics["bounded_diagnostics"]["status"] == (
        "not_applicable_internal_static_egnn_baseline"
    )
    assert metrics["node_count_strata"]["metrics"]
    assert metrics["nonzero_gradient_parameter_count"] > 0
    json.dumps(metrics, allow_nan=False)


def test_dynamic_egnn_runner_records_active_coordinate_diagnostics_without_test(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    symbols = _script_symbols()
    args = symbols["parse_args"](
        [
            "--benchmark-model",
            "internal_dynamic_egnn_baseline",
            "--num-samples",
            "12",
            "--train-size",
            "7",
            "--val-size",
            "2",
            "--batch-size",
            "2",
            "--steps",
            "2",
            "--hidden-dim",
            "8",
            "--num-layers",
            "2",
            "--bounded-diagnostics",
        ]
    )
    monkeypatch.setattr(
        symbols["argparse"].ArgumentParser,
        "parse_args",
        lambda self, argv=None: args,
    )

    symbols["main"]()
    metrics = json.loads(capsys.readouterr().out)

    assert metrics["model"] == "internal_dynamic_egnn_baseline"
    assert metrics["test_evaluated"] is False
    assert "test_mae" not in metrics
    assert metrics["baseline_details"]["coordinate_updates"] is True
    assert metrics["bounded_diagnostics"]["status"] == (
        "not_applicable_internal_dynamic_egnn_baseline"
    )
    coordinate = metrics["coordinate_diagnostics"]
    assert coordinate["enabled"] is True
    assert coordinate["active"] is True
    assert coordinate["displacement_rms_angstrom"] > 0.0
    assert coordinate["displacement_max_angstrom"] <= 0.25 + 1e-6
    assert coordinate["centroid_drift_max_angstrom"] < 1e-6
    assert len(coordinate["layers"]) == 1
    assert metrics["coordinate_gradient_parameters"][
        "nonzero_gradient_parameter_count"
    ] > 0
    json.dumps(metrics, allow_nan=False)


def test_factorized_coordinate_runner_records_dynamic_geometry_provenance(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    symbols = _script_symbols()
    args = symbols["parse_args"](
        [
            "--coordinate-updates",
            "--routing",
            "lgl",
            "--num-samples",
            "12",
            "--train-size",
            "7",
            "--val-size",
            "2",
            "--batch-size",
            "2",
            "--steps",
            "2",
            "--hidden-dim",
            "8",
            "--num-layers",
            "3",
            "--num-heads",
            "2",
        ]
    )
    monkeypatch.setattr(
        symbols["argparse"].ArgumentParser,
        "parse_args",
        lambda self, argv=None: args,
    )

    symbols["main"]()
    metrics = json.loads(capsys.readouterr().out)

    assert metrics["run_config"]["coordinate_updates"] is True
    assert metrics["run_config"]["geometry_recomputed_per_layer"] is True
    assert metrics["coordinate_diagnostics"]["enabled"] is True
    assert metrics["coordinate_diagnostics"]["active"] is True
    assert len(metrics["coordinate_diagnostics"]["layers"]) == 2
    assert metrics["test_evaluated"] is False
    json.dumps(metrics, allow_nan=False)


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


def test_static_dynamic_pairs_share_the_same_noncoordinate_initialization() -> None:
    symbols = _script_symbols()
    attention_static_args = symbols["parse_args"](["--routing", "lgl"])
    attention_dynamic_args = symbols["parse_args"](
        ["--routing", "lgl", "--coordinate-updates"]
    )
    egnn_static_args = symbols["parse_args"](
        ["--benchmark-model", "internal_static_egnn_baseline"]
    )
    egnn_dynamic_args = symbols["parse_args"](
        ["--benchmark-model", "internal_dynamic_egnn_baseline"]
    )

    hashes = []
    for args in (
        attention_static_args,
        attention_dynamic_args,
        egnn_static_args,
        egnn_dynamic_args,
    ):
        torch.manual_seed(43)
        model = symbols["_build_benchmark_model"](args, node_dim=11)
        hashes.append(symbols["_paired_base_state_hashes"](model))

    assert hashes[0] == hashes[1]
    assert hashes[2] == hashes[3]


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
    assert config["dataset_seed"] == args.seed


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
    assert len(metrics["final_state_sha256"]) == 64
    assert len(metrics["state_schema_sha256"]) == 64
    assert metrics["nonzero_gradient_parameter_count"] > 0
    assert metrics["gradient_clipping"]["measurement_point"] == "before_clipping"
    assert metrics["gradient_clipping"]["step_count"] == 1
    assert 0.0 <= metrics["gradient_clipping"]["clip_fraction"] <= 1.0
    assert metrics["train_probe"]["selection"] == "train_split_order_prefix"
    assert metrics["train_probe"]["sample_count"] == 7
    assert metrics["train_probe"]["mae"] >= 0.0
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


def test_pairwise_named_gradient_diagnostics_are_active() -> None:
    symbols = _script_symbols()
    model = symbols["build_regression_model"](
        node_dim=4,
        hidden_dim=8,
        num_layers=3,
        num_heads=2,
        local_head_counts=(2, 0, 2),
        use_pairwise_local_content=True,
    )
    sum(parameter.sum() for parameter in model.parameters()).backward()

    summary = symbols["_named_gradient_parameter_diagnostics"](
        model, "local_pairwise_content"
    )

    assert summary["parameter_count"] > 0
    assert summary["parameters_with_gradient_count"] == summary["parameter_count"]
    assert summary["nonzero_gradient_parameter_count"] == summary["parameter_count"]


@pytest.mark.parametrize("memory_count", [4, 8])
def test_bounded_diagnostics_connect_active_memory_without_mutating_model(
    memory_count: int,
) -> None:
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
        global_memory_count=memory_count,
        use_memory_interaction=True,
    )
    model.train()
    before_hashes = symbols["_model_state_hashes"](model)
    before_training = [module.training for module in model.modules()]
    before_gradients = [parameter.grad for parameter in model.parameters()]
    before_rng = torch.random.get_rng_state().clone()

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
    assert diagnostics["memory"]["assignment"]["memory_count"] == memory_count
    assert diagnostics["memory"]["assignment"]["marginal_entropy_over_log_m"] >= 0.0
    assert diagnostics["memory"]["assignment"]["mutual_information_over_log_m"] >= 0.0
    assert diagnostics["instrumentation"]["assignment_source"] == (
        "shared_invariant_router_exact_recompute"
    )
    assert diagnostics["memory"]["centers"]["scope"] == "single_graph_per_head"
    assert "coupling.q50" in diagnostics["memory"]["coupling"]
    pair_gate = diagnostics["memory"]["pair_gate"]
    assert pair_gate["min"] > 0.0
    assert pair_gate["max"] <= 1.0
    assert pair_gate["cv"] >= 0.0
    assert pair_gate["centered_frobenius_ratio"] == pytest.approx(
        pair_gate["cv"] / (1.0 + pair_gate["cv"] ** 2) ** 0.5
    )
    assert 0.0 <= pair_gate["nonconstant_fraction"] <= 1.0
    assert pair_gate["nonconstant_relative_tolerance"] == pytest.approx(1e-3)
    all_heads = diagnostics["memory"]["all_head_activation"]
    assert all_heads["scope"] == "single_graph_per_head"
    assert all_heads["head_count"] == 2
    assert len(all_heads["heads"]) == 2
    assert symbols["_model_state_hashes"](model) == before_hashes
    assert [module.training for module in model.modules()] == before_training
    assert [parameter.grad for parameter in model.parameters()] == before_gradients
    assert torch.equal(torch.random.get_rng_state(), before_rng)
    json.dumps(diagnostics, allow_nan=False)


def test_bounded_diagnostics_records_exact_single_memory_bypass() -> None:
    symbols = _script_symbols()
    dataset = symbols["SyntheticMoleculeDataset"](
        num_samples=4,
        node_dim=4,
        min_nodes=4,
        max_nodes=5,
        seed=89,
    )
    model = symbols["build_regression_model"](
        node_dim=4,
        hidden_dim=8,
        num_layers=3,
        num_heads=2,
        local_head_counts=(2, 0, 2),
        global_memory_count=1,
        use_memory_interaction=True,
    )

    diagnostics = symbols["_bounded_model_diagnostics"](
        model,
        dataset,
        [0],
        max_nodes=8,
        include_effective_rank=False,
    )

    memory = diagnostics["memory"]
    assert memory["status"] == "exact_single_memory_bypass"
    assert memory["transport_connected"] is False
    assert memory["pair_gate"]["mean"] == pytest.approx(1.0)
    assert memory["pair_gate"]["cv"] == pytest.approx(0.0)
    assert memory["pair_gate"]["centered_frobenius_ratio"] == pytest.approx(0.0)


def test_bounded_diagnostics_connects_one_trained_local_head() -> None:
    symbols = _script_symbols()
    dataset = symbols["SyntheticMoleculeDataset"](
        num_samples=4,
        node_dim=4,
        min_nodes=4,
        max_nodes=5,
        seed=97,
    )
    model = symbols["build_regression_model"](
        node_dim=4,
        hidden_dim=8,
        num_layers=3,
        num_heads=2,
        local_head_counts=(2, 0, 2),
    )

    diagnostics = symbols["_bounded_model_diagnostics"](
        model,
        dataset,
        [0],
        max_nodes=8,
        include_effective_rank=False,
    )

    local = diagnostics["local_attention"]
    assert local["status"] == "ok"
    assert local["layer_index"] == 0
    assert local["transport_connected"] is True
    assert local["summary"]["degree.min"] >= 1
    assert local["summary"]["attention.row_mass_max_abs_error"] < 1e-6
    assert local["summary"]["distance_over_cutoff.q100"] < 1.0
    assert diagnostics["batch"]["sample_id"] == dataset[0].sample_id
    assert "selected_trained_local_attention_weights" in diagnostics[
        "instrumentation"
    ]["connected"]
    assert "local_attention_weights" not in diagnostics["instrumentation"][
        "unconnected"
    ]
    json.dumps(diagnostics, allow_nan=False)


def test_bounded_diagnostics_cover_all_local_layers_and_validation_sample() -> None:
    symbols = _script_symbols()
    dataset = symbols["SyntheticMoleculeDataset"](
        num_samples=10,
        node_dim=4,
        min_nodes=3,
        max_nodes=8,
        seed=103,
    )
    validation_indices = [8, 1, 6, 2, 9, 4, 0]
    model = symbols["build_regression_model"](
        node_dim=4,
        hidden_dim=8,
        num_layers=3,
        num_heads=2,
        local_head_counts=(2, 0, 2),
    )
    model.train()
    before_training = [module.training for module in model.modules()]
    before_state = symbols["_model_state_hashes"](model)
    before_rng = torch.random.get_rng_state().clone()

    selected = symbols["_select_bounded_validation_indices"](
        dataset,
        validation_indices,
        max_nodes=8,
        sample_count=3,
    )
    ordered = sorted(
        validation_indices,
        key=lambda index: (dataset[index].node_feats.shape[0], index),
    )
    expected_positions = [0, (len(ordered) - 1) // 2, len(ordered) - 1]
    assert selected == [ordered[position] for position in expected_positions]

    diagnostics = symbols["_bounded_model_diagnostics"](
        model,
        dataset,
        validation_indices,
        max_nodes=8,
        include_effective_rank=False,
        local_sample_count=3,
    )

    local = diagnostics["local_attention"]
    assert local["status"] == "ok"
    assert [layer["layer_index"] for layer in local["layers"]] == [0, 2]
    assert all(len(layer["heads"]) == 2 for layer in local["layers"])
    distribution = local["validation_distribution"]
    assert distribution["selection"]["dataset_indices"] == selected
    assert distribution["selection"]["sample_count"] == 3
    assert distribution["selection"]["node_count.min"] >= 3
    assert distribution["selection"]["node_count.max"] <= 8
    assert [layer["layer_index"] for layer in distribution["layers"]] == [0, 2]
    for layer in distribution["layers"]:
        assert layer["sample_count"] == 3
        assert len(layer["heads"]) == 2
        assert layer["aggregate"]["attention.row_mass_max_abs_error.sample_max"] < 1e-6
    assert (
        "all_local_layers_and_heads_on_bounded_validation_sample"
        in diagnostics["instrumentation"]["connected"]
    )
    assert [module.training for module in model.modules()] == before_training
    assert symbols["_model_state_hashes"](model) == before_state
    assert torch.equal(torch.random.get_rng_state(), before_rng)
    json.dumps(diagnostics, allow_nan=False)


def test_diagnostic_sample_count_is_validated_and_recorded() -> None:
    symbols = _script_symbols()
    args = symbols["parse_args"](["--diagnostic-sample-count", "7"])
    config = symbols["_run_config"](args, split_seed=42, model_seed=43)

    assert args.diagnostic_sample_count == 7
    assert config["diagnostic_sample_count"] == 7

    with pytest.raises(SystemExit):
        symbols["parse_args"](["--diagnostic-sample-count", "0"])


@pytest.mark.parametrize("mode", ["uniform", "none"])
def test_bounded_diagnostics_reports_the_executed_transport_control(mode: str) -> None:
    symbols = _script_symbols()
    dataset = symbols["SyntheticMoleculeDataset"](
        num_samples=4, node_dim=4, min_nodes=4, max_nodes=5, seed=101
    )
    model = symbols["build_regression_model"](
        node_dim=4,
        hidden_dim=8,
        num_layers=3,
        num_heads=2,
        local_head_counts=(2, 0, 2),
        global_transport_mode=mode,
    )

    diagnostics = symbols["_bounded_model_diagnostics"](
        model, dataset, [0], max_nodes=8, include_effective_rank=True
    )

    assert diagnostics["instrumentation"]["global_transport_mode"] == mode
    assert diagnostics["local_attention"]["status"] == "ok"
    if mode == "uniform":
        kernel = diagnostics["kernel_attention"]
        assert kernel["status"] == "exact_uniform_transport"
        assert kernel["attention.entropy_over_log_n"] == pytest.approx(1.0)
        assert kernel["attention.effective_rank"] == pytest.approx(1.0)
    else:
        assert diagnostics["kernel_attention"]["status"] == (
            "disabled_no_global_transport"
        )
    json.dumps(diagnostics, allow_nan=False)
