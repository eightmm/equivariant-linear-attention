from pathlib import Path
import runpy

import pytest
import torch

from equivariant_attention import EquivariantAttention, EquivariantAttentionConfig
from equivariant_attention.benchmarking import (
    SyntheticMoleculeDataset,
    collate_graphs,
    split_dataset,
)
from equivariant_attention.training import (
    build_regression_model,
    evaluate_regression,
    fit_target_normalizer,
    train_regression_step,
)


def test_synthetic_dataset_collates_graph_batch() -> None:
    dataset = SyntheticMoleculeDataset(
        num_samples=3, node_dim=6, min_nodes=4, max_nodes=6, seed=11
    )
    batch = collate_graphs([dataset[0], dataset[1]])

    assert batch.node_feats.ndim == 2
    assert batch.pos.shape == (batch.node_feats.shape[0], 3)
    assert batch.target.shape == (2, 1)
    assert batch.batch.min().item() == 0
    assert batch.batch.max().item() == 1
    assert batch.sample_ids == (dataset[0].sample_id, dataset[1].sample_id)


def test_graph_batch_separates_feature_and_geometry_precision() -> None:
    dataset = SyntheticMoleculeDataset(num_samples=2, node_dim=4, seed=12)
    batch = collate_graphs([dataset[0], dataset[1]]).to("cpu", dtype=torch.float16)

    assert batch.node_feats.dtype == torch.float16
    assert batch.pos.dtype == torch.float32


@pytest.mark.parametrize("geometry_dtype", [torch.float16, torch.bfloat16, torch.int64])
def test_graph_batch_rejects_invalid_geometry_dtype(
    geometry_dtype: torch.dtype,
) -> None:
    dataset = SyntheticMoleculeDataset(num_samples=2, node_dim=4, seed=12)
    batch = collate_graphs([dataset[0], dataset[1]])

    with pytest.raises(
        (TypeError, ValueError), match="geometry_dtype.*float32 or float64"
    ):
        batch.to("cpu", dtype=torch.float16, geometry_dtype=geometry_dtype)


def test_split_dataset_is_deterministic_and_disjoint() -> None:
    dataset = SyntheticMoleculeDataset(num_samples=12, node_dim=4, seed=13)
    train_a, val_a, test_a = split_dataset(dataset, train_size=6, val_size=3, seed=17)
    train_b, val_b, test_b = split_dataset(dataset, train_size=6, val_size=3, seed=17)

    assert (train_a, val_a, test_a) == (train_b, val_b, test_b)
    assert set(train_a).isdisjoint(val_a)
    assert set(train_a).isdisjoint(test_a)
    assert set(val_a).isdisjoint(test_a)


def test_regression_train_and_eval_smoke() -> None:
    torch.manual_seed(29)
    dataset = SyntheticMoleculeDataset(
        num_samples=8, node_dim=5, min_nodes=4, max_nodes=6, seed=31
    )
    batch = collate_graphs([dataset[i] for i in range(4)])
    model = build_regression_model(node_dim=5, hidden_dim=16, num_layers=2, num_heads=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    normalizer = fit_target_normalizer(dataset[i] for i in range(4))

    loss = train_regression_step(model, batch, optimizer, target_normalizer=normalizer)
    metrics = evaluate_regression(model, [batch], target_normalizer=normalizer)

    assert torch.isfinite(torch.tensor(loss))
    assert metrics["mae"] >= 0.0
    assert metrics["rmse"] >= 0.0


def test_builder_uses_the_single_promoted_architecture() -> None:
    model = build_regression_model(node_dim=5, hidden_dim=32, num_layers=2, num_heads=4)

    assert model.attention_kind == "factorized_moment"
    assert model.symmetry == "O3"
    assert model.config.hidden_irreps == "32x0e + 2x1o"
    assert len(model.layers) == 2
    assert model.layers[0].ffn_out.in_features == 64


def test_config_and_training_cli_share_defaults() -> None:
    config = EquivariantAttentionConfig(node_dim=5)
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_compare.py"
    args = runpy.run_path(script)["parse_args"]([])

    assert config.num_layers == args.num_layers == 3
    assert config.num_heads == args.num_heads == 4
    assert config.linear_kernel_init == args.linear_kernel_init
    assert config.use_key_balancing == (not args.no_key_balancing)
    assert not hasattr(args, "model")
    assert not hasattr(args, "moment_sinkhorn_iterations")


def test_benchmark_cli_represents_real_batches() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "bench_attention.py"
    symbols = runpy.run_path(script)
    args = symbols["parse_args"]([])

    assert args.graphs == [1, 8, 32]
    assert args.nodes_per_graph == [16, 32]
    assert args.routing == "ggg"
    assert args.memory_count == 1
    assert not args.memory_interaction
    assert not args.radial_trace


def test_benchmark_cli_builds_registered_lgl_memory_config() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "bench_attention.py"
    symbols = runpy.run_path(script)
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
    config = symbols["benchmark_config"](args)
    model = EquivariantAttention(config)

    assert type(model) is EquivariantAttention
    assert config.local_head_counts == (4, 0, 4)
    assert config.global_memory_count == 4
    assert config.use_memory_interaction
    assert config.use_radial_trace


def test_benchmark_invalid_interaction_route_uses_model_config_validation() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "bench_attention.py"
    symbols = runpy.run_path(script)
    args = symbols["parse_args"](
        ["--routing", "lll", "--memory-count", "8", "--memory-interaction"]
    )

    with pytest.raises(ValueError, match="middle global stage.*lgl route"):
        EquivariantAttention(symbols["benchmark_config"](args))


def test_benchmark_rows_record_the_full_attention_configuration() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "bench_attention.py"
    symbols = runpy.run_path(script)
    args = symbols["parse_args"](
        [
            "--routing",
            "lgl",
            "--local-cutoff",
            "3.0",
            "--num-rbf",
            "12",
            "--memory-count",
            "8",
            "--memory-interaction",
            "--memory-assignment-temperature",
            "0.75",
            "--memory-assignment-scale",
            "3.5",
            "--memory-interaction-cutoff",
            "4.5",
            "--radial-trace",
            "--compile",
        ]
    )
    config = symbols["benchmark_config"](args)
    row = symbols["benchmark_row"](
        args=args,
        config=config,
        graphs=2,
        nodes_per_graph=5,
        backward=False,
        elapsed_ms=1.25,
        memory_mib=3.5,
    )
    record = dict(zip(symbols["BENCHMARK_COLUMNS"], row, strict=True))

    assert record == {
        "graphs": 2,
        "nodes_per_graph": 5,
        "total_nodes": 10,
        "pass": "forward",
        "ms": "1.250",
        "peak_mem_mib": "3.5",
        "implementation": "factorized_moment",
        "routing": "lgl",
        "local_head_counts": "4|0|4",
        "local_cutoff": 3.0,
        "num_rbf": 12,
        "memory_count": 8,
        "memory_interaction": True,
        "memory_assignment_temperature": 0.75,
        "memory_assignment_scale": 3.5,
        "memory_interaction_cutoff": 4.5,
        "radial_trace": True,
        "dtype": "float32",
        "compiled": True,
        "compile_mode": "reduce-overhead",
    }

    backward_row = symbols["benchmark_row"](
        args=args,
        config=config,
        graphs=2,
        nodes_per_graph=5,
        backward=True,
        elapsed_ms=2.5,
        memory_mib=4.5,
    )
    backward_record = dict(zip(symbols["BENCHMARK_COLUMNS"], backward_row, strict=True))
    assert not backward_record["compiled"]


def test_cpu_ci_runs_the_project_fast_gate() -> None:
    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"

    assert workflow.is_file()
    content = workflow.read_text()
    assert "scripts/check.sh fast" in content
    assert "astral-sh/setup-uv@v8.3.2" in content
    assert "astral-sh/setup-uv@v8\n" not in content


@pytest.mark.parametrize(
    ("dtype", "automatic", "expected"),
    [
        (torch.float64, False, 1e-10),
        (torch.float16, False, 5e-3),
        (torch.bfloat16, False, 1e-2),
        (torch.float16, True, 5e-3),
        (torch.bfloat16, True, 1e-2),
    ],
)
def test_ml_smoke_comparison_tolerance_tracks_numeric_precision(
    dtype: torch.dtype,
    automatic: bool,
    expected: float,
) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "ml_smoke.py"
    tolerance = runpy.run_path(script)["_comparison_tolerance"]

    assert tolerance(dtype, automatic=automatic) == expected
