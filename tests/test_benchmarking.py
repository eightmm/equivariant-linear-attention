from pathlib import Path
import json
import runpy
import subprocess

import pytest
import torch

from equivariant_attention import EquivariantAttention, EquivariantAttentionConfig
from equivariant_attention.benchmarking import (
    GraphSample,
    SyntheticMoleculeDataset,
    _radius_candidate_edge_index,
    _qm9_sample_id,
    collate_graphs,
    load_qm9_samples,
    split_dataset,
)
from equivariant_attention.training import (
    build_regression_model,
    evaluate_regression,
    fit_target_normalizer,
    train_regression_step,
)


def test_qm9_sample_id_records_processed_row_raw_index_and_name() -> None:
    data = type(
        "Data",
        (),
        {"idx": torch.tensor([3111]), "name": "gdb_3112"},
    )()

    assert _qm9_sample_id(data, row_index=3055) == (
        "qm9-row-3055-raw-index-3111-name-gdb_3112"
    )
    assert _qm9_sample_id(object(), row_index=7) == (
        "qm9-row-7-raw-index-unknown-name-unknown"
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


def test_sparse_edges_collate_with_offsets_and_move_with_batch() -> None:
    first = GraphSample(
        node_feats=torch.randn(2, 3),
        pos=torch.randn(2, 3),
        target=torch.tensor([1.0]),
        sample_id="first",
        edge_index=torch.tensor([[0, 1, 0], [0, 1, 1]]),
    )
    second = GraphSample(
        node_feats=torch.randn(3, 3),
        pos=torch.randn(3, 3),
        target=torch.tensor([2.0]),
        sample_id="second",
        edge_index=torch.tensor([[0, 1, 2, 2], [0, 1, 2, 0]]),
    )

    batch = collate_graphs([first, second])

    assert batch.edge_index_is_validated
    assert torch.equal(
        batch.edge_index,
        torch.tensor([[0, 1, 0, 2, 3, 4, 4], [0, 1, 1, 2, 3, 4, 2]]),
    )
    moved = batch.to("cpu", dtype=torch.float64)
    assert moved.edge_index is not None
    assert moved.edge_index.dtype == torch.long
    assert moved.edge_index.device == moved.node_feats.device
    assert torch.equal(moved.edge_index, batch.edge_index)
    assert moved.edge_index_is_validated


@pytest.mark.parametrize(
    ("edge_index", "match"),
    [
        (torch.tensor([[0, 0, 1], [0, 0, 1]]), "duplicate"),
        (torch.tensor([[0, 0], [0, 1]]), "self edge"),
    ],
)
def test_collate_rejects_edges_that_cannot_enter_validated_fast_path(
    edge_index: torch.Tensor,
    match: str,
) -> None:
    sample = GraphSample(
        node_feats=torch.randn(2, 3),
        pos=torch.randn(2, 3),
        target=torch.tensor([1.0]),
        sample_id="invalid",
        edge_index=edge_index,
    )

    with pytest.raises(ValueError, match=match):
        collate_graphs([sample])


def test_collate_rejects_mixed_sparse_edge_presence() -> None:
    dataset = SyntheticMoleculeDataset(num_samples=2, node_dim=4, seed=1209)
    sparse = GraphSample(
        node_feats=dataset[0].node_feats,
        pos=dataset[0].pos,
        target=dataset[0].target,
        sample_id=dataset[0].sample_id,
        edge_index=torch.arange(dataset[0].pos.shape[0]).repeat(2, 1),
    )

    with pytest.raises(ValueError, match="all samples.*edge_index"):
        collate_graphs([sparse, dataset[1]])


def test_radius_candidates_include_self_and_use_receiver_sender_rows() -> None:
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    )

    edge_index = _radius_candidate_edge_index(pos, cutoff=1.5)

    assert torch.equal(
        edge_index,
        torch.tensor([[0, 0, 1, 1, 2], [0, 1, 0, 1, 2]]),
    )


def test_qm9_loader_precomputes_requested_radius_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch_geometric.datasets

    data = type(
        "Data",
        (),
        {
            "x": torch.eye(3),
            "pos": torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
            ),
            "y": torch.arange(12, dtype=torch.float32).reshape(1, 12),
            "idx": torch.tensor([7]),
            "name": "gdb_8",
        },
    )()
    monkeypatch.setattr(torch_geometric.datasets, "QM9", lambda root: [data])

    sample = load_qm9_samples("unused", local_cutoff=1.5)[0]

    assert sample.edge_index is not None
    assert torch.equal(
        sample.edge_index,
        torch.tensor([[0, 0, 1, 1, 2], [0, 1, 0, 1, 2]]),
    )


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


def test_regression_step_records_preclip_norm_and_clip_fraction() -> None:
    torch.manual_seed(37)
    dataset = SyntheticMoleculeDataset(
        num_samples=6, node_dim=5, min_nodes=4, max_nodes=6, seed=41
    )
    batch = collate_graphs([dataset[i] for i in range(4)])
    model = build_regression_model(node_dim=5, hidden_dim=8, num_layers=1, num_heads=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    monitor: dict[str, float | int] = {}

    for _ in range(3):
        train_regression_step(
            model,
            batch,
            optimizer,
            grad_clip=1e-12,
            gradient_monitor=monitor,
        )

    assert monitor["step_count"] == 3
    assert monitor["clipped_step_count"] == 3
    assert monitor["pre_clip_grad_norm_last"] > 0.0
    assert monitor["pre_clip_grad_norm_max"] >= monitor["pre_clip_grad_norm_last"]
    assert monitor["pre_clip_grad_norm_sum"] > 0.0


def test_builder_uses_the_single_promoted_architecture() -> None:
    model = build_regression_model(node_dim=5, hidden_dim=32, num_layers=2, num_heads=4)

    assert model.attention_kind == "factorized_moment"
    assert model.symmetry == "O3"
    assert model.config.hidden_irreps == "32x0e + 2x1o"
    assert len(model.layers) == 2
    assert model.layers[0].ffn_out.in_features == 64


def test_builder_preserves_legacy_positional_arguments() -> None:
    model = build_regression_model(
        4,
        8,
        1,
        2,
        0.05,
        True,
        False,
        "fixed",
        (0,),
        3.5,
        7,
        False,
        1,
        False,
        1.0,
        2.5,
        2.5,
        False,
    )

    assert model.config.local_cutoff == 3.5
    assert model.config.num_rbf == 7
    assert model.config.global_transport_mode == "learned"


def test_config_and_training_cli_share_defaults() -> None:
    config = EquivariantAttentionConfig(node_dim=5)
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_compare.py"
    args = runpy.run_path(script)["parse_args"]([])

    assert config.num_layers == args.num_layers == 3
    assert config.num_heads == args.num_heads == 4
    assert config.linear_kernel_init == args.linear_kernel_init
    assert config.use_key_balancing == (not args.no_key_balancing)
    assert config.global_transport_mode == args.global_transport_mode == "learned"
    assert args.benchmark_model == "factorized_moment"
    assert not hasattr(args, "model")
    assert not hasattr(args, "moment_sinkhorn_iterations")


def test_benchmark_cli_represents_real_batches() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "bench_attention.py"
    symbols = runpy.run_path(script)
    args = symbols["parse_args"]([])

    assert args.graphs == [1, 8, 32]
    assert args.nodes_per_graph == [16, 32]
    assert args.routing == "ggg"
    assert args.global_transport_mode == "learned"
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


def test_training_cli_builds_sparse_edge_conditioned_lgl() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_compare.py"
    symbols = runpy.run_path(script)
    args = symbols["parse_args"](
        [
            "--dataset",
            "qm9",
            "--routing",
            "lgl",
            "--edge-conditioned-local-transport",
            "--precompute-local-edges",
        ]
    )

    model = symbols["_build_benchmark_model"](args, node_dim=11)

    assert model.config.local_head_counts == (4, 0, 4)
    assert model.config.use_edge_conditioned_local_transport
    assert args.precompute_local_edges


@pytest.mark.parametrize(
    ("routing", "expected"),
    [("lgg", (4, 0, 0)), ("ggl", (0, 0, 4))],
)
def test_benchmark_cli_uses_core_route_decomposition(
    routing: str,
    expected: tuple[int, ...],
) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "bench_attention.py"
    symbols = runpy.run_path(script)
    args = symbols["parse_args"](
        ["--routing", routing, "--global-transport-mode", "uniform"]
    )

    config = symbols["benchmark_config"](args)

    assert config.local_head_counts == expected
    assert config.global_transport_mode == "uniform"


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
        "global_transport_mode": "learned",
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


def test_bounded_control_screen_has_an_executable_ledger_command() -> None:
    root = Path(__file__).resolve().parents[1]
    runner = root / "scripts" / "run_bounded_control_screen.sh"

    assert runner.is_file()
    subprocess.run(["bash", "-n", str(runner)], check=True)
    expected_command = [
        "bash",
        "scripts/run_bounded_control_screen.sh",
        "artifacts/egnn-matched-baseline-development-20260718",
    ]
    records = [
        json.loads(line)
        for line in (root / "docs" / "EXPERIMENTS.jsonl").read_text().splitlines()
    ]

    assert any(record["cmd"] == expected_command for record in records)


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
