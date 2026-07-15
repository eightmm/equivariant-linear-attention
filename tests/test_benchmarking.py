from pathlib import Path
import runpy

import torch

from equivariant_attention import EquivariantAttentionConfig
from equivariant_attention.benchmarking import SyntheticMoleculeDataset, collate_graphs, split_dataset
from equivariant_attention.training import (
    build_regression_model,
    evaluate_regression,
    fit_target_normalizer,
    train_regression_step,
)


def test_synthetic_dataset_collates_graph_batch() -> None:
    dataset = SyntheticMoleculeDataset(num_samples=3, node_dim=6, min_nodes=4, max_nodes=6, seed=11)
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
    dataset = SyntheticMoleculeDataset(num_samples=8, node_dim=5, min_nodes=4, max_nodes=6, seed=31)
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
    assert not hasattr(args, "model")
    assert not hasattr(args, "moment_sinkhorn_iterations")


def test_benchmark_cli_represents_real_batches() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "bench_attention.py"
    symbols = runpy.run_path(script)
    args = symbols["parse_args"]([])

    assert args.graphs == [1, 8, 32]
    assert args.nodes_per_graph == [16, 32]


def test_cpu_ci_runs_the_project_fast_gate() -> None:
    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"

    assert workflow.is_file()
    assert "scripts/check.sh fast" in workflow.read_text()
