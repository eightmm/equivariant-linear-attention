from pathlib import Path
import runpy

import pytest
import torch

from equivariant_attention.baselines import EGNNBaseline, EGNNBaselineConfig
from equivariant_attention.benchmarking import GraphSample, SyntheticMoleculeDataset, collate_graphs, split_dataset
from equivariant_attention.moment import EquivariantMomentAttentionConfig
from equivariant_attention.training import build_regression_model, evaluate_regression, fit_target_normalizer, train_regression_step


def _random_rotation(dtype: torch.dtype) -> torch.Tensor:
    q = torch.randn(4, dtype=dtype)
    q = q / q.norm()
    w, x, y, z = q
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]),
            torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
            torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]),
        ]
    )


def test_synthetic_dataset_collates_full_neighbors() -> None:
    dataset = SyntheticMoleculeDataset(num_samples=3, node_dim=6, min_nodes=4, max_nodes=6, seed=11)
    batch = collate_graphs([dataset[0], dataset[1]], max_neighbors=None)

    assert batch.node_feats.ndim == 2
    assert batch.pos.shape == (batch.node_feats.shape[0], 3)
    assert batch.target.shape == (2, 1)
    assert batch.neighbor_index.shape == batch.neighbor_mask.shape
    assert batch.neighbor_mask.any(dim=1).all()
    assert batch.batch.max().item() == 1


def test_synthetic_dataset_collates_empty_neighbors_for_global_models() -> None:
    dataset = SyntheticMoleculeDataset(num_samples=2, node_dim=6, min_nodes=4, max_nodes=6, seed=12)
    batch = collate_graphs([dataset[0], dataset[1]], max_neighbors=0)

    assert batch.neighbor_index.shape == (batch.node_feats.shape[0], 0)
    assert batch.neighbor_mask.shape == (batch.node_feats.shape[0], 0)


def test_split_dataset_is_deterministic_and_disjoint() -> None:
    dataset = SyntheticMoleculeDataset(num_samples=12, node_dim=4, seed=13)
    train_a, val_a, test_a = split_dataset(dataset, train_size=6, val_size=3, seed=17)
    train_b, val_b, test_b = split_dataset(dataset, train_size=6, val_size=3, seed=17)

    assert train_a == train_b
    assert val_a == val_b
    assert test_a == test_b
    assert set(train_a).isdisjoint(val_a)
    assert set(train_a).isdisjoint(test_a)
    assert set(val_a).isdisjoint(test_a)


def test_egnn_graph_scalar_rotation_translation_invariance() -> None:
    torch.manual_seed(19)
    dataset = SyntheticMoleculeDataset(num_samples=2, node_dim=5, min_nodes=5, max_nodes=5, seed=23)
    batch = collate_graphs([dataset[0], dataset[1]], max_neighbors=None)
    model = EGNNBaseline(EGNNBaselineConfig(node_dim=5, hidden_dim=16, num_layers=2)).to(dtype=torch.float64).eval()
    out = model(batch.node_feats.double(), batch.pos.double(), batch=batch.batch, neighbor_index=batch.neighbor_index)

    rotation = _random_rotation(torch.float64)
    translation = torch.randn(1, 3, dtype=torch.float64)
    moved = model(
        batch.node_feats.double(),
        batch.pos.double() @ rotation.T + translation,
        batch=batch.batch,
        neighbor_index=batch.neighbor_index,
    )

    assert (moved["graph_scalar"] - out["graph_scalar"]).abs().max().item() < 1e-6


def test_regression_train_and_eval_smoke() -> None:
    torch.manual_seed(29)
    dataset = SyntheticMoleculeDataset(num_samples=8, node_dim=5, min_nodes=4, max_nodes=6, seed=31)
    batch = collate_graphs([dataset[i] for i in range(4)], max_neighbors=4)
    model = EGNNBaseline(EGNNBaselineConfig(node_dim=5, hidden_dim=16, num_layers=2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    normalizer = fit_target_normalizer(dataset[i] for i in range(4))
    loss = train_regression_step(model, batch, optimizer, target_normalizer=normalizer)
    metrics = evaluate_regression(model, [batch], target_normalizer=normalizer)

    assert torch.isfinite(torch.tensor(loss))
    assert metrics["mae"] >= 0.0
    assert metrics["rmse"] >= 0.0


@pytest.mark.parametrize("model_name", ["rich_local", "rich_linear", "rich_linear_light", "moment_linear"])
def test_rich_regression_default_lr_stays_finite_on_scaled_geometry(model_name: str) -> None:
    torch.manual_seed(101)
    base = SyntheticMoleculeDataset(num_samples=16, node_dim=5, min_nodes=4, max_nodes=8, seed=103)
    dataset = [GraphSample(sample.node_feats, sample.pos * 3.0, sample.target, sample.sample_id) for sample in base]
    model = build_regression_model(model_name, node_dim=5, hidden_dim=32, num_layers=3, num_heads=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    normalizer = fit_target_normalizer(dataset[i] for i in range(8))
    max_neighbors = 0 if model_name.startswith("rich_linear") or model_name == "moment_linear" else None

    losses = []
    for step in range(8):
        batch = collate_graphs([dataset[(step * 4 + i) % 8] for i in range(4)], max_neighbors=max_neighbors)
        losses.append(train_regression_step(model, batch, optimizer, target_normalizer=normalizer))

    loss_tensor = torch.tensor(losses)
    assert torch.isfinite(loss_tensor).all()
    assert loss_tensor.max().item() < 100.0


def test_moment_regression_builder_enables_enhanced_features() -> None:
    model = build_regression_model(
        "moment_linear",
        node_dim=5,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        moment_radial_trace=True,
        moment_full_gram_invariants=True,
        moment_shifted_angular_kernel=True,
        moment_learnable_balance_exponent=True,
        moment_radial_distance_kernel=True,
        moment_dynamic_moment_routing=True,
        moment_sinkhorn_iterations=2,
        moment_equivariant_ffn=True,
    )

    assert model.config.radial_trace
    assert model.config.full_gram_invariants
    assert model.config.shifted_angular_kernel
    assert model.config.learnable_balance_exponent
    assert model.config.radial_distance_kernel
    assert model.config.dynamic_moment_routing
    assert model.config.sinkhorn_iterations == 2
    assert model.config.equivariant_ffn
    assert model.config.ffn_hidden_ratio == 2.0


def test_moment_regression_builder_uses_ratio_two_ffn_by_default() -> None:
    model = build_regression_model("moment_linear", node_dim=5, hidden_dim=32, num_layers=2, num_heads=4)

    assert model.config.equivariant_ffn
    assert model.config.ffn_hidden_ratio == 2.0


def test_moment_config_and_cli_use_ratio_two_ffn_by_default() -> None:
    config = EquivariantMomentAttentionConfig(node_dim=5)
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_compare.py"
    args = runpy.run_path(script)["parse_args"]([])

    assert config.equivariant_ffn
    assert config.ffn_hidden_ratio == 2.0
    assert args.moment_equivariant_ffn
    assert args.moment_ffn_hidden_ratio == 2.0
    assert not args.moment_radial_distance_kernel
    assert not args.moment_dynamic_moment_routing
    assert config.sinkhorn_iterations == 1
    assert args.moment_sinkhorn_iterations == 1
    assert args.moment_radial_distance_shift_init == 1.1
    assert not args.skip_test_eval


def test_moment_regression_builder_enables_radial_distance_kernel() -> None:
    model = build_regression_model(
        "moment_linear",
        node_dim=5,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        moment_radial_distance_kernel=True,
        moment_radial_distance_shift_init=1.2,
    )

    assert model.config.radial_distance_kernel
    assert model.config.radial_distance_shift_init == 1.2


def test_moment_regression_builder_enables_dynamic_moment_routing() -> None:
    model = build_regression_model(
        "moment_linear",
        node_dim=5,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        moment_dynamic_moment_routing=True,
    )

    assert model.config.dynamic_moment_routing


def test_moment_regression_builder_enables_iterative_sinkhorn() -> None:
    model = build_regression_model(
        "moment_linear",
        node_dim=5,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        moment_sinkhorn_iterations=2,
    )

    assert model.config.sinkhorn_iterations == 2


def test_target_normalizer_round_trip() -> None:
    dataset = SyntheticMoleculeDataset(num_samples=5, node_dim=5, seed=37)
    normalizer = fit_target_normalizer(dataset[i] for i in [0, 1, 2])
    target = dataset[3].target.reshape(1, -1)

    assert torch.allclose(normalizer.inverse(normalizer.transform(target)), target)
