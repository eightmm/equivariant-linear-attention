import torch
import pytest

from equivariant_attention import EquivariantAttention, EquivariantAttentionConfig, prepare_for_inference
from equivariant_attention.backends import SphericalHarmonicsBackend

ATTENTION_MODES = ["dense", "linear", "linear_sh", "local"]


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


def _make_inputs(dtype: torch.dtype = torch.float64):
    torch.manual_seed(7)
    node_feats = torch.randn(7, 5, dtype=dtype)
    pos = torch.randn(7, 3, dtype=dtype)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    edge_feats = torch.randn(7, 7, 2, dtype=dtype)
    return node_feats, pos, edge_feats, batch


def _make_model(
    dtype: torch.dtype = torch.float64,
    attention_mode: str = "dense",
) -> EquivariantAttention:
    torch.manual_seed(11)
    edge_dim = 2 if attention_mode == "dense" else 0
    model = EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=5,
            edge_dim=edge_dim,
            hidden_dim=16,
            num_layers=2,
            num_heads=4,
            preferred_backend="cuequivariance",
            attention_mode=attention_mode,
            local_radius=4.0,
            max_neighbors=4,
        )
    )
    return model.to(dtype=dtype).eval()


def _max_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def _make_neighbor_index(batch: torch.Tensor, width: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    masks = []
    for i, graph_id in enumerate(batch.tolist()):
        members = torch.nonzero(batch == graph_id, as_tuple=False).flatten()
        local = members.tolist()
        start = local.index(i)
        ordered = local[start:] + local[:start]
        selected = ordered[:width]
        mask = [True] * len(selected)
        while len(selected) < width:
            selected.append(i)
            mask.append(False)
        rows.append(selected)
        masks.append(mask)
    return torch.tensor(rows, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)


def _permute_neighbor_index(neighbor_index: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    inv_perm = torch.argsort(perm)
    return inv_perm[neighbor_index[perm]]


def _rotate_tensor(tensor: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ab,...bc,dc->...ad", rotation, tensor, rotation)


@pytest.mark.parametrize("attention_mode", ATTENTION_MODES)
def test_forward_contract(attention_mode: str) -> None:
    model = _make_model(attention_mode=attention_mode)
    node_feats, pos, edge_feats, batch = _make_inputs()
    edge_feats = edge_feats if attention_mode == "dense" else None

    out = model(node_feats, pos, edge_feats=edge_feats, batch=batch)

    assert set(out) >= {
        "node_scalar",
        "node_vector",
        "node_tensor",
        "graph_scalar",
        "graph_vector",
        "graph_tensor",
        "backend",
        "attention_mode",
    }
    assert out["node_scalar"].shape == (7, 1)
    assert out["node_vector"].shape == (7, 3)
    assert out["node_tensor"].shape == (7, 3, 3)
    assert out["graph_scalar"].shape == (2, 1)
    assert out["graph_vector"].shape == (2, 3)
    assert out["graph_tensor"].shape == (2, 3, 3)
    assert isinstance(out["backend"], str)
    assert out["backend"] == "cuequivariance"
    assert out["attention_mode"] == attention_mode


def test_forward_without_edges_or_batch_uses_single_global_graph() -> None:
    torch.manual_seed(17)
    model = EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=4,
            edge_dim=0,
            hidden_dim=8,
            num_layers=1,
            num_heads=2,
            attention_mode="linear_sh",
        )
    ).to(dtype=torch.float64)

    out = model(
        torch.randn(5, 4, dtype=torch.float64),
        torch.randn(5, 3, dtype=torch.float64),
    )

    assert out["node_scalar"].shape == (5, 1)
    assert out["graph_scalar"].shape == (1, 1)
    assert torch.isfinite(out["node_vector"]).all()
    assert torch.isfinite(out["node_tensor"]).all()


@pytest.mark.parametrize("attention_mode", ATTENTION_MODES)
def test_rotation_translation_equivariance_error_below_1e_minus_6(attention_mode: str) -> None:
    model = _make_model(attention_mode=attention_mode)
    node_feats, pos, edge_feats, batch = _make_inputs()
    edge_feats = edge_feats if attention_mode == "dense" else None
    out = model(node_feats, pos, edge_feats=edge_feats, batch=batch)

    torch.manual_seed(13)
    rotation = _random_rotation(pos.dtype)
    translation = torch.randn(1, 3, dtype=pos.dtype)
    moved = model(
        node_feats,
        pos @ rotation.T + translation,
        edge_feats=edge_feats,
        batch=batch,
    )

    assert _max_error(moved["node_scalar"], out["node_scalar"]) < 1e-6
    assert _max_error(moved["graph_scalar"], out["graph_scalar"]) < 1e-6
    assert _max_error(moved["node_vector"], out["node_vector"] @ rotation.T) < 1e-6
    assert _max_error(moved["graph_vector"], out["graph_vector"] @ rotation.T) < 1e-6
    assert _max_error(moved["node_tensor"], _rotate_tensor(out["node_tensor"], rotation)) < 1e-6
    assert _max_error(moved["graph_tensor"], _rotate_tensor(out["graph_tensor"], rotation)) < 1e-6


def test_local_neighbor_index_rotation_translation_equivariance() -> None:
    model = _make_model(attention_mode="local")
    node_feats, pos, _edge_feats, batch = _make_inputs()
    neighbor_index, neighbor_mask = _make_neighbor_index(batch)
    out = model(node_feats, pos, batch=batch, neighbor_index=neighbor_index, neighbor_mask=neighbor_mask)

    torch.manual_seed(29)
    rotation = _random_rotation(pos.dtype)
    translation = torch.randn(1, 3, dtype=pos.dtype)
    moved = model(
        node_feats,
        pos @ rotation.T + translation,
        batch=batch,
        neighbor_index=neighbor_index,
        neighbor_mask=neighbor_mask,
    )

    assert _max_error(moved["node_scalar"], out["node_scalar"]) < 1e-6
    assert _max_error(moved["node_vector"], out["node_vector"] @ rotation.T) < 1e-6
    assert _max_error(moved["node_tensor"], _rotate_tensor(out["node_tensor"], rotation)) < 1e-6


@pytest.mark.parametrize("attention_mode", ATTENTION_MODES)
def test_permutation_consistency_error_below_1e_minus_6(attention_mode: str) -> None:
    model = _make_model(attention_mode=attention_mode)
    node_feats, pos, edge_feats, batch = _make_inputs()
    edge_feats = edge_feats if attention_mode == "dense" else None
    out = model(node_feats, pos, edge_feats=edge_feats, batch=batch)

    perm = torch.tensor([2, 0, 6, 4, 1, 5, 3], dtype=torch.long)
    inv_perm = torch.argsort(perm)
    moved = model(
        node_feats[perm],
        pos[perm],
        edge_feats=edge_feats[perm][:, perm] if edge_feats is not None else None,
        batch=batch[perm],
    )

    assert _max_error(moved["node_scalar"][inv_perm], out["node_scalar"]) < 1e-6
    assert _max_error(moved["node_vector"][inv_perm], out["node_vector"]) < 1e-6
    assert _max_error(moved["node_tensor"][inv_perm], out["node_tensor"]) < 1e-6
    assert _max_error(moved["graph_scalar"], out["graph_scalar"]) < 1e-6
    assert _max_error(moved["graph_vector"], out["graph_vector"]) < 1e-6
    assert _max_error(moved["graph_tensor"], out["graph_tensor"]) < 1e-6


def test_local_neighbor_index_permutation_consistency() -> None:
    model = _make_model(attention_mode="local")
    node_feats, pos, _edge_feats, batch = _make_inputs()
    neighbor_index, neighbor_mask = _make_neighbor_index(batch)
    out = model(node_feats, pos, batch=batch, neighbor_index=neighbor_index, neighbor_mask=neighbor_mask)

    perm = torch.tensor([2, 0, 6, 4, 1, 5, 3], dtype=torch.long)
    inv_perm = torch.argsort(perm)
    moved = model(
        node_feats[perm],
        pos[perm],
        batch=batch[perm],
        neighbor_index=_permute_neighbor_index(neighbor_index, perm),
        neighbor_mask=neighbor_mask[perm],
    )

    assert _max_error(moved["node_scalar"][inv_perm], out["node_scalar"]) < 1e-6
    assert _max_error(moved["node_vector"][inv_perm], out["node_vector"]) < 1e-6
    assert _max_error(moved["node_tensor"][inv_perm], out["node_tensor"]) < 1e-6


def test_invalid_config_rejected() -> None:
    bad_configs = [
        EquivariantAttentionConfig(node_dim=0),
        EquivariantAttentionConfig(node_dim=1, edge_dim=-1),
        EquivariantAttentionConfig(node_dim=1, hidden_dim=0),
        EquivariantAttentionConfig(node_dim=1, num_layers=0),
        EquivariantAttentionConfig(node_dim=1, num_heads=0),
        EquivariantAttentionConfig(node_dim=1, hidden_dim=10, num_heads=4),
        EquivariantAttentionConfig(node_dim=1, attention_mode="missing"),
        EquivariantAttentionConfig(node_dim=1, edge_dim=1, attention_mode="linear_sh"),
        EquivariantAttentionConfig(node_dim=1, edge_dim=1, attention_mode="local"),
        EquivariantAttentionConfig(node_dim=1, attention_mode="linear_sh", sh_lmax=3),
        EquivariantAttentionConfig(node_dim=1, attention_mode="local", local_radius=0.0),
        EquivariantAttentionConfig(node_dim=1, attention_mode="local", max_neighbors=0),
    ]
    for config in bad_configs:
        with pytest.raises(ValueError):
            EquivariantAttention(config)


def test_invalid_inputs_rejected() -> None:
    model = EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=3,
            edge_dim=1,
            hidden_dim=8,
            num_layers=1,
            num_heads=2,
            attention_mode="dense",
        )
    )
    node_feats = torch.randn(4, 3)
    pos = torch.randn(4, 3)
    edge_feats = torch.randn(4, 4, 1)

    with pytest.raises(ValueError, match="node_feats"):
        model(node_feats.unsqueeze(0), pos, edge_feats=edge_feats)
    with pytest.raises(ValueError, match="width"):
        model(torch.randn(4, 2), pos, edge_feats=edge_feats)
    with pytest.raises(ValueError, match="pos"):
        model(node_feats, torch.randn(4, 2), edge_feats=edge_feats)
    with pytest.raises(ValueError, match="at least one node"):
        model(torch.randn(0, 3), torch.randn(0, 3), edge_feats=torch.randn(0, 0, 1))
    with pytest.raises(TypeError, match="floating point"):
        model(node_feats.long(), pos, edge_feats=edge_feats)
    with pytest.raises(ValueError, match="finite"):
        model(node_feats.fill_(float("nan")), pos, edge_feats=edge_feats)
    with pytest.raises(ValueError, match="batch"):
        model(torch.randn(4, 3), pos, edge_feats=edge_feats, batch=torch.zeros(3, dtype=torch.long))
    with pytest.raises(ValueError, match="nonnegative"):
        model(torch.randn(4, 3), pos, edge_feats=edge_feats, batch=torch.tensor([0, 0, -1, 1]))
    with pytest.raises(ValueError, match="edge_feats"):
        model(torch.randn(4, 3), pos, edge_feats=torch.randn(4, 3, 1))
    with pytest.raises(ValueError, match="finite"):
        bad_edge = torch.randn(4, 4, 1)
        bad_edge[0, 0, 0] = float("inf")
        model(torch.randn(4, 3), pos, edge_feats=bad_edge)

    local = EquivariantAttention(
        EquivariantAttentionConfig(node_dim=3, hidden_dim=8, num_layers=1, num_heads=2, attention_mode="local")
    )
    with pytest.raises(ValueError, match="neighbor_index"):
        local(torch.randn(4, 3), pos, neighbor_index=torch.zeros(4, dtype=torch.long))
    with pytest.raises(ValueError, match="neighbor_mask"):
        local(torch.randn(4, 3), pos, neighbor_index=torch.zeros(4, 2, dtype=torch.long), neighbor_mask=torch.ones(4, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="range"):
        local(torch.randn(4, 3), pos, neighbor_index=torch.full((4, 2), 8, dtype=torch.long))


def test_edge_feature_contract_for_edge_dim_zero() -> None:
    model = EquivariantAttention(EquivariantAttentionConfig(node_dim=3, edge_dim=0, attention_mode="dense"))
    with pytest.raises(ValueError, match="edge_dim"):
        model(torch.randn(3, 3), torch.randn(3, 3), edge_feats=torch.randn(3, 3, 1))


def test_spherical_harmonics_backend_contracts() -> None:
    vectors = torch.randn(4, 3, dtype=torch.float64)

    for preferred in ("cuequivariance", "e3nn", "cartesian"):
        backend = SphericalHarmonicsBackend(preferred)
        out = backend(vectors)
        assert out.shape == (4, 4)
        assert torch.isfinite(out).all()

    backend = SphericalHarmonicsBackend("cartesian")
    assert backend(torch.empty(0, 3)).shape == (0, 4)
    assert SphericalHarmonicsBackend("cartesian", lmax=2)(vectors).shape == (4, 9)
    with pytest.raises(ValueError, match="shape"):
        backend(torch.randn(4, 2))
    with pytest.raises(ValueError, match="unknown backend"):
        SphericalHarmonicsBackend("missing")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lmax"):
        SphericalHarmonicsBackend("cartesian", lmax=3)


def test_prepare_for_inference_contract() -> None:
    model = EquivariantAttention(
        EquivariantAttentionConfig(node_dim=3, hidden_dim=8, num_layers=1, num_heads=2)
    )
    prepared = prepare_for_inference(model, device="cpu", dtype=torch.float32, compile_model=False)
    assert prepared.training is False
    assert all(not param.requires_grad for param in prepared.parameters())
