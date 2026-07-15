import torch
import pytest

from equivariant_attention import CartesianIrreps, RichEquivariantAttention, RichEquivariantAttentionConfig


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


def _rotate_tensor(tensor: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ab,...bc,dc->...ad", rotation, tensor, rotation)


def _make_inputs(dtype: torch.dtype = torch.float64):
    torch.manual_seed(31)
    node_feats = torch.randn(8, 6, dtype=dtype)
    pos = torch.randn(8, 3, dtype=dtype)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    return node_feats, pos, batch


def _make_neighbors(batch: torch.Tensor, width: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    masks = []
    for i, graph_id in enumerate(batch.tolist()):
        members = torch.nonzero(batch == graph_id, as_tuple=False).flatten().tolist()
        start = members.index(i)
        ordered = members[start:] + members[:start]
        selected = ordered[:width]
        mask = [True] * len(selected)
        while len(selected) < width:
            selected.append(i)
            mask.append(False)
        rows.append(selected)
        masks.append(mask)
    return torch.tensor(rows, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)


def _permute_neighbors(neighbor_index: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    inv_perm = torch.argsort(perm)
    return inv_perm[neighbor_index[perm]]


def _make_model(
    mode: str = "local",
    dtype: torch.dtype = torch.float64,
    *,
    vector_edge_bias: bool = False,
    vector_edge_bias_scale: float = 0.1,
) -> RichEquivariantAttention:
    torch.manual_seed(37)
    model = RichEquivariantAttention(
        RichEquivariantAttentionConfig(
            node_dim=6,
            hidden_irreps="12x0e + 4x1o + 3x2e",
            output_irreps="2x0e + 2x1o + 1x2e",
            num_layers=2,
            num_heads=3,
            attention_mode=mode,
            vector_edge_bias=vector_edge_bias,
            vector_edge_bias_scale=vector_edge_bias_scale,
        )
    )
    return model.to(dtype=dtype).eval()


def _activate_vector_edge_bias(model: RichEquivariantAttention) -> None:
    for layer in model.layers:
        layer.edge_vector_bias_weight.data[:, 0] = 0.05
        layer.edge_vector_bias_weight.data[:, 1] = 0.20
        layer.edge_vector_bias_weight.data[:, 2] = -0.15
        layer.edge_vector_bias_weight.data[:, 3] = 0.10
        layer.edge_vector_bias_offset.data.fill_(0.01)


def _max_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def test_cartesian_irreps_parse_and_dims() -> None:
    irreps = CartesianIrreps.parse("8x0e + 4x1o + 2x2e")

    assert irreps.scalars == 8
    assert irreps.vectors == 4
    assert irreps.tensors == 2
    assert irreps.dim == 8 + 4 * 3 + 2 * 5
    assert irreps.storage_dim == 8 + 4 * 3 + 2 * 9
    assert str(irreps) == "8x0e + 4x1o + 2x2e"


def test_cartesian_irreps_rejects_unsupported_terms() -> None:
    for spec in ["", "4x3e", "1x1e + bad", "-1x0e"]:
        with pytest.raises(ValueError):
            CartesianIrreps.parse(spec)


def test_rich_forward_contract_with_explicit_irreps() -> None:
    model = _make_model("local")
    node_feats, pos, batch = _make_inputs()
    neighbor_index, neighbor_mask = _make_neighbors(batch)

    out = model(node_feats, pos, batch=batch, neighbor_index=neighbor_index, neighbor_mask=neighbor_mask)

    assert out["node_scalars"].shape == (8, 2)
    assert out["node_vectors"].shape == (8, 2, 3)
    assert out["node_tensors"].shape == (8, 1, 3, 3)
    assert out["graph_scalars"].shape == (2, 2)
    assert out["graph_vectors"].shape == (2, 2, 3)
    assert out["graph_tensors"].shape == (2, 1, 3, 3)
    assert out["output_irreps"] == "2x0e + 2x1o + 1x2e"


def test_rich_vector_edge_bias_changes_local_attention() -> None:
    model = _make_model("local", vector_edge_bias=True, vector_edge_bias_scale=0.5)
    node_feats, pos, batch = _make_inputs()
    neighbor_index, neighbor_mask = _make_neighbors(batch)
    before = model(node_feats, pos, batch=batch, neighbor_index=neighbor_index, neighbor_mask=neighbor_mask)

    _activate_vector_edge_bias(model)
    after = model(node_feats, pos, batch=batch, neighbor_index=neighbor_index, neighbor_mask=neighbor_mask)

    assert _max_error(after["node_scalars"], before["node_scalars"]) > 1e-8


@pytest.mark.parametrize("mode", ["linear", "local"])
def test_rich_rotation_translation_equivariance(mode: str) -> None:
    model = _make_model(mode)
    node_feats, pos, batch = _make_inputs()
    neighbor_index, neighbor_mask = _make_neighbors(batch)
    kwargs = {"neighbor_index": neighbor_index, "neighbor_mask": neighbor_mask} if mode == "local" else {}
    out = model(node_feats, pos, batch=batch, **kwargs)

    torch.manual_seed(41)
    rotation = _random_rotation(pos.dtype)
    translation = torch.randn(1, 3, dtype=pos.dtype)
    moved = model(node_feats, pos @ rotation.T + translation, batch=batch, **kwargs)

    assert _max_error(moved["node_scalars"], out["node_scalars"]) < 1e-6
    assert _max_error(moved["graph_scalars"], out["graph_scalars"]) < 1e-6
    assert _max_error(moved["node_vectors"], torch.einsum("nca,ba->ncb", out["node_vectors"], rotation)) < 1e-6
    assert _max_error(moved["graph_vectors"], torch.einsum("gca,ba->gcb", out["graph_vectors"], rotation)) < 1e-6
    assert _max_error(moved["node_tensors"], _rotate_tensor(out["node_tensors"], rotation)) < 1e-6
    assert _max_error(moved["graph_tensors"], _rotate_tensor(out["graph_tensors"], rotation)) < 1e-6


def test_rich_vector_edge_bias_rotation_translation_equivariance() -> None:
    model = _make_model("local", vector_edge_bias=True, vector_edge_bias_scale=0.5)
    _activate_vector_edge_bias(model)
    node_feats, pos, batch = _make_inputs()
    neighbor_index, neighbor_mask = _make_neighbors(batch)
    out = model(node_feats, pos, batch=batch, neighbor_index=neighbor_index, neighbor_mask=neighbor_mask)

    torch.manual_seed(43)
    rotation = _random_rotation(pos.dtype)
    translation = torch.randn(1, 3, dtype=pos.dtype)
    moved = model(
        node_feats,
        pos @ rotation.T + translation,
        batch=batch,
        neighbor_index=neighbor_index,
        neighbor_mask=neighbor_mask,
    )

    assert _max_error(moved["node_scalars"], out["node_scalars"]) < 1e-6
    assert _max_error(moved["graph_scalars"], out["graph_scalars"]) < 1e-6
    assert _max_error(moved["node_vectors"], torch.einsum("nca,ba->ncb", out["node_vectors"], rotation)) < 1e-6
    assert _max_error(moved["graph_vectors"], torch.einsum("gca,ba->gcb", out["graph_vectors"], rotation)) < 1e-6
    assert _max_error(moved["node_tensors"], _rotate_tensor(out["node_tensors"], rotation)) < 1e-6
    assert _max_error(moved["graph_tensors"], _rotate_tensor(out["graph_tensors"], rotation)) < 1e-6


def test_rich_linear_batch_matches_independent_graphs() -> None:
    model = _make_model("linear")
    node_feats, pos, batch = _make_inputs()
    batched = model(node_feats, pos, batch=batch)

    node_scalars = []
    node_vectors = []
    node_tensors = []
    graph_scalars = []
    graph_vectors = []
    graph_tensors = []
    for graph_id in torch.unique(batch, sorted=True):
        idx = batch.eq(graph_id)
        out = model(node_feats[idx], pos[idx])
        node_scalars.append(out["node_scalars"])
        node_vectors.append(out["node_vectors"])
        node_tensors.append(out["node_tensors"])
        graph_scalars.append(out["graph_scalars"])
        graph_vectors.append(out["graph_vectors"])
        graph_tensors.append(out["graph_tensors"])

    assert _max_error(batched["node_scalars"], torch.cat(node_scalars, dim=0)) < 1e-6
    assert _max_error(batched["node_vectors"], torch.cat(node_vectors, dim=0)) < 1e-6
    assert _max_error(batched["node_tensors"], torch.cat(node_tensors, dim=0)) < 1e-6
    assert _max_error(batched["graph_scalars"], torch.cat(graph_scalars, dim=0)) < 1e-6
    assert _max_error(batched["graph_vectors"], torch.cat(graph_vectors, dim=0)) < 1e-6
    assert _max_error(batched["graph_tensors"], torch.cat(graph_tensors, dim=0)) < 1e-6


def test_rich_local_permutation_consistency() -> None:
    model = _make_model("local")
    node_feats, pos, batch = _make_inputs()
    neighbor_index, neighbor_mask = _make_neighbors(batch)
    out = model(node_feats, pos, batch=batch, neighbor_index=neighbor_index, neighbor_mask=neighbor_mask)

    perm = torch.tensor([3, 0, 6, 4, 1, 7, 2, 5], dtype=torch.long)
    inv_perm = torch.argsort(perm)
    moved = model(
        node_feats[perm],
        pos[perm],
        batch=batch[perm],
        neighbor_index=_permute_neighbors(neighbor_index, perm),
        neighbor_mask=neighbor_mask[perm],
    )

    assert _max_error(moved["node_scalars"][inv_perm], out["node_scalars"]) < 1e-6
    assert _max_error(moved["node_vectors"][inv_perm], out["node_vectors"]) < 1e-6
    assert _max_error(moved["node_tensors"][inv_perm], out["node_tensors"]) < 1e-6


def test_rich_vector_edge_bias_permutation_consistency() -> None:
    model = _make_model("local", vector_edge_bias=True, vector_edge_bias_scale=0.5)
    _activate_vector_edge_bias(model)
    node_feats, pos, batch = _make_inputs()
    neighbor_index, neighbor_mask = _make_neighbors(batch)
    out = model(node_feats, pos, batch=batch, neighbor_index=neighbor_index, neighbor_mask=neighbor_mask)

    perm = torch.tensor([3, 0, 6, 4, 1, 7, 2, 5], dtype=torch.long)
    inv_perm = torch.argsort(perm)
    moved = model(
        node_feats[perm],
        pos[perm],
        batch=batch[perm],
        neighbor_index=_permute_neighbors(neighbor_index, perm),
        neighbor_mask=neighbor_mask[perm],
    )

    assert _max_error(moved["node_scalars"][inv_perm], out["node_scalars"]) < 1e-6
    assert _max_error(moved["node_vectors"][inv_perm], out["node_vectors"]) < 1e-6
    assert _max_error(moved["node_tensors"][inv_perm], out["node_tensors"]) < 1e-6


def test_rich_vector_edge_bias_requires_local_vector_hidden() -> None:
    with pytest.raises(ValueError, match="vector_edge_bias requires local attention"):
        RichEquivariantAttention(RichEquivariantAttentionConfig(node_dim=3, attention_mode="linear", vector_edge_bias=True))

    with pytest.raises(ValueError, match="vector_edge_bias requires vector channels"):
        RichEquivariantAttention(
            RichEquivariantAttentionConfig(
                node_dim=3,
                hidden_irreps="8x0e + 2x2e",
                attention_mode="local",
                vector_edge_bias=True,
            )
        )
