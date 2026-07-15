import pytest
import torch

from equivariant_attention import EquivariantAttention, EquivariantAttentionConfig
from equivariant_attention.irreps import CartesianIrreps
from equivariant_attention.moment import (
    _bounded_kernel_scale,
    _bounded_irrep,
    _factorized_moment_attention,
    _st_features_to_matrix,
    _structured_key_mass,
    _structured_numerator,
    _symmetric_traceless_cross_features,
    _symmetric_traceless_features,
    _unit_ball,
)


def _make_model() -> EquivariantAttention:
    torch.manual_seed(101)
    return EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=6,
            hidden_irreps="12x0e + 3x1o",
            output_irreps="2x0e + 2x1o + 1x2e",
            num_layers=2,
            num_heads=3,
        )
    ).to(dtype=torch.float64).eval()


def _make_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(103)
    return (
        torch.randn(9, 6, dtype=torch.float64),
        torch.randn(9, 3, dtype=torch.float64),
        torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1]),
    )


def _orthogonal_matrix(*, reflection: bool = False) -> torch.Tensor:
    matrix, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if reflection and torch.linalg.det(matrix) > 0:
        matrix[:, 0] *= -1
    if not reflection and torch.linalg.det(matrix) < 0:
        matrix[:, 0] *= -1
    return matrix


def _rotate_tensor(value: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ab,...bc,dc->...ad", rotation, value, rotation)


def _max_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).detach().abs().max())


@pytest.mark.parametrize("reflection", [False, True])
def test_attention_is_o3_equivariant_and_translation_invariant(reflection: bool) -> None:
    model = _make_model()
    node_feats, pos, batch = _make_inputs()
    reference = model(node_feats, pos, batch=batch)
    transform = _orthogonal_matrix(reflection=reflection)
    translation = torch.randn(1, 3, dtype=torch.float64)

    moved = model(node_feats, pos @ transform.T + translation, batch=batch)

    assert _max_error(moved["node_scalars"], reference["node_scalars"]) < 1e-6
    assert _max_error(moved["graph_scalars"], reference["graph_scalars"]) < 1e-6
    assert _max_error(
        moved["node_vectors"],
        torch.einsum("nca,ba->ncb", reference["node_vectors"], transform),
    ) < 1e-6
    assert _max_error(
        moved["graph_vectors"],
        torch.einsum("gca,ba->gcb", reference["graph_vectors"], transform),
    ) < 1e-6
    assert _max_error(moved["node_tensors"], _rotate_tensor(reference["node_tensors"], transform)) < 1e-6
    assert _max_error(moved["graph_tensors"], _rotate_tensor(reference["graph_tensors"], transform)) < 1e-6


def test_attention_is_permutation_and_batch_consistent() -> None:
    model = _make_model()
    node_feats, pos, batch = _make_inputs()
    reference = model(node_feats, pos, batch=batch)
    permutation = torch.tensor([3, 0, 7, 5, 1, 8, 2, 6, 4])
    inverse = torch.argsort(permutation)

    permuted = model(node_feats[permutation], pos[permutation], batch=batch[permutation])

    for key in ("node_scalars", "node_vectors", "node_tensors"):
        assert _max_error(permuted[key][inverse], reference[key]) < 1e-6

    independent = [model(node_feats[batch == graph], pos[batch == graph]) for graph in range(2)]
    for key in ("node_scalars", "node_vectors", "node_tensors"):
        assert _max_error(reference[key], torch.cat([output[key] for output in independent])) < 1e-6
    for key in ("graph_scalars", "graph_vectors", "graph_tensors"):
        assert _max_error(reference[key], torch.cat([output[key] for output in independent])) < 1e-6


def test_forward_shapes_and_tensor_subspace() -> None:
    model = _make_model()
    node_feats, pos, batch = _make_inputs()

    output = model(node_feats, pos, batch=batch)

    assert output["node_scalars"].shape == (9, 2)
    assert output["node_vectors"].shape == (9, 2, 3)
    assert output["node_tensors"].shape == (9, 1, 3, 3)
    assert output["graph_scalars"].shape == (2, 2)
    assert output["graph_vectors"].shape == (2, 2, 3)
    assert output["graph_tensors"].shape == (2, 1, 3, 3)
    tensor = output["node_tensors"]
    assert _max_error(tensor, tensor.transpose(-1, -2)) < 1e-12
    trace_error = tensor.diagonal(dim1=-2, dim2=-1).sum(dim=-1).detach().abs().max()
    assert float(trace_error) < 1e-12


@pytest.mark.parametrize("balanced", [False, True])
def test_structured_attention_matches_explicit_dense_kernel(balanced: bool) -> None:
    torch.manual_seed(107)
    query_scalar = torch.rand(7, 2, 4, dtype=torch.float64)
    key_scalar = torch.rand(7, 2, 4, dtype=torch.float64)
    query_vector = _unit_ball(torch.randn(7, 2, 3, dtype=torch.float64), eps=1e-12)
    key_vector = _unit_ball(torch.randn(7, 2, 3, dtype=torch.float64), eps=1e-12)
    kernel_scale = torch.tensor([0.2, 0.7], dtype=torch.float64)
    value = torch.randn(7, 2, 5, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])

    actual = _factorized_moment_attention(
        query_scalar,
        key_scalar,
        query_vector,
        key_vector,
        kernel_scale,
        value,
        batch,
        num_graphs=2,
        balanced=balanced,
    )
    expected = torch.empty_like(value)
    for graph in range(2):
        index = batch == graph
        content = torch.einsum("ihd,jhd->hij", query_scalar[index], key_scalar[index])
        angular_dot = torch.einsum("iha,jha->hij", query_vector[index], key_vector[index])
        kernel = content + 1.0 + kernel_scale[:, None, None] * angular_dot.square()
        if balanced:
            kernel = kernel / kernel.sum(dim=1, keepdim=True)
        weights = kernel / kernel.sum(dim=2, keepdim=True)
        expected[index] = torch.einsum("hij,jhf->ihf", weights, value[index])

    assert _max_error(actual, expected) < 1e-10


def test_structured_mass_avoids_orthogonal_quadratic_cancellation() -> None:
    node_count = 10_000
    direction_q = torch.tensor([1.0, 1.0, 0.0]) / torch.sqrt(torch.tensor(2.0))
    direction_k = torch.tensor([1.0, -1.0, 0.0]) / torch.sqrt(torch.tensor(2.0))
    query_vector = _unit_ball((100.0 * direction_q).expand(node_count, 1, 3), eps=1e-12)
    key_vector = _unit_ball((100.0 * direction_k).expand(node_count, 1, 3), eps=1e-12)
    query_scalar = torch.zeros(node_count, 1, 1)
    key_scalar = torch.zeros_like(query_scalar)
    row_scale = torch.ones(node_count, 1)
    batch = torch.zeros(node_count, dtype=torch.long)

    mass = _structured_key_mass(
        query_scalar,
        key_scalar,
        query_vector,
        key_vector,
        torch.ones(1),
        row_scale,
        batch,
        num_graphs=1,
    )

    assert torch.isfinite(mass).all()
    assert torch.allclose(mass, torch.full_like(mass, float(node_count)), atol=2e-3, rtol=1e-6)


def test_vector_kernel_components_are_bounded() -> None:
    value = torch.tensor([[[1e6, -1e6, 1e6]]], dtype=torch.float64)
    bounded = _unit_ball(value, eps=1e-12)
    interior_scale = _bounded_kernel_scale(
        torch.tensor([-20.0, 0.0, 20.0], dtype=torch.float64),
        maximum=0.75,
    )
    saturated_value = torch.tensor([[[1e6, 0.0, 0.0]]], dtype=torch.float32)
    saturated_bound = _unit_ball(saturated_value, eps=1e-12)
    saturated_scale = _bounded_kernel_scale(torch.tensor([-100.0, 100.0]), maximum=0.75)

    assert float(bounded.norm(dim=-1).max()) < 1.0
    assert torch.all(interior_scale > 0.0)
    assert torch.all(interior_scale < 0.75)
    assert torch.all(saturated_bound.norm(dim=-1) <= 1.0)
    assert torch.all(saturated_scale >= 0.0)
    assert torch.all(saturated_scale <= 0.75)

    model = _make_model()
    initialized = _bounded_kernel_scale(
        model.layers[0].raw_vector_kernel,
        model.layers[0].vector_kernel_max,
    )
    assert torch.allclose(initialized, torch.full_like(initialized, 0.05))


def test_structured_numerator_preserves_signed_values() -> None:
    one = torch.ones(1, 1, 1, dtype=torch.float64)
    vector = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float64)
    numerator = _structured_numerator(
        one,
        one,
        vector,
        vector,
        torch.ones(1, dtype=torch.float64),
        torch.ones(1, 1, dtype=torch.float64),
        torch.full((1, 1, 1), -1.0, dtype=torch.float64),
        torch.zeros(1, dtype=torch.long),
        num_graphs=1,
    )

    assert torch.equal(numerator, torch.full_like(numerator, -3.0))


def test_exact_relative_second_moment_identity() -> None:
    torch.manual_seed(113)
    position = torch.randn(4, 3, dtype=torch.float64)
    weights = torch.rand(2, 4, 4, dtype=torch.float64)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    gate = torch.randn(2, 4, dtype=torch.float64)
    st_position = _symmetric_traceless_features(position)
    mass = torch.einsum("hij,hj->ih", weights, gate)
    first = torch.einsum("hij,hj,ja->iha", weights, gate, position)
    second = torch.einsum("hij,hj,jf->ihf", weights, gate, st_position)
    factorized = (
        second
        + st_position[:, None, :] * mass[..., None]
        - 2.0 * _symmetric_traceless_cross_features(first, position[:, None, :])
    )
    relative = position[None, :, :] - position[:, None, :]
    dense = torch.einsum(
        "hij,hj,ijf->ihf",
        weights,
        gate,
        _symmetric_traceless_features(relative),
    )

    assert _max_error(factorized, dense) < 1e-10


def test_symmetric_traceless_feature_storage_is_exact() -> None:
    value = torch.randn(8, 3, dtype=torch.float64)
    matrix = _st_features_to_matrix(_symmetric_traceless_features(value))
    direct = value[..., :, None] * value[..., None, :]
    direct = direct - value.square().sum(dim=-1)[..., None, None] * torch.eye(3) / 3.0

    assert _max_error(matrix, direct) < 1e-12


@pytest.mark.parametrize(
    "config, message",
    [
        (EquivariantAttentionConfig(node_dim=0), "node_dim"),
        (EquivariantAttentionConfig(node_dim=4, num_layers=0), "num_layers"),
        (EquivariantAttentionConfig(node_dim=4, num_heads=0), "num_heads"),
        (EquivariantAttentionConfig(node_dim=4, vector_kernel_max=0.0), "vector_kernel_max"),
        (
            EquivariantAttentionConfig(
                node_dim=4,
                vector_kernel_init=1.0,
                vector_kernel_max=1.0,
            ),
            "smaller",
        ),
        (EquivariantAttentionConfig(node_dim=4, hidden_irreps="7x0e + 2x1o", num_heads=2), "divisible"),
        (EquivariantAttentionConfig(node_dim=4, hidden_irreps="8x0e + 1x2e"), "scalar and vector"),
    ],
)
def test_config_rejects_invalid_architectures(config: EquivariantAttentionConfig, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        EquivariantAttention(config)


@pytest.mark.parametrize(
    "irreps",
    [
        "1x1e",
        "1x1o + 1x1e",
        "1x1e + 1x1o",
        "1x0o",
        "1x2o",
    ],
)
def test_cartesian_irreps_rejects_unsupported_parity(irreps: str) -> None:
    with pytest.raises(ValueError, match="supports only"):
        CartesianIrreps.parse(irreps)


def test_cartesian_irreps_direct_constructor_validates_fields() -> None:
    with pytest.raises(ValueError, match="nonnegative integers"):
        CartesianIrreps(vectors=-1)
    with pytest.raises(ValueError, match="parity"):
        CartesianIrreps(vector_parity="bad")


def test_bounded_irrep_uses_float32_for_fp16_norm_reduction() -> None:
    value = torch.full((2, 3, 3), 300.0, dtype=torch.float16)

    actual = _bounded_irrep(value, eps=1e-12)
    expected = _bounded_irrep(value.float(), eps=1e-12).half()

    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual) == actual.numel()
    assert torch.equal(actual, expected)
