import inspect
import math

import pytest
import torch

from equivariant_attention import EquivariantAttention, EquivariantAttentionConfig
import equivariant_attention.moment as moment
from equivariant_attention.inference import _AutocastInferenceModule


def _attention_inputs(seed: int = 211) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    query_scalar = (
        moment._normalize_positive_features(
            torch.rand(7, 2, 4, dtype=torch.float64),
            eps=1e-12,
        )
        .detach()
        .requires_grad_()
    )
    key_scalar = (
        moment._normalize_positive_features(
            torch.rand(7, 2, 4, dtype=torch.float64),
            eps=1e-12,
        )
        .detach()
        .requires_grad_()
    )
    query_vector = (
        moment._unit_ball(
            torch.randn(7, 2, 3, dtype=torch.float64),
            eps=1e-12,
        )
        .detach()
        .requires_grad_()
    )
    key_vector = (
        moment._unit_ball(
            torch.randn(7, 2, 3, dtype=torch.float64),
            eps=1e-12,
        )
        .detach()
        .requires_grad_()
    )
    value = torch.randn(7, 2, 5, dtype=torch.float64, requires_grad=True)
    beta = torch.tensor([0.1, 0.3], dtype=torch.float64, requires_grad=True)
    gamma = torch.tensor([0.2, 0.7], dtype=torch.float64, requires_grad=True)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    return query_scalar, key_scalar, query_vector, key_vector, value, beta, gamma, batch


def _dense_attention(
    query_scalar: torch.Tensor,
    key_scalar: torch.Tensor,
    query_vector: torch.Tensor,
    key_vector: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    batch: torch.Tensor,
    *,
    balanced: bool,
    alignment: bool,
    kernel_floor: float = 0.5,
    kernel_floor_mode: str = "fixed",
) -> torch.Tensor:
    output = torch.empty_like(value)
    for graph in range(2):
        index = batch == graph
        content = torch.einsum("ihd,jhd->hij", query_scalar[index], key_scalar[index])
        angular = torch.einsum("iha,jha->hij", query_vector[index], key_vector[index])
        graph_scale = 1.0 if kernel_floor_mode == "fixed" else 1.0 / int(index.sum())
        pair_floor = kernel_floor * graph_scale
        kernel = (
            pair_floor
            + content
            + graph_scale * beta[:, None, None]
            + gamma[:, None, None] * angular.square()
        )
        if alignment:
            kernel = kernel + graph_scale * beta[:, None, None] * angular
        if balanced:
            kernel = kernel / kernel.sum(dim=1, keepdim=True)
        weights = kernel / kernel.sum(dim=2, keepdim=True)
        output[index] = torch.einsum("hij,jhf->ihf", weights, value[index])
    return output


@pytest.mark.parametrize("balanced", [False, True])
@pytest.mark.parametrize("alignment", [False, True])
def test_structured_gradients_match_dense_reference_with_isolated_alignment(
    balanced: bool,
    alignment: bool,
) -> None:
    actual_inputs = _attention_inputs()
    dense_inputs = tuple(
        value.detach().clone().requires_grad_(value.requires_grad)
        if isinstance(value, torch.Tensor) and torch.is_floating_point(value)
        else value.clone()
        if isinstance(value, torch.Tensor)
        else value
        for value in actual_inputs
    )
    q0, k0, q1, k1, value, beta, gamma, batch = actual_inputs
    dq0, dk0, dq1, dk1, dvalue, dbeta, dgamma, dbatch = dense_inputs
    beta_dot = beta if alignment else torch.zeros_like(beta)
    actual = moment._factorized_moment_attention(
        q0,
        k0,
        q1,
        k1,
        gamma,
        value,
        batch,
        num_graphs=2,
        balanced=balanced,
        alignment_scale=beta,
        alignment_dot_scale=beta_dot,
        kernel_floor=0.5,
    )
    expected = _dense_attention(
        dq0,
        dk0,
        dq1,
        dk1,
        dvalue,
        dbeta,
        dgamma,
        dbatch,
        balanced=balanced,
        alignment=alignment,
    )
    probe = torch.randn_like(actual)
    actual_grad = torch.autograd.grad(
        (actual * probe).sum(), (q0, k0, q1, k1, value, beta, gamma)
    )
    dense_grad = torch.autograd.grad(
        (expected * probe).sum(),
        (dq0, dk0, dq1, dk1, dvalue, dbeta, dgamma),
    )

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-9)
    for left, right in zip(actual_grad, dense_grad, strict=True):
        assert torch.allclose(left, right, atol=1e-10, rtol=1e-9)
    if not alignment:
        assert float(actual_grad[-2].abs().max()) > 0.0


def test_alignment_off_retains_beta_constant_instead_of_removing_beta() -> None:
    q0, k0, q1, k1, value, beta, gamma, batch = _attention_inputs(seed=223)
    retained = moment._factorized_moment_attention(
        q0,
        k0,
        q1,
        k1,
        gamma,
        value,
        batch,
        num_graphs=2,
        balanced=False,
        alignment_scale=beta,
        alignment_dot_scale=torch.zeros_like(beta),
        kernel_floor=0.5,
    )
    removed = moment._factorized_moment_attention(
        q0,
        k0,
        q1,
        k1,
        gamma,
        value,
        batch,
        num_graphs=2,
        balanced=False,
        alignment_scale=torch.zeros_like(beta),
        alignment_dot_scale=torch.zeros_like(beta),
        kernel_floor=0.5,
    )

    assert not torch.allclose(retained, removed, atol=1e-12, rtol=0.0)


def test_key_balancing_can_remove_pure_key_side_alignment_preference() -> None:
    beta = torch.tensor([0.25], dtype=torch.float64)
    gamma = torch.tensor([0.0], dtype=torch.float64)
    query_scalar = torch.zeros(2, 1, 1, dtype=torch.float64)
    key_scalar = torch.zeros_like(query_scalar)
    query_vector = torch.tensor(
        [[[0.6, 0.0, 0.0]], [[0.6, 0.0, 0.0]]], dtype=torch.float64
    )
    key_vector = torch.tensor(
        [[[0.5, 0.0, 0.0]], [[-0.5, 0.0, 0.0]]], dtype=torch.float64
    )
    value = torch.tensor([[[1.0]], [[-1.0]]], dtype=torch.float64)
    batch = torch.zeros(2, dtype=torch.long)
    kwargs = dict(
        num_graphs=1,
        alignment_scale=beta,
        alignment_dot_scale=beta,
        kernel_floor=1.0,
    )

    row_only = moment._factorized_moment_attention(
        query_scalar,
        key_scalar,
        query_vector,
        key_vector,
        gamma,
        value,
        batch,
        balanced=False,
        **kwargs,
    )
    balanced = moment._factorized_moment_attention(
        query_scalar,
        key_scalar,
        query_vector,
        key_vector,
        gamma,
        value,
        batch,
        balanced=True,
        **kwargs,
    )

    assert float(row_only.abs().max()) > 0.03
    assert torch.allclose(balanced, torch.zeros_like(balanced), atol=1e-12, rtol=0.0)


def test_inverse_graph_size_floor_matches_dense_uneven_batch() -> None:
    q0, k0, q1, k1, value, beta, gamma, batch = _attention_inputs(seed=227)
    actual = moment._factorized_moment_attention(
        q0,
        k0,
        q1,
        k1,
        gamma,
        value,
        batch,
        num_graphs=2,
        balanced=False,
        alignment_scale=beta,
        alignment_dot_scale=beta,
        kernel_floor=0.75,
        kernel_floor_mode="inverse_graph_size",
        graph_counts=torch.tensor([3, 4]),
    )
    expected = _dense_attention(
        q0,
        k0,
        q1,
        k1,
        value,
        beta,
        gamma,
        batch,
        balanced=False,
        alignment=True,
        kernel_floor=0.75,
        kernel_floor_mode="inverse_graph_size",
    )

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-9)


def test_inverse_graph_size_floor_has_graph_size_independent_row_mass() -> None:
    batch = torch.tensor([0, 0, 1, 1, 1, 1, 1])
    query_scalar = torch.zeros(7, 1, 2, dtype=torch.float64)
    key_scalar = torch.zeros_like(query_scalar)
    query_vector = torch.zeros(7, 1, 3, dtype=torch.float64)
    key_vector = torch.zeros_like(query_vector)
    denominator = moment._structured_row_denominator(
        query_scalar,
        key_scalar,
        query_vector,
        key_vector,
        torch.zeros(1, dtype=torch.float64),
        torch.ones(7, 1, dtype=torch.float64),
        batch,
        2,
        alignment_scale=torch.zeros(1, dtype=torch.float64),
        alignment_dot_scale=torch.zeros(1, dtype=torch.float64),
        kernel_floor=0.75,
        kernel_floor_mode="inverse_graph_size",
        graph_counts=torch.tensor([2, 5]),
    )

    assert torch.equal(denominator, torch.full_like(denominator, 0.75))


def test_inverse_graph_size_scales_the_positive_alignment_baseline() -> None:
    batch = torch.tensor([0, 0, 1, 1, 1, 1, 1])
    query_scalar = torch.zeros(7, 1, 2, dtype=torch.float64)
    key_scalar = torch.zeros_like(query_scalar)
    query_vector = torch.zeros(7, 1, 3, dtype=torch.float64)
    key_vector = torch.zeros_like(query_vector)
    denominator = moment._structured_row_denominator(
        query_scalar,
        key_scalar,
        query_vector,
        key_vector,
        torch.zeros(1, dtype=torch.float64),
        torch.ones(7, 1, dtype=torch.float64),
        batch,
        2,
        alignment_scale=torch.tensor([0.25], dtype=torch.float64),
        alignment_dot_scale=torch.tensor([0.25], dtype=torch.float64),
        kernel_floor=0.75,
        kernel_floor_mode="inverse_graph_size",
        graph_counts=torch.tensor([2, 5]),
    )

    assert torch.equal(denominator, torch.full_like(denominator, 1.0))


def test_inverse_graph_size_floor_rejects_key_balancing() -> None:
    with pytest.raises(ValueError, match="inverse_graph_size.*balancing"):
        EquivariantAttention(
            EquivariantAttentionConfig(
                node_dim=4,
                kernel_floor_mode="inverse_graph_size",
                use_key_balancing=True,
            )
        )


def test_scale_first_geometry_matches_direct_float64_formula() -> None:
    torch.manual_seed(229)
    pos = torch.randn(9, 3, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1])
    counts = torch.tensor([4, 5])
    normalized, log_radius, log_scale, log_normalized_square = (
        moment._scale_first_geometry(
            pos,
            batch,
            num_graphs=2,
            graph_counts=counts,
        )
    )
    centers = torch.stack([pos[batch == graph].mean(dim=0) for graph in range(2)])
    centered = pos - centers[batch]
    radius = centered.norm(dim=-1, keepdim=True)
    direct_scale = torch.stack(
        [
            centered[batch == graph].square().sum(dim=-1).mean().sqrt()
            for graph in range(2)
        ]
    ).unsqueeze(-1)
    direct_normalized = centered / direct_scale[batch]

    assert torch.allclose(normalized, direct_normalized, atol=1e-12, rtol=1e-12)
    assert torch.allclose(log_radius, torch.log1p(radius), atol=1e-12, rtol=1e-12)
    assert torch.allclose(log_scale, torch.log1p(direct_scale), atol=1e-12, rtol=1e-12)
    assert torch.allclose(
        log_normalized_square,
        torch.log1p(direct_normalized.square().sum(dim=-1, keepdim=True)),
        atol=1e-12,
        rtol=1e-12,
    )


def test_scale_first_geometry_stays_finite_for_extreme_float32_reductions() -> None:
    node_count = 100_000
    pos = torch.full((node_count, 3), 1e34, dtype=torch.float32)
    pos[-1, 0] = -1e34
    batch = torch.zeros(node_count, dtype=torch.long)
    outputs = moment._scale_first_geometry(
        pos,
        batch,
        num_graphs=1,
        graph_counts=torch.tensor([node_count]),
    )

    assert all(torch.isfinite(value).all() for value in outputs)


@pytest.mark.parametrize("node_count", [1, 5])
@pytest.mark.parametrize(
    "route_kwargs",
    [
        {},
        {
            "local_head_counts": (2, 0, 2),
            "global_memory_count": 4,
            "use_memory_interaction": True,
        },
    ],
)
def test_zero_rms_graph_has_finite_coordinate_gradients(
    node_count: int,
    route_kwargs: dict[str, object],
) -> None:
    torch.manual_seed(239)
    model = EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=3,
            hidden_irreps="8x0e + 2x1o",
            num_heads=2,
            **route_kwargs,
        )
    ).double()
    node_feats = torch.randn(node_count, 3, dtype=torch.float64)
    pos = torch.zeros(node_count, 3, dtype=torch.float64, requires_grad=True)
    outputs = model(node_feats, pos)
    loss = sum(value.square().sum() for value in outputs.values())
    coordinate_gradient = torch.autograd.grad(loss, pos)[0]

    assert torch.isfinite(coordinate_gradient).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_coordinate_contract_rejects_low_precision_geometry(dtype: torch.dtype) -> None:
    model = EquivariantAttention(EquivariantAttentionConfig(node_dim=3))
    with pytest.raises(TypeError, match="float32 or float64"):
        model(torch.randn(4, 3), torch.randn(4, 3, dtype=dtype))


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        ({"node_dim": True}, TypeError, "node_dim"),
        ({"node_dim": 4, "num_layers": 1.5}, TypeError, "num_layers"),
        (
            {"node_dim": 4, "use_alignment_linear_term": "yes"},
            TypeError,
            "use_alignment",
        ),
        ({"node_dim": 4, "eps": math.nan}, ValueError, "eps"),
        ({"node_dim": 4, "eps": math.inf}, ValueError, "eps"),
        ({"node_dim": 4, "kernel_floor": 1e-100}, ValueError, "kernel_floor"),
        (
            {"node_dim": 4, "residual_scale_init": math.inf},
            ValueError,
            "residual_scale_init",
        ),
        (
            {
                "node_dim": 4,
                "linear_kernel_init": 1.00000001,
                "linear_kernel_max": 1.00000002,
            },
            ValueError,
            "linear_kernel_init",
        ),
        (
            {
                "node_dim": 4,
                "kernel_floor": 3e38,
                "linear_kernel_max": 1e38,
                "linear_kernel_init": 1.0,
                "vector_kernel_max": 1e38,
                "vector_kernel_init": 1.0,
            },
            ValueError,
            "upper bound",
        ),
    ],
)
def test_config_rejects_invalid_runtime_and_float32_values(
    kwargs: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        EquivariantAttention(EquivariantAttentionConfig(**kwargs))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kernel_floor": 1e-44},
        {"linear_kernel_init": 1e-40},
        {"vector_kernel_init": 1e-40},
        {"linear_kernel_init": 1.0, "linear_kernel_max": 1e38},
    ],
)
def test_kernel_controls_reject_subnormal_float32_values_and_ratios(
    kwargs: dict[str, float],
) -> None:
    with pytest.raises(ValueError, match="normal float32"):
        EquivariantAttention(EquivariantAttentionConfig(node_dim=4, **kwargs))


def test_smallest_normal_float32_kernel_controls_round_trip_positive() -> None:
    tiny = torch.finfo(torch.float32).tiny
    model = EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=4,
            hidden_irreps="8x0e + 2x1o",
            num_heads=2,
            kernel_floor=tiny,
            linear_kernel_init=tiny,
            vector_kernel_init=tiny,
        )
    )
    beta = moment._bounded_kernel_scale(
        model.layers[0].raw_linear_kernel,
        model.layers[0].linear_kernel_max,
    )
    gamma = moment._bounded_kernel_scale(
        model.layers[0].raw_vector_kernel,
        model.layers[0].vector_kernel_max,
    )

    assert torch.all(beta > 0.0)
    assert torch.all(gamma > 0.0)


def test_smallest_normal_kernel_controls_keep_zero_content_model_finite() -> None:
    tiny = torch.finfo(torch.float32).tiny
    model = EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=4,
            hidden_irreps="8x0e + 2x1o",
            num_heads=2,
            kernel_floor=tiny,
            linear_kernel_init=tiny,
            vector_kernel_init=tiny,
            kernel_floor_mode="inverse_graph_size",
            use_key_balancing=False,
        )
    )
    with torch.no_grad():
        for layer in model.layers:
            layer.query_scalar.weight.zero_()
            layer.query_scalar.bias.fill_(-100.0)
            layer.key_scalar.weight.zero_()
            layer.key_scalar.bias.fill_(-100.0)
            layer.query_vector.weight.zero_()
            layer.key_vector.weight.zero_()

    outputs = model(torch.randn(2, 4), torch.zeros(2, 3))

    assert all(torch.isfinite(value).all() for value in outputs.values())


def test_private_attention_has_only_one_cycle_or_no_balancing() -> None:
    parameters = inspect.signature(moment._factorized_moment_attention).parameters

    assert "balance_exponent" not in parameters
    assert "sinkhorn_iterations" not in parameters


def test_autocast_wrapper_preserves_eval_state_and_model_metadata() -> None:
    model = EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=3,
            hidden_irreps="8x0e + 2x1o",
            num_layers=1,
            num_heads=2,
        )
    ).eval()
    wrapper = _AutocastInferenceModule(model, torch.bfloat16)

    assert not wrapper.training
    assert wrapper.device_type == "cpu"
    for name in (
        "attention_kind",
        "symmetry",
        "config",
        "hidden_irreps",
        "output_irreps",
    ):
        assert getattr(wrapper, name) == getattr(model, name)
