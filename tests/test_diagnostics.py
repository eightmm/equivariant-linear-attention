import json

import pytest
import torch

from equivariant_attention.diagnostics import (
    attention_weight_summary,
    dense_kernel_attention_summary,
    kernel_component_quantiles,
    kernel_parameter_summary,
    matrix_effective_rank,
    memory_assignment_summary,
    memory_pair_gate_summary,
    pair_gate_summary,
)


def test_uniform_attention_has_interpretable_scalar_diagnostics() -> None:
    num_nodes = 5
    weights = torch.full((num_nodes, num_nodes), 1.0 / num_nodes, dtype=torch.float64)

    summary = attention_weight_summary(weights)

    assert summary["num_queries"] == num_nodes
    assert summary["num_keys"] == num_nodes
    assert summary["entropy_over_log_n"] == pytest.approx(1.0)
    assert summary["max_weight"] == pytest.approx(1.0 / num_nodes)
    assert summary["effective_support"] == pytest.approx(float(num_nodes))
    assert summary["column_cv"] == pytest.approx(0.0)
    assert all(not isinstance(value, torch.Tensor) for value in summary.values())
    json.dumps(summary, allow_nan=False)


def test_singleton_attention_diagnostics_are_finite() -> None:
    summary = attention_weight_summary(torch.ones(1, 1))

    assert summary == {
        "num_queries": 1,
        "num_keys": 1,
        "entropy_over_log_n": 1.0,
        "max_weight": 1.0,
        "effective_support": 1.0,
        "column_cv": 0.0,
    }
    assert all(torch.isfinite(torch.tensor(float(value))) for value in summary.values())
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize(
    "weights, match",
    [
        (torch.ones(2, 2, 2), "two-dimensional"),
        (torch.tensor([[1.0, -0.1]]), "nonnegative"),
        (torch.tensor([[0.0, 0.0]]), "positive mass"),
        (torch.tensor([[1.0, float("nan")]]), "finite"),
    ],
)
def test_attention_diagnostics_reject_invalid_weights(
    weights: torch.Tensor, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        attention_weight_summary(weights)


def test_kernel_component_quantiles_are_flat_json_scalars() -> None:
    summary = kernel_component_quantiles(
        {
            "content": torch.tensor([0.0, 1.0, 2.0, 3.0]),
            "angular": torch.tensor([-1.0, 0.0, 1.0]),
        },
        quantiles=(0.0, 0.5, 1.0),
    )

    assert summary == {
        "content.q00": 0.0,
        "content.q50": 1.5,
        "content.q100": 3.0,
        "angular.q00": -1.0,
        "angular.q50": 0.0,
        "angular.q100": 1.0,
    }
    assert all(type(value) is float for value in summary.values())
    json.dumps(summary, allow_nan=False)


def test_kernel_parameter_summary_reports_beta_and_gamma_without_grad_state() -> None:
    beta = torch.tensor([0.25, 0.75], requires_grad=True)
    gamma = torch.tensor([0.5, 1.5], requires_grad=True)

    summary = kernel_parameter_summary(beta, gamma)

    assert summary == {
        "beta.min": 0.25,
        "beta.mean": 0.5,
        "beta.max": 0.75,
        "gamma.min": 0.5,
        "gamma.mean": 1.0,
        "gamma.max": 1.5,
    }
    assert beta.requires_grad
    assert gamma.requires_grad
    assert all(type(value) is float for value in summary.values())
    json.dumps(summary, allow_nan=False)


def test_matrix_effective_rank_is_bounded_and_interpretable() -> None:
    assert matrix_effective_rank(torch.eye(4), max_matrix_size=4) == pytest.approx(4.0)
    assert matrix_effective_rank(torch.ones(4, 4), max_matrix_size=4) == pytest.approx(
        1.0
    )

    with pytest.raises(ValueError, match="max_matrix_size=4"):
        matrix_effective_rank(torch.ones(5, 2), max_matrix_size=4)


def test_attention_effective_rank_is_opt_in() -> None:
    weights = torch.eye(3)

    without_rank = attention_weight_summary(weights)
    with_rank = attention_weight_summary(
        weights, include_effective_rank=True, effective_rank_max_size=3
    )

    assert "effective_rank" not in without_rank
    assert with_rank["effective_rank"] == pytest.approx(3.0)


@pytest.mark.parametrize("balanced", [False, True])
def test_dense_kernel_diagnostics_cover_mass_denominator_and_attention(
    balanced: bool,
) -> None:
    num_nodes = 4
    summary = dense_kernel_attention_summary(
        torch.zeros(num_nodes, 2, dtype=torch.float64),
        torch.zeros(num_nodes, 2, dtype=torch.float64),
        torch.zeros(num_nodes, 3, dtype=torch.float64),
        torch.zeros(num_nodes, 3, dtype=torch.float64),
        beta=0.0,
        gamma=0.0,
        kernel_floor=1.0,
        alignment_linear_term=True,
        balanced=balanced,
        include_effective_rank=True,
        max_nodes=8,
    )

    assert summary["attention.entropy_over_log_n"] == pytest.approx(1.0)
    assert summary["attention.max_weight"] == pytest.approx(0.25)
    assert summary["attention.effective_support"] == pytest.approx(4.0)
    assert summary["attention.column_cv"] == pytest.approx(0.0)
    assert summary["attention.effective_rank"] == pytest.approx(1.0)
    assert summary["kernel.q00"] == pytest.approx(1.0)
    assert summary["key_mass.min"] > 0.0
    assert summary["row_denominator.min"] > 0.0
    json.dumps(summary, allow_nan=False)


def test_dense_kernel_diagnostics_reject_unrepresentable_condition_ratio() -> None:
    query_scalar = torch.ones(2, 1, dtype=torch.float64)
    key_scalar = torch.tensor([[1e-300], [1e300]], dtype=torch.float64)
    vectors = torch.zeros(2, 3, dtype=torch.float64)

    with pytest.raises(ValueError, match="key_mass.*dynamic range"):
        dense_kernel_attention_summary(
            query_scalar,
            key_scalar,
            vectors,
            vectors,
            beta=0.0,
            gamma=0.0,
            kernel_floor=1e-300,
            alignment_linear_term=False,
            balanced=False,
            max_nodes=2,
        )


@pytest.mark.parametrize("balanced", [False, True])
def test_positive_constant_pair_gate_cancels_from_attention_normalization(
    balanced: bool,
) -> None:
    torch.manual_seed(97)
    query_scalar = torch.rand(5, 3, dtype=torch.float64)
    key_scalar = torch.rand(5, 3, dtype=torch.float64)
    query_vector = torch.randn(5, 3, dtype=torch.float64) * 0.1
    key_vector = torch.randn(5, 3, dtype=torch.float64) * 0.1
    kwargs = {
        "beta": 0.2,
        "gamma": 0.3,
        "kernel_floor": 0.5,
        "alignment_linear_term": True,
        "balanced": balanced,
        "max_nodes": 5,
    }

    baseline = dense_kernel_attention_summary(
        query_scalar,
        key_scalar,
        query_vector,
        key_vector,
        **kwargs,
    )
    gated = dense_kernel_attention_summary(
        query_scalar,
        key_scalar,
        query_vector,
        key_vector,
        pair_gate=torch.full((5, 5), 0.25, dtype=torch.float64),
        **kwargs,
    )

    for name, value in baseline.items():
        if name.startswith("attention."):
            assert gated[name] == pytest.approx(value, abs=1e-12, rel=1e-12)


def test_memory_assignment_summary_reports_occupancy_and_entropy() -> None:
    assignment = torch.full((5, 2, 4), 0.25, dtype=torch.float64)

    summary = memory_assignment_summary(assignment)

    assert summary["memory_count"] == 4
    assert summary["occupancy.min"] == pytest.approx(1.25)
    assert summary["occupancy.max"] == pytest.approx(1.25)
    assert summary["assignment_entropy_over_log_m"] == pytest.approx(1.0)
    json.dumps(summary, allow_nan=False)


def test_constant_pair_gate_reports_exactly_zero_variation() -> None:
    summary = pair_gate_summary(torch.full((4, 4), 0.25, dtype=torch.float64))

    assert summary["min"] == pytest.approx(0.25)
    assert summary["p01"] == pytest.approx(0.25)
    assert summary["median"] == pytest.approx(0.25)
    assert summary["p99"] == pytest.approx(0.25)
    assert summary["max"] == pytest.approx(0.25)
    assert summary["cv"] == pytest.approx(0.0)
    assert summary["centered_frobenius_ratio"] == pytest.approx(0.0)
    assert summary["nonconstant_fraction"] == pytest.approx(0.0)
    assert summary["nonconstant_relative_tolerance"] == pytest.approx(1e-3)
    json.dumps(summary, allow_nan=False)


def test_pair_gate_variation_is_scale_stable_and_internally_consistent() -> None:
    gate = torch.tensor([[1.0, 0.5], [0.5, 1.0]], dtype=torch.float64)
    summary = pair_gate_summary(gate, nonconstant_relative_tolerance=0.1)
    tiny_summary = pair_gate_summary(
        gate * 1e-300,
        nonconstant_relative_tolerance=0.1,
    )

    expected_cv = 1.0 / 3.0
    expected_frobenius = 1.0 / 10.0**0.5
    assert summary["mean"] == pytest.approx(0.75)
    assert summary["median"] == pytest.approx(0.75)
    assert summary["cv"] == pytest.approx(expected_cv)
    assert summary["centered_frobenius_ratio"] == pytest.approx(expected_frobenius)
    assert summary["nonconstant_fraction"] == pytest.approx(1.0)
    assert tiny_summary["cv"] == pytest.approx(summary["cv"])
    assert tiny_summary["centered_frobenius_ratio"] == pytest.approx(
        summary["centered_frobenius_ratio"]
    )
    assert tiny_summary["nonconstant_fraction"] == pytest.approx(
        summary["nonconstant_fraction"]
    )
    assert summary["centered_frobenius_ratio"] == pytest.approx(
        summary["cv"] / (1.0 + summary["cv"] ** 2) ** 0.5
    )
    json.dumps(summary, allow_nan=False)
    json.dumps(tiny_summary, allow_nan=False)


def test_memory_pair_gate_summary_never_pools_head_normalization_domains() -> None:
    assignment = torch.zeros(3, 2, 2, dtype=torch.float64)
    assignment[..., 0] = 1.0
    coupling = torch.stack(
        [
            torch.eye(2, dtype=torch.float64) * 0.2,
            torch.eye(2, dtype=torch.float64) * 0.8,
        ]
    )

    summary = memory_pair_gate_summary(assignment, coupling)

    assert summary["scope"] == "single_graph_per_head"
    assert summary["head_count"] == 2
    assert len(summary["heads"]) == 2
    assert summary["heads"][0]["pair_gate"]["mean"] == pytest.approx(0.2)
    assert summary["heads"][1]["pair_gate"]["mean"] == pytest.approx(0.8)
    assert all(
        head["pair_gate"]["cv"] == pytest.approx(0.0) for head in summary["heads"]
    )
    assert summary["worst_case"]["pair_gate.cv"] == pytest.approx(0.0)
    json.dumps(summary, allow_nan=False)


def test_single_memory_and_all_ones_coupling_have_constant_unit_pair_gate() -> None:
    assignment = torch.softmax(torch.randn(5, 2, 4, dtype=torch.float64), dim=-1)
    summary = memory_pair_gate_summary(
        assignment,
        torch.ones(2, 4, 4, dtype=torch.float64),
    )

    assert all(
        head["pair_gate"]["mean"] == pytest.approx(1.0)
        and head["pair_gate"]["cv"] == pytest.approx(0.0)
        for head in summary["heads"]
    )


@pytest.mark.parametrize(
    "gate, match",
    [
        (torch.ones(2, 2, 2), "two-dimensional"),
        (torch.ones(2, 3), "square"),
        (torch.tensor([[1.0, -0.1], [-0.1, 1.0]]), "nonnegative"),
        (torch.tensor([[1.0, float("nan")], [float("nan"), 1.0]]), "finite"),
        (torch.zeros(2, 2), "positive mass"),
        (torch.ones(2, 2, dtype=torch.complex64), "real-valued"),
        (torch.tensor([[1.0, 0.0], [0.5, 1.0]]), "symmetric"),
        (torch.full((2, 2), 1.1), "at most one"),
    ],
)
def test_pair_gate_summary_rejects_invalid_gates(
    gate: torch.Tensor,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        pair_gate_summary(gate)


@pytest.mark.parametrize("tolerance", [True, 0.0, -0.1, 1.0, float("nan")])
def test_pair_gate_summary_rejects_invalid_relative_tolerance(
    tolerance: object,
) -> None:
    error = TypeError if isinstance(tolerance, bool) else ValueError
    with pytest.raises(error, match="nonconstant_relative_tolerance"):
        pair_gate_summary(
            torch.ones(2, 2),
            nonconstant_relative_tolerance=tolerance,  # type: ignore[arg-type]
        )


def test_inverse_graph_size_diagnostic_scales_positive_baseline_only() -> None:
    query_scalar = torch.tensor([[0.2], [0.8]], dtype=torch.float64)
    key_scalar = torch.tensor([[0.3], [0.7]], dtype=torch.float64)
    query_vector = torch.tensor(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float64
    )
    key_vector = torch.tensor([[0.5, 0.0, 0.0], [-0.25, 0.0, 0.0]], dtype=torch.float64)
    beta = 0.4
    gamma = 0.6
    kernel_floor = 0.8
    graph_size = 2

    summary = dense_kernel_attention_summary(
        query_scalar,
        key_scalar,
        query_vector,
        key_vector,
        beta=beta,
        gamma=gamma,
        kernel_floor=kernel_floor,
        kernel_floor_mode="inverse_graph_size",
        graph_size=graph_size,
        alignment_linear_term=True,
        balanced=False,
        max_nodes=graph_size,
    )

    content = query_scalar @ key_scalar.T
    angular = query_vector @ key_vector.T
    expected_kernel = (
        content
        + (kernel_floor + beta * (1.0 + angular)) / graph_size
        + gamma * angular.square()
    )
    expected_attention = expected_kernel / expected_kernel.sum(dim=-1, keepdim=True)
    expected_summary = attention_weight_summary(expected_attention)

    assert summary["floor.q00"] == pytest.approx(kernel_floor / graph_size)
    assert summary["beta_constant.q00"] == pytest.approx(beta / graph_size)
    assert summary["quadratic.q100"] == pytest.approx(
        float((gamma * angular.square()).max())
    )
    assert summary["kernel.q00"] == pytest.approx(float(expected_kernel.min()))
    for name, value in expected_summary.items():
        assert summary[f"attention.{name}"] == pytest.approx(value)


def test_inverse_graph_size_diagnostic_rejects_balancing_and_missing_size() -> None:
    inputs = (
        torch.ones(2, 1),
        torch.ones(2, 1),
        torch.zeros(2, 3),
        torch.zeros(2, 3),
    )

    with pytest.raises(ValueError, match="graph_size"):
        dense_kernel_attention_summary(
            *inputs,
            beta=0.1,
            gamma=0.1,
            kernel_floor=0.5,
            kernel_floor_mode="inverse_graph_size",
            alignment_linear_term=True,
            balanced=False,
        )
    with pytest.raises(ValueError, match="key balancing"):
        dense_kernel_attention_summary(
            *inputs,
            beta=0.1,
            gamma=0.1,
            kernel_floor=0.5,
            kernel_floor_mode="inverse_graph_size",
            graph_size=2,
            alignment_linear_term=True,
            balanced=True,
        )
