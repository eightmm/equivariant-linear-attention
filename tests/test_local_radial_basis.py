"""Radial-basis spacing and receptive-field contracts.

The incumbent local basis places Gaussian centers uniformly in the normalized
squared distance ``u = ||d||^2 / R_c^2``. That makes short-range radial
resolution depend on the cutoff: raising ``R_c`` to cover a whole QM9 molecule
compresses every covalent distance into the first one or two basis functions.
The opt-in ``distance`` spacing keeps the same parameter schema and
square-root-free evaluation while placing centers uniformly in ``r / R_c`` and
using the corresponding variable widths in squared-distance coordinates.
"""

from __future__ import annotations

import math

import pytest
import torch

from equivariant_attention.benchmarking import _radius_candidate_edge_index
from equivariant_attention.moment import (
    EquivariantAttention,
    EquivariantAttentionConfig,
    _radial_basis,
    _radial_basis_parameters,
)

NUM_RBF = 16


def _reference_squared_basis(
    squared_distance: torch.Tensor, num_rbf: int
) -> torch.Tensor:
    centers = torch.linspace(
        0.0,
        1.0,
        num_rbf,
        dtype=squared_distance.dtype,
        device=squared_distance.device,
    )
    width = 1.0 / max(1, num_rbf - 1)
    return torch.exp(
        -0.5 * ((squared_distance.unsqueeze(-1) - centers) / width).square()
    )


def _peak_radii(num_rbf: int, spacing: str, cutoff: float) -> torch.Tensor:
    """Return the physical radius maximizing each basis function."""

    radius = torch.linspace(0.0, cutoff, 20001, dtype=torch.float64)
    basis = _radial_basis(
        (radius / cutoff).square(),
        num_rbf=num_rbf,
        spacing=spacing,
    )
    return radius[basis.argmax(dim=0)]


def _fwhm_in_radius(num_rbf: int, spacing: str, cutoff: float) -> torch.Tensor:
    """Return each basis function's full width at half maximum in radius units."""

    radius = torch.linspace(0.0, cutoff, 40001, dtype=torch.float64)
    basis = _radial_basis(
        (radius / cutoff).square(),
        num_rbf=num_rbf,
        spacing=spacing,
    )
    widths = []
    for index in range(num_rbf):
        column = basis[:, index]
        inside = (column >= 0.5 * column.max()).nonzero().flatten()
        widths.append(float(radius[inside[-1]] - radius[inside[0]]))
    return torch.tensor(widths, dtype=torch.float64)


def _model(**overrides: object) -> EquivariantAttention:
    config: dict[str, object] = {
        "node_dim": 5,
        "hidden_irreps": "12x0e + 3x1o",
        "output_irreps": "2x0e + 1x1o",
        "num_layers": 2,
        "num_heads": 3,
        "local_head_counts": (3, 3),
        "use_key_balancing": False,
        "use_gated_local_transport": True,
        "local_cutoff": 2.0,
        "num_rbf": NUM_RBF,
    }
    config.update(overrides)
    return EquivariantAttention(EquivariantAttentionConfig(**config)).double()


def _legacy_model(**overrides: object) -> EquivariantAttention:
    """The pre-gated local route, whose learned radial gate consumes the basis."""

    return _model(
        use_gated_local_transport=False,
        learn_local_radial_gate=True,
        **overrides,
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260725)
    scalars = torch.randn(9, 5, generator=generator, dtype=torch.float64)
    pos = torch.randn(9, 3, generator=generator, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2])
    return scalars, pos, batch


def _reflection() -> torch.Tensor:
    generator = torch.Generator().manual_seed(4093)
    matrix = torch.randn(3, 3, generator=generator, dtype=torch.float64)
    orthogonal, _ = torch.linalg.qr(matrix)
    if torch.linalg.det(orthogonal) > 0:
        orthogonal[:, 0] = -orthogonal[:, 0]
    return orthogonal


def test_squared_spacing_matches_the_incumbent_reference() -> None:
    squared_distance = torch.linspace(0.0, 1.0, 257, dtype=torch.float64)

    actual = _radial_basis(squared_distance, num_rbf=NUM_RBF, spacing="squared")

    expected = _reference_squared_basis(squared_distance, NUM_RBF)
    assert torch.equal(actual, expected)


def test_distance_spacing_matches_its_explicit_reference() -> None:
    squared_distance = torch.linspace(0.0, 1.0, 129, dtype=torch.float64)
    step = 1.0 / (NUM_RBF - 1)
    knots = torch.linspace(0.0, 1.0, NUM_RBF, dtype=torch.float64)
    centers = knots.square()
    widths = (knots + step).square() - centers

    actual = _radial_basis(squared_distance, num_rbf=NUM_RBF, spacing="distance")

    expected = torch.exp(
        -0.5 * ((squared_distance.unsqueeze(-1) - centers) / widths).square()
    )
    assert torch.equal(actual, expected)


def test_single_basis_function_is_spacing_independent() -> None:
    squared_distance = torch.linspace(0.0, 1.0, 17, dtype=torch.float64)

    squared = _radial_basis(squared_distance, num_rbf=1, spacing="squared")
    distance = _radial_basis(squared_distance, num_rbf=1, spacing="distance")

    assert squared.shape == (17, 1)
    assert torch.equal(squared, distance)


@pytest.mark.parametrize("spacing", ["squared", "distance"])
def test_precomputed_radial_parameters_match_dynamic_reference(
    spacing: str,
) -> None:
    squared_distance = torch.linspace(0.0, 1.0, 37, dtype=torch.float64)
    centers, widths = _radial_basis_parameters(
        NUM_RBF,
        spacing,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    expected = _radial_basis(
        squared_distance,
        num_rbf=NUM_RBF,
        spacing=spacing,
    )
    actual = _radial_basis(
        squared_distance,
        num_rbf=NUM_RBF,
        spacing=spacing,
        centers=centers,
        widths=widths,
    )

    assert torch.equal(actual, expected)


def test_model_forward_reuses_nonpersistent_geometry_constants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    scalars, pos, batch = _inputs()
    state_keys = tuple(model.state_dict())

    assert "_local_rbf_centers" not in state_keys
    assert "_local_rbf_widths" not in state_keys

    def forbidden(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("forward must reuse precomputed geometry constants")

    monkeypatch.setattr(torch, "linspace", forbidden)
    monkeypatch.setattr(torch, "triu_indices", forbidden)

    output = model(scalars, pos, batch=batch)

    assert torch.isfinite(output["graph_scalars"]).all()


def test_geometry_constant_buffers_follow_model_dtype_and_device() -> None:
    model = _model().float()

    assert model._local_rbf_centers.dtype == torch.float32
    assert model._local_rbf_widths.dtype == torch.float32
    model = model.double()
    assert model._local_rbf_centers.dtype == torch.float64
    assert model._local_rbf_widths.dtype == torch.float64
    moved = model.to("meta")
    assert moved._local_rbf_centers.device.type == "meta"
    assert moved._local_rbf_widths.device.type == "meta"


def test_rank_two_global_forward_uses_cached_python_upper_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(
        local_head_counts=(0, 0),
        use_gated_local_transport=False,
        angular_feature_rank=2,
    )
    scalars, pos, batch = _inputs()

    def forbidden(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("forward must not allocate triangular index tensors")

    monkeypatch.setattr(torch, "triu_indices", forbidden)

    output = model(scalars, pos, batch=batch)

    assert torch.isfinite(output["graph_scalars"]).all()


def test_distance_spacing_peaks_are_uniform_in_physical_radius() -> None:
    cutoff = 5.0

    peaks = _peak_radii(NUM_RBF, "distance", cutoff)

    expected = torch.linspace(0.0, cutoff, NUM_RBF, dtype=torch.float64)
    assert torch.allclose(peaks, expected, atol=cutoff / 20000.0 * 2)
    separations = peaks.diff()
    assert torch.allclose(separations, separations[:1], atol=1e-3)


def test_squared_spacing_resolution_degrades_at_short_range() -> None:
    """The incumbent basis is coarse where chemistry lives and fine at R_c."""

    cutoff = 5.0

    peaks = _peak_radii(NUM_RBF, "squared", cutoff)

    separations = peaks.diff()
    assert separations[0] > separations[-1]
    assert bool(torch.all(separations.diff() < 0.0))


def test_distance_spacing_resolves_the_covalent_range_more_finely() -> None:
    """With R_c=5 A the incumbent basis cannot separate covalent distances."""

    cutoff = 5.0
    covalent = torch.tensor([1.09, 1.54, 2.15, 2.60], dtype=torch.float64)
    normalized = (covalent / cutoff).square()

    squared = _radial_basis(normalized, num_rbf=NUM_RBF, spacing="squared")
    distance = _radial_basis(normalized, num_rbf=NUM_RBF, spacing="distance")

    assert int(distance.argmax(dim=-1).unique().numel()) == covalent.numel()
    assert int(squared.argmax(dim=-1).unique().numel()) < covalent.numel()
    squared_width = _fwhm_in_radius(NUM_RBF, "squared", cutoff)
    distance_width = _fwhm_in_radius(NUM_RBF, "distance", cutoff)
    near_bond = int((_peak_radii(NUM_RBF, "distance", cutoff) - 1.5).abs().argmin())
    squared_near_bond = int((_peak_radii(NUM_RBF, "squared", cutoff) - 1.5).abs().argmin())
    assert float(distance_width[near_bond]) < 0.55 * float(
        squared_width[squared_near_bond]
    )


def test_distance_spacing_resolution_is_more_uniform_and_cutoff_covariant() -> None:
    narrow = _fwhm_in_radius(NUM_RBF, "distance", 2.5)
    wide = _fwhm_in_radius(NUM_RBF, "distance", 5.0)

    for cutoff in (2.5, 5.0):
        squared_width = _fwhm_in_radius(NUM_RBF, "squared", cutoff)
        distance_width = _fwhm_in_radius(NUM_RBF, "distance", cutoff)
        squared_spread = float(squared_width.max() / squared_width.min())
        distance_spread = float(distance_width.max() / distance_width.min())
        assert distance_spread < squared_spread
    assert torch.allclose(wide, 2.0 * narrow, rtol=5e-3, atol=1e-3)


def test_distance_spacing_is_finite_and_bounded_including_coincident_nodes() -> None:
    squared_distance = torch.tensor([0.0, 1e-30, 0.5, 1.0], dtype=torch.float64)

    basis = _radial_basis(squared_distance, num_rbf=NUM_RBF, spacing="distance")

    assert torch.isfinite(basis).all()
    assert bool((basis >= 0.0).all() and (basis <= 1.0).all())
    assert math.isclose(float(basis[0, 0]), 1.0, abs_tol=0.0)


def test_radial_basis_rejects_unknown_spacing() -> None:
    squared_distance = torch.zeros(3, dtype=torch.float64)

    with pytest.raises(ValueError, match="local_rbf_spacing"):
        _radial_basis(squared_distance, num_rbf=NUM_RBF, spacing="linear")


def test_config_rejects_invalid_local_rbf_spacing() -> None:
    with pytest.raises(ValueError, match="local_rbf_spacing"):
        _model(local_rbf_spacing="linear")
    with pytest.raises(TypeError, match="local_rbf_spacing"):
        _model(local_rbf_spacing=2)


def test_default_spacing_is_the_incumbent_and_leaves_outputs_unchanged() -> None:
    scalars, pos, batch = _inputs()
    assert EquivariantAttentionConfig(node_dim=3).local_rbf_spacing == "squared"

    torch.manual_seed(11)
    default = _model()
    torch.manual_seed(11)
    explicit = _model(local_rbf_spacing="squared")

    assert default.state_dict().keys() == explicit.state_dict().keys()
    expected = default(scalars, pos, batch=batch)
    actual = explicit(scalars, pos, batch=batch)
    for key, value in expected.items():
        assert torch.equal(actual[key], value)


def test_distance_spacing_keeps_the_parameter_schema_byte_compatible() -> None:
    torch.manual_seed(23)
    incumbent = _model()
    torch.manual_seed(23)
    candidate = _model(local_rbf_spacing="distance")

    incumbent_state = incumbent.state_dict()
    candidate_state = candidate.state_dict()
    assert incumbent_state.keys() == candidate_state.keys()
    for key, value in incumbent_state.items():
        assert torch.equal(candidate_state[key], value)
    candidate.load_state_dict(incumbent_state)


def test_distance_spacing_changes_the_gated_local_function() -> None:
    scalars, pos, batch = _inputs()
    torch.manual_seed(23)
    incumbent = _model()
    torch.manual_seed(23)
    candidate = _model(local_rbf_spacing="distance")

    incumbent_out = incumbent(scalars, pos, batch=batch)["graph_scalars"]
    candidate_out = candidate(scalars, pos, batch=batch)["graph_scalars"]

    assert not torch.allclose(incumbent_out, candidate_out)


def test_distance_spacing_changes_the_legacy_radial_gate_function() -> None:
    scalars, pos, batch = _inputs()
    torch.manual_seed(23)
    incumbent = _legacy_model()
    torch.manual_seed(23)
    candidate = _legacy_model(local_rbf_spacing="distance")
    with torch.no_grad():
        weights = torch.randn_like(incumbent.layers[0].local_radial_weight)
        for model in (incumbent, candidate):
            model.layers[0].local_radial_weight.copy_(weights)

    incumbent_out = incumbent(scalars, pos, batch=batch)["graph_scalars"]
    candidate_out = candidate(scalars, pos, batch=batch)["graph_scalars"]

    assert not torch.allclose(incumbent_out, candidate_out)


@pytest.mark.parametrize("spacing", ["squared", "distance"])
def test_spacing_preserves_o3_translation_and_permutation(spacing: str) -> None:
    scalars, pos, batch = _inputs()
    model = _model(local_rbf_spacing=spacing)
    rotation = _reflection()
    shift = torch.tensor([0.3, -1.2, 0.7], dtype=torch.float64)

    base = model(scalars, pos, batch=batch)
    transformed = model(scalars, pos @ rotation.T + shift, batch=batch)
    permutation = torch.tensor([3, 1, 0, 2, 5, 4, 6, 8, 7])
    permuted = model(scalars[permutation], pos[permutation], batch=batch[permutation])

    assert torch.allclose(
        transformed["graph_scalars"], base["graph_scalars"], atol=1e-6
    )
    assert torch.allclose(
        transformed["node_vectors"],
        base["node_vectors"] @ rotation.T,
        atol=1e-6,
    )
    assert torch.allclose(
        permuted["node_scalars"], base["node_scalars"][permutation], atol=1e-6
    )


@pytest.mark.parametrize("spacing", ["squared", "distance"])
def test_spacing_keeps_graphs_isolated(spacing: str) -> None:
    scalars, pos, batch = _inputs()
    model = _model(local_rbf_spacing=spacing)

    base = model(scalars, pos, batch=batch)
    moved = pos.clone()
    moved[batch == 1] += torch.tensor([12.0, -9.0, 7.0], dtype=torch.float64)
    shifted = model(scalars, moved, batch=batch)

    assert torch.allclose(
        shifted["node_scalars"][batch == 0],
        base["node_scalars"][batch == 0],
        atol=1e-6,
    )


def test_distance_spacing_has_finite_nonzero_coordinate_gradients() -> None:
    scalars, pos, batch = _inputs()
    model = _legacy_model(local_rbf_spacing="distance")
    pos = pos.clone().requires_grad_(True)

    model(scalars, pos, batch=batch)["graph_scalars"].sum().backward()

    assert pos.grad is not None
    assert torch.isfinite(pos.grad).all()
    assert float(pos.grad.abs().max()) > 0.0
    weight_grad = model.layers[0].local_radial_weight.grad
    assert weight_grad is not None
    assert torch.isfinite(weight_grad).all()
    assert float(weight_grad.abs().max()) > 0.0


def test_distance_spacing_survives_coincident_coordinates() -> None:
    scalars = torch.zeros(4, 5, dtype=torch.float64)
    pos = torch.zeros(4, 3, dtype=torch.float64, requires_grad=True)
    batch = torch.tensor([0, 0, 1, 1])
    model = _model(local_rbf_spacing="distance")

    out = model(scalars, pos, batch=batch)
    out["graph_scalars"].sum().backward()

    assert torch.isfinite(out["graph_scalars"]).all()
    assert pos.grad is not None
    assert torch.isfinite(pos.grad).all()


def test_radius_candidates_grow_monotonically_with_the_cutoff() -> None:
    generator = torch.Generator().manual_seed(7)
    pos = torch.randn(24, 3, generator=generator, dtype=torch.float32) * 2.0

    narrow = _radius_candidate_edge_index(pos, cutoff=2.5)
    wide = _radius_candidate_edge_index(pos, cutoff=5.0)

    narrow_pairs = {(int(r), int(s)) for r, s in zip(*narrow.tolist())}
    wide_pairs = {(int(r), int(s)) for r, s in zip(*wide.tolist())}
    assert narrow_pairs < wide_pairs
    assert all((index, index) in narrow_pairs for index in range(pos.shape[0]))
