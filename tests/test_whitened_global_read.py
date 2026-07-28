"""Whitened (ridge-regression) global read contract.

The registered diagnosis of this project's global path is that its exact
factorized kernel is numerically uniform: normalized entropy over `log N` was
`0.999759` and the selectivity-bearing alignment and quadratic terms were
`0.3%` of a kernel dominated by its positive constant. Two attempts to make the
*weights* selective regressed by about six times the admission threshold.

This lane changes the *metric of the read* instead. It replaces the pooled sum
`phi_i^T S` with the ridge solution `phi_i^T (G + lambda I)^-1 S`, where `G` is
the graph-mean Gram matrix of the same key feature map. Whitening divides out
the dominant near-constant direction rather than averaging along it, and at
large `lambda` it returns to the incumbent sum-pooling limit.

Two properties are load bearing and tested here:

1. the read is exactly `O(3)` equivariant, which requires the *isometric*
   symmetric-quadratic basis; the incumbent's asymmetric `1x`/`2x` compression
   pairs to the same kernel but breaks whitening covariance;
2. the mechanism must measurably reduce kernel uniformity, which is a
   falsifiable prediction independent of any accuracy outcome.
"""

from __future__ import annotations

from math import sqrt

import pytest
import torch

from equivariant_attention import EquivariantAttention, EquivariantAttentionConfig
from equivariant_attention.moment import (
    _factorized_moment_attention,
    _isometric_quadratic_features,
    _kernel_feature_map,
    _segment_sum,
    _symmetric_quadratic_features,
    _whitened_global_read,
)


_RIDGE = 0.75


def _max_error(left: torch.Tensor, right: torch.Tensor) -> float:
    difference = (left.detach() - right.detach()).abs()
    if difference.numel() == 0:
        return 0.0
    return float(difference.max())


def _rotation(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    angle = torch.tensor(0.7431, dtype=dtype)
    cos, sin = torch.cos(angle), torch.sin(angle)
    about_z = torch.tensor(
        [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]], dtype=dtype
    )
    about_x = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]], dtype=dtype
    )
    return about_z @ about_x


def _reflection(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.diag(torch.tensor([1.0, -1.0, 1.0], dtype=dtype))


def _read_inputs(
    *,
    nodes: int = 9,
    heads: int = 2,
    content: int = 4,
    values: int = 3,
    seed: int = 20260727,
    dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "query_scalar": torch.rand(
            (nodes, heads, content), generator=generator, dtype=dtype
        ),
        "key_scalar": torch.rand(
            (nodes, heads, content), generator=generator, dtype=dtype
        ),
        "query_vector": torch.randn(
            (nodes, heads, 3), generator=generator, dtype=dtype
        )
        * 0.4,
        "key_vector": torch.randn((nodes, heads, 3), generator=generator, dtype=dtype)
        * 0.4,
        "kernel_scale": torch.tensor([0.2, 0.7], dtype=dtype)[:heads],
        "alignment_scale": torch.tensor([0.1, 0.3], dtype=dtype)[:heads],
        "scalar_value": torch.randn(
            (nodes, heads, values), generator=generator, dtype=dtype
        ),
        "vector_value": torch.randn(
            (nodes, heads, 3), generator=generator, dtype=dtype
        ),
        "batch": torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2])[:nodes],
    }


def _call_read(
    inputs: dict[str, torch.Tensor],
    *,
    ridge: float = _RIDGE,
    kernel_floor: float = 1.0,
    rank_reliability_gate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = inputs["batch"]
    num_graphs = int(batch.max()) + 1
    return _whitened_global_read(
        inputs["query_scalar"],
        inputs["key_scalar"],
        inputs["query_vector"],
        inputs["key_vector"],
        inputs["kernel_scale"],
        inputs["scalar_value"],
        inputs["vector_value"],
        batch,
        num_graphs=num_graphs,
        graph_counts=torch.bincount(batch, minlength=num_graphs),
        alignment_scale=inputs["alignment_scale"],
        alignment_dot_scale=inputs["alignment_scale"],
        kernel_floor=kernel_floor,
        ridge=ridge,
        rank_reliability_gate=rank_reliability_gate,
    )


def test_rank_reliability_gate_disables_underdetermined_graphs() -> None:
    inputs = _read_inputs(nodes=9, heads=1, values=3)
    scalar, vector = _call_read(inputs, rank_reliability_gate=True)

    assert torch.equal(scalar, torch.zeros_like(scalar))
    assert torch.equal(vector, torch.zeros_like(vector))


def test_rank_reliability_gate_uses_the_graphwise_degrees_of_freedom_fraction() -> None:
    generator = torch.Generator().manual_seed(23)
    nodes = 40
    inputs = {
        "query_scalar": torch.rand(
            (nodes, 1, 2), generator=generator, dtype=torch.float64
        ),
        "key_scalar": torch.rand(
            (nodes, 1, 2), generator=generator, dtype=torch.float64
        ),
        "query_vector": torch.randn(
            (nodes, 1, 3), generator=generator, dtype=torch.float64
        ),
        "key_vector": torch.randn(
            (nodes, 1, 3), generator=generator, dtype=torch.float64
        ),
        "kernel_scale": torch.tensor([0.2], dtype=torch.float64),
        "alignment_scale": torch.tensor([0.1], dtype=torch.float64),
        "scalar_value": torch.randn(
            (nodes, 1, 3), generator=generator, dtype=torch.float64
        ),
        "vector_value": torch.randn(
            (nodes, 1, 3), generator=generator, dtype=torch.float64
        ),
        "batch": torch.zeros(nodes, dtype=torch.long),
    }
    scalar, vector = _call_read(inputs)
    gated_scalar, gated_vector = _call_read(
        inputs, rank_reliability_gate=True
    )
    feature_count = inputs["query_scalar"].shape[-1] + 1 + 3 + 6
    expected = (nodes - feature_count) / nodes

    torch.testing.assert_close(gated_scalar, expected * scalar)
    torch.testing.assert_close(gated_vector, expected * vector)


def _dense_kernel(
    inputs: dict[str, torch.Tensor], *, kernel_floor: float = 1.0
) -> torch.Tensor:
    """Explicit `(heads, nodes, nodes)` incumbent kernel, per graph elsewhere."""
    content = torch.einsum(
        "ihd,jhd->hij", inputs["query_scalar"], inputs["key_scalar"]
    )
    dot = torch.einsum("iha,jha->hij", inputs["query_vector"], inputs["key_vector"])
    alignment = inputs["alignment_scale"][:, None, None]
    return (
        kernel_floor
        + content
        + alignment * (1.0 + dot)
        + inputs["kernel_scale"][:, None, None] * dot.square()
    )


def test_isometric_quadratic_features_pair_to_the_squared_dot_product() -> None:
    generator = torch.Generator().manual_seed(11)
    left = torch.randn((5, 3), generator=generator, dtype=torch.float64)
    right = torch.randn((5, 3), generator=generator, dtype=torch.float64)

    paired = (
        _isometric_quadratic_features(left) * _isometric_quadratic_features(right)
    ).sum(dim=-1)

    assert _max_error(paired, (left * right).sum(dim=-1).square()) < 1e-12


def test_isometric_basis_preserves_norms_but_the_compressed_basis_does_not() -> None:
    generator = torch.Generator().manual_seed(12)
    value = torch.randn((6, 3), generator=generator, dtype=torch.float64)
    rotation = _rotation()

    isometric = _isometric_quadratic_features(value)
    rotated = _isometric_quadratic_features(value @ rotation.T)
    assert _max_error(
        isometric.square().sum(dim=-1), rotated.square().sum(dim=-1)
    ) < 1e-12

    compressed = _symmetric_quadratic_features(value, left_factor=False)
    rotated_compressed = _symmetric_quadratic_features(
        value @ rotation.T, left_factor=False
    )
    assert (
        _max_error(
            compressed.square().sum(dim=-1),
            rotated_compressed.square().sum(dim=-1),
        )
        > 1e-6
    )


def test_feature_map_reproduces_the_incumbent_dense_kernel() -> None:
    inputs = _read_inputs()
    query_features = _kernel_feature_map(
        inputs["query_scalar"],
        inputs["query_vector"],
        inputs["kernel_scale"],
        inputs["alignment_scale"],
        inputs["alignment_scale"],
        kernel_floor=1.0,
    )
    key_features = _kernel_feature_map(
        inputs["key_scalar"],
        inputs["key_vector"],
        inputs["kernel_scale"],
        inputs["alignment_scale"],
        inputs["alignment_scale"],
        kernel_floor=1.0,
    )

    factorized = torch.einsum("ihd,jhd->hij", query_features, key_features)

    assert _max_error(factorized, _dense_kernel(inputs)) < 1e-12


def test_whitened_read_matches_the_dense_ridge_reference() -> None:
    inputs = _read_inputs()
    scalar_read, vector_read = _call_read(inputs)

    query_features = _kernel_feature_map(
        inputs["query_scalar"],
        inputs["query_vector"],
        inputs["kernel_scale"],
        inputs["alignment_scale"],
        inputs["alignment_scale"],
        kernel_floor=1.0,
    )
    key_features = _kernel_feature_map(
        inputs["key_scalar"],
        inputs["key_vector"],
        inputs["kernel_scale"],
        inputs["alignment_scale"],
        inputs["alignment_scale"],
        kernel_floor=1.0,
    )
    value = torch.cat([inputs["scalar_value"], inputs["vector_value"]], dim=-1)
    batch = inputs["batch"]
    expected = torch.zeros_like(value)
    for graph in range(int(batch.max()) + 1):
        index = batch == graph
        count = float(index.sum())
        for head in range(value.shape[1]):
            psi = key_features[index, head]
            phi = query_features[index, head]
            gram = psi.T @ psi / count
            cross = psi.T @ value[index, head] / count
            eye = torch.eye(gram.shape[0], dtype=gram.dtype)
            shrinkage = _RIDGE * float(gram.diagonal().sum()) / gram.shape[0]
            expected[index, head] = phi @ torch.linalg.solve(
                gram + shrinkage * eye, cross
            )

    observed = torch.cat([scalar_read, vector_read], dim=-1)
    assert _max_error(observed, expected) < 1e-10


def test_padded_and_per_node_reductions_agree() -> None:
    """The padded batched-matmul path must equal the per-node fallback exactly.

    The padded layout exists only to remove the `(N, H, F, F)` intermediate that
    the recorded profile showed to dominate this lane, so it may not change the
    function.
    """
    inputs = _read_inputs()
    padded = _call_read(inputs)

    from equivariant_attention import moment

    original = moment._graph_padded_layout
    moment._graph_padded_layout = lambda *args, **kwargs: None
    try:
        per_node = _call_read(inputs)
    finally:
        moment._graph_padded_layout = original

    assert _max_error(padded[0], per_node[0]) < 1e-12
    assert _max_error(padded[1], per_node[1]) < 1e-12


def test_padded_layout_declines_extreme_graph_size_skew() -> None:
    from equivariant_attention.moment import _graph_padded_layout

    balanced = torch.tensor([0, 0, 1, 1, 2, 2])
    counts = torch.bincount(balanced, minlength=3)
    assert _graph_padded_layout(balanced, counts, 3) is not None

    skewed = torch.cat([torch.zeros(40, dtype=torch.long), torch.arange(1, 20)])
    skewed_counts = torch.bincount(skewed, minlength=20)
    assert _graph_padded_layout(skewed, skewed_counts, 20) is None


def test_padded_layout_positions_every_node_once() -> None:
    from equivariant_attention.moment import _graph_padded_layout, _pad_by_graph

    batch = torch.tensor([2, 0, 1, 0, 2, 2, 1])
    counts = torch.bincount(batch, minlength=3)
    layout = _graph_padded_layout(batch, counts, 3)
    assert layout is not None
    value = torch.arange(7, dtype=torch.float64).reshape(7, 1, 1)

    padded = _pad_by_graph(value, layout)
    assert padded.shape == (3, 3, 1, 1)
    for graph in range(3):
        expected = sorted(value[batch == graph].reshape(-1).tolist())
        occupied = padded[graph].reshape(-1)[: int(counts[graph])]
        assert sorted(occupied.tolist()) == expected
        assert float(padded[graph].reshape(-1)[int(counts[graph]) :].abs().sum()) == 0.0


def test_whitened_read_is_an_exact_row_stochasticfree_linear_map() -> None:
    """The read is a fixed linear map of the values with no N x N tensor."""
    inputs = _read_inputs()
    scalar_read, _ = _call_read(inputs)

    scaled = dict(inputs)
    scaled["scalar_value"] = inputs["scalar_value"] * 3.0
    scaled_read, _ = _call_read(scaled)

    assert _max_error(scaled_read, 3.0 * scalar_read) < 1e-10


def test_large_ridge_limit_is_not_the_normalized_incumbent_read() -> None:
    """The large-shrinkage limit is the kernel *numerator*, not the incumbent.

    An independent review found the packet's original wording wrong: the
    incumbent read is `phi_i^T S / phi_i^T m`, while this lane tends to
    `(1/lambda) phi_i^T S`. The missing factor is the query-dependent
    denominator, which varies across nodes of one graph, so no rescaling can
    turn the limit into the incumbent function.
    """
    inputs = _read_inputs()
    ridge = 1.0e9
    scalar_read, vector_read = _call_read(inputs, ridge=ridge)
    value = torch.cat([inputs["scalar_value"], inputs["vector_value"]], dim=-1)
    batch = inputs["batch"]
    num_graphs = int(batch.max()) + 1

    incumbent = _factorized_moment_attention(
        inputs["query_scalar"],
        inputs["key_scalar"],
        inputs["query_vector"],
        inputs["key_vector"],
        inputs["kernel_scale"],
        value,
        batch,
        num_graphs=num_graphs,
        balanced=False,
        alignment_scale=inputs["alignment_scale"],
        alignment_dot_scale=inputs["alignment_scale"],
        kernel_floor=1.0,
    )

    query_features = _kernel_feature_map(
        inputs["query_scalar"],
        inputs["query_vector"],
        inputs["kernel_scale"],
        inputs["alignment_scale"],
        inputs["alignment_scale"],
        kernel_floor=1.0,
    )
    key_features = _kernel_feature_map(
        inputs["key_scalar"],
        inputs["key_vector"],
        inputs["kernel_scale"],
        inputs["alignment_scale"],
        inputs["alignment_scale"],
        kernel_floor=1.0,
    )
    counts = torch.bincount(batch, minlength=num_graphs).to(dtype=value.dtype)
    gram = _segment_sum(
        key_features.unsqueeze(-1) * key_features.unsqueeze(-2), batch, num_graphs
    ) / counts[:, None, None, None]
    shrinkage = ridge * gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / gram.shape[-1]
    observed = shrinkage[batch][..., None] * torch.cat(
        [scalar_read, vector_read], dim=-1
    )

    same_graph = batch.unsqueeze(1) == batch.unsqueeze(0)
    kernel = torch.einsum("ihd,jhd->hij", query_features, key_features) * same_graph
    numerator = torch.einsum("hij,jhf->ihf", kernel, value) / counts[batch][
        None, :, None
    ].permute(1, 0, 2)
    assert _max_error(observed, numerator) < 1e-7

    # The incumbent divides by a denominator that is not constant within a graph.
    denominator = kernel.sum(dim=-1).permute(1, 0)[batch == 0, 0] / counts[0]
    assert float(denominator.max() - denominator.min()) > 0.1
    assert _max_error(observed, incumbent) > 0.1


def test_large_ridge_recovers_the_kernel_moment_limit() -> None:
    """At large shrinkage the read is the pooled kernel moment, times a scale.

    The shrinkage is `ridge * tr(G)/F` per graph and head, so the recovered
    constant is that quantity rather than `ridge` alone.
    """
    inputs = _read_inputs()
    ridge = 1.0e8
    scalar_read, vector_read = _call_read(inputs, ridge=ridge)

    query_features = _kernel_feature_map(
        inputs["query_scalar"],
        inputs["query_vector"],
        inputs["kernel_scale"],
        inputs["alignment_scale"],
        inputs["alignment_scale"],
        kernel_floor=1.0,
    )
    key_features = _kernel_feature_map(
        inputs["key_scalar"],
        inputs["key_vector"],
        inputs["kernel_scale"],
        inputs["alignment_scale"],
        inputs["alignment_scale"],
        kernel_floor=1.0,
    )
    value = torch.cat([inputs["scalar_value"], inputs["vector_value"]], dim=-1)
    batch = inputs["batch"]
    num_graphs = int(batch.max()) + 1
    counts = torch.bincount(batch, minlength=num_graphs).to(dtype=value.dtype)
    cross = _segment_sum(
        key_features.unsqueeze(-1) * value.unsqueeze(-2), batch, num_graphs
    ) / counts[:, None, None, None]
    pooled = torch.einsum("nhd,nhdv->nhv", query_features, cross[batch])
    gram = _segment_sum(
        key_features.unsqueeze(-1) * key_features.unsqueeze(-2), batch, num_graphs
    ) / counts[:, None, None, None]
    features = gram.shape[-1]
    shrinkage = ridge * gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / features

    observed = shrinkage[batch][..., None] * torch.cat(
        [scalar_read, vector_read], dim=-1
    )
    assert _max_error(observed, pooled) < 1e-6


def test_whitened_read_is_invariant_and_covariant_under_full_o3() -> None:
    inputs = _read_inputs()
    scalar_read, vector_read = _call_read(inputs)

    for transform in (_rotation(), _reflection(), _rotation() @ _reflection()):
        moved = dict(inputs)
        moved["query_vector"] = inputs["query_vector"] @ transform.T
        moved["key_vector"] = inputs["key_vector"] @ transform.T
        moved["vector_value"] = inputs["vector_value"] @ transform.T
        moved_scalar, moved_vector = _call_read(moved)

        assert _max_error(moved_scalar, scalar_read) < 1e-10
        assert _max_error(moved_vector, vector_read @ transform.T) < 1e-10


def test_whitening_with_the_compressed_basis_would_break_o3() -> None:
    """Documents why the isometric basis is required rather than cosmetic."""
    inputs = _read_inputs()

    def compressed_read(data: dict[str, torch.Tensor]) -> torch.Tensor:
        heads = data["kernel_scale"].shape[0]
        root = data["kernel_scale"].sqrt()[None, :, None]
        query = torch.cat(
            [
                data["query_scalar"],
                root * _symmetric_quadratic_features(
                    data["query_vector"], left_factor=True
                ),
            ],
            dim=-1,
        )
        key = torch.cat(
            [
                data["key_scalar"],
                root * _symmetric_quadratic_features(
                    data["key_vector"], left_factor=False
                ),
            ],
            dim=-1,
        )
        batch = data["batch"]
        num_graphs = int(batch.max()) + 1
        counts = torch.bincount(batch, minlength=num_graphs).to(dtype=key.dtype)
        gram = _segment_sum(
            key.unsqueeze(-1) * key.unsqueeze(-2), batch, num_graphs
        ) / counts[:, None, None, None]
        cross = _segment_sum(
            key.unsqueeze(-1) * data["scalar_value"].unsqueeze(-2),
            batch,
            num_graphs,
        ) / counts[:, None, None, None]
        eye = torch.eye(gram.shape[-1], dtype=gram.dtype)
        solved = torch.linalg.solve(gram + _RIDGE * eye, cross)
        del heads
        return torch.einsum("nhd,nhdv->nhv", query, solved[batch])

    rotation = _rotation()
    moved = dict(inputs)
    moved["query_vector"] = inputs["query_vector"] @ rotation.T
    moved["key_vector"] = inputs["key_vector"] @ rotation.T

    assert _max_error(compressed_read(moved), compressed_read(inputs)) > 1e-6


def test_whitened_read_is_permutation_equivariant_and_graph_isolated() -> None:
    inputs = _read_inputs()
    scalar_read, vector_read = _call_read(inputs)

    permutation = torch.tensor([1, 0, 3, 2, 6, 4, 5, 8, 7])
    permuted = {
        key: (value[permutation] if value.shape[:1] == permutation.shape else value)
        for key, value in inputs.items()
    }
    permuted_scalar, permuted_vector = _call_read(permuted)
    assert _max_error(permuted_scalar, scalar_read[permutation]) < 1e-12
    assert _max_error(permuted_vector, vector_read[permutation]) < 1e-12

    isolated = dict(inputs)
    keep = inputs["batch"] == 0
    isolated = {
        key: (value[keep] if value.shape[:1] == keep.shape else value)
        for key, value in inputs.items()
    }
    isolated_scalar, _ = _call_read(isolated)
    assert _max_error(isolated_scalar, scalar_read[keep]) < 1e-10


def test_whitened_read_reduces_kernel_uniformity() -> None:
    """The activity witness: whitening must sharpen a near-uniform kernel."""
    generator = torch.Generator().manual_seed(4242)
    nodes = 48
    content = 4
    scalar = 0.5 + 0.02 * torch.rand(
        (nodes, 1, content), generator=generator, dtype=torch.float64
    )
    vector = 0.05 * torch.randn((nodes, 1, 3), generator=generator, dtype=torch.float64)
    inputs = {
        "query_scalar": scalar,
        "key_scalar": scalar.clone(),
        "query_vector": vector,
        "key_vector": vector.clone(),
        "kernel_scale": torch.tensor([0.2], dtype=torch.float64),
        "alignment_scale": torch.tensor([0.1], dtype=torch.float64),
        "scalar_value": torch.randn(
            (nodes, 1, 2), generator=generator, dtype=torch.float64
        ),
        "vector_value": torch.randn(
            (nodes, 1, 3), generator=generator, dtype=torch.float64
        ),
        "batch": torch.zeros(nodes, dtype=torch.long),
    }

    def row_dispersion(weights: torch.Tensor) -> float:
        """Mean within-row coefficient of variation of the normalized weights.

        The registered global diagnostics report both normalized entropy and
        column CV because entropy is insensitive at this scale: a CV of `2.6e-3`
        still leaves entropy at `0.99999`. CV is therefore the measure that can
        resolve whether the read is dominated by its constant direction.
        """
        rows = weights / weights.sum(dim=-1, keepdim=True)
        return float((rows.std(dim=-1) / rows.mean(dim=-1)).abs().mean())

    kernel = _dense_kernel(inputs)
    incumbent_dispersion = row_dispersion(kernel)

    query_features = _kernel_feature_map(
        inputs["query_scalar"],
        inputs["query_vector"],
        inputs["kernel_scale"],
        inputs["alignment_scale"],
        inputs["alignment_scale"],
        kernel_floor=1.0,
    )
    key_features = _kernel_feature_map(
        inputs["key_scalar"],
        inputs["key_vector"],
        inputs["kernel_scale"],
        inputs["alignment_scale"],
        inputs["alignment_scale"],
        kernel_floor=1.0,
    )
    psi = key_features[:, 0]
    phi = query_features[:, 0]
    gram = psi.T @ psi / float(nodes)
    eye = torch.eye(gram.shape[0], dtype=gram.dtype)
    shrinkage = 0.05 * float(gram.diagonal().sum()) / gram.shape[0]
    whitened = (phi @ torch.linalg.solve(gram + shrinkage * eye, psi.T)).unsqueeze(0)
    whitened_dispersion = row_dispersion(whitened)

    assert incumbent_dispersion < 0.01
    assert whitened_dispersion > 5.0 * incumbent_dispersion


def test_enabled_model_matches_the_incumbent_at_initialization() -> None:
    incumbent = _model(whitened=False)
    whitened = _model(whitened=True)

    shared = dict(incumbent.state_dict())
    enabled = dict(whitened.state_dict())
    extra = set(enabled) - set(shared)
    assert extra == {
        f"layers.{index}.whitened_scalar_mix" for index in range(2)
    } | {f"layers.{index}.whitened_vector_mix" for index in range(2)}
    for name, value in shared.items():
        assert torch.equal(value, enabled[name]), name

    node_feats, pos, batch = _inputs()
    reference = incumbent(node_feats, pos, batch)
    observed = whitened(node_feats, pos, batch)
    for key, value in reference.items():
        assert _max_error(observed[key], value) == 0.0, key


def test_enabled_lane_receives_finite_nonzero_gradients() -> None:
    model = _model(whitened=True)
    with torch.no_grad():
        for layer in model.layers:
            if layer.whitened_scalar_mix is not None:
                layer.whitened_scalar_mix.fill_(0.3)
                layer.whitened_vector_mix.fill_(0.2)

    node_feats, pos, batch = _inputs()
    pos = pos.clone().requires_grad_(True)
    output = model(node_feats, pos, batch)
    loss = output["graph_scalars"].square().sum() + output["node_vectors"].square().sum()
    loss.backward()

    assert bool(torch.isfinite(pos.grad).all())
    for layer in model.layers:
        for name in ("whitened_scalar_mix", "whitened_vector_mix"):
            gradient = getattr(layer, name).grad
            assert gradient is not None and bool(torch.isfinite(gradient).all())
            assert float(gradient.abs().max()) > 0.0


def test_enabled_model_stays_finite_on_degenerate_graphs() -> None:
    model = _model(whitened=True)
    with torch.no_grad():
        for layer in model.layers:
            layer.whitened_scalar_mix.fill_(0.5)
            layer.whitened_vector_mix.fill_(0.5)

    node_feats = torch.ones((4, 6), dtype=torch.float64)
    pos = torch.zeros((4, 3), dtype=torch.float64)
    batch = torch.tensor([0, 1, 1, 2])
    output = model(node_feats, pos, batch)
    for value in output.values():
        assert bool(torch.isfinite(value).all())

    single = model(
        torch.ones((1, 6), dtype=torch.float64),
        torch.zeros((1, 3), dtype=torch.float64),
        torch.zeros(1, dtype=torch.long),
    )
    for value in single.values():
        assert bool(torch.isfinite(value).all())


def test_whitened_model_is_o3_and_translation_consistent() -> None:
    model = _model(whitened=True)
    with torch.no_grad():
        for layer in model.layers:
            layer.whitened_scalar_mix.fill_(0.4)
            layer.whitened_vector_mix.fill_(0.35)

    node_feats, pos, batch = _inputs()
    reference = model(node_feats, pos, batch)
    for transform in (_rotation(), _reflection()):
        moved = model(node_feats, pos @ transform.T, batch)
        assert _max_error(moved["graph_scalars"], reference["graph_scalars"]) < 1e-8
        assert (
            _max_error(
                moved["node_vectors"], reference["node_vectors"] @ transform.T
            )
            < 1e-8
        )
    shifted = model(node_feats, pos + torch.tensor([1.5, -2.0, 0.25]), batch)
    assert _max_error(shifted["graph_scalars"], reference["graph_scalars"]) < 1e-8


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"use_key_balancing": True}, "key balancing"),
        ({"global_transport_mode": "uniform"}, "learned global transport"),
        ({"kernel_floor_mode": "inverse_graph_size"}, "fixed kernel baseline"),
        ({"use_multiscale_spatial_kernel": True}, "spatial"),
        (
            {"global_memory_count": 4, "use_memory_interaction": True},
            "memory interaction",
        ),
        ({"whitened_global_ridge": 0.0}, "positive"),
        ({"whitened_global_ridge": float("inf")}, "finite"),
    ],
)
def test_config_rejects_unsupported_whitened_combinations(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises((ValueError, TypeError), match=message):
        _model(whitened=True, **overrides)


def test_ridge_must_be_a_real_number() -> None:
    with pytest.raises(TypeError, match="whitened_global_ridge"):
        _model(whitened=True, whitened_global_ridge="1.0")


def test_lgl_route_places_the_lane_only_on_stages_with_global_heads() -> None:
    from equivariant_attention.training import build_regression_model

    settings = {
        "node_dim": 12,
        "hidden_dim": 64,
        "num_layers": 3,
        "num_heads": 4,
        "local_head_counts": (4, 0, 4),
        "local_cutoff": 6.0,
        "use_key_balancing": False,
        "use_gated_local_transport": True,
        "use_grouped_invariant_normalization": True,
    }
    torch.manual_seed(41)
    incumbent = build_regression_model(**settings)
    torch.manual_seed(41)
    whitened = build_regression_model(**settings, use_whitened_global_read=True)

    lanes = [
        index
        for index, layer in enumerate(whitened.layers)
        if layer.whitened_scalar_mix is not None
    ]
    assert lanes == [1]

    incumbent_parameters = sum(value.numel() for value in incumbent.parameters())
    whitened_parameters = sum(value.numel() for value in whitened.parameters())
    assert whitened_parameters - incumbent_parameters == 8
    assert whitened_parameters / incumbent_parameters < 1.001
    shared = dict(incumbent.state_dict())
    enabled = dict(whitened.state_dict())
    assert all(torch.equal(shared[name], enabled[name]) for name in shared)


def test_lane_receives_gradient_once_the_zero_init_readout_activates() -> None:
    """The regression readout is zero initialized, so step zero is blind.

    Every arm of this project starts with an exactly zero prediction, so no
    upstream parameter has gradient on the first update. The honest activity
    claim is therefore that the lane receives finite nonzero gradient as soon as
    the readout is nonzero, which is what this checks.
    """
    from equivariant_attention.training import build_regression_model

    torch.manual_seed(41)
    model = build_regression_model(
        node_dim=12,
        hidden_dim=64,
        num_layers=3,
        num_heads=4,
        local_head_counts=(4, 0, 4),
        local_cutoff=6.0,
        use_key_balancing=False,
        use_gated_local_transport=True,
        use_grouped_invariant_normalization=True,
        use_whitened_global_read=True,
    ).to(dtype=torch.float64)
    generator = torch.Generator().manual_seed(7)
    node_feats = torch.randn((40, 12), generator=generator, dtype=torch.float64)
    pos = torch.randn((40, 3), generator=generator, dtype=torch.float64) * 4.0
    batch = torch.cat([torch.zeros(20, dtype=torch.long), torch.ones(20, dtype=torch.long)])
    target = torch.randn((2, 1), generator=generator, dtype=torch.float64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    layer = model.layers[1]

    observed: list[float] = []
    for _ in range(3):
        optimizer.zero_grad()
        loss = (model(node_feats, pos, batch)["graph_scalars"] - target).square().mean()
        loss.backward()
        gradient = layer.whitened_scalar_mix.grad
        assert gradient is not None and bool(torch.isfinite(gradient).all())
        observed.append(float(gradient.abs().max()))
        optimizer.step()

    assert observed[0] == 0.0
    assert observed[-1] > 0.0
    assert float(layer.whitened_scalar_mix.abs().max()) > 0.0


def _model(*, whitened: bool, **overrides: object) -> EquivariantAttention:
    settings: dict[str, object] = {
        "node_dim": 6,
        "hidden_irreps": "12x0e + 3x1o",
        "output_irreps": "2x0e + 2x1o",
        "num_layers": 2,
        "num_heads": 3,
        "use_key_balancing": False,
        "use_whitened_global_read": whitened,
    }
    settings.update(overrides)
    torch.manual_seed(101)
    return (
        EquivariantAttention(EquivariantAttentionConfig(**settings))
        .to(dtype=torch.float64)
        .eval()
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(103)
    node_feats = torch.randn((7, 6), generator=generator, dtype=torch.float64)
    pos = torch.randn((7, 3), generator=generator, dtype=torch.float64) * sqrt(2.0)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    return node_feats, pos, batch
