from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import pytest
import torch

from equivariant_linear_attention.geometry.layout import PackedGraphLayout, pack_graph_layout


def _batch_from_counts(counts: Sequence[int]) -> torch.Tensor:
    count_tensor = torch.tensor(counts, dtype=torch.long)
    return torch.repeat_interleave(torch.arange(len(counts)), count_tensor)


@pytest.mark.parametrize(
    ("counts", "expected_structure"),
    [
        ([37], "direct"),
        ([14, 16, 15, 16], "padded"),
        ([1, 2, 7, 9, 31, 33], "bucketed"),
        ([4096, *([1] * 63)], "extreme"),
    ],
)
def test_pack_graph_layout_selects_deterministic_structure(
    counts: list[int],
    expected_structure: str,
) -> None:
    batch = _batch_from_counts(counts)

    layout = pack_graph_layout(batch)

    assert isinstance(layout, PackedGraphLayout)
    assert layout.batch is batch
    assert layout.structure == expected_structure
    assert layout.num_nodes == sum(counts)
    assert layout.num_graphs == len(counts)
    assert layout.max_nodes == max(counts)
    assert layout.graph_counts.tolist() == counts
    assert layout.graph_ptr.tolist() == [
        0,
        *torch.tensor(counts).cumsum(0).tolist(),
    ]
    assert layout.graph_ptr.dtype == torch.int32
    assert layout.order is None
    assert layout.inverse_order is None


def test_grouped_count_fast_path_does_not_sort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = torch.tensor([3, 5, 2], dtype=torch.long)
    batch = _batch_from_counts(counts.tolist())

    def forbidden_argsort(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("known-grouped collation must not sort")

    monkeypatch.setattr(torch, "argsort", forbidden_argsort)
    layout = pack_graph_layout(
        batch,
        graph_counts=counts,
        assume_grouped=True,
    )

    assert layout.batch is batch
    assert layout.order is None
    assert layout.inverse_order is None
    torch.testing.assert_close(layout.graph_counts, counts)


def test_pack_graph_layout_selects_ragged_grouped_gemm_before_extreme_fallback() -> None:
    batch = _batch_from_counts([1, 2, 7, 9, 31, 33])

    layout = pack_graph_layout(batch, maximum_buckets=2)

    assert layout.structure == "ragged"
    assert layout.graph_spans == (
        (0, 1),
        (1, 2),
        (3, 7),
        (10, 9),
        (19, 31),
        (50, 33),
    )
    assert layout.select_lane(
        backend="feature_gemm",
        dtype=torch.float32,
        device="cpu",
        num_heads=2,
        feature_width=16,
        value_width=12,
    ) == "ragged_gemm"


def test_unordered_layout_groups_and_restores_nodes_once() -> None:
    grouped_batch = _batch_from_counts([3, 2, 4])
    permutation = torch.tensor([5, 0, 3, 7, 1, 8, 4, 2, 6])
    batch = grouped_batch[permutation]
    values = torch.arange(batch.numel() * 2).reshape(batch.numel(), 2)

    layout = pack_graph_layout(batch)
    grouped = layout.group_nodes(values)
    restored = layout.ungroup_nodes(grouped)

    assert layout.order is not None
    assert layout.inverse_order is not None
    assert torch.equal(batch.index_select(0, layout.order), grouped_batch)
    assert torch.equal(restored, values)


def test_padded_plan_has_exact_zero_padding_and_cached_mask() -> None:
    batch = _batch_from_counts([3, 5, 4])
    layout = pack_graph_layout(batch)
    values = torch.arange(12, dtype=torch.float64).unsqueeze(-1)

    padded = layout.gather_dense(values)

    assert layout.structure == "padded"
    assert layout.dense_index is not None
    assert layout.dense_mask is not None
    assert padded.shape == (3, 5, 1)
    assert torch.equal(padded[layout.dense_mask].flatten(), values.flatten())
    assert torch.count_nonzero(padded[~layout.dense_mask]) == 0


def test_bucket_plans_cover_every_grouped_node_once_with_zero_padding() -> None:
    batch = _batch_from_counts([1, 2, 7, 9, 31, 33])
    layout = pack_graph_layout(batch)
    values = torch.arange(batch.numel(), dtype=torch.float64).unsqueeze(-1)
    covered: list[torch.Tensor] = []

    assert layout.structure == "bucketed"
    assert layout.dense_index is None
    assert len(layout.buckets) == 6
    for bucket in layout.buckets:
        padded = bucket.gather(layout.group_nodes(values))
        assert padded.shape[:2] == bucket.node_index.shape
        assert torch.count_nonzero(padded[~bucket.mask]) == 0
        covered.append(bucket.node_index[bucket.mask])

    assert torch.equal(
        torch.sort(torch.cat(covered)).values,
        torch.arange(batch.numel()),
    )
    assert layout.packed_slots == sum(
        bucket.node_index.numel() for bucket in layout.buckets
    )


def test_layout_requires_the_exact_batch_tensor() -> None:
    batch = _batch_from_counts([2, 3])
    layout = pack_graph_layout(batch)

    layout.validate_batch(batch)
    with pytest.raises(ValueError, match="exact batch tensor"):
        layout.validate_batch(batch.clone())


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"packed_slots": 1}, "packed_slots"),
        ({"max_nodes": 99}, "max_nodes"),
        ({"graph_counts": torch.tensor([1, 4])}, "graph_counts"),
        ({"graph_ptr": torch.tensor([0, 1, 5], dtype=torch.int32)}, "graph_ptr"),
        ({"structure": "extreme"}, "structure"),
    ],
)
def test_public_layout_constructor_rejects_forged_execution_metadata(
    updates: dict[str, object],
    message: str,
) -> None:
    layout = pack_graph_layout(_batch_from_counts([2, 3]))

    with pytest.raises(ValueError, match=message):
        replace(layout, **updates)


def test_layout_to_same_device_is_identity_and_cross_device_is_trusted() -> None:
    batch = _batch_from_counts([1, 2, 7, 9, 31, 33])
    layout = pack_graph_layout(batch)

    assert layout.to("cpu") is layout
    moved = layout.to("meta")

    assert moved is not layout
    assert moved.batch.device.type == "meta"
    assert moved.graph_ptr.device.type == "meta"
    assert all(bucket.node_index.device.type == "meta" for bucket in moved.buckets)
    assert moved.validated
    moved.validate_batch(moved.batch)


@pytest.mark.parametrize(
    ("dtype", "device", "expected"),
    [
        (torch.bfloat16, "cuda", (32, 48)),
        (torch.float16, "cuda", (32, 48)),
        (torch.float32, "cuda", (32, 40)),
        (torch.float64, "cuda", (26, 37)),
        (torch.bfloat16, "cpu", (26, 37)),
        (torch.float32, "cpu", (26, 37)),
    ],
)
def test_width_padding_policy_is_exact_and_device_aware(
    dtype: torch.dtype,
    device: str,
    expected: tuple[int, int],
) -> None:
    layout = pack_graph_layout(_batch_from_counts([8]))

    assert layout.padded_widths(
        feature_width=26,
        augmented_value_width=37,
        dtype=dtype,
        device=device,
    ) == expected


def test_cost_selector_covers_direct_padded_bucket_and_fallback() -> None:
    small = pack_graph_layout(_batch_from_counts([3]))
    direct = pack_graph_layout(_batch_from_counts([512]))
    padded = pack_graph_layout(_batch_from_counts([120, 128, 124, 127]))
    bucketed = pack_graph_layout(_batch_from_counts([1, 2, 7, 9, 31, 33]))
    extreme = pack_graph_layout(_batch_from_counts([4096, *([1] * 63)]))
    common = {
        "dtype": torch.float32,
        "device": "cpu",
        "num_heads": 4,
        "feature_width": 32,
        "value_width": 40,
    }

    assert small.select_lane(backend="auto", **common) == "outer_scatter"
    assert small.select_lane(backend="feature_gemm", **common) == "direct"
    assert direct.select_lane(backend="auto", **common) == "direct"
    assert padded.select_lane(backend="auto", **common) == "padded_bmm"
    assert bucketed.select_lane(backend="auto", **common) == "bucket_bmm"
    assert extreme.select_lane(backend="auto", **common) == "outer_scatter"
    assert extreme.select_lane(backend="feature_gemm", **common) == "outer_scatter"
    assert direct.select_lane(backend="outer_scatter", **common) == "outer_scatter"


@pytest.mark.parametrize(
    ("updates", "exception", "message"),
    [
        ({"backend": "unknown"}, ValueError, "backend"),
        ({"num_heads": 0}, ValueError, "num_heads"),
        ({"feature_width": 0}, ValueError, "feature_width"),
        ({"value_width": -1}, ValueError, "value_width"),
        ({"dtype": torch.int64}, TypeError, "floating"),
    ],
)
def test_cost_selector_validates_controls(
    updates: dict[str, object],
    exception: type[Exception],
    message: str,
) -> None:
    layout = pack_graph_layout(_batch_from_counts([64]))
    controls: dict[str, object] = {
        "backend": "auto",
        "dtype": torch.float32,
        "device": "cpu",
        "num_heads": 2,
        "feature_width": 16,
        "value_width": 24,
    }
    controls.update(updates)

    with pytest.raises(exception, match=message):
        layout.select_lane(**controls)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("batch", "message"),
    [
        (torch.empty(0, dtype=torch.long), "nonempty"),
        (torch.tensor([0, -1]), "nonnegative"),
        (torch.tensor([0, 2]), "contiguous"),
        (torch.tensor([False, True]), "integer"),
    ],
)
def test_pack_graph_layout_rejects_invalid_batch(
    batch: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        pack_graph_layout(batch)


def test_grouped_fast_path_rejects_count_sum_mismatch() -> None:
    batch = _batch_from_counts([2, 3])

    with pytest.raises(ValueError, match="sum"):
        pack_graph_layout(
            batch,
            graph_counts=torch.tensor([2, 2]),
            assume_grouped=True,
        )
