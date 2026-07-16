import json
import math
import runpy
from pathlib import Path
import sys
from typing import Any

import pytest
import torch


PROBE = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "probe_kernel_scaling.py")
)


def _assert_finite_json_scalars(value: Any) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _assert_finite_json_scalars(child)
    elif isinstance(value, list):
        for child in value:
            _assert_finite_json_scalars(child)
    elif isinstance(value, float):
        assert math.isfinite(value)


@pytest.mark.parametrize("alignment_linear_term", [False, True])
def test_corrected_inverse_kernel_scales_complete_positive_baseline(
    alignment_linear_term: bool,
) -> None:
    query_scalar = torch.tensor([[0.2, 0.4]], dtype=torch.float64)
    key_scalar = torch.tensor([[0.3, 0.5], [0.7, 0.1]], dtype=torch.float64)
    query_vector = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    key_vector = torch.tensor([[0.5, 0.0, 0.0], [-0.25, 0.0, 0.0]], dtype=torch.float64)

    actual = PROBE["_kernel_block"](
        query_scalar,
        key_scalar,
        query_vector,
        key_vector,
        kernel_floor=0.8,
        beta=torch.tensor(0.4, dtype=torch.float64),
        gamma=torch.tensor(0.6, dtype=torch.float64),
        alignment_linear_term=alignment_linear_term,
        mode="inverse_graph_size",
        graph_size=2,
    )

    content = query_scalar @ key_scalar.T
    angular = query_vector @ key_vector.T
    delta = float(alignment_linear_term)
    expected = (
        content + (0.8 + 0.4 * (1.0 + delta * angular)) / 2 + 0.6 * angular.square()
    )
    assert torch.allclose(actual, expected, atol=1e-12, rtol=0.0)


def test_streamed_statistics_match_a_dense_reference_exactly() -> None:
    inputs = PROBE["_synthetic_inputs"](
        11, device=torch.device("cpu"), dtype=torch.float64
    )
    beta = torch.tensor(0.35, dtype=torch.float64)
    gamma = torch.tensor(0.45, dtype=torch.float64)

    streamed = PROBE["_stream_attention_stats"](
        *inputs[:4],
        kernel_floor=0.7,
        beta=beta,
        gamma=gamma,
        alignment_linear_term=True,
        mode="inverse_graph_size",
        block_rows=3,
    )
    kernel = PROBE["_kernel_block"](
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        kernel_floor=0.7,
        beta=beta,
        gamma=gamma,
        alignment_linear_term=True,
        mode="inverse_graph_size",
        graph_size=11,
    )
    weights = kernel / kernel.sum(dim=-1, keepdim=True)
    entropy = -(weights * weights.log()).sum(dim=-1)

    assert streamed["max_weight"] == pytest.approx(
        float(weights.amax(dim=-1).mean()), abs=1e-12
    )
    assert streamed["entropy_over_log_n"] == pytest.approx(
        float(entropy.mean() / math.log(11)), abs=1e-12
    )


def test_probe_compares_both_modes_with_bounded_json_safe_outputs() -> None:
    result = PROBE["run_probe"](
        sizes=(8, 17),
        device="cpu",
        dtype="float64",
        block_rows=5,
        probe_rows=3,
        warmup=0,
        repeats=1,
    )

    assert result["schema_version"] == 1
    assert result["sizes"] == [8, 17]
    assert result["key_balancing"] is False
    assert result["effective_rank_computed"] is False
    assert len(result["runs"]) == 4
    assert {(run["size"], run["mode"]) for run in result["runs"]} == {
        (8, "fixed"),
        (8, "inverse_graph_size"),
        (17, "fixed"),
        (17, "inverse_graph_size"),
    }

    for run in result["runs"]:
        size = run["size"]
        resource = run["resource_bound"]
        assert run["formula"] == ("a_dot_b + s_N*(c + beta*(1 + delta*t)) + gamma*t^2")
        assert run["config"]["baseline_scale"] == pytest.approx(
            1.0 if run["mode"] == "fixed" else 1.0 / size
        )
        assert 0.0 < run["statistics"]["max_weight"] <= 1.0
        assert 0.0 <= run["statistics"]["entropy_over_log_n"] <= 1.0
        assert run["runtime_ms"] >= 0.0
        assert resource["full_attention_matrix_persisted"] is False
        assert resource["statistics_block_covers_all_rows"] is False
        assert resource["statistics_pair_elements"] <= min(5, size) * size
        assert resource["gradient_pair_elements"] <= min(3, size) * size
        assert resource["largest_explicit_pair_tensor_bytes"] == (
            max(
                resource["statistics_pair_elements"],
                resource["gradient_pair_elements"],
            )
            * 8
        )
        assert "effective_rank" not in json.dumps(run)
        for name in ("beta", "gamma", "values"):
            assert run["gradient_norms"][name] >= 0.0
        assert run["output_probe_norm"] >= 0.0

    _assert_finite_json_scalars(result)
    json.dumps(result, allow_nan=False)


def test_probe_statistics_and_gradients_are_deterministic() -> None:
    kwargs = {
        "sizes": (13,),
        "device": "cpu",
        "dtype": "float64",
        "block_rows": 4,
        "probe_rows": 4,
        "warmup": 0,
        "repeats": 1,
    }
    first = PROBE["run_probe"](**kwargs)
    second = PROBE["run_probe"](**kwargs)

    for first_run, second_run in zip(first["runs"], second["runs"], strict=True):
        assert first_run["statistics"] == second_run["statistics"]
        assert first_run["gradient_norms"] == second_run["gradient_norms"]
        assert first_run["output_probe_norm"] == second_run["output_probe_norm"]


def test_probe_cli_writes_strict_json_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "nested" / "probe.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe_kernel_scaling.py",
            "--sizes",
            "8",
            "--device",
            "cpu",
            "--warmup",
            "0",
            "--repeats",
            "1",
            "--metrics-out",
            str(output),
        ],
    )

    PROBE["main"]()
    result = json.loads(output.read_text())

    assert result["sizes"] == [8]
    assert len(result["runs"]) == 2


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"sizes": ()}, "nonempty"),
        ({"sizes": (1,)}, "at least two"),
        ({"sizes": (8, 8)}, "unique"),
        ({"sizes": (8,), "block_rows": 0}, "block_rows"),
        ({"sizes": (8,), "probe_rows": 0}, "probe_rows"),
        ({"sizes": (8,), "repeats": 0}, "repeats"),
    ],
)
def test_probe_rejects_invalid_resource_contracts(
    kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        PROBE["run_probe"](**kwargs)
