from __future__ import annotations

import hashlib
import json
import math
import os
import random
from statistics import fmean, stdev
from typing import Mapping, Sequence

import torch


_DETERMINISM_MODES = frozenset({"seeded", "strict"})
_VALID_CUBLAS_WORKSPACE_CONFIGS = frozenset({":4096:8", ":16:8"})
_REPEAT_IDENTITY_FIELDS = (
    "dataset",
    "source_sha256",
    "initial_state_sha256",
    "model_seed",
    "split_hashes",
    "run_config",
    "reproducibility",
)


def configure_reproducibility(*, seed: int, mode: str) -> dict[str, object]:
    """Configure one process before model construction or CUDA execution."""
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    if mode not in _DETERMINISM_MODES:
        raise ValueError("mode must be 'seeded' or 'strict'")

    random.seed(seed)
    torch.manual_seed(seed)
    strict = mode == "strict"
    if strict:
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace is None:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        elif workspace not in _VALID_CUBLAS_WORKSPACE_CONFIGS:
            raise ValueError(
                "strict determinism requires CUBLAS_WORKSPACE_CONFIG to be "
                "':4096:8' or ':16:8'"
            )
    torch.use_deterministic_algorithms(strict, warn_only=False)
    torch.backends.cudnn.deterministic = strict
    torch.backends.cudnn.benchmark = False

    return {
        "seed": seed,
        "mode": mode,
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def summarize_repeated_runs(
    runs: Sequence[Mapping[str, object]],
    *,
    metric_path: str,
    max_metric_span: float,
    min_runs: int = 5,
    expected_mode: str | None = None,
) -> dict[str, object]:
    """Validate identical run identities and quantify same-seed metric drift."""
    if type(min_runs) is not int or min_runs < 2:
        raise ValueError("min_runs must be an integer at least 2")
    if len(runs) < min_runs:
        raise ValueError(f"at least {min_runs} runs are required")
    if (
        isinstance(max_metric_span, bool)
        or not isinstance(max_metric_span, (int, float))
        or not math.isfinite(float(max_metric_span))
        or float(max_metric_span) < 0.0
    ):
        raise ValueError("max_metric_span must be finite and nonnegative")
    if not metric_path:
        raise ValueError("metric_path must be nonempty")

    reference = runs[0]
    identity_fields = list(_REPEAT_IDENTITY_FIELDS)
    if reference.get("dataset") == "qm9":
        identity_fields.append("data_identity")
    for index, run in enumerate(runs):
        for field in identity_fields:
            if field not in run:
                raise ValueError(
                    f"run {index} is missing required identity field {field}"
                )
        _validate_identity_values(run, index=index)
    for index, run in enumerate(runs[1:], start=1):
        for field in identity_fields:
            if run[field] != reference[field]:
                raise ValueError(
                    f"run {index} differs from run 0 at identity field {field}"
                )
    reproducibility = reference["reproducibility"]
    assert isinstance(reproducibility, Mapping)
    recorded_mode = str(reproducibility["mode"])
    if expected_mode is not None:
        if expected_mode not in _DETERMINISM_MODES:
            raise ValueError("expected_mode must be 'seeded' or 'strict'")
        if expected_mode != recorded_mode:
            raise ValueError(
                f"expected_mode={expected_mode} does not match "
                f"recorded mode={recorded_mode}"
            )

    values = [_finite_metric(run, metric_path) for run in runs]
    metric_min = min(values)
    metric_max = max(values)
    metric_span = metric_max - metric_min
    final_hashes = [
        _sha256_text(run.get("final_state_sha256"), "final_state_sha256")
        for run in runs
    ]
    unique_final_state_count = len(set(final_hashes))
    bitwise_reproducible = metric_span == 0.0 and unique_final_state_count == 1
    span_passed = metric_span <= float(max_metric_span)
    gate_mode = "bitwise" if recorded_mode == "strict" else "metric_span"
    gate_passed = bitwise_reproducible if gate_mode == "bitwise" else span_passed

    identity = {field: reference[field] for field in identity_fields}
    identity_sha256 = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "run_count": len(runs),
        "metric_path": metric_path,
        "metric": {
            "values": values,
            "mean": fmean(values),
            "sample_std": stdev(values),
            "min": metric_min,
            "max": metric_max,
            "span": metric_span,
        },
        "max_metric_span": float(max_metric_span),
        "span_passed": span_passed,
        "unique_final_state_count": unique_final_state_count,
        "bitwise_reproducible": bitwise_reproducible,
        "recorded_determinism_mode": recorded_mode,
        "gate_mode": gate_mode,
        "gate_passed": gate_passed,
        "identity_sha256": identity_sha256,
    }


def _validate_identity_values(run: Mapping[str, object], *, index: int) -> None:
    prefix = f"run {index}"
    dataset = _nonempty_text(run["dataset"], f"{prefix} dataset")
    _sha256_text(run["source_sha256"], f"{prefix} source_sha256")
    _sha256_text(run["initial_state_sha256"], f"{prefix} initial_state_sha256")
    model_seed = _integer(run["model_seed"], f"{prefix} model_seed")

    split_hashes = _mapping(run["split_hashes"], f"{prefix} split_hashes")
    for name in ("train", "validation", "test"):
        if name not in split_hashes:
            raise ValueError(f"{prefix} split_hashes is missing {name}")
        _sha256_text(split_hashes[name], f"{prefix} split_hashes.{name}")

    run_config = _mapping(run["run_config"], f"{prefix} run_config")
    for field in ("dataset", "model_seed", "split_seed", "determinism", "device"):
        if field not in run_config:
            raise ValueError(f"{prefix} run_config is missing {field}")
    if _nonempty_text(run_config["dataset"], f"{prefix} run_config.dataset") != dataset:
        raise ValueError(f"{prefix} run_config.dataset does not match dataset")
    if _integer(run_config["model_seed"], f"{prefix} run_config.model_seed") != model_seed:
        raise ValueError(f"{prefix} run_config.model_seed does not match model_seed")
    _integer(run_config["split_seed"], f"{prefix} run_config.split_seed")
    mode = _nonempty_text(
        run_config["determinism"],
        f"{prefix} run_config.determinism",
    )
    _nonempty_text(run_config["device"], f"{prefix} run_config.device")

    reproducibility = _mapping(
        run["reproducibility"],
        f"{prefix} reproducibility",
    )
    required_reproducibility = (
        "seed",
        "mode",
        "deterministic_algorithms",
        "deterministic_warn_only",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "cublas_workspace_config",
    )
    for field in required_reproducibility:
        if field not in reproducibility:
            raise ValueError(f"{prefix} reproducibility is missing {field}")
    if _integer(reproducibility["seed"], f"{prefix} reproducibility.seed") != model_seed:
        raise ValueError(f"{prefix} reproducibility.seed does not match model_seed")
    recorded_mode = _nonempty_text(
        reproducibility["mode"],
        f"{prefix} reproducibility.mode",
    )
    if recorded_mode not in _DETERMINISM_MODES:
        raise ValueError(f"{prefix} reproducibility.mode is invalid")
    if mode != recorded_mode:
        raise ValueError(
            f"{prefix} run_config.determinism does not match reproducibility.mode"
        )
    flag_values = {
        field: _boolean(reproducibility[field], f"{prefix} reproducibility.{field}")
        for field in (
            "deterministic_algorithms",
            "deterministic_warn_only",
            "cudnn_deterministic",
            "cudnn_benchmark",
        )
    }
    expected_flags = {
        "deterministic_algorithms": recorded_mode == "strict",
        "deterministic_warn_only": False,
        "cudnn_deterministic": recorded_mode == "strict",
        "cudnn_benchmark": False,
    }
    if flag_values != expected_flags:
        raise ValueError(
            f"{prefix} reproducibility flags do not match mode={recorded_mode}"
        )
    workspace = reproducibility["cublas_workspace_config"]
    if recorded_mode == "strict":
        if workspace not in _VALID_CUBLAS_WORKSPACE_CONFIGS:
            raise ValueError(
                f"{prefix} reproducibility.cublas_workspace_config is invalid"
            )
    elif workspace is not None and not isinstance(workspace, str):
        raise TypeError(
            f"{prefix} reproducibility.cublas_workspace_config must be text or null"
        )

    if dataset == "qm9":
        data_identity = _mapping(run["data_identity"], f"{prefix} data_identity")
        if not data_identity:
            raise ValueError(f"{prefix} data_identity must be nonempty")
        for name, digest in data_identity.items():
            _nonempty_text(name, f"{prefix} data_identity key")
            _sha256_text(digest, f"{prefix} data_identity.{name}")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not value:
        raise TypeError(f"{label} must be a nonempty mapping")
    return value


def _nonempty_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be nonempty text")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be boolean")
    return value


def _sha256_text(value: object, label: str) -> str:
    text = _nonempty_text(value, label)
    if len(text) != 64:
        raise ValueError(f"{label} must be 64 hexadecimal characters")
    try:
        int(text, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be 64 hexadecimal characters") from error
    return text


def _finite_metric(run: Mapping[str, object], path: str) -> float:
    value: object = run
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise ValueError(f"metric path not found: {path}")
        value = value[component]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"metric must be a finite number: {path}")
    return float(value)
