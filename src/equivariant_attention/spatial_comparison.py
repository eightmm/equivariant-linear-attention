from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Any


ARMS = ("explicit", "implicit", "hybrid")
REQUIRED_TASKS = ("local_directional", "smooth_gaussian", "mixed")


@dataclass(frozen=True, slots=True)
class SpatialPromotionThresholds:
    """Bounded synthetic-screen thresholds, all relative to explicit local."""

    min_seeds: int = 3
    max_initial_equivalence_abs: float = 1e-7
    max_edge_independence_abs: float = 1e-7
    max_hybrid_local_regression: float = 0.02
    min_hybrid_smooth_improvement: float = 0.05
    min_hybrid_mixed_improvement: float = 0.02
    max_hybrid_train_time_overhead: float = 0.25
    max_hybrid_inference_time_overhead: float = 0.25
    max_hybrid_training_memory_overhead: float = 0.25
    max_implicit_local_regression: float = 0.02
    min_implicit_smooth_improvement: float = 0.05
    max_implicit_inference_time_ratio: float = 1.0
    max_implicit_training_memory_ratio: float = 1.0

    def __post_init__(self) -> None:
        if self.min_seeds <= 0:
            raise ValueError("min_seeds must be positive")
        for name, value in self.__dict__.items():
            if name == "min_seeds":
                continue
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")


def _safe_relative(candidate: float, reference: float) -> float:
    if not math.isfinite(candidate) or not math.isfinite(reference):
        return float("nan")
    if reference == 0.0:
        return 0.0 if candidate == 0.0 else float("inf")
    return (candidate - reference) / abs(reference)


def _safe_ratio(candidate: float, reference: float) -> float:
    if not math.isfinite(candidate) or not math.isfinite(reference):
        return float("nan")
    if reference == 0.0:
        return 1.0 if candidate == 0.0 else float("inf")
    return candidate / reference


def validate_spatial_comparison(payload: dict[str, Any]) -> list[str]:
    """Return protocol violations instead of silently accepting partial results."""

    errors: list[str] = []
    if payload.get("experiment") != "spatial_operator_comparison":
        errors.append("unexpected experiment name")
    if int(payload.get("schema_version", -1)) < 2:
        errors.append("schema_version must be at least two")
    if tuple(payload.get("arms", ())) != ARMS:
        errors.append("arms must be explicit, implicit, hybrid in canonical order")
    protocol = payload.get("protocol", {})
    for key in (
        "same_parameter_schema",
        "same_initial_state_per_task_seed",
        "same_train_validation_data_per_task_seed",
        "validation_or_test_labels_used_for_training",
        "no_edge_graph_prepared_outside_timed_forward",
    ):
        if key not in protocol:
            errors.append(f"missing protocol field: {key}")
    if protocol.get("same_parameter_schema") is not True:
        errors.append("parameter schemas were not matched")
    if protocol.get("same_initial_state_per_task_seed") is not True:
        errors.append("initial states were not paired")
    if protocol.get("same_train_validation_data_per_task_seed") is not True:
        errors.append("data were not paired")
    if protocol.get("validation_or_test_labels_used_for_training") is not False:
        errors.append("validation/test labels may have influenced training")
    if protocol.get("no_edge_graph_prepared_outside_timed_forward") is not True:
        errors.append("implicit timing includes avoidable graph preparation")

    runs = payload.get("runs", [])
    observed: dict[tuple[str, int], set[str]] = {}
    hashes: dict[tuple[str, int], set[str]] = {}
    counts: dict[tuple[str, int], set[int]] = {}
    for run in runs:
        key = (str(run.get("task")), int(run.get("seed", -1)))
        observed.setdefault(key, set()).add(str(run.get("arm")))
        hashes.setdefault(key, set()).add(str(run.get("initial_state_sha256")))
        audit = run.get("audit", {})
        counts.setdefault(key, set()).add(int(audit.get("parameter_count", -1)))
    for key, arms in observed.items():
        if arms != set(ARMS):
            errors.append(f"incomplete arm set for task/seed {key}: {sorted(arms)}")
        if len(hashes[key]) != 1:
            errors.append(f"initial-state hash mismatch for task/seed {key}")
        if len(counts[key]) != 1:
            errors.append(f"parameter-count mismatch for task/seed {key}")
    return errors


def _paired_runs(payload: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    pairs: dict[tuple[str, int], dict[str, Any]] = {}
    for run in payload.get("runs", []):
        key = (str(run["task"]), int(run["seed"]))
        pairs.setdefault(key, {})[str(run["arm"])] = run
    return pairs


def paired_spatial_deltas(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Compute per-seed paired accuracy and resource differences."""

    rows: list[dict[str, Any]] = []
    for (task, seed), arms in sorted(_paired_runs(payload).items()):
        if set(arms) != set(ARMS):
            continue
        explicit = arms["explicit"]
        for arm in ("implicit", "hybrid"):
            candidate = arms[arm]
            rows.append(
                {
                    "task": task,
                    "seed": seed,
                    "arm": arm,
                    "mae_relative": _safe_relative(
                        float(candidate["best_validation"]["mae"]),
                        float(explicit["best_validation"]["mae"]),
                    ),
                    "rmse_relative": _safe_relative(
                        float(candidate["best_validation"]["rmse"]),
                        float(explicit["best_validation"]["rmse"]),
                    ),
                    "train_time_ratio": _safe_ratio(
                        float(candidate["median_train_step_ms"]),
                        float(explicit["median_train_step_ms"]),
                    ),
                    "inference_time_ratio": _safe_ratio(
                        float(candidate["inference_ms"]),
                        float(explicit["inference_ms"]),
                    ),
                    "training_memory_ratio": _safe_ratio(
                        float(candidate["training_peak_allocated_bytes"]),
                        float(explicit["training_peak_allocated_bytes"]),
                    ),
                    "inference_memory_ratio": _safe_ratio(
                        float(candidate["inference_peak_allocated_bytes"]),
                        float(explicit["inference_peak_allocated_bytes"]),
                    ),
                }
            )
    return rows


def _mean_for(
    rows: list[dict[str, Any]],
    *,
    task: str,
    arm: str,
    key: str,
) -> float:
    values = [
        float(row[key])
        for row in rows
        if row["task"] == task
        and row["arm"] == arm
        and math.isfinite(float(row[key]))
    ]
    return statistics.mean(values) if values else float("nan")


def _max_audit_value(payload: dict[str, Any], key: str) -> float:
    values = [
        float(audit.get("initial_equivalence", {}).get(key, float("nan")))
        for audit in payload.get("audits", [])
    ]
    finite = [value for value in values if math.isfinite(value)]
    return max(finite, default=float("nan"))


def spatial_promotion_decision(
    payload: dict[str, Any],
    thresholds: SpatialPromotionThresholds = SpatialPromotionThresholds(),
) -> dict[str, Any]:
    """Apply a synthetic-screen gate without claiming real-task superiority."""

    protocol_errors = validate_spatial_comparison(payload)
    paired = paired_spatial_deltas(payload)
    seeds = {int(seed) for seed in payload.get("seeds", [])}
    tasks = set(str(task) for task in payload.get("tasks", []))
    evidence_complete = (
        len(seeds) >= thresholds.min_seeds
        and set(REQUIRED_TASKS).issubset(tasks)
        and not protocol_errors
    )

    explicit_hybrid_error = _max_audit_value(
        payload,
        "explicit_vs_hybrid_max_abs",
    )
    implicit_edge_error = _max_audit_value(
        payload,
        "implicit_edge_independence_max_abs",
    )
    audits_pass = (
        math.isfinite(explicit_hybrid_error)
        and explicit_hybrid_error <= thresholds.max_initial_equivalence_abs
        and math.isfinite(implicit_edge_error)
        and implicit_edge_error <= thresholds.max_edge_independence_abs
    )

    hybrid_local = _mean_for(
        paired,
        task="local_directional",
        arm="hybrid",
        key="mae_relative",
    )
    hybrid_smooth = _mean_for(
        paired,
        task="smooth_gaussian",
        arm="hybrid",
        key="mae_relative",
    )
    hybrid_mixed = _mean_for(
        paired,
        task="mixed",
        arm="hybrid",
        key="mae_relative",
    )
    hybrid_train = statistics.mean(
        [
            float(row["train_time_ratio"])
            for row in paired
            if row["arm"] == "hybrid"
            and math.isfinite(float(row["train_time_ratio"]))
        ]
        or [float("nan")]
    )
    hybrid_inference = statistics.mean(
        [
            float(row["inference_time_ratio"])
            for row in paired
            if row["arm"] == "hybrid"
            and math.isfinite(float(row["inference_time_ratio"]))
        ]
        or [float("nan")]
    )
    hybrid_memory = statistics.mean(
        [
            float(row["training_memory_ratio"])
            for row in paired
            if row["arm"] == "hybrid"
            and math.isfinite(float(row["training_memory_ratio"]))
        ]
        or [float("nan")]
    )

    implicit_local = _mean_for(
        paired,
        task="local_directional",
        arm="implicit",
        key="mae_relative",
    )
    implicit_smooth = _mean_for(
        paired,
        task="smooth_gaussian",
        arm="implicit",
        key="mae_relative",
    )
    implicit_inference = statistics.mean(
        [
            float(row["inference_time_ratio"])
            for row in paired
            if row["arm"] == "implicit"
            and math.isfinite(float(row["inference_time_ratio"]))
        ]
        or [float("nan")]
    )
    implicit_memory = statistics.mean(
        [
            float(row["training_memory_ratio"])
            for row in paired
            if row["arm"] == "implicit"
            and math.isfinite(float(row["training_memory_ratio"]))
        ]
        or [float("nan")]
    )

    hybrid_checks = {
        "local_regression": hybrid_local <= thresholds.max_hybrid_local_regression,
        "smooth_improvement": hybrid_smooth
        <= -thresholds.min_hybrid_smooth_improvement,
        "mixed_improvement": hybrid_mixed
        <= -thresholds.min_hybrid_mixed_improvement,
        "train_time": hybrid_train
        <= 1.0 + thresholds.max_hybrid_train_time_overhead,
        "inference_time": hybrid_inference
        <= 1.0 + thresholds.max_hybrid_inference_time_overhead,
        "training_memory": hybrid_memory
        <= 1.0 + thresholds.max_hybrid_training_memory_overhead,
    }
    implicit_checks = {
        "local_regression": implicit_local
        <= thresholds.max_implicit_local_regression,
        "smooth_improvement": implicit_smooth
        <= -thresholds.min_implicit_smooth_improvement,
        "inference_time": implicit_inference
        <= thresholds.max_implicit_inference_time_ratio,
        "training_memory": implicit_memory
        <= thresholds.max_implicit_training_memory_ratio,
    }

    hybrid_pass = evidence_complete and audits_pass and all(hybrid_checks.values())
    implicit_pass = evidence_complete and audits_pass and all(
        implicit_checks.values()
    )
    if not evidence_complete:
        verdict = "insufficient_synthetic_evidence"
    elif implicit_pass:
        verdict = "implicit_passes_synthetic_replacement_gate"
    elif hybrid_pass:
        verdict = "hybrid_passes_synthetic_candidate_gate"
    else:
        verdict = "retain_explicit_as_canonical"

    return {
        "verdict": verdict,
        "synthetic_only": True,
        "real_task_validation_required": True,
        "evidence_complete": evidence_complete,
        "audits_pass": audits_pass,
        "protocol_errors": protocol_errors,
        "audit_metrics": {
            "explicit_vs_hybrid_max_abs": explicit_hybrid_error,
            "implicit_edge_independence_max_abs": implicit_edge_error,
        },
        "hybrid_metrics": {
            "local_mae_relative": hybrid_local,
            "smooth_mae_relative": hybrid_smooth,
            "mixed_mae_relative": hybrid_mixed,
            "train_time_ratio": hybrid_train,
            "inference_time_ratio": hybrid_inference,
            "training_memory_ratio": hybrid_memory,
        },
        "hybrid_checks": hybrid_checks,
        "implicit_metrics": {
            "local_mae_relative": implicit_local,
            "smooth_mae_relative": implicit_smooth,
            "inference_time_ratio": implicit_inference,
            "training_memory_ratio": implicit_memory,
        },
        "implicit_checks": implicit_checks,
        "thresholds": {
            field: getattr(thresholds, field)
            for field in thresholds.__dataclass_fields__
        },
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    numeric = float(value)
    if not math.isfinite(numeric):
        return "—"
    return f"{numeric:.{digits}f}"


def _percent(value: Any) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{100.0 * float(value):+.2f}%"


def render_spatial_comparison_report(
    payload: dict[str, Any],
    thresholds: SpatialPromotionThresholds = SpatialPromotionThresholds(),
) -> str:
    decision = spatial_promotion_decision(payload, thresholds)
    summaries = payload.get("summaries", [])
    paired = paired_spatial_deltas(payload)
    lines = [
        "# Spatial operator comparison",
        "",
        f"**Synthetic gate verdict:** `{decision['verdict']}`",
        "",
        "> This report is a controlled synthetic operator attribution. It does not "
        "promote an architecture without downstream, leakage-controlled validation.",
        "",
        "## Protocol audit",
        "",
        f"- Device: `{payload.get('device')}`",
        f"- Compute dtype: `{payload.get('compute_dtype')}`",
        f"- Seeds: `{payload.get('seeds')}`",
        f"- Tasks: `{payload.get('tasks')}`",
        f"- Neighbor discovery included: `{payload.get('neighbor_discovery_included')}`",
        f"- Protocol errors: `{decision['protocol_errors']}`",
        f"- Zero-init explicit/hybrid max error: "
        f"`{_fmt(decision['audit_metrics']['explicit_vs_hybrid_max_abs'], 8)}`",
        f"- Implicit edge-independence max error: "
        f"`{_fmt(decision['audit_metrics']['implicit_edge_independence_max_abs'], 8)}`",
        "",
        "## Accuracy and resources",
        "",
        "| Task | Arm | Seeds | MAE | RMSE | Pearson | Train ms | Inference ms | Train peak MiB | Clip fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {task} | {arm} | {seeds} | {mae} ± {mae_std} | "
            "{rmse} ± {rmse_std} | {pearson} | {train} | {infer} | "
            "{memory} | {clip} |".format(
                task=row["task"],
                arm=row["arm"],
                seeds=row["seeds"],
                mae=_fmt(row.get("mean_mae")),
                mae_std=_fmt(row.get("std_mae")),
                rmse=_fmt(row.get("mean_rmse")),
                rmse_std=_fmt(row.get("std_rmse")),
                pearson=_fmt(row.get("mean_pearson")),
                train=_fmt(row.get("mean_median_train_step_ms"), 2),
                infer=_fmt(row.get("mean_inference_ms"), 2),
                memory=_fmt(
                    float(row.get("mean_training_peak_allocated_bytes", 0.0))
                    / (1024.0**2),
                    1,
                ),
                clip=_percent(row.get("mean_clip_fraction")),
            )
        )

    lines.extend(
        [
            "",
            "## Paired differences versus explicit",
            "",
            "Negative MAE difference is better; resource ratios below one are better.",
            "",
            "| Task | Seed | Arm | MAE Δ | Train ratio | Inference ratio | Train-memory ratio |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in paired:
        lines.append(
            f"| {row['task']} | {row['seed']} | {row['arm']} | "
            f"{_percent(row['mae_relative'])} | "
            f"{_fmt(row['train_time_ratio'])} | "
            f"{_fmt(row['inference_time_ratio'])} | "
            f"{_fmt(row['training_memory_ratio'])} |"
        )

    lines.extend(
        [
            "",
            "## Synthetic promotion gates",
            "",
            "### Hybrid candidate",
            "",
        ]
    )
    for key, passed in decision["hybrid_checks"].items():
        lines.append(f"- [{'x' if passed else ' '}] `{key}`")
    lines.extend(["", "### Implicit replacement candidate", ""])
    for key, passed in decision["implicit_checks"].items():
        lines.append(f"- [{'x' if passed else ' '}] `{key}`")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `explicit` remains the reference for sharp, directional local interactions.",
            "- `implicit` tests whether a smooth edge-free spatial kernel can replace the sparse local route.",
            "- `hybrid` tests whether the implicit kernel adds long-range value without damaging local fidelity.",
            "- Passing this report only advances an arm to real-task validation; it is not a canonical promotion.",
            "",
            "## Required next stage",
            "",
            "1. Repeat the winning synthetic arm on at least one point-cloud/field task and one molecular or protein task.",
            "2. Keep split, initialization, parameter schema, optimizer, and compute budget paired.",
            "3. Report task metric, step time, peak memory, gradient clipping, and scaling slopes.",
            "4. Use target- or scaffold/cluster-disjoint validation where leakage is possible.",
            "5. Promote only after the result is stable across seeds and the intended workload family.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "SpatialPromotionThresholds",
    "paired_spatial_deltas",
    "render_spatial_comparison_report",
    "spatial_promotion_decision",
    "validate_spatial_comparison",
]
