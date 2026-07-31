from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
import statistics
from typing import Any


ARMS = ("explicit", "implicit", "hybrid")
REQUIRED_TASKS = ("local_directional", "smooth_gaussian", "mixed")


@dataclass(frozen=True, slots=True)
class SpatialPromotionThresholds:
    """Synthetic-screen thresholds relative to the explicit arm."""

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
        for field in fields(self):
            if field.name == "min_seeds":
                continue
            value = float(getattr(self, field.name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{field.name} must be finite and nonnegative"
                )


def _relative(candidate: float, reference: float) -> float:
    if not math.isfinite(candidate) or not math.isfinite(reference):
        return float("nan")
    if reference == 0.0:
        return 0.0 if candidate == 0.0 else float("inf")
    return (candidate - reference) / abs(reference)


def _ratio(candidate: float, reference: float) -> float:
    if not math.isfinite(candidate) or not math.isfinite(reference):
        return float("nan")
    if reference == 0.0:
        return 1.0 if candidate == 0.0 else float("inf")
    return candidate / reference


def validate_spatial_comparison(payload: dict[str, Any]) -> list[str]:
    """Return all protocol violations found in one result bundle."""

    errors: list[str] = []
    if payload.get("experiment") != "spatial_operator_comparison":
        errors.append("unexpected experiment name")
    if int(payload.get("schema_version", -1)) < 2:
        errors.append("schema_version must be at least two")
    if tuple(payload.get("arms", ())) != ARMS:
        errors.append("canonical arm order is explicit, implicit, hybrid")

    protocol = payload.get("protocol", {})
    expected = {
        "same_parameter_schema": True,
        "same_initial_state_per_task_seed": True,
        "same_train_validation_data_per_task_seed": True,
        "validation_or_test_labels_used_for_training": False,
        "no_edge_graph_prepared_outside_timed_forward": True,
    }
    for key, required in expected.items():
        if protocol.get(key) is not required:
            errors.append(f"protocol mismatch: {key} must be {required}")

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for run in payload.get("runs", []):
        grouped.setdefault(
            (str(run.get("task")), int(run.get("seed", -1))),
            [],
        ).append(run)
    for key, runs in grouped.items():
        if {str(run.get("arm")) for run in runs} != set(ARMS):
            errors.append(f"incomplete arm set for {key}")
        if len({str(run.get("initial_state_sha256")) for run in runs}) != 1:
            errors.append(f"initial-state hash mismatch for {key}")
        counts = {
            int(run.get("audit", {}).get("parameter_count", -1))
            for run in runs
        }
        if len(counts) != 1:
            errors.append(f"parameter-count mismatch for {key}")
    return errors


def _paired_runs(payload: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for run in payload.get("runs", []):
        key = (str(run["task"]), int(run["seed"]))
        result.setdefault(key, {})[str(run["arm"])] = run
    return result


def paired_spatial_deltas(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return paired candidate-versus-explicit differences per task and seed."""

    rows: list[dict[str, Any]] = []
    for (task, seed), arms in sorted(_paired_runs(payload).items()):
        if set(arms) != set(ARMS):
            continue
        reference = arms["explicit"]
        for arm in ("implicit", "hybrid"):
            candidate = arms[arm]
            rows.append(
                {
                    "task": task,
                    "seed": seed,
                    "arm": arm,
                    "mae_relative": _relative(
                        float(candidate["best_validation"]["mae"]),
                        float(reference["best_validation"]["mae"]),
                    ),
                    "rmse_relative": _relative(
                        float(candidate["best_validation"]["rmse"]),
                        float(reference["best_validation"]["rmse"]),
                    ),
                    "train_time_ratio": _ratio(
                        float(candidate["median_train_step_ms"]),
                        float(reference["median_train_step_ms"]),
                    ),
                    "inference_time_ratio": _ratio(
                        float(candidate["inference_ms"]),
                        float(reference["inference_ms"]),
                    ),
                    "training_memory_ratio": _ratio(
                        float(candidate["training_peak_allocated_bytes"]),
                        float(reference["training_peak_allocated_bytes"]),
                    ),
                    "inference_memory_ratio": _ratio(
                        float(candidate["inference_peak_allocated_bytes"]),
                        float(reference["inference_peak_allocated_bytes"]),
                    ),
                }
            )
    return rows


def _mean(
    rows: list[dict[str, Any]],
    *,
    arm: str,
    key: str,
    task: str | None = None,
) -> float:
    values = [
        float(row[key])
        for row in rows
        if row["arm"] == arm
        and (task is None or row["task"] == task)
        and math.isfinite(float(row[key]))
    ]
    return statistics.mean(values) if values else float("nan")


def _max_audit(payload: dict[str, Any], key: str) -> float:
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
    """Apply a bounded synthetic gate, never a downstream promotion claim."""

    errors = validate_spatial_comparison(payload)
    paired = paired_spatial_deltas(payload)
    evidence_complete = (
        len(set(int(seed) for seed in payload.get("seeds", [])))
        >= thresholds.min_seeds
        and set(REQUIRED_TASKS).issubset(
            str(task) for task in payload.get("tasks", [])
        )
        and not errors
    )
    hybrid_identity = _max_audit(
        payload,
        "explicit_vs_hybrid_max_abs",
    )
    implicit_independence = _max_audit(
        payload,
        "implicit_edge_independence_max_abs",
    )
    audits_pass = (
        math.isfinite(hybrid_identity)
        and hybrid_identity <= thresholds.max_initial_equivalence_abs
        and math.isfinite(implicit_independence)
        and implicit_independence <= thresholds.max_edge_independence_abs
    )

    hybrid_metrics = {
        "local_mae_relative": _mean(
            paired,
            task="local_directional",
            arm="hybrid",
            key="mae_relative",
        ),
        "smooth_mae_relative": _mean(
            paired,
            task="smooth_gaussian",
            arm="hybrid",
            key="mae_relative",
        ),
        "mixed_mae_relative": _mean(
            paired,
            task="mixed",
            arm="hybrid",
            key="mae_relative",
        ),
        "train_time_ratio": _mean(
            paired,
            arm="hybrid",
            key="train_time_ratio",
        ),
        "inference_time_ratio": _mean(
            paired,
            arm="hybrid",
            key="inference_time_ratio",
        ),
        "training_memory_ratio": _mean(
            paired,
            arm="hybrid",
            key="training_memory_ratio",
        ),
    }
    implicit_metrics = {
        "local_mae_relative": _mean(
            paired,
            task="local_directional",
            arm="implicit",
            key="mae_relative",
        ),
        "smooth_mae_relative": _mean(
            paired,
            task="smooth_gaussian",
            arm="implicit",
            key="mae_relative",
        ),
        "inference_time_ratio": _mean(
            paired,
            arm="implicit",
            key="inference_time_ratio",
        ),
        "training_memory_ratio": _mean(
            paired,
            arm="implicit",
            key="training_memory_ratio",
        ),
    }

    hybrid_checks = {
        "local_regression": hybrid_metrics["local_mae_relative"]
        <= thresholds.max_hybrid_local_regression,
        "smooth_improvement": hybrid_metrics["smooth_mae_relative"]
        <= -thresholds.min_hybrid_smooth_improvement,
        "mixed_improvement": hybrid_metrics["mixed_mae_relative"]
        <= -thresholds.min_hybrid_mixed_improvement,
        "train_time": hybrid_metrics["train_time_ratio"]
        <= 1.0 + thresholds.max_hybrid_train_time_overhead,
        "inference_time": hybrid_metrics["inference_time_ratio"]
        <= 1.0 + thresholds.max_hybrid_inference_time_overhead,
        "training_memory": hybrid_metrics["training_memory_ratio"]
        <= 1.0 + thresholds.max_hybrid_training_memory_overhead,
    }
    implicit_checks = {
        "local_regression": implicit_metrics["local_mae_relative"]
        <= thresholds.max_implicit_local_regression,
        "smooth_improvement": implicit_metrics["smooth_mae_relative"]
        <= -thresholds.min_implicit_smooth_improvement,
        "inference_time": implicit_metrics["inference_time_ratio"]
        <= thresholds.max_implicit_inference_time_ratio,
        "training_memory": implicit_metrics["training_memory_ratio"]
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
        "protocol_errors": errors,
        "audit_metrics": {
            "explicit_vs_hybrid_max_abs": hybrid_identity,
            "implicit_edge_independence_max_abs": implicit_independence,
        },
        "hybrid_metrics": hybrid_metrics,
        "hybrid_checks": hybrid_checks,
        "implicit_metrics": implicit_metrics,
        "implicit_checks": implicit_checks,
        "thresholds": asdict(thresholds),
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def _percent(value: Any) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{100.0 * float(value):+.2f}%"


def render_spatial_comparison_report(
    payload: dict[str, Any],
    thresholds: SpatialPromotionThresholds = SpatialPromotionThresholds(),
) -> str:
    decision = spatial_promotion_decision(payload, thresholds)
    lines = [
        "# Spatial operator comparison",
        "",
        f"**Synthetic gate verdict:** `{decision['verdict']}`",
        "",
        "> This is a controlled synthetic operator attribution, not a downstream "
        "architecture promotion.",
        "",
        "## Protocol audit",
        "",
        f"- Device: `{payload.get('device')}`",
        f"- Compute dtype: `{payload.get('compute_dtype')}`",
        f"- Seeds: `{payload.get('seeds')}`",
        f"- Tasks: `{payload.get('tasks')}`",
        f"- Protocol errors: `{decision['protocol_errors']}`",
        f"- Explicit/hybrid zero-init error: "
        f"`{_fmt(decision['audit_metrics']['explicit_vs_hybrid_max_abs'], 8)}`",
        f"- Implicit edge-independence error: "
        f"`{_fmt(decision['audit_metrics']['implicit_edge_independence_max_abs'], 8)}`",
        "",
        "## Accuracy and resources",
        "",
        "| Task | Arm | Seeds | MAE | RMSE | Pearson | Train ms | Inference ms | Train peak MiB | Clip |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("summaries", []):
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
    for row in paired_spatial_deltas(payload):
        lines.append(
            f"| {row['task']} | {row['seed']} | {row['arm']} | "
            f"{_percent(row['mae_relative'])} | "
            f"{_fmt(row['train_time_ratio'])} | "
            f"{_fmt(row['inference_time_ratio'])} | "
            f"{_fmt(row['training_memory_ratio'])} |"
        )

    lines.extend(["", "## Synthetic gates", "", "### Hybrid", ""])
    for key, passed in decision["hybrid_checks"].items():
        lines.append(f"- [{'x' if passed else ' '}] `{key}`")
    lines.extend(["", "### Implicit replacement", ""])
    for key, passed in decision["implicit_checks"].items():
        lines.append(f"- [{'x' if passed else ' '}] `{key}`")

    lines.extend(
        [
            "",
            "## Interpretation and next stage",
            "",
            "- `explicit` is the reference for sharp directional local interactions.",
            "- `implicit` tests replacement by a smooth edge-free kernel.",
            "- `hybrid` tests an additional smooth long-range residual.",
            "- A passing arm must next be tested on at least one general 3D task and one molecular/protein task.",
            "- Real-task validation must preserve paired splits, initialization, parameter schema, optimizer, and compute budget.",
            "- Leakage-controlled splits and task-specific latency/memory measurements are mandatory before promotion.",
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
