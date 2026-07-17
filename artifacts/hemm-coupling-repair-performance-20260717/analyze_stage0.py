#!/usr/bin/env python3
"""Create compact decision evidence from incumbent and repaired Stage-0 outputs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


COUPLINGS = ("radial", "identity", "lambda_0.10", "lambda_0.25", "lambda_0.50")


def _minimum(heads: list[dict[str, object]], path: tuple[str, ...]) -> float:
    values = []
    for head in heads:
        value: object = head
        for key in path:
            value = value[key]  # type: ignore[index]
        values.append(float(value))
    return min(values)


def _summarize_counterfactuals(suite: dict[str, object]) -> dict[str, object]:
    summaries: dict[str, object] = {}
    lanes = suite["lanes"]
    for scenario in suite["scenarios"]:
        records: dict[str, list[dict[str, object]]] = {name: [] for name in COUPLINGS}
        diagnoses: Counter[str] = Counter()
        for lane in lanes:
            if lane["scenario"] != scenario:
                continue
            for arm in lane["arms"]:
                diagnoses[str(arm["diagnosis"])] += 1
                for name in COUPLINGS:
                    counterfactual = arm["counterfactuals"][name]
                    heads = counterfactual["activation"]["heads"]
                    records[name].append(
                        {
                            "passed": bool(counterfactual["decision"]["passed"]),
                            "mi": _minimum(
                                heads,
                                ("assignment", "mutual_information_over_log_m"),
                            ),
                            "pair_gate_d": _minimum(
                                heads,
                                ("pair_gate", "centered_frobenius_ratio"),
                            ),
                            "messages": float(
                                counterfactual["mechanism"]["messages"]["aggregate"]
                            ),
                            "post_middle": float(
                                counterfactual["mechanism"]["post_middle"]["aggregate"]
                            ),
                            "gradient_min": min(
                                float(counterfactual["mechanism"]["gradients"][key])
                                for key in ("scalars", "vectors", "positions")
                            ),
                            "output": float(counterfactual["relative_output_rms"]),
                        }
                    )
        summaries[str(scenario)] = {
            "diagnoses": dict(sorted(diagnoses.items())),
            "counterfactuals": {
                name: {
                    "arm_count": len(values),
                    "pass_count": sum(value["passed"] for value in values),
                    "mutual_information.min": min(value["mi"] for value in values),
                    "mutual_information.max": max(value["mi"] for value in values),
                    "pair_gate_d.min": min(value["pair_gate_d"] for value in values),
                    "pair_gate_d.max": max(value["pair_gate_d"] for value in values),
                    "messages_rsym.min": min(value["messages"] for value in values),
                    "post_middle_rsym.min": min(
                        value["post_middle"] for value in values
                    ),
                    "gradient_rsym.min": min(
                        value["gradient_min"] for value in values
                    ),
                    "output_relative_rms.min": min(
                        value["output"] for value in values
                    ),
                }
                for name, values in records.items()
            },
        }
    return summaries


def _summarize_incumbent(incumbent: dict[str, object]) -> dict[str, object]:
    records: dict[str, list[float]] = {
        name: [] for name in ("radial", "identity", "lambda_0.10", "lambda_0.25", "lambda_0.50")
    }
    mutual_information: list[float] = []
    for row in incumbent["rows"]:
        for head in row["heads"]:
            mutual_information.append(
                float(head["assignment"]["mutual_information_over_log_m"])
            )
            for name in records:
                records[name].append(
                    float(head["pair_gates"][name]["centered_frobenius_ratio"])
                )
    return {
        "base_commit": incumbent["base_commit"],
        "source_sha256": incumbent["source_sha256"],
        "row_count": len(incumbent["rows"]),
        "mutual_information.min": min(mutual_information),
        "mutual_information.max": max(mutual_information),
        "pair_gate_d": {
            name: {"min": min(values), "max": max(values)}
            for name, values in records.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    incumbent = json.loads(args.incumbent.read_text(encoding="utf-8"))
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    scenarios = _summarize_counterfactuals(suite)
    summary = {
        "schema_version": 1,
        "test_evaluated": False,
        "incumbent": _summarize_incumbent(incumbent),
        "candidate": {
            "source_sha256": suite["source_sha256"],
            "lane_count": len(suite["lanes"]),
            "hidden_dims": suite["hidden_dims"],
            "seeds": suite["seeds"],
            "scenarios": scenarios,
            "semantic_identity_active": suite["semantic_identity_active"],
            "selected_identity_mix": suite["selected_identity_mix"],
            "scientific_decision": suite["scientific_decision"],
        },
    }
    args.summary_out.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    aligned = scenarios["aligned"]["counterfactuals"]
    rows = []
    for name in ("identity", "lambda_0.10", "lambda_0.25", "lambda_0.50"):
        metric = aligned[name]
        rows.append(
            f"| {name} | {metric['pass_count']}/{metric['arm_count']} | "
            f"{metric['mutual_information.min']:.3e} | "
            f"{metric['pair_gate_d.min']:.3e} | "
            f"{metric['messages_rsym.min']:.3e} | "
            f"{metric['post_middle_rsym.min']:.3e} | "
            f"{metric['gradient_rsym.min']:.3e} | "
            f"{metric['output_relative_rms.min']:.3e} |"
        )
    report = "\n".join(
        [
            "# HEMM counterfactual and repair result",
            "",
            "Decision: `block_interacting_memory_arms`",
            "Test labels evaluated: no",
            "",
            "The clean incumbent reproduction confirms radial-coupling collapse. "
            "Identity coupling is numerically nonconstant, but the registered-width "
            "worst heads remain below the frozen material-activation threshold. The "
            "preregistered shared invariant MLP router increases assignment diversity "
            "in some seeds but is not robust: no residual candidate passes all aligned "
            "width/seed/M arms, and the semantic-only identity diagnostic also fails.",
            "",
            "| Coupling | Aligned arms passed | min I_slot | min gate D | min message R_sym | min post R_sym | min gradient R_sym | min output RMS |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "The fixed candidates are therefore rejected without changing lambda or any "
            "threshold. M=4/M=8 accuracy runs and memory-performance claims remain "
            "blocked. This does not block the independent ggg-versus-lgl backbone study.",
            "",
            "Raw evidence: `stage0-suite.json` (compressed for publication). Compact "
            "machine-readable evidence: `stage0-summary.json`.",
        ]
    )
    args.report_out.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
