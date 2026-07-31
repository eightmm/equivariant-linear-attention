#!/usr/bin/env python3
"""Aggregate fresh-process AB/BA canonical ELA resource receipts."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import statistics


EXPECTED_SHAPES = {
    (128, 8, 64, 3),
    (512, 32, 64, 3),
}


def _strict_load(path: Path) -> dict[str, object]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"nonfinite JSON constant in {path}: {value}")
        ),
    )
    if not isinstance(payload, dict):
        raise TypeError(f"resource receipt must be an object: {path}")
    if payload.get("schema_version") != 3:
        raise ValueError(f"unsupported resource receipt schema: {path}")
    if payload.get("experiment") != "canonical_ela_overhead":
        raise ValueError(f"unexpected resource experiment: {path}")
    if payload.get("source_verified") is not True:
        raise ValueError(f"resource receipt did not verify its source: {path}")
    reproducibility = payload.get("reproducibility")
    if not isinstance(reproducibility, Mapping):
        raise ValueError(f"resource receipt lacks reproducibility: {path}")
    if (
        reproducibility.get("mode") != "strict"
        or reproducibility.get("deterministic_algorithms") is not True
        or reproducibility.get("deterministic_warn_only") is not False
        or reproducibility.get("cudnn_deterministic") is not True
        or reproducibility.get("cudnn_benchmark") is not False
    ):
        raise ValueError(f"resource receipt is not strictly deterministic: {path}")
    if payload.get("git_dirty") is not False:
        raise ValueError(f"resource receipt must come from a clean tree: {path}")
    if not str(payload.get("device", "")).startswith("cuda"):
        raise ValueError(f"resource receipt must use CUDA: {path}")
    if payload.get("dtype") != "float32":
        raise ValueError(f"resource receipt must use FP32: {path}")
    if int(payload.get("warmup", -1)) < 10:
        raise ValueError(f"resource receipt requires at least 10 warmups: {path}")
    if int(payload.get("repeats", 0)) < 30:
        raise ValueError(f"resource receipt requires at least 30 repeats: {path}")
    if not isinstance(payload.get("device_fingerprint"), Mapping):
        raise ValueError(f"resource receipt lacks a GPU fingerprint: {path}")
    if payload.get("neighbor_discovery_included") is not False:
        raise ValueError(f"neighbor discovery must be excluded: {path}")
    if payload.get("graph_packing_included") is not False:
        raise ValueError(f"graph packing must be excluded: {path}")
    if payload.get("host_device_preparation_included") is not False:
        raise ValueError(f"host/device preparation must be excluded: {path}")
    if payload.get("models_profiled_one_at_a_time") is not True:
        raise ValueError(f"models were not profiled one at a time: {path}")
    if payload.get("same_common_weights") is not True:
        raise ValueError(f"receipt is not common-weight paired: {path}")
    equivalence = payload.get("functional_equivalence")
    if not isinstance(equivalence, Mapping):
        raise ValueError(f"missing functional equivalence: {path}")
    if any(
        float(equivalence[name]) != 0.0
        for name in (
            "node_output_max_abs",
            "graph_output_max_abs",
            "feature_gradient_max_abs",
            "position_gradient_max_abs",
            "common_parameter_gradient_max_abs",
        )
    ):
        raise ValueError(f"functional pairing is not exact: {path}")
    if equivalence.get("candidate_branch_gradients_finite") is not True:
        raise ValueError(f"candidate branch gradients are not finite: {path}")
    if equivalence.get("candidate_branch_gradients_nonzero") is not True:
        raise ValueError(f"candidate branch gradients are zero: {path}")
    ratios = payload.get("ratios")
    if not isinstance(ratios, Mapping):
        raise ValueError(f"resource receipt lacks ratios: {path}")
    if any(
        ratios.get(name) is None
        for name in (
            "inference_peak_allocated",
            "optimizer_train_step_peak_allocated",
        )
    ):
        raise ValueError(f"CUDA memory measurement is missing: {path}")
    for name in (
        "parameters",
        "inference_median",
        "optimizer_train_step_median",
        "inference_peak_allocated",
        "optimizer_train_step_peak_allocated",
    ):
        value = float(ratios[name])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"resource ratio {name} must be finite and positive")
    for name in ("common_state_sha256", "input_sha256"):
        digest = payload.get(name)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"resource receipt lacks valid {name}: {path}")
    return payload


def _shape_key(payload: Mapping[str, object]) -> tuple[int, int, int, int]:
    return (
        int(payload["nodes"]),
        int(payload["supplied_candidate_degree"]),
        int(payload["width"]),
        int(payload["depth"]),
    )


def summarize(paths: Sequence[Path]) -> dict[str, object]:
    if not paths:
        raise ValueError("at least one resource receipt is required")
    receipts = [(path, _strict_load(path)) for path in paths]
    git_shas = {str(payload["git_sha"]) for _, payload in receipts}
    source_files = {str(payload["source_file"]) for _, payload in receipts}
    fingerprints = {
        json.dumps(payload["device_fingerprint"], sort_keys=True)
        for _, payload in receipts
    }
    if len(git_shas) != 1 or len(source_files) != 1:
        raise ValueError("resource receipts must share one source commit")
    if len(fingerprints) != 1:
        raise ValueError("resource receipts must share one GPU fingerprint")

    groups: dict[
        tuple[int, int, int, int],
        dict[int, dict[str, Mapping[str, object]]],
    ] = defaultdict(lambda: defaultdict(dict))
    for path, payload in receipts:
        order = str(payload["profile_order"])
        if order not in {"control-first", "candidate-first"}:
            raise ValueError(f"invalid profile order: {path}")
        seed = int(payload["seed"])
        pair = groups[_shape_key(payload)][seed]
        if order in pair:
            raise ValueError(f"duplicate shape/seed/order receipt: {path}")
        pair[order] = payload

    if set(groups) != EXPECTED_SHAPES:
        raise ValueError(
            "resource receipts must contain exactly the registered shapes; "
            f"observed={sorted(groups)}"
        )

    shape_results = []
    all_passed = True
    for shape, pairs in sorted(groups.items()):
        if set(pairs) != set(range(5)):
            raise ValueError(f"shape {shape} requires exact seeds 0..4")
        pair_results = []
        for seed, pair in sorted(pairs.items()):
            if set(pair) != {"control-first", "candidate-first"}:
                raise ValueError(f"shape {shape} seed {seed} lacks AB or BA order")
            first_payload = pair["control-first"]
            second_payload = pair["candidate-first"]
            for identity_name in ("common_state_sha256", "input_sha256"):
                if first_payload[identity_name] != second_payload[identity_name]:
                    raise ValueError(
                        f"shape {shape} seed {seed} has mismatched "
                        f"{identity_name}"
                    )
            first = pair["control-first"]["ratios"]
            second = pair["candidate-first"]["ratios"]
            if not isinstance(first, Mapping) or not isinstance(second, Mapping):
                raise ValueError("resource ratios must be objects")

            def balanced(name: str) -> float:
                left = float(first[name])
                right = float(second[name])
                return math.sqrt(left * right)

            memory_values = [
                float(value)
                for receipt in pair.values()
                for name in (
                    "inference_peak_allocated",
                    "optimizer_train_step_peak_allocated",
                )
                if (value := receipt["ratios"][name]) is not None
            ]
            pair_results.append(
                {
                    "seed": seed,
                    "inference_ratio_geometric_mean": balanced(
                        "inference_median"
                    ),
                    "train_step_ratio_geometric_mean": balanced(
                        "optimizer_train_step_median"
                    ),
                    "maximum_individual_latency_ratio": max(
                        float(receipt["ratios"][name])
                        for receipt in pair.values()
                        for name in (
                            "inference_median",
                            "optimizer_train_step_median",
                        )
                    ),
                    "maximum_memory_ratio": (
                        max(memory_values) if memory_values else None
                    ),
                }
            )

        parameter_ratio = max(
            float(payload["ratios"]["parameters"])
            for pair in pairs.values()
            for payload in pair.values()
        )
        inference_ratio = statistics.median(
            result["inference_ratio_geometric_mean"]
            for result in pair_results
        )
        train_ratio = statistics.median(
            result["train_step_ratio_geometric_mean"]
            for result in pair_results
        )
        individual_max = max(
            result["maximum_individual_latency_ratio"]
            for result in pair_results
        )
        memory_values = [
            result["maximum_memory_ratio"]
            for result in pair_results
            if result["maximum_memory_ratio"] is not None
        ]
        memory_max = max(memory_values) if memory_values else None
        checks = {
            "parameters": parameter_ratio <= 1.05,
            "median_inference": inference_ratio <= 1.10,
            "median_optimizer_train_step": train_ratio <= 1.15,
            "maximum_memory": memory_max is None or memory_max <= 1.10,
            "no_individual_latency_outlier": individual_max <= 1.20,
        }
        passed = all(checks.values())
        all_passed = all_passed and passed
        shape_results.append(
            {
                "nodes": shape[0],
                "degree": shape[1],
                "width": shape[2],
                "depth": shape[3],
                "pair_count": len(pair_results),
                "parameter_ratio": parameter_ratio,
                "median_inference_ratio": inference_ratio,
                "median_optimizer_train_step_ratio": train_ratio,
                "maximum_memory_ratio": memory_max,
                "maximum_individual_latency_ratio": individual_max,
                "pairs": pair_results,
                "checks": checks,
                "passed": passed,
            }
        )

    return {
        "schema_version": 1,
        "experiment": "canonical_ela_resource_ab_ba",
        "git_sha": next(iter(git_shas)),
        "source_file": next(iter(source_files)),
        "environment_contract": {
            "device": "cuda",
            "dtype": "float32",
            "minimum_warmup": 10,
            "minimum_repeats": 30,
            "device_fingerprint": json.loads(next(iter(fingerprints))),
        },
        "same_common_weights": True,
        "receipts": [
            {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "shape": list(_shape_key(payload)),
                "seed": int(payload["seed"]),
                "profile_order": str(payload["profile_order"]),
            }
            for path, payload in receipts
        ],
        "shape_results": shape_results,
        "resource_gate": {
            "limits": {
                "parameters": 1.05,
                "median_inference": 1.10,
                "median_optimizer_train_step": 1.15,
                "maximum_memory": 1.10,
                "individual_latency": 1.20,
            },
            "passed": all_passed,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("receipts", nargs="+", type=Path)
    args = parser.parse_args()
    summary = summarize(args.receipts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(summary, indent=2, allow_nan=False)
    args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    if not summary["resource_gate"]["passed"]:
        raise SystemExit("canonical resource AB/BA gate failed")


if __name__ == "__main__":
    main()
