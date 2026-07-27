#!/usr/bin/env python3
"""Merge isolated real-LBA resource profiles into one CTP gate decision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy


PROFILE = runpy.run_path(
    str(Path(__file__).with_name("profile_lba_train_step.py"))
)
EXPECTED_ARMS = ("candidate", "persistent_2e", "ctp")
IDENTITY_FIELDS = (
    "dataset",
    "dataset_revision",
    "split",
    "sample_ids",
    "sample_identity_sha256",
    "batch_size",
    "node_count",
    "edge_count",
    "edge_index_sha256",
    "device",
    "dtype",
    "model_seed",
    "warmup",
    "profiled_steps",
    "timed_steps",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("profiles", nargs=3, type=Path)
    args = parser.parse_args()

    records = [json.loads(path.read_text()) for path in args.profiles]
    reference = records[0]
    for field in IDENTITY_FIELDS:
        if any(record[field] != reference[field] for record in records[1:]):
            raise ValueError(f"isolated profiles disagree on {field}")

    arms: dict[str, object] = {}
    sources: dict[str, object] = {}
    for path, record in zip(args.profiles, records, strict=True):
        if record.get("status") != "completed":
            raise ValueError(f"profile did not complete: {path}")
        profile_arms = record.get("arms")
        if not isinstance(profile_arms, dict) or len(profile_arms) != 1:
            raise ValueError(f"profile must contain exactly one arm: {path}")
        arm, arm_record = next(iter(profile_arms.items()))
        if arm not in EXPECTED_ARMS or arm in arms:
            raise ValueError(f"unexpected or duplicate arm: {arm}")
        arms[arm] = arm_record
        sources[arm] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    if set(arms) != set(EXPECTED_ARMS):
        raise ValueError("isolated profiles do not cover the registered arms")

    comparison = PROFILE["_ctp_resource_comparison"](arms)
    result = {
        "schema_version": 1,
        "status": "completed",
        **{field: reference[field] for field in IDENTITY_FIELDS},
        "measurement_route": "isolated_fresh_process_per_arm",
        "arms": arms,
        "comparison": comparison,
        "source_profiles": sources,
        "validation_evaluated": False,
        "test_evaluated": False,
        "limitations": {
            "single_real_train_batch": True,
            "accuracy_not_inferred": True,
            "host_order_drift_avoided": True,
            "wall_and_device_metrics_retained": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
