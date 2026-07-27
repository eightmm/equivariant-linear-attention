#!/usr/bin/env python3
"""Run the preregistered receptive-field/radial-resolution QM9 screen.

The incumbent hybrid route gives its local heads a 2.5-Angstrom cutoff, which
covers only 41.8% of same-molecule atom pairs on cached QM9, while the private
static EGNN control has been consuming the complete same-graph edge list. This
screen separates two confounded factors at fixed features, split, seed, and
optimizer: how much of the molecule the local heads may see (`local_cutoff`),
and how finely the radial basis resolves the covalent range at that cutoff
(`local_rbf_spacing`). It also runs the EGNN control twice -- once with its
historical complete topology and once on the same 5.0-Angstrom candidates as
the wide attention arms -- so the reported EGNN gap is decomposed into an
architecture part and a topology part.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence


PACKET_ID = "receptive-field-20260725"
MAX_STEPS = 500
MAX_GPU_SECONDS = 900.0
MINIMUM_IMPROVEMENT_EV = 0.020
# A wider cutoff buys pairs it must pay for: cached QM9 mean non-self candidates
# per node rise from 5.14 at 2.5 A to 12.09 at 5.0 A. The registered ceiling is
# therefore looser than the v3 packet's 1.25x, and is a declared cost budget
# rather than a claim that the wide arms are free.
MAXIMUM_RESOURCE_RATIO = 2.0

ATTENTION_ARMS: dict[str, tuple[str, str]] = {
    "incumbent": ("2.5", "squared"),
    "wide_cutoff": ("5.0", "squared"),
    "distance_rbf": ("2.5", "distance"),
    "wide_distance": ("5.0", "distance"),
}
EGNN_ARMS: dict[str, str | None] = {
    "egnn_complete": None,
    "egnn_matched": "5.0",
}
ARMS: dict[str, str] = {
    **{arm: "attention" for arm in ATTENTION_ARMS},
    **{arm: "egnn" for arm in EGNN_ARMS},
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--steps", type=int, default=MAX_STEPS)
    parser.add_argument("--budget-seconds", type=float, default=MAX_GPU_SECONDS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.steps <= MAX_STEPS:
        parser.error(f"--steps must lie in [1, {MAX_STEPS}]")
    if not 0.0 < args.budget_seconds <= MAX_GPU_SECONDS:
        parser.error(f"--budget-seconds must lie in (0, {MAX_GPU_SECONDS}]")
    return args


def build_command(
    arm: str,
    *,
    output: Path,
    steps: int,
    device: str,
) -> list[str]:
    try:
        family = ARMS[arm]
    except KeyError as error:
        raise ValueError(f"unknown arm: {arm}") from error
    if family == "attention":
        cutoff, spacing = ATTENTION_ARMS[arm]
        switches = [
            "--routing",
            "lgl",
            "--global-transport-mode",
            "learned",
            "--hidden-dim",
            "64",
            "--num-heads",
            "4",
            "--gated-local-transport",
            "--grouped-invariant-normalization",
            "--local-cutoff",
            cutoff,
            "--local-rbf-spacing",
            spacing,
            "--precompute-local-edges",
        ]
    else:
        cutoff = EGNN_ARMS[arm]
        switches = [
            "--benchmark-model",
            "internal_static_egnn_baseline",
            "--hidden-dim",
            "91",
        ]
        if cutoff is not None:
            switches += ["--local-cutoff", cutoff, "--precompute-local-edges"]
    return [
        "uv",
        "run",
        "--locked",
        "python",
        "scripts/train_compare.py",
        "--dataset",
        "qm9",
        "--data-root",
        "data/qm9",
        "--qm9-target-index",
        "4",
        "--num-samples",
        "130000",
        "--train-size",
        "110000",
        "--val-size",
        "10000",
        "--batch-size",
        "64",
        "--steps",
        str(steps),
        "--num-layers",
        "3",
        "--split-seed",
        "42",
        "--model-seed",
        "42",
        "--determinism",
        "strict",
        "--device",
        device,
        "--amp-dtype",
        "none",
        "--skip-test-eval",
        "--metrics-out",
        str(output),
        *switches,
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    commands = {
        arm: build_command(
            arm,
            output=args.output_dir / f"{arm}.json",
            steps=args.steps,
            device=args.device,
        )
        for arm in ARMS
    }
    plan = {
        "schema_version": 1,
        "packet_id": PACKET_ID,
        "dataset": "QM9 gap",
        "split": "seeded_random_row_warm_start_seed42",
        "raw_feature_contract": (
            "identical node_feats, pos, target, and split; candidate topology is "
            "the screened factor and is reported per arm"
        ),
        "factors": {
            "local_cutoff_angstrom": ["2.5", "5.0"],
            "local_rbf_spacing": ["squared", "distance"],
        },
        "arms": list(ARMS),
        "arm_families": dict(ARMS),
        "steps": args.steps,
        "device": args.device,
        "budget_seconds": args.budget_seconds,
        "minimum_improvement_eV": MINIMUM_IMPROVEMENT_EV,
        "maximum_train_step_latency_ratio": MAXIMUM_RESOURCE_RATIO,
        "maximum_peak_allocation_ratio": MAXIMUM_RESOURCE_RATIO,
        "validation_evaluated": True,
        "test_evaluated": False,
        "commands": commands,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "plan.json", plan)
    records: dict[str, dict[str, object]] = {}
    started = time.monotonic()
    for arm, command in commands.items():
        remaining = args.budget_seconds - (time.monotonic() - started)
        if remaining <= 0.0:
            raise TimeoutError("receptive-field QM9 GPU budget exhausted")
        _run_command(command, timeout=remaining)
        record = _load_json(args.output_dir / f"{arm}.json")
        _validate_record(record, arm=arm, steps=args.steps)
        records[arm] = record

    attention_hashes = {
        str(records[arm]["initial_state_sha256"]) for arm in ATTENTION_ARMS
    }
    if len(attention_hashes) != 1:
        raise RuntimeError("attention arms do not share one initialization")
    decision = screen_decision(records)
    summary = {
        **plan,
        "status": "completed",
        "elapsed_seconds": time.monotonic() - started,
        "source_sha256": _source_hash(),
        "records": {
            arm: {
                "val_mae_eV": record["val_mae"],
                "train_probe_mae_eV": record["train_probe"]["mae"],
                "parameter_count": record["parameter_count"],
                "step_latency_median_seconds": record[
                    "step_latency_median_seconds"
                ],
                "peak_cuda_memory_bytes": record["peak_cuda_memory_bytes"],
                "edge_topology": record["run_config"].get("edge_topology"),
                "local_cutoff": record["run_config"].get("local_cutoff"),
                "local_rbf_spacing": record["run_config"].get("local_rbf_spacing"),
                "clip_fraction": record["gradient_clipping"]["clip_fraction"],
                "mean_pre_clip_grad_norm": record["gradient_clipping"][
                    "pre_clip_grad_norm_mean"
                ],
                "initial_state_sha256": record["initial_state_sha256"],
            }
            for arm, record in records.items()
        },
        "decision": decision,
        "validation_evaluated": True,
        "test_evaluated": False,
    }
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def screen_decision(
    records: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    incumbent = records["incumbent"]
    incumbent_mae = float(incumbent["val_mae"])
    incumbent_latency = float(incumbent["step_latency_median_seconds"])
    incumbent_memory = float(incumbent["peak_cuda_memory_bytes"])
    candidates: list[dict[str, object]] = []
    for arm in ("wide_cutoff", "distance_rbf", "wide_distance"):
        record = records[arm]
        mae = float(record["val_mae"])
        improvement = incumbent_mae - mae
        latency_ratio = (
            float(record["step_latency_median_seconds"]) / incumbent_latency
        )
        memory_ratio = float(record["peak_cuda_memory_bytes"]) / incumbent_memory
        finite = all(
            math.isfinite(value)
            for value in (mae, improvement, latency_ratio, memory_ratio)
        )
        candidates.append(
            {
                "arm": arm,
                "val_mae_eV": mae,
                "improvement_eV": improvement,
                "step_latency_ratio": latency_ratio,
                "peak_allocation_ratio": memory_ratio,
                "finite": finite,
                "promotion_screen_passed": (
                    finite
                    and improvement >= MINIMUM_IMPROVEMENT_EV
                    and latency_ratio <= MAXIMUM_RESOURCE_RATIO
                    and memory_ratio <= MAXIMUM_RESOURCE_RATIO
                ),
            }
        )
    admitted = [row for row in candidates if row["promotion_screen_passed"]]
    selected = (
        min(admitted, key=lambda row: float(row["val_mae_eV"]))
        if admitted
        else None
    )
    egnn_complete = float(records["egnn_complete"]["val_mae"])
    egnn_matched = float(records["egnn_matched"]["val_mae"])
    best_attention = min(
        float(records[arm]["val_mae"]) for arm in ATTENTION_ARMS
    )
    return {
        "incumbent_val_mae_eV": incumbent_mae,
        "candidates": candidates,
        "promotion_screen_passed": selected is not None,
        "promotion_selected_arm": None if selected is None else selected["arm"],
        "egnn_complete_val_mae_eV": egnn_complete,
        "egnn_matched_val_mae_eV": egnn_matched,
        # Positive means the historical complete-graph EGNN number was helped by
        # its topology advantage rather than by its architecture alone.
        "egnn_topology_confound_eV": egnn_matched - egnn_complete,
        "best_attention_val_mae_eV": best_attention,
        "best_attention_minus_matched_egnn_eV": best_attention - egnn_matched,
    }


def _validate_record(
    record: Mapping[str, object],
    *,
    arm: str,
    steps: int,
) -> None:
    if record.get("dataset") != "qm9" or record.get("steps") != steps:
        raise RuntimeError(f"{arm} emitted an unexpected dataset or update count")
    if record.get("model_seed") != 42 or record.get("split_seed") != 42:
        raise RuntimeError(f"{arm} changed a registered seed")
    if record.get("test_evaluated") is not False:
        raise RuntimeError(f"{arm} evaluated test labels")
    config = record.get("run_config")
    if not isinstance(config, Mapping):
        raise RuntimeError(f"{arm} is missing run_config")
    if ARMS[arm] == "attention":
        cutoff, spacing = ATTENTION_ARMS[arm]
        expected: dict[str, object] = {
            "routing": "lgl",
            "precompute_local_edges": True,
            "gated_local_transport": True,
            "grouped_invariant_normalization": True,
            "local_cutoff": float(cutoff),
            "local_rbf_spacing": spacing,
            "test_evaluated": False,
        }
    else:
        matched_cutoff = EGNN_ARMS[arm]
        expected = {
            "model": "internal_static_egnn_baseline",
            "precompute_local_edges": matched_cutoff is not None,
            "local_cutoff": (
                float(matched_cutoff)
                if matched_cutoff is not None
                else "not_applicable"
            ),
            "test_evaluated": False,
        }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"{arm} emitted unexpected {key}")


def _run_command(command: Sequence[str], *, timeout: float) -> None:
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    print(f"running: {shlex.join(command)}", flush=True)
    process = subprocess.Popen(
        list(command),
        env=environment,
        start_new_session=True,
    )
    try:
        return_code = process.wait(timeout=timeout)
    except BaseException:
        _terminate_process_group(process)
        raise
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=2.0)


def _source_hash() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__).resolve(),
        root / "scripts" / "train_compare.py",
        root / "src" / "equivariant_attention" / "moment.py",
        root / "src" / "equivariant_attention" / "training.py",
        root / "PROJECT.md",
        root / "uv.lock",
    ]
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"receptive-field QM9 screen failed: {error}", file=sys.stderr)
        raise
