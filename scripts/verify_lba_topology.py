"""Verify the frozen ATOM3D-LBA ID30 candidate-topology identity.

The LBA packets recorded a cross-run topology defect: identical samples and
identical code produced 32,303,245 and 32,303,244 directed edges in different
processes. Run this before any multi-seed LBA claim. It rebuilds the official
train and validation topology in fresh subprocesses under different BLAS thread
budgets and requires one edge count and one hash.

    uv run python scripts/verify_lba_topology.py \
      artifacts/topology-contract/id30-identity.json

Add `--expect-edge <sha256>` to require the edge-only identity. The legacy
`--expect` option still checks the historical joint sample-ID/edge digest. No
label is read, no GPU is used, and the test split remains inadmissible.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from equivariant_attention.benchmarking import GraphSample  # noqa: E402
from equivariant_attention.pdbbind import (  # noqa: E402
    ATOM3D_LBA_REVISION,
    edge_topology_sha256,
    load_atom3d_lba_split_samples,
    sample_identity_sha256,
    segment_balanced_knn_edge_index,
    topology_sha256,
)

LOCAL_CUTOFF_ANGSTROM = 6.0
INTRA_K = 16
CROSS_K = 16
THREAD_BUDGETS = ("1", "4")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/atom3d_lba"))
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--expect", default=None)
    parser.add_argument("--expect-edge", default=None)
    parser.add_argument("--worker", action="store_true")
    return parser.parse_args(argv)


def _load(root: Path, split: str, limit: int | None) -> list[GraphSample]:
    indices = None if limit is None else tuple(range(limit))
    return load_atom3d_lba_split_samples(
        root,
        split=split,
        revision=ATOM3D_LBA_REVISION,
        indices=indices,
    )


def _with_edges(samples: Sequence[GraphSample]) -> list[GraphSample]:
    result: list[GraphSample] = []
    for sample in samples:
        if sample.readout_mask is None:
            raise ValueError("ATOM3D-LBA sample requires a ligand readout mask")
        result.append(
            replace(
                sample,
                edge_index=segment_balanced_knn_edge_index(
                    sample.pos,
                    sample.readout_mask,
                    intra_k=INTRA_K,
                    cross_k=CROSS_K,
                    cutoff=LOCAL_CUTOFF_ANGSTROM,
                ),
            )
        )
    return result


def build_identity(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    train = _with_edges(_load(args.data_root, "train", args.train_limit))
    validation = _with_edges(_load(args.data_root, "val", args.val_limit))
    samples = [*train, *validation]
    edge_count = sum(
        int(sample.edge_index.shape[1])
        for sample in samples
        if sample.edge_index is not None
    )
    return {
        "train_size": len(train),
        "validation_size": len(validation),
        "edge_count": edge_count,
        "topology_sha256": topology_sha256(samples),
        "edge_topology_sha256": edge_topology_sha256(samples),
        "sample_identity_sha256": sample_identity_sha256(samples),
        "build_seconds": time.perf_counter() - started,
        "torch_threads": int(os.environ.get("OMP_NUM_THREADS", "0")),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker:
        print(json.dumps(build_identity(args)))
        return 0

    runs: list[dict[str, object]] = []
    for threads in THREAD_BUDGETS:
        environment = dict(os.environ)
        environment["OMP_NUM_THREADS"] = threads
        environment["MKL_NUM_THREADS"] = threads
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            str(args.output_path),
            "--data-root",
            str(args.data_root),
            "--worker",
        ]
        if args.train_limit is not None:
            command += ["--train-limit", str(args.train_limit)]
        if args.val_limit is not None:
            command += ["--val-limit", str(args.val_limit)]
        completed = subprocess.run(
            command,
            capture_output=True,
            check=True,
            env=environment,
            text=True,
        )
        record = json.loads(completed.stdout.strip().splitlines()[-1])
        record["requested_threads"] = int(threads)
        runs.append(record)
        print(
            f"threads={threads} edges={record['edge_count']} "
            f"edge_sha256={record['edge_topology_sha256']} "
            f"seconds={float(record['build_seconds']):.1f}",
            file=sys.stderr,
            flush=True,
        )

    hashes = {str(run["topology_sha256"]) for run in runs}
    edge_hashes = {str(run["edge_topology_sha256"]) for run in runs}
    sample_hashes = {str(run["sample_identity_sha256"]) for run in runs}
    counts = {str(run["edge_count"]) for run in runs}
    reproducible = (
        len(hashes) == 1
        and len(edge_hashes) == 1
        and len(sample_hashes) == 1
        and len(counts) == 1
    )
    matches_expected = None if args.expect is None else args.expect in hashes
    matches_expected_edge = (
        None if args.expect_edge is None else args.expect_edge in edge_hashes
    )
    summary = {
        "revision": ATOM3D_LBA_REVISION,
        "cutoff_angstrom": LOCAL_CUTOFF_ANGSTROM,
        "intra_k": INTRA_K,
        "cross_k": CROSS_K,
        "runs": runs,
        "reproducible": reproducible,
        "expected_topology_sha256": args.expect,
        "matches_expected": matches_expected,
        "expected_edge_topology_sha256": args.expect_edge,
        "matches_expected_edge": matches_expected_edge,
        "status": (
            "passed"
            if reproducible
            and matches_expected is not False
            and matches_expected_edge is not False
            else "failed"
        ),
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(runs[0], sort_keys=True))
    if not reproducible:
        print("topology identity drifted across processes", file=sys.stderr)
        return 1
    if matches_expected is False:
        print("topology identity differs from the expected hash", file=sys.stderr)
        return 1
    if matches_expected_edge is False:
        print("edge topology differs from the expected hash", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
