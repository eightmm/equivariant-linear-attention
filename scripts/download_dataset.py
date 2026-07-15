from __future__ import annotations

import argparse
import json
from pathlib import Path

from equivariant_attention.benchmarking import load_qm9_samples


def main() -> None:
    args = parse_args()
    if args.dataset != "qm9":
        raise ValueError(f"unsupported dataset: {args.dataset}")

    samples = load_qm9_samples(args.data_root, target_index=args.qm9_target_index, limit=args.limit)
    metadata = {
        "dataset": args.dataset,
        "data_root": str(args.data_root),
        "num_loaded": len(samples),
        "node_dim": samples[0].node_feats.shape[1] if samples else None,
        "target_dim": samples[0].target.numel() if samples else None,
        "first_sample_id": samples[0].sample_id if samples else None,
    }
    text = json.dumps(metadata, indent=2, sort_keys=True)
    print(text)
    if args.metadata_out is not None:
        args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_out.write_text(text + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and verify benchmark datasets.")
    parser.add_argument("--dataset", choices=["qm9"], default="qm9")
    parser.add_argument("--data-root", type=Path, default=Path("data/qm9"))
    parser.add_argument("--qm9-target-index", type=int, default=4)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--metadata-out", type=Path, default=Path("outputs/qm9_metadata.json"))
    return parser.parse_args()


if __name__ == "__main__":
    main()
