from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from equivariant_attention.reproducibility import summarize_repeated_runs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate identical run identities and summarize same-seed metric drift."
        )
    )
    parser.add_argument("metrics", type=Path, nargs="+")
    parser.add_argument("--metric-path", default="val_mae")
    parser.add_argument("--max-metric-span", type=float, required=True)
    parser.add_argument("--min-runs", type=int, default=5)
    parser.add_argument("--expected-mode", choices=["seeded", "strict"], default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        runs = [json.loads(path.read_text()) for path in args.metrics]
        summary = summarize_repeated_runs(
            runs,
            metric_path=args.metric_path,
            max_metric_span=args.max_metric_span,
            min_runs=args.min_runs,
            expected_mode=args.expected_mode,
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        parser.error(str(error))
    summary["input_paths"] = [str(path) for path in args.metrics]
    text = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n")
    print(text)
    return 0 if summary["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
