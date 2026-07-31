#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from equivariant_attention.spatial_comparison import (
    SpatialPromotionThresholds,
    render_spatial_comparison_report,
    spatial_promotion_decision,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render an auditable explicit/implicit/hybrid comparison"
    )
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decision-json", type=Path)
    parser.add_argument("--min-seeds", type=int, default=3)
    parser.add_argument("--max-local-regression", type=float, default=0.02)
    parser.add_argument("--min-smooth-improvement", type=float, default=0.05)
    parser.add_argument("--min-mixed-improvement", type=float, default=0.02)
    parser.add_argument("--max-time-overhead", type=float, default=0.25)
    parser.add_argument("--max-memory-overhead", type=float, default=0.25)
    args = parser.parse_args()

    payload = json.loads(args.result.read_text(encoding="utf-8"))
    thresholds = SpatialPromotionThresholds(
        min_seeds=args.min_seeds,
        max_hybrid_local_regression=args.max_local_regression,
        min_hybrid_smooth_improvement=args.min_smooth_improvement,
        min_hybrid_mixed_improvement=args.min_mixed_improvement,
        max_hybrid_train_time_overhead=args.max_time_overhead,
        max_hybrid_inference_time_overhead=args.max_time_overhead,
        max_hybrid_training_memory_overhead=args.max_memory_overhead,
        max_implicit_local_regression=args.max_local_regression,
        min_implicit_smooth_improvement=args.min_smooth_improvement,
        max_implicit_inference_time_ratio=1.0,
        max_implicit_training_memory_ratio=1.0,
    )
    report = render_spatial_comparison_report(payload, thresholds)
    decision = spatial_promotion_decision(payload, thresholds)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    if args.decision_json is not None:
        args.decision_json.parent.mkdir(parents=True, exist_ok=True)
        args.decision_json.write_text(
            json.dumps(decision, indent=2),
            encoding="utf-8",
        )
    print(report)


if __name__ == "__main__":
    main()
