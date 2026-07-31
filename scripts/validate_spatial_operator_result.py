#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from equivariant_attention import validate_spatial_comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a spatial operator comparison result bundle"
    )
    parser.add_argument("result", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.result.read_text(encoding="utf-8"))
    errors = validate_spatial_comparison(payload)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("spatial operator comparison protocol: valid")


if __name__ == "__main__":
    main()
