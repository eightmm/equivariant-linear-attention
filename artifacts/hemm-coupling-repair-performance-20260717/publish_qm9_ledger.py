#!/usr/bin/env python3
"""Create a public ledger copy without a machine-local repository path."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[1]
SOURCE = RUN_DIR / "qm9-runs.jsonl"
OUTPUT = RUN_DIR / "qm9-runs-public.jsonl"
RECEIPT = RUN_DIR / "qm9-runs-public-provenance.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root_prefix = f"{ROOT.resolve()}/"
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines()]
    public_rows = []
    replacements = 0
    for row in rows:
        public = json.loads(json.dumps(row))
        command = []
        for value in public["cmd"]:
            if isinstance(value, str) and root_prefix in value:
                replacements += value.count(root_prefix)
                value = value.replace(root_prefix, "")
            command.append(value)
        public["cmd"] = command
        public_rows.append(public)
    assert replacements == len(rows)
    serialized = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
        for row in public_rows
    )
    assert root_prefix not in serialized
    OUTPUT.write_text(serialized, encoding="utf-8")
    receipt = {
        "schema_version": 1,
        "source_sha256": sha256(SOURCE),
        "public_sha256": sha256(OUTPUT),
        "row_count": len(rows),
        "replacement_count": replacements,
        "transformation": (
            "Replace the exact machine-local repository-root prefix only in "
            "command arguments; preserve every note, metric, hash, status, and row."
        ),
    }
    RECEIPT.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
