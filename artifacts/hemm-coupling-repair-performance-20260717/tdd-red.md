# TDD red phase

Command:

```text
uv run --locked pytest -q tests/test_diagnostics.py tests/test_local_memory.py tests/test_probe_memory_activation.py
```

Result: exit 2 in 0.88 seconds on 2026-07-17. Collection failed because the
newly specified `memory_center_summary` did not exist. The same test change also
specifies marginal/conditional/MI assignment diagnostics, coupling variation,
shared invariant-router schema, residual-coupling invariants and validation,
the four probe scenarios, and symmetric relative RMS. No production code had
been changed when this failure was observed.
