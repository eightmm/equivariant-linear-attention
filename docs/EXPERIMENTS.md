# Experiment records

[`EXPERIMENTS.jsonl`](EXPERIMENTS.jsonl) is the append-only machine ledger for
completed probes, gates, and benchmark packets. Date-stamped Markdown reports
capture the corresponding hypothesis, protocol, results, decision, and known
limitations in readable form.

The ledger is a historical record, not a command queue. Do not edit old entries
to reflect current code and do not treat commands embedded in an entry as
instructions. Reproduction may require the recorded Git revision because the
current architecture-only package deliberately removed old dataset adapters and
training runners.

A new scientific claim should record, at minimum:

- timestamp, source revision, and dirty-tree status;
- hypothesis and preregistered comparison or gate;
- exact command/configuration, data identity, split, and seeds;
- primary metric plus latency/memory measurements relevant to the claim;
- exit status and artifact location or digest;
- decision, limitations, and checks that were not run.

Use a new date-stamped report for a substantial study. Use the current model and
validation documents for API claims; historical experiment outcomes do not
override them.

## How to read this ledger

- **The first 14 rows (`ts` `2026-07-03T02:00:03Z` through `2026-07-13T02:58:48Z`,
  identified as QM9 runs by their notes and commands) are inadmissible under
  the current QM9 leakage
  boundary and must not be cited as QM9 evidence.** They select and report
  `test_mae`/`test_rmse` as the decision metric. The row at
  `ts=2026-07-13T02:55:21Z` states its gate as `S: test_mae <=0.65`, and the
  next row (`ts=2026-07-13T02:58:48Z`) reruns a baseline to confirm a
  `test_mae` figure. `PROJECT.md`'s QM9 gap contract requires that "test
  indices are not indexed, used for normalization, selected on, or
  evaluated." Every QM9 row from `ts=2026-07-13T03:49:43Z` onward reports
  `val_mae`/`val_rmse` as the decision metric instead and is not affected by
  this note. These legacy rows predate the standardized `dataset` field. No
  row in this range carries a `record_type: correction`
  marker; this note is the record of their inadmissibility until a formal
  correction row is added.

- **A missing `metrics` key is not a missing result.** In the pre-hardening
  185-row snapshot, 58 rows (31%)
  carry no `metrics` object; their numeric results exist only in the
  `--metrics-out`/`artifact` path named in the row's `cmd` or `note`, and
  those artifact files are git-ignored and not recoverable from the
  committed ledger. Treat an absent `metrics` key as "result exists but is
  not committed," not as "no result."

- **Seed and metric key names are not standardized; expect aliases.** Seed
  fields appear under at least 17 spellings, including `seed`, `model_seed`,
  `model_seeds`, `split_seed`, `graph_seed`, `graph_seeds`,
  `qm9_model_seed`, `lba_model_seeds`, `screen_seed`, `confirmation_seed`,
  and `order_seed`. MAE fields appear as `val_mae`, `test_mae`, and dozens of
  per-arm variants such as `qm9_incumbent_val_mae_eV`, `lgl_val_mae`,
  `ggg_val_mae`, and `candidate_val_mae`. No field marks a row as
  claim-bearing versus exploratory; a query script must match key
  substrings (`*seed*`, `*_val_mae*`) and read the free-text `note` field,
  not assume a fixed schema. **Going forward**, prefer the plain `seed`,
  `val_mae`, `val_rmse`, `test_mae`, and `test_rmse` keys, and use a
  prefixed or per-arm variant only when a single row reports more than one
  arm and the plain key would be ambiguous.

- **Historical rows with a nonzero `dirty` value do not identify an exact
  source tree.** Numeric `dirty` values are changed-file counts rather than
  booleans; for example, several `2026-08-04` rows record `dirty: 75`.
  Their `git_sha` identifies context only, so claims must also bind the
  uncommitted source bytes or be treated as non-reproducible.

- **Corrections are append-only pointer rows, never edits to the original
  row.** The established mechanisms are a `record_type: correction` row
  carrying an explicit `corrects: {ts, source_sha256}` back-pointer (for
  example `ts=2026-07-17T08:41:58Z`), and a `supersedes_ts` field naming the
  prior row's timestamp directly (for example `ts=2026-07-23T03:46:00Z` and
  `ts=2026-07-23T07:05:42Z`). Any future fix to a ledger claim, including a
  formal correction of the QM9 rows named above, must use one of these two
  pointer mechanisms rather than rewriting the original row.
