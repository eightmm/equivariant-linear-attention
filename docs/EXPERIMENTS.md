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
