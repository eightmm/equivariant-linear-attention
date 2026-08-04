# Relation-conditioned transport ablation

`scripts/ablate_relation_transport.py` is a deterministic paired mechanics
ablation for the canonical ELA relation conditioner. It answers a narrow
question: can the private relation-conditioned local transport use invariant
edge labels when geometry, node features, topology, model state, and relation
cutoffs are otherwise identical?

The synthetic dataset contains repeated four-node 3D graphs. Every graph has
the same features, coordinates, and complete directed topology. All edges in a
graph have relation ID zero or one, and the graph target is respectively `-1`
or `+1`. Therefore `edge_type` is the only varying model input.

The two arms have identical parameter schemas and bit-identical initial state:

- `candidate` trains the complete ELA;
- `control` keeps only `relation_score_bias`, `relation_radial_scale`, and
  `relation_value_gate` at their zero/identity initialization.

Both arms still receive the same relation IDs, use two declared relation types,
share one cutoff, and use the same sparse edges. The local scalar output
projection is set to `0.1` in both arms before cloning. This shared bootstrap
opens the deliberately zero-initialized local residual so the small experiment
measures the relation conditioner instead of spending its first updates opening
an unrelated downstream projection.

Run the bounded CPU default:

```bash
uv run --locked python scripts/ablate_relation_transport.py \
  --output artifacts/relation-transport-ablation.json
```

The JSON receipt includes the seed, deterministic mode, task hash, full initial
state and parameter-schema hashes, initial output agreement, exact frozen
parameter names, per-arm losses and timings, and post-training relation
parameter magnitudes. Repeating the command with the same software and CPU
configuration should reproduce the numerical fields; wall-clock timings are
observational and need not be identical.

This is deliberately not a molecular benchmark. A positive result establishes
that the relation path is active and useful on a label-only identification
task. It does not establish downstream accuracy, generalization, efficiency,
or superiority over another architecture. Those claims require a preregistered
real-data paired experiment with an untouched evaluation split.
