# Evidence-only provenance correction

The completed gated and gated-plus-grouped QM9 JSON records contain the
descriptive `run_config.gated_local_aggregation` value
`cutoff_sum_over_sqrt_degree_plus_explicit_mass`.

That label was stale. The hashed source executed by those records already used

```text
sum_j f_c(u_ij) message_ij / sqrt(1 + sum_j f_c(u_ij))
```

and retained `sum_j f_c(u_ij)^2` only as an input to the explicit mass
projection. The implementation, singleton/degree tests, and raw source hash are
authoritative. The subsequent source changes only the descriptive run-config
string to
`cutoff_sum_over_sqrt_one_plus_cutoff_mass_plus_explicit_mass`; no metric or
raw record was rewritten.
