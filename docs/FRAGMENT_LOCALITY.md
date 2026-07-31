# Fragment and size-locality diagnostic

The explicit compact-cutoff local operator depends only on relative coordinates
of candidate pairs. Adding a disconnected fragment farther than the cutoff does
not change the kernel or message among existing nodes.

The edge-free Gaussian--Taylor approximation uses graph-centered coordinates

\[
z_i=x_i-\mu_g.
\]

The exact Gaussian factorization is origin independent after all terms cancel,
but a finite Taylor truncation does not preserve that cancellation exactly.
Adding a distant fragment can therefore change the approximate kernel between
nodes that were already present. A soft long-range kernel can additionally
transmit the fragment values even when it is far away.

This diagnostic separates those effects.

## Quantities

Let `A` be the original fragment and `B` a distant added fragment. The script
records:

1. approximate implicit kernel drift within the original `A x A` block;
2. original-node message drift when `B` has zero values;
3. original-node message drift when `B` has random values;
4. compact-cutoff kernel drift within `A x A`;
5. compact-cutoff message drift when `B` has zero values.

The zero-value implicit message drift isolates feature-origin/truncation effects.
The random-value drift additionally includes intended soft long-range coupling.
The compact-cutoff values should be numerically zero when the fragment distance
exceeds the cutoff.

## Command

```bash
uv run python scripts/evaluate_fragment_locality.py \
  --base-nodes 64 \
  --fragment-nodes 16 \
  --fragment-distance 20 \
  --value-width 16 \
  --cutoff 1.75 \
  --scales 2,4,8 \
  --output artifacts/spatial-operator-comparison/<run-id>/fragment-locality.json
```

## Interpretation

There is no universal acceptable implicit drift threshold. It depends on the
intended semantics:

- atomistic energy, force, disconnected-fragment, and size-extensive tasks
  require a very strict threshold;
- global shape or field tasks may intentionally permit long-range coupling;
- a hybrid short-range/long-range model should preserve the explicit path for
  local or extensive quantities and gate the implicit path by task semantics.

For molecular and protein validation, repeat the diagnostic over several graph
sizes and fragment distances. At minimum use

```text
fragment distance: 2, 4, 8, 16 times the explicit cutoff
fragment size:     1, 8, 32, and a size comparable to the base graph
```

Record the drift curve rather than a single point.

## Promotion rule

An implicit-only replacement must not be promoted for a size-consistent task if
zero-value fragment insertion changes the predicted original-subsystem quantity
outside the predeclared numerical tolerance.

A hybrid model may still be acceptable when:

1. the task is not required to be additive over disconnected fragments; or
2. the implicit contribution is disabled for extensive/conservative heads; or
3. the spatial feature is reformulated using an origin-independent or periodic
   Fourier/Ewald basis and passes the same diagnostic.

This test complements, but does not replace, real-task fragment/additivity and
force-consistency evaluation.
