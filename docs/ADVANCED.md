# Advanced configuration

Ordinary users construct `ELA` directly. The advanced module exists for
reproducible geometry and feature policies without expanding the package-root
API:

```python
from equivariant_linear_attention import ELA
from equivariant_linear_attention.advanced import (
    ELAConfig,
    ELAFeatures,
    SparseGeometry,
)

config = ELAConfig(
    input_irreps="32x0e",
    output_irreps="1x0e",
    width=128,
    depth=8,
    geometry=SparseGeometry(
        cutoff=6.0,
        max_neighbors=64,
    ),
    features=ELAFeatures(condition_dim=16),
)
model = ELA.from_config(config)
```

`max_neighbors` is part of `SparseGeometry` because it changes the graph, not
just execution. `skin` may be used for uncapped moving-coordinate radius
graphs; it cannot be combined with `max_neighbors`, because a capped cached
candidate list cannot preserve exact nearest-neighbour membership.

Mixed irreps, invariant ST5 losses, semantic order, and conservative-force
helpers also live in `equivariant_linear_attention.advanced`. They do not
create alternate model classes or execution paths.

The numerical `ELABatch`, receiver-major CSR graph, layer implementation, and
backend controls are internal interfaces. They remain importable from their
implementation modules for repository benchmarks, but are not stable user API.
