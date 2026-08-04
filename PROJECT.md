# Project contract

Equivariant Linear Attention exposes exactly one public execution path:

```text
ELAGraph -> ELA -> ELAGraph
```

```python
from equivariant_linear_attention import ELA, ELAGraph
```

`ELA` is the only public model. `ELAGraph` is the only public input, batch, and
output container. Coordinate updates are declared on `ELA` with
`update_positions`; there is no runtime refinement object or alternate model.

The numerical core uses a private packed receiver-major CSR representation,
but that representation is not part of ordinary user code. Public edges are
always sender-to-receiver. Representations are declared only with irreps through
`l <= 2`.

The canonical layer is fixed global-plus-local exact transport with parity-valid
updates, tensor closure, residuals, and an equivariant FFN. Learned branch
routing, persistent edge state, dense attention, and automatically promoted
Triton execution are not part of the canonical architecture.
