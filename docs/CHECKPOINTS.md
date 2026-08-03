# Checkpoint contract

ELA supports portable checkpoints made from plain configuration data and a
`state_dict`. Pickled model, config, or batch objects are not a supported
cross-version format.

## Minimal schema

```python
from dataclasses import asdict
import torch

payload = {
    "format_version": 1,
    "package": "equivariant-linear-attention",
    "config": asdict(model.config),
    "state_dict": model.state_dict(),
    "metadata": {
        "source_revision": git_sha,
        "step": step,
    },
}
torch.save(payload, checkpoint_path)
```

Training applications may add optimizer, scheduler, scaler, and random-number
states under their own documented keys. Those fields are not part of the model
API and should not be required for inference.

## Safe reconstruction

```python
import torch

from equivariant_linear_attention import (
    ELA,
    ELAConfig,
    ELAFeatures,
    SparseGeometry,
)

payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
if payload["format_version"] != 1:
    raise ValueError("unsupported ELA checkpoint format")

raw = dict(payload["config"])
geometry = SparseGeometry(**raw.pop("geometry"))
features = ELAFeatures(**raw.pop("features"))
model = ELA(ELAConfig(**raw, geometry=geometry, features=features))
model.load_state_dict(payload["state_dict"], strict=True)
```

Only load checkpoints from a trusted source. `weights_only=True` narrows the
deserialization surface, while the explicit format check and strict state load
prevent silent partial reconstruction.

## Compatibility

- Source-module moves and private class renames must preserve `state_dict` keys.
- Public configuration changes require a format-version decision and migration
  note.
- A tensor shape or key change requires an explicit migration; never use
  `strict=False` as an undocumented compatibility policy.
- Historical advanced-stack checkpoints use the guarded migration described in
  [`MIGRATION_TO_ELA.md`](MIGRATION_TO_ELA.md).

Store large checkpoints outside Git. Record the source revision, data identity,
and training configuration alongside any scientific result.
