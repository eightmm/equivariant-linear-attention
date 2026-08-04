# Architecture-lane paired ablations

`scripts/ablate_architecture_lanes.py` provides bounded deterministic CPU
ablations for the two canonical lanes that cannot be judged by a private toggle
test alone:

- `cg12`: the Cartesian `l=1 (x) l=2 -> l=1,2` tensor-closure projections;
- `multiscale`: the two nested radial envelopes over one prepared sparse CSR.

Each experiment starts from two bit-identical ELA instances with the same
parameter schema and task tensors. All shared parameters are frozen. The
candidate enables and learns only the selected zero-initialized lane; the
control keeps those same parameters at zero and privately disables the lane.
The private toggles are not added to the root package or the public `ELA`
constructor.

The `cg12` task has zero scalar inputs and nonzero polar/axial vector plus
even/odd rank-2 tensor inputs. Its deterministic teacher differs from the
identity model only through the four `l1_l2_*_out` projections, so matching the
teacher requires the intended vector-tensor coupling.

The `multiscale` task repeats one four-node feature template and one complete
directed CSR pattern at eight radial scales. Its teacher differs only through
`local_scale_score_mix` and `local_scale_value_mix`. A shared nonzero local
scalar readout is applied before cloning so the radial lane is observable in a
small frozen-backbone run.

Run both bounded defaults:

```bash
uv run --locked python scripts/ablate_architecture_lanes.py \
  --lane all \
  --output artifacts/architecture-lane-ablations.json
```

Use `--lane cg12` or `--lane multiscale` to run one arm pair. The JSON receipt
records state/schema/task hashes, exact initial output agreement, target and
trainable parameter names, teacher signal strength, losses, timing, and the
post-training lane magnitudes. The existing relation-conditioned experiment
remains separate in `scripts/ablate_relation_transport.py` because its label
identification control can keep the shared backbone trainable without a bypass.

These are teacher-reconstruction mechanics tests. They establish that a lane
is connected, trainable, and functionally distinguishable from its identity
control under a deliberately diagnostic input. They do not establish molecular
accuracy, generalization, efficiency, or superiority. Such claims require a
preregistered real-data paired experiment with an untouched evaluation split.
