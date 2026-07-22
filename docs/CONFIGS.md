# Configuration

## Model

| Field | Default | Constraint |
|---|---:|---|
| `node_dim` | required | positive |
| `hidden_irreps` | `64x0e + 4x1o` | positive scalar/vector channels, no persistent tensors |
| `output_irreps` | `1x0e` | at least one supported `0e/1o/2e` term |
| `num_layers` | 3 | positive |
| `num_heads` | 4 | positive; divides scalar channels |
| `linear_kernel_init` | 0.05 | normal float32; ratio to max is normal and below one |
| `linear_kernel_max` | 1.0 | finite normal positive linear-scale bound |
| `vector_kernel_init` | 0.05 | normal float32; ratio to max is normal and below one |
| `vector_kernel_max` | 1.0 | finite normal positive quadratic-scale bound |
| `kernel_floor` | 1.0 | normal float32 strictly positive pair-kernel floor |
| `kernel_floor_mode` | `fixed` | fixed or graph-size-scaled positive kernel baseline |
| `use_alignment_linear_term` | true | false removes only `beta * (q dot k)` and retains the `beta` constant |
| `use_key_balancing` | true | exactly one key-balancing cycle when true |
| `local_head_counts` | `None` | `None` means all-global; otherwise a length-`num_layers` tuple with each entry in `[0, num_heads]` |
| `global_transport_mode` | `learned` | `learned`, exact graph-mean `uniform`, or attention-residual `none` |
| `local_cutoff` | 2.5 | positive raw-coordinate local cutoff |
| `num_rbf` | 16 | positive number of local Gaussian RBFs |
| `learn_local_radial_gate` | false | frozen radial gate for registered routing studies |
| `global_memory_count` | 1 | positive; CLI registers 1, 4, and 8 |
| `use_memory_interaction` | false | registered only for the middle global stage of three-layer `lgl` |
| `memory_assignment_temperature` | 1.0 | positive bounded-assignment temperature |
| `memory_assignment_scale` | 2.5 | positive centroid-refinement distance scale |
| `memory_interaction_cutoff` | 2.5 | positive fixed center-coupling cutoff |
| `use_radial_trace` | false | reserve and populate the exact relative radial second-moment scalar |
| `residual_scale_init` | 0.1 | nonnegative |
| `eps` | `1e-12` | positive; normalization occurs in float32+ |

The ratio-2 equivariant FFN remains fixed. All settings configure the same
`EquivariantAttention` class. `local_head_counts=None`, one memory, memory
interaction off, and radial trace off are the public defaults. The training
presets map `ggg` to `(0,0,0)`, `lgg` to `(H,0,0)`, `ggl` to `(0,0,H)`,
`lgl` to `(H,0,H)`, and `lll` to `(H,H,H)` for three layers. `uniform`
broadcasts the exact graph mean of the same moment sufficient statistics.
`none` bypasses the attention updater in all-global blocks and keeps only the
pointwise FFN; it does not execute global geometry preprocessing.

`inverse_graph_size` keeps the compatibility-oriented option name but scales
the full shifted positive global baseline `(c + beta + delta*beta*t)` by
`1/N_g`; content and `gamma*t^2` stay unscaled. It is rejected with key
balancing and is never used as a proxy for local receiver degree. Enabling
memory interaction requires learned transport and the registered `lgl` route;
merely raising the
memory count with interaction off is algebraically the incumbent. The M=4/8
CLI values are implemented diagnostic arms, not promoted configurations: the
frozen Stage-0 pair-gate probe currently blocks broader interacting-memory
experiments and constructing one emits a Stage-0-blocked warning. The internal
M-shared invariant router is present with the same
parameter schema for every memory count, but no residual-coupling value is a
public configuration because none of the registered candidates passed the
unchanged full Stage-0 thresholds.

Kernel controls deliberately reject positive float32 subnormals: inverse-logit
initialization and `c/N_g` can otherwise round them to zero and invalidate the
strictly positive denominator contract. This does not apply to the scale-first
local cutoff and memory geometry controls, whose subnormal behavior has
separate finite-value and gradient tests.

## Training probe

The CLI exposes dataset/split/model seeds, width, depth, heads, AdamW learning
rate and weight decay, gradient clipping, device, bf16 autocast, target
normalization, alignment/balancing/floor controls, routing, local cutoff/RBF,
memory, radial-trace, and test-evaluation policy. The explicit public flags are
`--routing ggg/lgg/ggl/lgl/lll`,
`--global-transport-mode learned/uniform/none`, `--memory-count 1/4/8`,
`--memory-interaction`, `--radial-trace`,
`--edge-conditioned-local-transport`, `--precompute-local-edges`, and opt-in
`--evaluate-test`. Edge-conditioned local transport requires all local heads,
matching hidden vector/head counts, and cannot be combined with
`--pairwise-local-content` or `--learn-local-radial-gate`. Precomputed QM9
radius candidates avoid forward-time pair discovery but do not make the data
loader asymptotically linear. `GraphBatch` collation validates supplied edge
contents and enables the model's trusted linear hot path; direct model calls
validate by default, and callers must not assert `edge_index_is_validated` for
unchecked external tensors.
`--bounded-diagnostics` recomputes every active local layer/head outside timed
training; `--diagnostic-sample-count` defaults to 32 deterministic validation
graphs selected after sorting by `(node_count, dataset_index)`.
`--benchmark-model internal_static_egnn_baseline` selects only the private
same-harness comparison baseline; it does not add a public model API. Its
model-specific default is width 91 (the factorized default remains 64), and
nondefault factorized-only controls are rejected instead of silently ignored.
`metrics.run_config` records every run-defining argument and marks the EGNN arm
`official_reproduction=false`. It also records whether global transport and
global geometry actually executed; local-only routes report
`not_applicable_no_global_heads` regardless of the configured transport label.

`scripts/run_registered_transport_study.py` freezes the 2026-07-19 six-arm
screen, fifteen-arm five-seed transport confirmation, conditional five-arm
private EGNN comparison, exact QM9 data/split hashes, test-disabled policy, and
1,500-second GPU ceiling. A scientific threshold failure exits successfully
but records `transport_locked=false`; infrastructure, provenance, nonfinite, or
budget failures stop the runner.
