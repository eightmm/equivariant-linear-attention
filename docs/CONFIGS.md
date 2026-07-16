# Configuration

## Model

| Field | Default | Constraint |
|---|---:|---|
| `node_dim` | required | positive |
| `hidden_irreps` | `64x0e + 4x1o` | positive scalar/vector channels, no persistent tensors |
| `output_irreps` | `1x0e` | at least one supported `0e/1o/2e` term |
| `num_layers` | 3 | positive |
| `num_heads` | 4 | positive; divides scalar channels |
| `linear_kernel_init` | 0.05 | positive and smaller than `linear_kernel_max` |
| `linear_kernel_max` | 1.0 | positive upper bound for the linear angular scale |
| `vector_kernel_init` | 0.05 | positive and smaller than `vector_kernel_max` |
| `vector_kernel_max` | 1.0 | positive upper bound for the quadratic angular scale |
| `kernel_floor` | 1.0 | strictly positive pair-kernel floor |
| `use_linear_kernel` | true | false only for the quadratic-only P1 control |
| `use_key_balancing` | true | false only for the row-normalized P1 control |
| `residual_scale_init` | 0.1 | nonnegative |
| `eps` | `1e-12` | positive; normalization occurs in float32+ |

The ratio-2 equivariant FFN remains fixed. Linear-kernel and balancing switches
exist only to run the four registered P1 variants within the same implementation.

## Training probe

The CLI exposes dataset/split/model seeds, width, depth, heads, AdamW learning
rate and weight decay, gradient clipping, device, bf16 autocast, target
normalization, linear-kernel/balancing controls, and test-evaluation policy.
`metrics.run_config` records every run-defining argument.
