# Configuration

## Model

| Field | Default | Constraint |
|---|---:|---|
| `node_dim` | required | positive |
| `hidden_irreps` | `64x0e + 4x1o` | positive scalar/vector channels, no persistent tensors |
| `output_irreps` | `1x0e` | at least one supported `0e/1o/2e` term |
| `num_layers` | 3 | positive |
| `num_heads` | 4 | positive; divides scalar channels |
| `vector_kernel_init` | 0.05 | positive and smaller than `vector_kernel_max` |
| `vector_kernel_max` | 1.0 | positive upper bound for the quadratic angular scale |
| `residual_scale_init` | 0.1 | nonnegative |
| `eps` | `1e-12` | positive; normalization occurs in float32+ |

The balancing cycle and ratio-2 equivariant FFN are fixed architecture choices,
not user-facing switches.

## Training probe

The CLI exposes dataset/split/model seeds, width, depth, heads, AdamW learning
rate and weight decay, gradient clipping, device, bf16 autocast, target
normalization, and test-evaluation policy. `metrics.run_config` records every
run-defining argument.
