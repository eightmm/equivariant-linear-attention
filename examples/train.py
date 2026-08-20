"""Minimal token-level TriELA training step."""

import torch

from equivariant_linear_attention import ELAGraph, TriELA

model = TriELA(
    "16x0e",
    "1x0e",
    width=32,
    pair_width=16,
    triangle_hidden=16,
    num_stages=1,
    pair_blocks_per_stage=1,
    local_blocks_per_stage=1,
    local_points=8,
    max_pair_tokens=64,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

graph = ELAGraph(
    x=torch.randn(24, 16),
    pos=torch.randn(24, 3),
    batch=torch.tensor([0] * 12 + [1] * 12),
    y=torch.randn(2, 1),
)
output = model.forward_with_aux(graph)
assert output.graph.graph_x is not None and graph.y is not None
loss = torch.nn.functional.mse_loss(output.graph.graph_x, graph.y)
loss.backward()
optimizer.step()
