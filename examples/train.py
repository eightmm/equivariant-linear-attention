"""Minimal graph-property training example."""

from __future__ import annotations

import torch

from equivariant_linear_attention import ELA, ELAGraph

model = ELA("16x0e", "1x0e", width=64, depth=4)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for step in range(20):
    graph = ELAGraph(
        x=torch.randn(48, 16),
        pos=torch.randn(48, 3),
        batch=torch.arange(4).repeat_interleave(12),
        y=torch.randn(4, 1),
    )
    output = model(graph)
    assert output.graph_x is not None and graph.y is not None
    loss = torch.nn.functional.mse_loss(output.graph_x, graph.y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    print(step, float(loss.detach()))
