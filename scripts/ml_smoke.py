"""One-batch forward/backward smoke for canonical ELA."""

from __future__ import annotations

import torch

from equivariant_linear_attention import ELA, ELAGraph


def main() -> None:
    torch.manual_seed(7)
    model = ELA(
        "6x0e",
        "2x0e + 1x1o",
        width=32,
        depth=2,
        condition_dim=3,
        order_dim=1,
        update_positions=True,
    ).double()
    assert model.coordinate_update is not None
    with torch.no_grad():
        model.coordinate_update.vector.base_weight.fill_(0.1)
    graph = ELAGraph(
        x=torch.randn(9, 6, dtype=torch.float64),
        pos=torch.randn(9, 3, dtype=torch.float64, requires_grad=True),
        batch=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1]),
        group=torch.tensor([0, 0, 1, 1, 0, 0, 0, 1, 1]),
        condition=torch.randn(2, 3, dtype=torch.float64),
        order=torch.linspace(0.0, 1.0, 9, dtype=torch.float64).unsqueeze(-1),
        update_mask=torch.tensor([True, True, False, False, True, True, True, True, True]),
    )
    output = model(graph)
    assert output.graph_x is not None and output.delta is not None
    loss = output.graph_x.square().mean() + output.delta.square().mean()
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise RuntimeError("non-finite or missing gradients")
    print("ml_smoke: ok")


if __name__ == "__main__":
    main()
