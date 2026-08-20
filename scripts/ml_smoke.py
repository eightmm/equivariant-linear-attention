"""One tiny CPU forward/backward smoke for canonical TriELA."""

from __future__ import annotations

import torch

from equivariant_linear_attention import (
    BiomolecularPairContext,
    ELAGraph,
    TriELA,
)


def main() -> None:
    torch.manual_seed(7)
    model = TriELA(
        "6x0e",
        "2x0e + 1x1o",
        width=16,
        pair_width=8,
        triangle_hidden=8,
        num_stages=1,
        pair_blocks_per_stage=1,
        local_blocks_per_stage=1,
        pair_transition_factor=2,
        pair_dropout=0.0,
        local_points=3,
        max_pair_tokens=6,
        condition_dim=3,
        order_dim=1,
        update_positions=True,
    ).double()
    coordinate_updates = model.stages[0].coordinate_updates
    assert coordinate_updates is not None
    with torch.no_grad():
        coordinate_updates[0].vector.base_weight.fill_(0.1)
    graph = ELAGraph(
        x=torch.randn(9, 6, dtype=torch.float64),
        pos=torch.randn(9, 3, dtype=torch.float64, requires_grad=True),
        batch=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1]),
        group=torch.tensor([0, 0, 1, 1, 0, 0, 0, 1, 1]),
        condition=torch.randn(2, 3, dtype=torch.float64),
        order=torch.linspace(0.0, 1.0, 9, dtype=torch.float64).unsqueeze(-1),
        update_mask=torch.tensor(
            [True, True, False, False, True, True, True, True, True]
        ),
    )
    metadata = BiomolecularPairContext(
        token_index=torch.arange(9),
        chain_id=torch.tensor([0, 0, 1, 1, 0, 0, 0, 1, 1]),
        molecule_type=torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1]),
    )
    output = model.forward_with_aux(graph, metadata)
    assert output.graph.graph_x is not None and output.graph.delta is not None
    loss = (
        output.graph.graph_x.square().mean()
        + output.graph.delta.square().mean()
        + output.distogram_logits.square().mean()
    )
    loss.backward()
    gradients = [
        p.grad for p in model.parameters() if p.requires_grad and p.grad is not None
    ]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise RuntimeError("non-finite or missing gradients")
    print("ml_smoke: ok")


if __name__ == "__main__":
    main()
