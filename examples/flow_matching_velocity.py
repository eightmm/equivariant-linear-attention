from __future__ import annotations

import torch

from equivariant_linear_attention import ELA, ELAGraph


def center_per_graph(
    coordinates: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    totals = coordinates.new_zeros((num_graphs, 3))
    totals.index_add_(0, batch, coordinates)
    counts = torch.bincount(batch, minlength=num_graphs).clamp_min(1)
    return coordinates - totals[batch] / counts[batch, None].to(coordinates.dtype)


def graph_balanced_velocity_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    per_node = (prediction - target).square().mean(dim=-1)
    per_graph = per_node.new_zeros((num_graphs,))
    per_graph.index_add_(0, batch, per_node)
    counts = torch.bincount(batch, minlength=num_graphs).clamp_min(1)
    return (per_graph / counts.to(per_graph.dtype)).mean()


def main() -> None:
    torch.manual_seed(7)
    counts = torch.tensor([5, 7])
    batch = torch.repeat_interleave(torch.arange(2), counts)
    features = torch.randn(12, 8)
    x0 = center_per_graph(torch.randn(12, 3), batch, num_graphs=2)
    x1 = center_per_graph(torch.randn(12, 3), batch, num_graphs=2)

    time = torch.rand(2, 1)
    x_t = (1.0 - time[batch]) * x0 + time[batch] * x1
    target_velocity = x1 - x0

    model = ELA(
        "8x0e",
        "1x1o",
        condition_dim=1,
        width=32,
        depth=2,
        cutoff=5.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    graph = ELAGraph(x=features, pos=x_t, batch=batch, condition=time)
    output = model(graph)
    velocity = model.split_output(output.x)["1o"].squeeze(-2)
    loss = graph_balanced_velocity_loss(
        velocity,
        target_velocity,
        batch,
        num_graphs=2,
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    dt = 0.01
    x_next = output.pos + dt * velocity.detach()
    print(
        f"loss={float(loss.detach()):.6f}, "
        f"velocity_shape={tuple(velocity.shape)}, "
        f"next_shape={tuple(x_next.shape)}"
    )


if __name__ == "__main__":
    main()
