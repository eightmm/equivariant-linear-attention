from __future__ import annotations

import torch

from equivariant_attention import ELA, ELABatch


def center_per_graph(
    coordinates: torch.Tensor,
    batch_index: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    """Remove the per-graph translation gauge from packed coordinates."""

    totals = coordinates.new_zeros((num_graphs, 3))
    totals.index_add_(0, batch_index, coordinates)
    counts = torch.bincount(batch_index, minlength=num_graphs).clamp_min(1)
    return coordinates - totals[batch_index] / counts[batch_index, None].to(
        coordinates.dtype
    )


def graph_balanced_velocity_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch_index: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    """Average nodes within each graph before averaging the mini-batch."""

    per_node = (prediction - target).square().mean(dim=-1)
    per_graph = per_node.new_zeros((num_graphs,))
    per_graph.index_add_(0, batch_index, per_node)
    counts = torch.bincount(batch_index, minlength=num_graphs).clamp_min(1)
    return (per_graph / counts.to(per_graph.dtype)).mean()


def main() -> None:
    torch.manual_seed(7)

    # Two packed point clouds with fixed node correspondence between x0 and x1.
    ptr = torch.tensor([0, 5, 12], dtype=torch.long)
    counts = ptr[1:] - ptr[:-1]
    batch_index = torch.repeat_interleave(torch.arange(2), counts)
    node_features = torch.randn(12, 8)
    x0 = center_per_graph(torch.randn(12, 3), batch_index, num_graphs=2)
    x1 = center_per_graph(torch.randn(12, 3), batch_index, num_graphs=2)

    time = torch.rand(2, 1)
    time_per_node = time[batch_index]
    x_t = (1.0 - time_per_node) * x0 + time_per_node * x1
    target_velocity = x1 - x0

    model = ELA(
        input_irreps="8x0e",
        output_irreps="1x1o",
        condition_dim=1,
        width=32,
        depth=2,
        cutoff=5.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    batch = ELABatch(
        node_irreps=node_features,
        positions=x_t,
        ptr=ptr,
        condition=time,
    )

    output = model(batch)
    velocity = model.split_output(output["node"])["1o"].squeeze(-2)
    loss = graph_balanced_velocity_loss(
        velocity,
        target_velocity,
        batch.batch,
        batch.num_graphs,
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # For an Euler sampler, displacement is dt * velocity, not velocity itself.
    dt = 0.01
    x_next = x_t + dt * velocity.detach()
    print(
        f"loss={float(loss.detach()):.6f}, "
        f"velocity_shape={tuple(velocity.shape)}, "
        f"next_shape={tuple(x_next.shape)}"
    )


if __name__ == "__main__":
    main()
