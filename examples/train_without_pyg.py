from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from equivariant_linear_attention import ELA, ELAGraph


class ToyMoleculeDataset(Dataset[ELAGraph]):
    """Variable-size examples using the same object accepted by the model."""

    def __init__(self, size: int = 128, seed: int = 0) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.samples: list[ELAGraph] = []
        for sample_id in range(size):
            nodes = int(torch.randint(6, 18, (), generator=generator).item())
            features = torch.randn(nodes, 8, generator=generator)
            positions = torch.randn(nodes, 3, generator=generator)
            target = (
                features[:, 0].sum() + 0.1 * positions.square().sum().sqrt()
            ).reshape(1)
            # edge_index is omitted: ELA constructs the radius graph internally.
            self.samples.append(
                ELAGraph(
                    x=features,
                    pos=positions,
                    y=target,
                    ids=(sample_id,),
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ELAGraph:
        return self.samples[index]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16 if device.type == "cuda" else torch.float32
    )
    loader = DataLoader(
        ToyMoleculeDataset(),
        batch_size=16,
        shuffle=True,
        collate_fn=ELAGraph.collate,
        pin_memory=device.type == "cuda",
    )

    model = ELA(
        "8x0e",
        "1x0e",
        width=64,
        depth=4,
        cutoff=4.5,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss()

    model.train()
    for graph in loader:
        graph = graph.to(device, non_blocking=True)
        if graph.y is None:
            raise RuntimeError("training graph is missing y")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=compute_dtype,
            enabled=device.type == "cuda",
        ):
            output = model(graph)
            if output.graph_x is None:
                raise RuntimeError("model did not produce graph predictions")
            loss = loss_fn(output.graph_x.float(), graph.y.float())
        loss.backward()
        optimizer.step()

    print(f"last loss: {float(loss.detach()):.6f}")


if __name__ == "__main__":
    main()
