from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from equivariant_attention import ELA


class ToyMoleculeDataset(Dataset[dict[str, torch.Tensor]]):
    """Example variable-size graph dataset using ordinary tensor dictionaries."""

    def __init__(self, size: int = 128, seed: int = 0) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.samples: list[dict[str, torch.Tensor]] = []
        for _ in range(size):
            nodes = int(torch.randint(6, 18, (), generator=generator).item())
            atomic_features = torch.randn(nodes, 8, generator=generator)
            positions = torch.randn(nodes, 3, generator=generator)
            target = (
                atomic_features[:, 0].sum()
                + 0.1 * positions.square().sum().sqrt()
            ).reshape(1)
            # edge_index is deliberately omitted. ELA will construct radius
            # candidates for this small example batch.
            self.samples.append(
                {
                    "x": atomic_features,
                    "pos": positions,
                    "target": target,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    dataset = ToyMoleculeDataset()
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=ELA.collate,
        pin_memory=device.type == "cuda",
    )

    model = ELA.scalar(
        node_dim=8,
        output_dim=1,
        width=64,
        depth=4,
        cutoff=4.5,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss()

    model.train()
    for batch in loader:
        batch = batch.to(
            device,
            dtype=compute_dtype,
            non_blocking=True,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=compute_dtype,
            enabled=device.type == "cuda",
        ):
            output = model(batch)
            loss = loss_fn(
                output["graph_irreps"].float(),
                batch["target"].float(),
            )
        loss.backward()
        optimizer.step()

    print(f"last loss: {float(loss.detach()):.6f}")


if __name__ == "__main__":
    main()
