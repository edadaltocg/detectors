import torch
from torch import Tensor, nn


def energy(x: Tensor, model: nn.Module, temperature: float = 1, *args, **kwargs):
    """https://arxiv.org/pdf/2010.03759.pdf"""
    model.eval()
    with torch.no_grad():
        logits = model(x)
    return temperature * torch.logsumexp(logits / temperature, dim=-1)


if __name__ == "__main__":
    batch_size = 2
    x = torch.randn(batch_size, 3, 32, 32, requires_grad=True)
    targets = torch.randint(0, 10, (batch_size,), dtype=torch.long, requires_grad=False)

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 6, 5)
            self.linear = torch.nn.Linear(6 * 28 * 28, 10)

        def forward(self, x: Tensor) -> Tensor:
            x = self.conv1(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            return x

    model = Model()
    scores = energy(x, model, temperature=1, eps=0.0)
    assert scores.shape == (batch_size,)
