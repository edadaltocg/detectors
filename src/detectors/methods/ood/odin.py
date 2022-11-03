import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F


def odin_baseline_implementation(
    x: Tensor, model: nn.Module, temperature: float = 1, eps: float = 0.0, *args, **kwargs
):
    x.requires_grad_(True)
    model.eval()
    if eps > 0:
        outputs = model(x)
        labels = torch.argmax(outputs, dim=1)
        loss = F.cross_entropy(outputs / temperature, labels)
        loss.backward()

        grad_sign = x.grad.data.sign()

        # Adding small perturbations to images
        x = x - eps * grad_sign

    with torch.no_grad():
        outputs = model(x)
    return torch.softmax(outputs / temperature, dim=1).max(dim=1)[0]


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
    scores = odin_baseline_implementation(x, model, temperature=1, eps=0.0)
    assert scores.shape == (batch_size,)
