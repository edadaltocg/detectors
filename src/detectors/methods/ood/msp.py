import torch
from torch import Tensor


def msp(input: Tensor, model: torch.nn.Module, *args, **kwargs) -> Tensor:
    """
    Compute the Maximum Softmax Response
    """
    device = input.device
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(input)
        probs = torch.softmax(logits, dim=1)
    return torch.max(probs, dim=1)[0]


if __name__ == "__main__":
    x = torch.randn(10, 3, 32, 32)

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
    print(msp(x, model))
