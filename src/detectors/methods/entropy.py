import torch
from torch import Tensor, nn


def entropy(x: Tensor, model: nn.Module, **kwargs) -> Tensor:
    with torch.no_grad():
        logits = model(x)
    return torch.sum(-torch.softmax(logits, dim=1) * torch.log_softmax(logits, dim=1), dim=1)
