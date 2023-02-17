import torch
from torch import Tensor, nn


def energy(x: Tensor, model: nn.Module, temperature: float = 1.0, *args, **kwargs):
    """https://arxiv.org/pdf/2010.03759.pdf"""
    model.eval()
    with torch.no_grad():
        logits = model(x)
    return temperature * torch.logsumexp(logits / temperature, dim=-1)
