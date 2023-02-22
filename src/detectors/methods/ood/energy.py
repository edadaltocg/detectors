import torch
from torch import Tensor, nn


@torch.no_grad()
def energy(x: Tensor, model: nn.Module, temperature: float = 1.0, *args, **kwargs):
    """https://arxiv.org/pdf/2010.03759.pdf"""
    model.eval()
    logits = model(x)
    return temperature * torch.logsumexp(logits / temperature, dim=-1)
