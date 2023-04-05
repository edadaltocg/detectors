import torch
from torch import Tensor


def logit_norm(model, x: Tensor, p="fro", **kwargs):
    with torch.no_grad():
        logits = model(x)
    return torch.norm(logits, p=p, dim=-1)
