import torch
import torch.nn.functional as F
from torch import Tensor, nn


def odin(x: Tensor, model: nn.Module, temperature: float = 1, eps: float = 0.0, *args, **kwargs):
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
