import torch
import torch.nn.functional as F
from torch import Tensor, nn


def odin(x: Tensor, model: nn.Module, temperature: float = 1000, eps: float = 0.0, **kwargs) -> Tensor:
    """`ODIN <ODIN_PAPER_URL>` OOD detector.

    Args:
        x (Tensor): input tensor.
        model (nn.Module): classifier.
        temperature (float, optional): softmax temperature parameter. Defaults to 1000.
        eps (float, optional): input preprocessing noise value. Defaults to 0.0 (no input preprocessing).

    Returns:
        Tensor: OOD scores for each input.
    """
    x = x.requires_grad_()
    model.eval()
    if eps > 0:
        # input preprocessing
        outputs = model(x)
        labels = torch.argmax(outputs, dim=1)
        loss = F.nll_loss(outputs / temperature, labels)
        loss.backward()

        grad_sign = x.grad.data.sign()
        # Adding small perturbations to images
        x = x - eps * grad_sign

    with torch.no_grad():
        outputs = model(x)
    return torch.softmax(outputs / temperature, dim=1).max(dim=1)[0]
