import torch
import torch.nn.functional as F
from torch import Tensor, nn

DefaultConfig = dict(temperature=1.3, eps=0.0)


def doctor(x: Tensor, model: nn.Module, temperature: float = 1000, eps: float = 0.0, **kwargs) -> Tensor:
    """`Doctor <DOCTOR_PAPER_URL>` detector.

    Args:
        x (Tensor): input tensor.
        model (nn.Module): classifier.
        temperature (float, optional): softmax temperature parameter. Defaults to 1000.
        eps (float, optional): input preprocessing noise value. Defaults to 0.0 (no input preprocessing).

    Returns:
        Tensor: scores for each input.
    """
    model.eval()
    if eps > 0:
        x.requires_grad_()
        # input preprocessing
        outputs = model(x)
        labels = torch.argmax(outputs, dim=1)
        loss = torch.mean(1 - torch.softmax(outputs / temperature, dim=1).square().sum(dim=1))
        loss.backward()

        grad_sign = x.grad.data.sign()
        # Adding small perturbations to images
        x = x - eps * grad_sign

    with torch.no_grad():
        outputs = model(x)
    return 1 - torch.softmax(outputs / temperature, dim=1).square().sum(dim=1)
