import torch
from torch import Tensor


@torch.no_grad()
def argm(input: Tensor, model: torch.nn.Module, **kwargs) -> Tensor:
    """Argmax.

    Args:
        input (Tensor): input tensor.
        model (nn.Module): classifier.

    Returns:
        Tensor: OOD scores for each input.
    """
    model.eval()
    logits = model(input)
    am = torch.argmax(logits, dim=1)
    return am
