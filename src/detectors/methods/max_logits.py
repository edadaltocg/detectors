import torch
from torch import Tensor


@torch.no_grad()
def max_logits(input: Tensor, model: torch.nn.Module, **kwargs) -> Tensor:
    """Max Logits [https://arxiv.org/abs/1911.11132] OOD detector.

    Args:
        logits (Tensor): input tensor.

    Returns:
        Tensor: OOD scores for each input.
    """
    model.eval()
    logits = model(input)
    return torch.max(logits, dim=1)[0]
