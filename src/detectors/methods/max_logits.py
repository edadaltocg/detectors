import torch
from torch import Tensor


@torch.no_grad()
def max_logits(input: Tensor, model: torch.nn.Module, **kwargs) -> Tensor:
    """Max Logits OOD detector.

    Args:
        logits (Tensor): input tensor.

    Returns:
        Tensor: OOD scores for each input.

    References:
        [1] https://arxiv.org/abs/1911.11132
    """
    model.eval()
    logits = model(input)
    return torch.max(logits, dim=1)[0]
