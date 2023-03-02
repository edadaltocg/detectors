import torch
from torch import Tensor


@torch.no_grad()
def msp(input: Tensor, model: torch.nn.Module, **kwargs) -> Tensor:
    """
    `Maximum Softmax Response <MSP_PAPER_URL>` OOD detector.

    Args:
        input (Tensor): input tensor.
        model (nn.Module): classifier.

    Returns:
        Tensor: OOD scores for each input.
    """
    model.eval()
    logits = model(input)
    probs = torch.softmax(logits, dim=1)
    return torch.max(probs, dim=1)[0]
