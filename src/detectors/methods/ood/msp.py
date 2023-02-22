import torch
from torch import Tensor


@torch.no_grad()
def msp(input: Tensor, model: torch.nn.Module, *args, **kwargs) -> Tensor:
    """
    Compute the Maximum Softmax Response
    """
    model.eval()
    logits = model(input)
    probs = torch.softmax(logits, dim=1)
    return torch.max(probs, dim=1)[0]
