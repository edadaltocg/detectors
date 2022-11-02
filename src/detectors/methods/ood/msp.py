import torch
from torch import Tensor


def msp(input: Tensor, model: torch.nn.Module, *args, **kwargs) -> Tensor:
    """
    Compute the Maximum Softmax Response
    """
    device = input.device
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(input)
        probs = torch.nn.functional.softmax(logits, dim=1)
        return torch.max(probs, dim=1)[0]
