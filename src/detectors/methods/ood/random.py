import torch
from torch import Tensor


@torch.no_grad()
def random_score(input: Tensor, model: torch.nn.Module, *args, **kwargs) -> Tensor:
    model.eval()
    logits = model(input)
    return torch.rand((logits.shape[0],))
