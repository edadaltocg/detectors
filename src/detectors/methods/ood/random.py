import torch
from torch import Tensor


def random_score(input: Tensor, *args, **kwargs) -> Tensor:
    return torch.rand((input.shape[0],))
