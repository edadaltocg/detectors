import torch
from torch import Tensor


def random_score(input: Tensor, **kwargs) -> Tensor:
    return torch.rand((input.shape[0],), device=input.device)


def always_one(input: Tensor, **kwargs) -> Tensor:
    return torch.ones((input.shape[0],), device=input.device)


def always_zero(input: Tensor, **kwargs) -> Tensor:
    return torch.zeros((input.shape[0],), device=input.device)
