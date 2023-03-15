import torch
from torch import Tensor


def mean_aggregation(x: Tensor, *args, **kwargs) -> Tensor:
    return x.mean(dim=1, keepdim=True)


def max_aggregation(x: Tensor, *args, **kwargs) -> Tensor:
    return x.max(dim=1, keepdim=True)[0]


def min_aggregation(x: Tensor, *args, **kwargs) -> Tensor:
    return x.min(dim=1, keepdim=True)[0]


def avg_topk_aggregation(x: Tensor, k: int, *args, **kwargs) -> Tensor:
    return x.topk(k, dim=1)[0].mean(dim=1, keepdim=True)


def topk_aggregation(x: Tensor, k: int, *args, **kwargs) -> Tensor:
    return x.topk(k, dim=1)[0]
