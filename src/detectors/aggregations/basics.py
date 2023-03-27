import torch
from torch import Tensor


def mean_aggregation(x: Tensor, *args, **kwargs) -> Tensor:
    return x.mean(dim=1, keepdim=True)


def median_aggregation(x: Tensor, *args, **kwargs) -> Tensor:
    return x.median(dim=1, keepdim=True)[0]


def max_aggregation(x: Tensor, *args, **kwargs) -> Tensor:
    return x.max(dim=1, keepdim=True)[0]


def min_aggregation(x: Tensor, *args, **kwargs) -> Tensor:
    return x.min(dim=1, keepdim=True)[0]


def avg_topk_aggregation(x: Tensor, k: int, *args, **kwargs) -> Tensor:
    return x.topk(k, dim=1)[0].mean(dim=1, keepdim=True)


def topk_aggregation(x: Tensor, k: int, *args, **kwargs) -> Tensor:
    return x.topk(k, dim=1)[0]


def layer_idx(x: Tensor, layer_idx: int = -1, *args, **kwargs) -> Tensor:
    return x[:, layer_idx]


def depth_weighted_sum(x: Tensor, *args, **kwargs) -> Tensor:
    w = torch.arange(0, x.shape[1], 1, device=x.device, dtype=x.dtype) / x.shape[1]
    x = x * w
    return x.sum(dim=1, keepdim=True)
