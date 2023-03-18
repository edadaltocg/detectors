from functools import partial
from typing import Callable

import torch
from torch import Tensor, nn


def input_pre_processing(method: Callable, x: Tensor, eps: float):
    x.requires_grad_()
    assert x.requires_grad
    assert eps >= 0

    scores = method(x)
    scores.sum().backward()
    x = x - eps * torch.sign(-x.grad)
    return x


# Reductions
def flatten(data: Tensor, **kwargs):
    return torch.flatten(data, 1)


def adaptive_avg_pool2d(data: Tensor, **kwargs):
    if len(data.shape) > 2:
        return torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(data), 1)
    return data


def adaptive_max_pool2d(data: Tensor, **kwargs):
    if len(data.shape) > 2:
        return torch.flatten(nn.AdaptiveMaxPool2d((1, 1))(data), 1)
    return data


def getitem(data: Tensor, **kwargs):
    return data[:, 0].clone().contiguous()


def none_reduction(data: Tensor, **kwargs):
    return data


reductions_registry = {
    "flatten": flatten,
    "avg": adaptive_avg_pool2d,
    "max": adaptive_max_pool2d,
    "getitem": getitem,
    "none": none_reduction,
}


def create_reduction(reduction: str, **kwargs):
    return partial(reductions_registry[reduction], **kwargs)


def get_penultimate_layer_name(model: nn.Module):
    return list(model._modules.keys())[-2]


def get_last_layer(model: nn.Module):
    return list(model._modules.values())[-1]
