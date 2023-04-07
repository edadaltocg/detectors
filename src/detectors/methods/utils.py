import logging
from functools import partial
from typing import Callable, Literal

import numpy as np
import torch
from torch import Tensor, nn

_logger = logging.getLogger(__name__)


def input_pre_processing(score_fn: Callable, x: Tensor, eps: float):
    x.requires_grad_()
    assert x.requires_grad
    assert eps >= 0

    scores = score_fn(x)
    scores.mean().backward()
    x = x - eps * torch.sign(-x.grad)
    return x


# Feature reductions
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
    if data.dim() == 3:
        return data[:, 0].clone().contiguous()
    elif data.dim() > 3:
        raise ValueError("Data must be a 3D or 2D tensor")
    return data


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


def get_last_layer_name(model: nn.Module):
    return list(model._modules.keys())[-1]


def get_last_layer(model: nn.Module):
    return list(model._modules.values())[-1]


# matrix operations
def torch_reduction_matrix(sigma: Tensor, reduction_method="pseudo"):
    import torch

    if reduction_method == "cholesky":
        C = torch.linalg.cholesky(sigma)
        return torch.linalg.inv(C.T)
    elif reduction_method == "svd":
        u, s, _ = torch.linalg.svd(sigma)
        return u @ torch.diag(torch.sqrt(1 / s))
    elif reduction_method == "pseudo" or reduction_method == "pinv":
        return torch.linalg.pinv(sigma)
    elif reduction_method == "inverse" or reduction_method == "inv":
        return torch.linalg.inv(sigma)
    else:
        raise ValueError(f"Unknown reduction method {reduction_method}")


def sklearn_cov_matrix_estimarion(
    x: np.ndarray,
    method: Literal[
        "EmpiricalCovariance",
        "GraphicalLasso",
        "GraphicalLassoCV",
        "LedoitWolf",
        "MinCovDet",
        "ShrunkCovariance",
        "OAS",
    ] = "EmpiricalCovariance",
    **method_kwargs,
):
    import sklearn.covariance

    try:
        method = getattr(sklearn.covariance, method)(**method_kwargs)
    except AttributeError:
        raise ValueError(f"Unknown method {method}")

    method.fit(x)
    cov_mat = method.covariance_
    _logger.debug("Cov mat determinant %s", np.linalg.det(cov_mat))
    _logger.debug("Cov mat rank %s", np.linalg.matrix_rank(cov_mat))
    _logger.debug("Cov mat condition number %s", np.linalg.cond(cov_mat))
    _logger.debug("Cov mat norm %s", np.linalg.norm(cov_mat))
    _logger.debug("Cov mat trace %s", np.trace(cov_mat))
    _logger.debug("Cov mat eigvals %s", np.linalg.eigvalsh(cov_mat))
    return method.location_, method.covariance_, method.precision_
