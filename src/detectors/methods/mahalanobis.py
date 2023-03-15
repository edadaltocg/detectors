import logging
from functools import partial
from typing import List, Literal, Optional

import numpy as np
import torch
from torch import Tensor, nn
from torchvision.models.feature_extraction import create_feature_extractor

_logger = logging.getLogger(__name__)


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


def mahalanobis_distance_inv(x: Tensor, y: Tensor, precision: Tensor):
    """Mahalanobis distance betwee x and y normalized to the interval [0,1].

    Args:
        x (Tensor): first point.
        y (Tensor): second point.
        precision (Tensor): inverse of the covariance matrix.
    """

    d_squared = torch.mm(torch.mm(x - y, precision), (x - y).T).diag()
    return torch.sqrt(d_squared)


def mahalanobis_inv_layer_score(x: Tensor, mus: Tensor, inv: Tensor) -> Tensor:
    stack = torch.zeros((x.shape[0], mus.shape[0]), device=x.device, dtype=torch.float32)
    for i, mu in enumerate(mus):
        stack[:, i] = mahalanobis_distance_inv(x, mu.reshape(1, -1), inv).reshape(-1)

    return -stack.min(1, keepdim=True)[0]


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
    return method.location_, method.covariance_, method.precision_


def class_cond_mus_cov_inv_matrix(
    x: Tensor, targets: Tensor, cov_method: str = "EmpiricalCovariance", inv_method="pseudo", device=torch.device("cpu")
):
    unique_classes = sorted(torch.unique(targets.detach().cpu()).numpy().tolist())
    class_cond_mean = {}
    centered_data_per_class = {}
    for c in unique_classes:
        filt = targets == c
        temp = x[filt].to(device)
        class_cond_mean[c] = temp.mean(0, keepdim=True)
        centered_data_per_class[c] = temp - class_cond_mean[c]

        class_cond_mean[c] = class_cond_mean[c].detach().cpu()
        centered_data_per_class[c] = centered_data_per_class[c].detach().cpu()

    centered_data_per_class = torch.vstack(list(centered_data_per_class.values()))
    mus = torch.vstack(list(class_cond_mean.values()))

    mu, cov_mat, inv_mat = sklearn_cov_matrix_estimarion(centered_data_per_class.numpy(), method=cov_method)
    cov_mat = torch.from_numpy(cov_mat).float()
    inv_mat = torch.from_numpy(inv_mat).float()

    _logger.debug("Cov mat determinant %s", torch.linalg.det(cov_mat))
    _logger.debug("Cov mat rank %s", torch.linalg.matrix_rank(cov_mat))
    _logger.debug("Cov mat condition number %s", torch.linalg.cond(cov_mat))
    _logger.debug("Cov mat norm %s", torch.linalg.norm(cov_mat))
    _logger.debug("Cov mat trace %s", torch.trace(cov_mat))
    _logger.debug("Cov mat eigvals %s", torch.linalg.eigvalsh(cov_mat))

    return mus, cov_mat, inv_mat


class Mahalanobis:
    """`Mahalanobis <MAHALANOBIS_PAPER_URL>` OOD detector.

    Args:
        model (nn.Module): Model to be used to extract features
        features_nodes (Optional[List[str]]): List of strings that represent the feature nodes.
            Defaults to None.
        cov_mat_method (str, optional): Covariance matrix estimation method. Can be one of
            ["EmpiricalCovariance", "GraphicalLasso", "GraphicalLassoCV", "LedoitWolf", "MinCovDet", "ShrunkCovariance", "OAS"].
            Defaults to "EmpiricalCovariance".
        inv_mat_method (str, optional): Inverse matrix estimation method. Can be one of ["cholesky", "svd", "pseudo", "inverse"].
            Defaults to "pseudo".
        pooling_op_name (str, optional): Pooling operation to be applied to the features. Can be one of ["max", "avg", "flatten", "getitem", "none"].
            Defaults to "avg".
        aggregation_method (None, optional): Aggregation method to be applied to the features. Defaults to None.
        mu_cov_inv_est_fn (function, optional): Function to be used to estimate the means, covariance and inverse matrix.
            Defaults to `class_cond_mus_cov_inv_matrix`.
    """

    def __init__(
        self,
        model: nn.Module,
        features_nodes: Optional[List[str]] = None,
        cov_mat_method: Literal[
            "EmpiricalCovariance",
            "GraphicalLasso",
            "GraphicalLassoCV",
            "LedoitWolf",
            "MinCovDet",
            "ShrunkCovariance",
            "OAS",
        ] = "EmpiricalCovariance",
        inv_mat_method: Literal["cholesky", "svd", "pseudo", "inverse"] = "pseudo",
        pooling_op_name: Literal["max", "avg", "flatten", "getitem", "none"] = "avg",
        aggregation_method=None,
        mu_cov_inv_est_fn=class_cond_mus_cov_inv_matrix,
        **kwargs,
    ) -> None:
        self.model = model
        self.model.eval()
        self.features_nodes = features_nodes
        if self.features_nodes is not None:
            self.feature_extractor = create_feature_extractor(self.model, self.features_nodes)
        else:
            if not hasattr(self.model, "forward_features"):
                raise ValueError(
                    "Model does not have a forward_features method. "
                    "Please provide a list of feature nodes to extract features from."
                )
            self.feature_extractor: nn.Module = self.model.forward_features  # type: ignore
        self.reduction_method = inv_mat_method
        self.aggregation_method = aggregation_method
        if aggregation_method is not None and features_nodes is not None and len(features_nodes) > 1:
            _logger.warning("Disabling aggregation method because only one feature is used.")
            self.aggregation_method = None

        self.pooling_name = pooling_op_name
        self.pooling_op = reductions_registry[pooling_op_name]
        self.device = next(self.model.parameters()).device
        self.mu_cov_inv_est_fn = partial(mu_cov_inv_est_fn, cov_method=cov_mat_method, inv_method=inv_mat_method)

        self.mus = []
        self.invs = []

        self.training_features = {}

    def start(self, *args, **kwargs):
        self.training_features = {}

    @torch.no_grad()
    def update(self, x: Tensor, y: Tensor) -> None:
        if self.features_nodes is None:
            features = {"penultimate": self.feature_extractor(x)}
        else:
            features = self.feature_extractor(x)

        for k in features:
            features[k] = self.pooling_op(features[k])

        # accumulate training features
        if len(self.training_features) == 0:
            for k in features:
                self.training_features[k] = features[k].cpu()
            self.training_features["targets"] = y.cpu()
        else:
            for k in features:
                self.training_features[k] = torch.cat((self.training_features[k], features[k].cpu()), dim=0)
            self.training_features["targets"] = torch.cat((self.training_features["targets"], y.cpu()), dim=0)

    def end(self, *args, **kwargs):
        _logger.info("Computing inverse matrix.")
        for k in self.training_features:
            if k == "targets":
                continue

            mu, cov, inv = self.mu_cov_inv_est_fn(
                self.training_features[k], self.training_features["targets"], device=self.device
            )
            self.mus.append(mu.to(self.device))
            self.invs.append(inv.to(self.device))

        del self.training_features

    def __call__(self, x: Tensor) -> Tensor:
        if len(self.invs) == 0 or len(self.mus) == 0:
            raise ValueError("You must properly fit the Mahalanobis method first.")

        with torch.no_grad():
            if self.features_nodes is None:
                features = {"penultimate": self.feature_extractor(x)}
            else:
                features = self.feature_extractor(x)

        for k in features:
            features[k] = self.pooling_op(features[k])

        features_keys = list(features.keys())
        stack = None
        for k, mu, inv in zip(features_keys, self.mus, self.invs):
            device = features[k].device
            scores = mahalanobis_inv_layer_score(features[k], mu.to(device), inv.to(device))
            if stack is None:
                stack = scores
            else:
                stack = torch.cat((stack, scores), dim=1)  # type: ignore

        if stack is None:
            raise ValueError("Stack is None, this should not happen.")

        if stack.shape[1] > 1 and self.aggregation_method is None:
            stack = stack.mean(1, keepdim=True)
        elif stack.shape[1] > 1 and self.aggregation_method is not None:
            stack = self.aggregation_method(stack)

        return stack.view(-1)