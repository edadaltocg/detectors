import logging
from functools import partial
from typing import List, Literal, Optional

import numpy as np
import torch
from torch import Tensor, nn
from torchvision.models.feature_extraction import create_feature_extractor

_logger = logging.getLogger(__name__)


def flatten(data: Tensor, *args, **kwargs):
    return torch.flatten(data, 1)


def adaptive_avg_pool2d(data: Tensor, *args, **kwargs):
    if len(data.shape) > 2:
        return torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(data), 1)
    return data


def adaptive_max_pool2d(data: Tensor, *args, **kwargs):
    if len(data.shape) > 2:
        return torch.flatten(nn.AdaptiveMaxPool2d((1, 1))(data), 1)
    return data


def getitem(data: Tensor, *args, **kwargs):
    return data[:, 0].clone().contiguous()


def none_reduction(data: Tensor, *args, **kwargs):
    return data


reductions_registry = {
    "flatten": flatten,
    "avg": adaptive_avg_pool2d,
    "max": adaptive_max_pool2d,
    "getitem": getitem,
    "none": none_reduction,
}


def mahalanobis_distance_inv(x: Tensor, y: Tensor, inverse: Tensor):
    return torch.nan_to_num(torch.sqrt(((x - y).T * (inverse @ (x - y).T)).sum(0)), 1e9)


def mahalanobis_inv_layer_score(x: Tensor, mus: Tensor, inv: Tensor):
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
    elif reduction_method == "pseudo":
        return torch.linalg.pinv(sigma)
    elif reduction_method == "inverse":
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
    x: Tensor, targets: Tensor, cov_method="EmpiricalCovariance", inv_method="pseudo", device="cpu"
):
    unique_classes = torch.unique(targets).detach().cpu().numpy().tolist()
    class_cond_cov = {}
    class_cond_mean = {}
    for c in unique_classes:
        filt = targets == c
        if filt.sum() == 0:
            continue
        temp = x[filt]
        mu, cov, inv = sklearn_cov_matrix_estimarion(temp.detach().cpu().numpy(), method=cov_method)
        class_cond_cov[c] = torch.from_numpy(cov).float().to(device)
        class_cond_mean[c] = torch.from_numpy(mu).float().to(device)
    cov_mat = sum(list(class_cond_cov.values())) / x.shape[0]
    inv_mat = torch_reduction_matrix(cov_mat, reduction_method=inv_method)
    mus = torch.vstack(list(class_cond_mean.values()))
    return mus, cov_mat, inv_mat


class Mahalanobis:
    def __init__(
        self,
        model: torch.nn.Module,
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
        *args,
        **kwargs,
    ) -> None:
        self.model = model
        self.features_nodes = features_nodes
        if self.features_nodes is not None:
            self.feature_extractor = create_feature_extractor(self.model, self.features_nodes)
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
            features = {"penultimate": self.model.forward_features(x)}
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
                features = {"penultimate": self.model.forward_features(x)}
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


def inverse_estimation():
    X = torch.rand(100, 10)
    batch_size = 5
    cov_true = torch.cov(X.T)
    inv_true = torch.inverse(cov_true)
    print(inv_true)
    n = X.shape[1]
    lbd = 0.999
    lbd_sum = 0
    inv = torch.eye(n)
    mu = torch.zeros(n)

    for i in range(0, X.shape[0], batch_size):
        batch = X[i : i + batch_size]
        lbd_sum = lbd * lbd_sum + (1 - lbd) * batch.shape[0]
        delta = batch - mu
        mu = mu + delta / lbd_sum
        inv = lbd * inv + delta
        sigma = inv / lbd_sum

        # estimation of the inverse
        num = 1 / (lbd**2)
    return
