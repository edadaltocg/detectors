import logging
from typing import List, Literal, Optional

import torch
from torch import Tensor, nn

from detectors.methods.templates import DetectorWithFeatureExtraction

from .utils import sklearn_cov_matrix_estimarion

_logger = logging.getLogger(__name__)


def mahalanobis_dist_forward_substitution(x: Tensor, y: Tensor, L: Tensor):
    return torch.sqrt(torch.sum(torch.square(torch.mm(x, L).unsqueeze(1) - torch.mm(y, L).unsqueeze(0)), dim=-1)).min(
        dim=1, keepdim=True
    )[0]


def mahalanobis_distance_inv_fast(x: Tensor, y: Tensor, precision: Tensor):
    """Mahalanobis distance betwee x and y with an accelerated implementation.

    Args:
        x (Tensor): first point.
        y (Tensor): second point.
        precision (Tensor): inverse of the covariance matrix.
    """
    d_squared = torch.mm(torch.mm(x - y, precision), (x - y).T).diag()
    return torch.sqrt(d_squared)


def mahalanobis_distance_inv(x: Tensor, y: Tensor, precision: Tensor):
    """Mahalanobis distance betwee x and y.

    Args:
        x (Tensor): first point.
        y (Tensor): second point.
        precision (Tensor): inverse of the covariance matrix.
    """

    d_squared = torch.sum((x - y).T * torch.mm(precision, (x - y).T), dim=0)
    return torch.sqrt(d_squared)


def mahalanobis_inv_layer_score(x: Tensor, mus: Tensor, inv: Tensor) -> Tensor:
    stack = torch.zeros((x.shape[0], mus.shape[0]), device=x.device, dtype=x.dtype)
    for i, mu in enumerate(mus):
        stack[:, i] = mahalanobis_distance_inv(x, mu.reshape(1, -1), inv).reshape(-1)

    return -torch.nan_to_num(stack.min(1, keepdim=True)[0], nan=1e6)


def mahalanobis_inv_layer_score_fast(x: Tensor, mus: Tensor, inv: Tensor) -> Tensor:
    stack = torch.zeros((x.shape[0], mus.shape[0]), device=x.device, dtype=x.dtype)
    for i, mu in enumerate(mus):
        stack[:, i] = mahalanobis_distance_inv_fast(x, mu.reshape(1, -1), inv).reshape(-1)

    return -torch.nan_to_num(stack.min(1, keepdim=True)[0], nan=1e6)


def class_cond_mus_cov_inv_matrix(
    x: Tensor, targets: Tensor, cov_method: str = "EmpiricalCovariance", device=torch.device("cpu")
):
    class_cond_mean = {}
    centered_data_per_class = {}
    unique_classes = sorted(torch.unique(targets.detach().cpu()).numpy().tolist())
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

    return mus, cov_mat, inv_mat


HYPERPARAMETERS = dict(
    cov_mat_method=[
        "EmpiricalCovariance",
        "GraphicalLasso",
        "GraphicalLassoCV",
        "LedoitWolf",
        "ShrunkCovariance",
        "OAS",
    ]
)


class Mahalanobis(DetectorWithFeatureExtraction):
    """Mahalanobis OOD detector.

    Args:
        model (nn.Module): Model to be used to extract features.
        features_nodes (Optional[List[str]]): List of strings that represent the feature nodes.
            Defaults to None.
        all_blocks (bool, optional): If True, use all blocks of the model. Defaults to False.
        last_layer (bool, optional): If True, use also the last layer of the model. Defaults to False.
        pooling_op_name (str, optional): Pooling operation to be applied to the features.
            Can be one of `max`, `avg`, `flatten`, `getitem`, `avg_or_getitem`, `max_or_getitem`, `none`. Defaults to `avg`.
        aggregation_method_name (str, optional): Aggregation method to be applied to the features. Defaults to None.
        cov_mat_method (str, optional): Covariance matrix estimation method. Can be one of:
            `EmpiricalCovariance`, `GraphicalLasso`, `GraphicalLassoCV`, `LedoitWolf`, `MinCovDet`, `ShrunkCovariance`, `OAS`.
            Defaults to `EmpiricalCovariance`.
        mu_cov_inv_est_fn (function, optional): Function to be used to estimate the means, covariance and inverse matrix.
            Defaults to `class_cond_mus_cov_inv_matrix`.
        cov_reg (float, optional): Covariance regularization. Defaults to 1e-6.

    References:
        [1] https://arxiv.org/abs/1807.03888
    """

    def __init__(
        self,
        model: nn.Module,
        features_nodes: Optional[List[str]] = None,
        all_blocks: bool = False,
        last_layer: bool = False,
        pooling_op_name: str = "avg_or_getitem",
        aggregation_method_name: Optional[str] = "mean",
        cov_mat_method: Literal[
            "EmpiricalCovariance",
            "GraphicalLasso",
            "GraphicalLassoCV",
            "LedoitWolf",
            "MinCovDet",
            "ShrunkCovariance",
            "OAS",
        ] = "EmpiricalCovariance",
        mu_cov_inv_est_fn=class_cond_mus_cov_inv_matrix,
        cov_reg: float = 1e-6,
        **kwargs,
    ):
        super().__init__(
            model, features_nodes, all_blocks, last_layer, pooling_op_name, aggregation_method_name, **kwargs
        )
        self.cov_mat_method = cov_mat_method
        self.mu_cov_inv_est_fn = mu_cov_inv_est_fn
        self.cov_reg = cov_reg

    def _layer_score(self, x: Tensor, layer_name: Optional[str] = None, index: Optional[int] = None):
        return mahalanobis_inv_layer_score(
            x, self.mus[layer_name].to(x.device), self.precision_chols[layer_name].to(x.device)
        )

    def _fit_params(self) -> None:
        self.mus = {}
        self.invs = {}
        self.precision_chols = {}
        device = next(self.model.parameters()).device
        for layer_name, layer_features in self.train_features.items():
            self.mus[layer_name], cov, self.invs[layer_name] = self.mu_cov_inv_est_fn(
                layer_features, self.train_targets, self.cov_mat_method, device=device
            )
            cov_chol = torch.linalg.cholesky(cov.to(device) + self.cov_reg * torch.eye(cov.shape[1], device=device))
            self.precision_chols[layer_name] = torch.linalg.solve_triangular(
                cov_chol, torch.eye(cov_chol.shape[1], device=device), upper=False
            ).T