import logging
from typing import Optional
from torch import Tensor

import torch

from .mahalanobis import Mahalanobis, mahalanobis_distance_inv

_logger = logging.getLogger(__name__)


def relative_mahalanobis_inv_layer_score(
    x: Tensor, mus: Tensor, inv: Tensor, background_mu: Tensor, background_inv: Tensor
):
    stack = torch.zeros((x.shape[0], mus.shape[0]), device=x.device, dtype=torch.float32)
    for i, mu in enumerate(mus):
        stack[:, i] = mahalanobis_distance_inv(x, mu.reshape(1, -1), inv).reshape(-1) - mahalanobis_distance_inv(
            x, background_mu.reshape(1, -1), background_inv
        ).reshape(-1)

    return -torch.nan_to_num(stack.min(1, keepdim=True)[0], nan=1e6)


class RelativeMahalanobis(Mahalanobis):
    """`RelativeMahalanobis <PAPER_URL>` detector.

    Args:
        model (nn.Module): Model to be used to extract features
        features_nodes (Optional[List[str]]): List of strings that represent the feature nodes.
            Defaults to None.
        all_blocks (bool, optional): If True, use all blocks of the model. Defaults to False.
        last_layer (bool, optional): If True, use also the last layer of the model. Defaults to False.
        pooling_op_name (str, optional): Pooling operation to be applied to the features. Can be one of ["max", "avg", "flatten", "getitem", "none"].
            Defaults to "avg".
        aggregation_method_name (str, optional): Aggregation method to be applied to the features. Defaults to None.
        cov_mat_method (str, optional): Covariance matrix estimation method. Can be one of
            ["EmpiricalCovariance", "GraphicalLasso", "GraphicalLassoCV", "LedoitWolf", "MinCovDet", "ShrunkCovariance", "OAS"].
            Defaults to "EmpiricalCovariance".
        mu_cov_inv_est_fn (function, optional): Function to be used to estimate the means, covariance and inverse matrix.
            Defaults to `class_cond_mus_cov_inv_matrix`.
        cov_reg (float, optional): Covariance regularization. Defaults to 1e-6.
        **kwargs
    """

    def _layer_score(self, x: Tensor, layer_name: Optional[str] = None, index: Optional[int] = None):
        return relative_mahalanobis_inv_layer_score(
            x,
            self.mus[layer_name].to(x.device),
            self.invs[layer_name].to(x.device),
            self.background_mus[layer_name].to(x.device),
            self.background_invs[layer_name].to(x.device),
        )

    def _fit_params(self) -> None:
        self.mus = {}
        self.invs = {}
        self.background_mus = {}
        self.background_invs = {}
        device = next(self.model.parameters()).device
        for layer_name, layer_features in self.train_features.items():
            self.mus[layer_name], cov, self.invs[layer_name] = self.mu_cov_inv_est_fn(
                layer_features, self.train_targets, self.cov_mat_method, device=device
            )
            self.background_mus[layer_name] = torch.mean(layer_features, dim=0, keepdim=True).to(device)
            self.background_invs[layer_name] = torch.pinverse(torch.cov(layer_features.T).to(device))
