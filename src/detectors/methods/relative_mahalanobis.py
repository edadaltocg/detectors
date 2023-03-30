import logging
from functools import partial
from typing import List, Literal, Optional
from .mahalanobis import (
    Mahalanobis,
    class_cond_mus_cov_inv_matrix,
    mahalanobis_distance_inv,
    mahalanobis_inv_layer_score,
    mahalanobis_inv_layer_score_fast,
)

import numpy as np
import torch
from torch import Tensor, nn
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm


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
    def __init__(
        self,
        model: nn.Module,
        features_nodes: Optional[List[str]] = None,
        all_blocks: bool = False,
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
        aggregation_method_name=None,
        mu_cov_inv_est_fn=class_cond_mus_cov_inv_matrix,
        **kwargs,
    ) -> None:
        super().__init__(
            model,
            features_nodes,
            all_blocks,
            cov_mat_method,
            inv_mat_method,
            pooling_op_name,
            aggregation_method_name,
            mu_cov_inv_est_fn,
            **kwargs,
        )
        self.background_mus = []
        self.background_invs = []

    def start(self, *args, **kwargs):
        self.training_features = {}
        self.mus = []
        self.invs = []
        self.background_mus = []
        self.background_invs = []

    def end(self, *args, **kwargs):
        _logger.info("Computing inverse matrix.")
        targets = self.training_features.pop("targets")
        for k in self.training_features:
            _logger.info("Training features shape for key %s is %s", k, self.training_features[k].shape)
            mu, cov, inv = self.mu_cov_inv_est_fn(self.training_features[k], targets, device=self.device)
            self.mus.append(mu.to(self.device))
            self.invs.append(inv.to(self.device))
            self.background_mus.append(torch.mean(self.training_features[k], dim=0, keepdim=True).to(self.device))
            self.background_invs.append(torch.pinverse(torch.cov(self.training_features[k].T).to(self.device)))

        if self.aggregation_method is not None and hasattr:
            _logger.info("Fitting aggregator %s...", self.aggregation_method_name)
            all_scores = []
            for i, k in enumerate(self.training_features):
                train_scores = []
                self.batch_size = self.training_features[k].shape[0]
                idx = 0
                for idx in tqdm(range(0, self.training_features[k].shape[0], self.batch_size)):
                    score = relative_mahalanobis_inv_layer_score(
                        self.training_features[k][idx : idx + self.batch_size].to(self.device),
                        self.mus[i].to(self.device),
                        self.invs[i].to(self.device),
                        self.background_mus[i].to(self.device),
                        self.background_invs[i].to(self.device),
                    )
                    train_scores.append(score)
                train_scores = torch.cat(train_scores, dim=0)
                all_scores.append(train_scores.view(-1, 1))
            stack = torch.cat(all_scores, dim=1)
            self.aggregation_method.fit(stack, targets)

        del self.training_features

    def __call__(self, x: Tensor) -> Tensor:
        self.feature_extractor = self.feature_extractor.to(x.device)
        if len(self.invs) == 0 or len(self.mus) == 0:
            raise ValueError("You must properly fit the Mahalanobis method first.")

        with torch.no_grad():
            features = self.feature_extractor(x)
            if not isinstance(features, dict):
                features = {"penultimate": features}

        for k in features:
            features[k] = self.pooling_op(features[k])

        features_keys = list(features.keys())
        stack = None
        for k, mu, inv, bmu, binv in zip(features_keys, self.mus, self.invs, self.background_mus, self.background_invs):
            device = features[k].device
            scores = relative_mahalanobis_inv_layer_score(
                features[k], mu.to(device), inv.to(device), bmu.to(device), binv.to(device)
            ).view(-1, 1)
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
